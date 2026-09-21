# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None

logger = logging.get_logger(__name__)


def sdpa_causal(q, k, v, window_size):
    """Dense causal attention through torch SDPA, for stacks without flash-attn.

    Covers only what flash_attn_func's dense call does; the varlen paths (padded
    batches, cu_seqlens) still require flash-attn and raise below. q/k/v arrive
    as (b, l, h, d) and SDPA wants (b, h, l, d).
    """
    if window_size is not None:
        raise NotImplementedError(
            "Sliding-window attention requires flash-attn; the SDPA fallback "
            "implements full causal attention only.")

    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    q_len, k_len = q.shape[2], k.shape[2]

    # SDPA's is_causal aligns the mask top-left, flash-attn aligns it
    # bottom-right. They agree only when q_len == k_len; a single query step
    # against a filled cache needs no mask at all. Anything else would be
    # silently wrong, so refuse it rather than guess.
    if q_len == k_len:
        is_causal = True
    elif q_len == 1:
        is_causal = False
    else:
        raise NotImplementedError(
            f"SDPA fallback cannot mask q_len={q_len} against k_len={k_len}; "
            "install flash-attn for chunked prefill against a cache.")

    kwargs = {"enable_gqa": True} if q.shape[1] != k.shape[1] else {}
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, **kwargs)
    return o.transpose(1, 2)


class Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        output_gate: bool = False,
        use_rope: bool = True,
        window_size: int | None = None,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.output_gate = output_gate
        self.use_rope = use_rope

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            logger.warning_once(
                "Flash Attention is not installed; falling back to torch SDPA for dense "
                "causal attention. Varlen paths (padding masks, cu_seqlens) still require it.")

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Qwen3-Next's output gate: a sigmoid computed from the LAYER INPUT,
        # applied elementwise to the attention output before o_proj. It is the
        # variant the current hybrids use (Qwen3-Next, Kimi), and costs one
        # hidden_size x hidden_size matrix per attention layer.
        #
        # Before o_proj and not after: gating the projected output is a
        # different operation -- it scales the residual contribution rather
        # than the per-head mixture, and the two are not equivalent because
        # o_proj mixes heads.
        if output_gate:
            self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        # NoPE when `use_rope` is False: no rotary at all, as in Kimi's hybrid.
        # The attention layers of a hybrid do not need to carry position when
        # the linear layers already do -- a gated-delta recurrence is
        # order-dependent by construction, so position reaches the attention
        # layers through the residual stream rather than through a rotation.
        # It is parameter-free either way, so the parameter match is
        # unaffected.
        self.rotary = (RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
                       if use_rope else None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        if self.rotary is not None:
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Contains at least one padding token in the sequence
        if cu_seqlens is None and attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            if flash_attn_varlen_func is None:
                raise ImportError(
                    "Attention over a padding mask needs flash-attn; the SDPA fallback "
                    "covers dense causal attention only.")
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            if flash_attn_varlen_func is None:
                raise ImportError(
                    "Variable-length attention (cu_seqlens / intra-doc masking) needs "
                    "flash-attn; the SDPA fallback covers dense causal attention only.")
            o = flash_attn_varlen_func(
                rearrange(q, 'b l ... -> (b l) ...').contiguous(),
                rearrange(k, 'b l ... -> (b l) ...').contiguous(),
                rearrange(v, 'b l ... -> (b l) ...').contiguous(),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
            o = rearrange(o, '(b l) h d -> b l h d', b=batch_size, l=q_len)
        elif flash_attn_func is not None:
            o = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0),
            )
        else:
            o = sdpa_causal(q, k, v, self.window_size)
        o = o.reshape(batch_size, q_len, -1)
        if self.output_gate:
            o = o * torch.sigmoid(self.g_proj(hidden_states))
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
