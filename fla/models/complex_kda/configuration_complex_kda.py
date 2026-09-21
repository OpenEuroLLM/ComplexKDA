# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers.configuration_utils import PretrainedConfig

from fla.models.hybrid import HybridAttentionConfig, _HybridAttentionConfigMixin


class ComplexKDAConfig(_HybridAttentionConfigMixin, PretrainedConfig):
    model_type = 'complex_kda'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        attn_mode: str = "chunk",
        hidden_size: int = 2048,
        expand_v: float = 1.0,
        use_short_conv: bool = True,
        drop_silu: bool = False,
        drop_key_silu: bool = False,
        allow_neg_eigval: bool = True,
        gate: str = "signed_sigmoid2",
        gate_init_style: str = "shipped",
        # "lowrank" (fla's and Kimi Linear's factored output gate) or "linear"
        # (Kimi K3's full-rank one). Default stays lowrank so checkpoints and
        # runs predating this keep loading; the study's arms set it explicitly.
        output_gate: str = "lowrank",
        conv_silu: str = "qkv",
        beta_init_style: str = "standard",
        lower_bound: float = -5.0,
        conv_size: int = 4,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        num_hidden_layers: int = 24,
        norm_eps: float = 1e-6,
        attn: HybridAttentionConfig = None,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        attnres_block_size: int | None = None,
        **kwargs,
    ):
        self.attn_mode = attn_mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v
        self.use_short_conv = use_short_conv
        self.drop_silu = drop_silu
        self.drop_key_silu = drop_key_silu
        self.conv_size = conv_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.norm_eps = norm_eps
        self.attn = attn
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.allow_neg_eigval = allow_neg_eigval
        self.gate = gate
        self.gate_init_style = gate_init_style
        self.output_gate = output_gate
        self.conv_silu = conv_silu
        self.beta_init_style = beta_init_style
        self.lower_bound = lower_bound
        self.attnres_block_size = attnres_block_size

        if output_gate not in ("lowrank", "linear"):
            raise ValueError(f"`output_gate` must be `lowrank` or `linear`, got {output_gate!r}.")
        if beta_init_style not in ("standard", "spread"):
            raise ValueError(f"`beta_init_style` must be `standard` or `spread`, got {beta_init_style!r}.")
        if beta_init_style == "spread" and not allow_neg_eigval:
            raise ValueError("`beta_init_style='spread'` requires `allow_neg_eigval=True`.")

        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in [-5, 0), got {lower_bound}.")

        if attnres_block_size is not None and attnres_block_size != 1:
            if attnres_block_size < 2 or attnres_block_size % 2 != 0:
                raise ValueError(
                    "`attnres_block_size` must be `None`, `1` (full mode), or an even integer; "
                    f"got {attnres_block_size}."
                )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
