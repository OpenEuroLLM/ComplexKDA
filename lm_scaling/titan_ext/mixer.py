"""Adapt an ``fla`` linear-attention layer to torchtitan's attention contract.

torchtitan (and titan-oellm's ``qwen3_custom``) call the mixer as

    x = x + self.attention(self.attention_norm(x), rope_cache, masks, positions)

and initialize it through ``init_weights(init_std)``. The ``fla`` layers instead
follow the HuggingFace convention, taking ``(hidden_states, attention_mask,
past_key_values, ...)`` and returning a 3-tuple. This module is the shim between
the two, and nothing else: the mixer maths, the Triton kernels and the gate
initialization all stay in ``fla``, so a titan run and an ``lm/`` run execute
the same layer code. That is what makes the training-curve comparison between
backends meaningful rather than a re-implementation bake-off.

RoPE and attention masks are accepted and dropped. Linear attention carries
position implicitly through the recurrence, which is why the study's linear arms
are NoPE; a dense-attention arm should keep torchtitan's own ``Attention``
rather than route through here.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MIXERS = ("kda", "complex-kda", "gdn")


def build_fla_layer(mixer: str, args) -> nn.Module:
    """Instantiate the ``fla`` layer named by ``mixer``."""
    cls, kw = fla_layer_kwargs(mixer, args)
    return cls(**kw)


def fla_layer_kwargs(mixer: str, args):
    """(layer class, constructor kwargs) for ``mixer``.

    Separate from construction so a caller can LOG what the layer is actually
    given. Logging the flavor's fields instead is not the same thing and hid a
    real defect: the anchor's Complex-KDA arm reads `output_gate` with a default
    of "linear", and adding an `output_gate` field to the shared args dataclass
    -- defaulting to "lowrank" for the KDA family -- made `getattr` find that
    instead. The arm silently built 331,103,056 parameters against the paper's
    344,865,616, and the flavor's own fields looked entirely correct.

    ``args`` is a torchtitan model-args object; only the fields read below are
    required, so this works with any of titan-oellm's arg dataclasses.
    """
    hidden_size = args.dim
    head_dim = args.head_dim
    num_heads = hidden_size // head_dim

    common = dict(
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_heads=num_heads,
        norm_eps=args.norm_eps,
        layer_idx=getattr(args, "_layer_idx", 0),
    )

    if mixer == "attn":
        # THE HYBRID'S ATTENTION LAYERS, and they are not torchtitan's.
        #
        # Routing them through fla means `lm/` and torchtitan run the SAME
        # attention code in a hybrid, so the two backends agree by
        # construction rather than by a test that compares them. That is worth
        # more here than anywhere else: the parameter match is computed from
        # this very class, so a difference in what the two stacks build would
        # show up as a model whose count is right and whose layers are not.
        #
        # Gated and NoPE, as in Qwen3-Next and Kimi's hybrids:
        #
        #   output_gate  a sigmoid of the layer input, applied to the
        #                attention output before o_proj. One hidden x hidden
        #                matrix per attention layer, ~1.3% of total N at 302M,
        #                paid for out of d_ffn by the parameter match.
        #   use_rope     off. The linear layers already carry position -- a
        #                gated-delta recurrence is order-dependent by
        #                construction -- so the attention layers reach it
        #                through the residual stream instead of a rotation.
        #                Parameter-free, so the match does not move.
        #
        # The DENSE `attn` arm keeps torchtitan's own attention, with RoPE and
        # no gate. That is deliberate: it is the classical baseline, and the
        # comparison the hybrids are in is against the linear arms, not
        # against it.
        from fla.layers.attn import Attention

        # No `head_dim`: fla's Attention does not take one, it derives
        # hidden_size // num_heads. That is the same number here by
        # construction -- num_heads is computed as hidden_size // head_dim
        # above -- but passing it is a TypeError, and the assertion below is
        # what keeps the two definitions from drifting apart silently.
        assert hidden_size == num_heads * head_dim, (
            f"attention head geometry does not close: {hidden_size} != "
            f"{num_heads} x {head_dim}")
        return Attention, dict(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            qk_norm=bool(getattr(args, "qk_norm", False)),
            output_gate=bool(getattr(args, "attn_output_gate", False)),
            use_rope=bool(getattr(args, "attn_use_rope", False)),
            rope_theta=getattr(args, "rope_theta", 10000.0),
            layer_idx=getattr(args, "_layer_idx", 0),
        )
    if mixer == "kda":
        from fla.layers.kda import KimiDeltaAttention

        return KimiDeltaAttention, dict(
            **common,
            allow_neg_eigval=getattr(args, "allow_neg_eigval", True),
            lower_bound=getattr(args, "lower_bound", -5.0),
        )
    if mixer == "complex-kda":
        from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention

        return ComplexKimiDeltaAttention, dict(
            **common,
            allow_neg_eigval=getattr(args, "allow_neg_eigval", True),
            lower_bound=getattr(args, "lower_bound", -5.0),
            gate=getattr(args, "gate", "signed_sigmoid2"),
            gate_init_style=getattr(args, "gate_init_style", "spread"),
            beta_init_style=getattr(args, "beta_init_style", "standard"),
            drop_silu=bool(getattr(args, "drop_silu", False)),
            conv_silu=str(getattr(args, "conv_silu", "qkv")),
            drop_key_silu=bool(getattr(args, "drop_key_silu", False)),
            # The KDA family's arms share this layer, so the output gate has to
            # come from the flavor: "linear" is K3's full-rank map and the
            # layer's own default is Kimi Linear's low-rank one.
            output_gate=getattr(args, "output_gate", "lowrank"),
        )
    if mixer == "gdn":
        from fla.layers.gated_deltanet import GatedDeltaNet

        return GatedDeltaNet, dict(
            **common,
            # expand_v defaults to 2.0 on this layer while the whole ladder and
            # the paper's gdn.json run 1. Left unset it silently doubles the
            # value stream and the parameter count stops matching.
            expand_v=float(getattr(args, "expand_v", 1)),
            allow_neg_eigval=getattr(args, "allow_neg_eigval", True),
        )
    raise ValueError(f"unknown mixer {mixer!r}; expected one of {MIXERS}")
# The fla model whose `_init_weights` HF's post_init runs over each mixer --
# how `lm/` and flame (fla-org's reference trainer) initialize it. "attn" is
# a hybrid's attention layer, which under `lm/` lives inside a ComplexKDA
# model.
_HF_MODEL = {
    "kda": ("fla.models.kda.modeling_kda", "KDAPreTrainedModel"),
    "complex-kda": ("fla.models.complex_kda.modeling_complex_kda", "ComplexKDAPreTrainedModel"),
    "attn": ("fla.models.complex_kda.modeling_complex_kda", "ComplexKDAPreTrainedModel"),
    "gdn": ("fla.models.gated_deltanet.modeling_gated_deltanet", "GatedDeltaNetPreTrainedModel"),
}

# fla's default initializer_range, and what the flame reference configs set.
INITIALIZER_RANGE = 0.02


def hf_init_(layer: nn.Module, mixer: str, n_layers: int = 0) -> nn.Module:
    """Run the fla model's own `_init_weights` over `layer`, children first.

    That is what HF's post_init does to every module of an fla model, so it is
    what `lm/` and flame start from: every Linear and Conv1d N(0, 0.02) with
    zero biases, norms at 1, and the model's gate init (A_log, dt_bias, the
    spread) applied last, on the module itself. Constructed but never run
    through this, a layer carries PyTorch's defaults instead --
    U(+-1/sqrt(fan_in)) for the projections, U(+-0.5) for a 4-tap depthwise
    conv, 14x fla's std -- and every torchtitan run before 2026-09-11 started
    from that.

    Draws from the ambient RNG, so reference_layer's seed covers them.
    `prenorm_residual_strategy` stays at the fla models' default, None, which
    is what post_init passes.
    """
    import importlib
    from types import SimpleNamespace

    module, name = _HF_MODEL[mixer]
    init = getattr(importlib.import_module(module), name)._init_weights
    stub = SimpleNamespace(config=SimpleNamespace(
        initializer_range=INITIALIZER_RANGE, num_hidden_layers=n_layers))
    with torch.no_grad():
        layer.apply(lambda m: init(stub, m))
    return layer


def reference_layer(mixer: str, args, layer_idx: int):
    """Build the layer `fla` would have built, identically on every rank.

    Seeded from the layer index rather than the ambient RNG: under FSDP each
    rank holds a shard of the same logical parameter and derives it from its
    own reference, so the references must agree or the assembled weight is
    noise -- and nothing reports it, because each shard is individually
    well-formed. Megatron's wrapper does the same thing for the same reason.
    """
    state = torch.random.get_rng_state()
    torch.manual_seed(0x5EED + int(layer_idx))
    try:
        return hf_init_(build_fla_layer(mixer, args), mixer,
                        int(getattr(args, "n_layers", 0) or 0))
    finally:
        torch.random.set_rng_state(state)


def _copy_into(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Write a full tensor into `dst`, which may be an FSDP shard.

    torchtitan applies `fully_shard` before `init_weights`, so on more than one
    GPU the parameters are DTensors and `dst.copy_(plain_tensor)` raises
    "got mixed torch.Tensor and DTensor". Sharding the source the same way
    first gives each rank exactly its own slice. On one GPU there is no
    DTensor and this is a plain copy -- which is why the single-GPU loss
    matching passed while every multi-GPU run died on the first layer.
    """
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError:  # older torch
        DTensor = ()
        distribute_tensor = None

    if DTensor and isinstance(dst, DTensor):
        full = src.to(dst.device, dst.dtype)
        dst.copy_(distribute_tensor(full, dst.device_mesh, dst.placements))
    else:
        dst.copy_(src.to(dst.device, dst.dtype))


class FLAMixer(nn.Module):
    """torchtitan-shaped wrapper around one ``fla`` layer."""

    def __init__(self, mixer: str, args, layer_idx: int = 0):
        super().__init__()
        args._layer_idx = layer_idx
        self.mixer_name = mixer
        # Kept so init_weights can rebuild a reference layer: torchtitan
        # constructs on meta and to_empty()s, so the values set in __init__ do
        # not survive to training.
        self._args = args
        self._layer_idx = layer_idx
        self.inner = build_fla_layer(mixer, args)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor | None = None,
        attention_masks=None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # rope_cache/positions are unused: these mixers are NoPE by design.
        # attention_masks is torchtitan's causal-mask object, not fla's [B, T]
        # padding mask -- passing it through would trip fla's dim assertion, and
        # it carries no information here because the recurrence is causal by
        # construction and titan feeds packed, equal-length sequences.
        o, _, _ = self.inner(hidden_states=x, attention_mask=None, past_key_values=None)
        return o

    def init_weights(self, init_std: float):
        """Give every parameter `fla`'s own initialization.

        torchtitan builds the model on the **meta** device and then calls
        ``model.to_empty(device=...)`` before ``init_weights()``. ``to_empty``
        allocates *uninitialized* memory, so any parameter this method does not
        write keeps whatever was in those bytes.

        An earlier version of this method deliberately skipped ``A_log``,
        ``dt_bias``, ``f_proj``, the convs and the norms, on the reasoning that
        their initialization is the thing under study and must not be
        overwritten by a truncated normal. The reasoning was right and the
        implementation inverted it: under ``to_empty`` those parameters were
        not preserved, they were garbage. Every linear arm trained under
        torchtitan with an uninitialized decay gate. It converged -- and sat
        **37.5% above** the same configuration under `lm/` at 0.2BT, which is
        how it was caught.

        The fix builds a reference layer and copies its values in. The
        reference is constructed and then run through the fla model's own
        ``_init_weights`` (``hf_init_``), as HF's post_init does -- so it is
        what `lm/` and flame start from, including whatever
        ``gate_init_style`` does, without restating it here where it would
        drift. Until 2026-09-11 the reference was only constructed, which left
        PyTorch's default init: projections U(+-1/sqrt(fan_in)) and short
        convs at 14x fla's std. ``init_std`` is deliberately unused: matching
        `lm/` means fla's initialization, not torchtitan's.
        """
        # Seeded from the layer index, not the ambient RNG. Under FSDP each
        # rank holds a shard of the same logical parameter and builds its own
        # reference, so the references have to agree across ranks or the shards
        # come from different draws and the assembled weight is noise.
        ref = reference_layer(self.mixer_name, self._args, self._layer_idx)
        ref_params = dict(ref.named_parameters())
        ref_buffers = dict(ref.named_buffers())
        with torch.no_grad():
            for name, p in self.inner.named_parameters():
                src = ref_params.get(name)
                if src is None:
                    raise RuntimeError(
                        f"{self.mixer_name}: no reference value for {name!r}; "
                        f"it would keep uninitialized memory from to_empty()"
                    )
                _copy_into(p, src)
            for name, b in self.inner.named_buffers():
                src = ref_buffers.get(name)
                if src is not None:
                    _copy_into(b, src)
