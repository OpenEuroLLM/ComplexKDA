"""Parameter-match the linear-attention variants to the transformer baseline.

The mixer block of KDA / Complex-KDA / GDN has a different parameter count
than the dense attention it replaces, so a like-for-like comparison at fixed N
requires resizing something else.  Following the paper (Table 2) we resize the
SwiGLU MLP's latent dimension ``intermediate_size`` and leave depth, width and
head geometry alone.

Counting is done by instantiating the real ``fla`` config on the meta device,
so the number is whatever the model actually allocates -- not an analytic
approximation that can silently drift from the implementation.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

import fla.layers.attn as _fla_attn
from fla.models import GatedDeltaNetConfig, ComplexKDAConfig, TransformerConfig

# Counting instantiates models on the meta device and never runs a forward, so a
# missing flash-attn (no CPU wheel, and irrelevant to parameter shapes) must not
# stop us from sizing the transformer baseline.  `Attention.__init__` only
# asserts the symbol is non-None, so satisfying that is enough.  Patched on the
# module after import rather than by stubbing `sys.modules["flash_attn"]`, which
# would also flip transformers' own flash-attn feature detection.
if _fla_attn.flash_attn_func is None:
    def _unavailable(*args, **kwargs):
        raise RuntimeError("flash-attn is stubbed for parameter counting only")

    _fla_attn.flash_attn_func = _unavailable
    _fla_attn.flash_attn_varlen_func = _unavailable

# One arch name -> the config class, the titan mixer, and the constructor
# kwargs. This is the single definition of what each arm IS; `ladder_flavors`
# reads it back out of `replication_1p3B.json` rather than keeping its own
# copy, because two typed copies of "complex-kda means gate=X" is precisely
# how two "identical" runs stop being identical.
#
# The KDA family is ONE layer. `kda` and `complex-kda` are both
# `ComplexKimiDeltaAttention` with K3's full-rank output gate, differing only in
# the decay gate's activation and its init -- which is the whole claim the
# ordering complex-kda > kda is supposed to support. Built from two different
# files they also differed in the output gate's rank, in the MLP width chosen
# to compensate, and in whatever else drifted between them; an ordering
# measured that way is not attributable to the gate.
#
# Attention and GDN stay upstream on purpose: they are the external baselines,
# and a baseline reimplemented inside the study's own layer is no longer
# external.
ARCHS = {
    "attn": {
        "cls": TransformerConfig,
        # `arch` is the id the trainer builds from; `mixer` is the layer
        # torchtitan swaps in (None = torchtitan's own Attention, no swap).
        # They are separate because the KDA family shares one layer under
        # several run names, so neither field can be derived from the arm's
        # name.
        #
        # This arm is the PARAMETER REFERENCE and not a run: the published
        # Transformer++ 1.3B row is the baseline the table is read against.
        # Every arm below is matched to the count fla builds from it.
        "arch": "attn",
        "mixer": None,
        "kwargs": {},
    },
    "gdn": {
        "cls": GatedDeltaNetConfig,
        "arch": "gdn",
        "mixer": "gdn",
        # The second external baseline, on the scaling ladder only. expand_v is
        # pinned because GatedDeltaNetConfig defaults it to 2.0, which sizes a
        # materially larger mixer and silently shrinks the matched MLP to
        # compensate. allow_neg_eigval False is the paper's own `gdn`; its
        # `gdn-neg` is the True one, and running the wrong one would turn a
        # baseline into a different architecture under the same name.
        "kwargs": {"allow_neg_eigval": False, "expand_v": 1.0},
    },
    # ------------------------------------------------------------------
    # The four arms of the 1.3B / 100BT FineWeb-Edu campaign, in two pairs.
    #
    # The KDA family is ONE layer. Both members of a pair are
    # `ComplexKimiDeltaAttention`, differing only in the decay gate's
    # activation and in whether negative eigenvalues are allowed -- which is
    # the whole claim a signed-vs-baseline ordering is supposed to support.
    # Built from two different files they would also differ in the output
    # gate's rank, in the MLP width chosen to compensate, and in whatever
    # else drifted between them; an ordering measured that way is not
    # attributable to the gate.
    #
    # `output_gate` is "lowrank" throughout: fla's and Kimi Linear's factored
    # output gate. It is stated rather than left to the layer default so that
    # a change to the layer cannot silently move the campaign -- and
    # `_check_kwargs` refuses it outright if the config ever stops declaring
    # it, which is the failure this whole table exists to prevent.
    "kda-sig-lowrank": {
        "cls": ComplexKDAConfig,
        "arch": "complex-kda",
        "mixer": "complex-kda",
        # THE BASELINE: the bounded sigmoid gate, at the shipped init.
        #
        # Its log-decay is lower_bound * sigmoid(A*u), bounded in
        # [lower_bound, 0] -- the positive half of the family
        # signed_sigmoid2 belongs to. Against it, a signed-vs-positive
        # comparison changes the gate's SIGN only; against fla's softplus
        # gate it would change the gate family too, and the ordering could
        # not be attributed to the sign.
        "kwargs": {"allow_neg_eigval": False, "lower_bound": -5.0,
                   "expand_v": 1.0, "gate": "sigmoid",
                   "gate_init_style": "shipped", "output_gate": "lowrank"},
    },
    "ckda-shipped-lowrank": {
        "cls": ComplexKDAConfig,
        "arch": "complex-kda",
        "mixer": "complex-kda",
        # THE SIGNED ARM: alpha in [-1, 1] and beta in [0, 2], at the same
        # shipped init as the baseline -- every channel starts at alpha > 0
        # and training discovers the negative ones. Same parameters as the
        # baseline, so the matched width comes out identical.
        "kwargs": {"allow_neg_eigval": True, "lower_bound": -5.0,
                   "expand_v": 1.0, "gate": "signed_sigmoid2",
                   "gate_init_style": "shipped", "output_gate": "lowrank"},
    },
    # --- the same pair inside Kimi Linear's 3:1 stack -------------------
    #
    # `attn_every` counts from 1 like Megatron's --linear-attention-freq, so
    # layer i is FULL attention when (i + 1) % attn_every == 0 -- at 24
    # layers that is 6 attention and 18 linear, the published 3:1.
    #
    # It is K3's PATTERN, not K3: their full layers are MLA and ours are the
    # same GQA block the `attn` reference uses, gated (Qwen3-Next's sigmoid
    # on the attention output) and NoPE. Calling it a K3 reproduction would
    # claim something this does not test. What it does test is whether the
    # hybrid ratio that makes Kimi Linear work also carries the signed gate,
    # which is the question worth asking -- a gate that helps a pure linear
    # stack need not help one that already has attention layers to carry the
    # retrieval it is bad at.
    #
    # The mixer kwargs are the pure arms', to the key. A hybrid that
    # differed from its pure arm in anything but the stack would make the
    # hybrid comparison a different experiment from the pure one.
    "kda-sig-hybrid-lowrank": {
        "cls": ComplexKDAConfig,
        "arch": "complex-kda",
        "mixer": "complex-kda",
        "hybrid": {"attn_every": 4, "attn_output_gate": True, "attn_use_rope": False},
        "kwargs": {"allow_neg_eigval": False, "lower_bound": -5.0,
                   "expand_v": 1.0, "gate": "sigmoid",
                   "gate_init_style": "shipped", "output_gate": "lowrank"},
    },
    "ckda-shipped-hybrid-lowrank": {
        "cls": ComplexKDAConfig,
        "arch": "complex-kda",
        "mixer": "complex-kda",
        "hybrid": {"attn_every": 4, "attn_output_gate": True, "attn_use_rope": False},
        "kwargs": {"allow_neg_eigval": True, "lower_bound": -5.0,
                   "expand_v": 1.0, "gate": "signed_sigmoid2",
                   "gate_init_style": "shipped", "output_gate": "lowrank"},
    },
}


CONFIG_MAP = {k: v["cls"] for k, v in ARCHS.items()}
EXTRA = {k: v["kwargs"] for k, v in ARCHS.items()}


def _check_kwargs(cls, kwargs):
    """Reject kwargs the config does not actually declare.

    ``PretrainedConfig.__init__`` swallows unknown keyword arguments and sets
    them as attributes, so a knob that upstream has renamed or deleted keeps
    "working" while changing nothing about the model.  That turns an upstream
    refactor into a silently wrong ladder, which is exactly what happened when
    `output_gate` was dropped from ComplexKDAConfig.
    """
    import inspect

    declared = inspect.signature(cls.__init__).parameters
    unknown = [k for k in kwargs if k not in declared]
    if unknown:
        raise TypeError(
            f"{cls.__name__} does not declare {unknown}; upstream probably "
            f"renamed or removed it. Passing it would be silently ignored."
        )


def build_config(arch, d_model, n_heads, n_layers, d_ffn, head_dim, vocab_size, seq_len, tie):
    _check_kwargs(CONFIG_MAP[arch], EXTRA[arch])
    return CONFIG_MAP[arch](
        hidden_size=d_model,
        num_heads=n_heads,
        num_hidden_layers=n_layers,
        intermediate_size=d_ffn,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
        vocab_size=vocab_size,
        tie_word_embeddings=tie,
        **EXTRA[arch],
    )


def count_params(cfg) -> int:
    """Total parameters, counting a tied embedding once."""
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def _per_layer(arch, d_model, n_heads, n_layers, d_ffn, head_dim, vocab_size,
               seq_len, tie) -> int:
    """Parameters in ONE block, by difference.

    Taking N(L) - N(L-1) rather than dividing N by L: the difference is exact
    whatever the embeddings, the final norm and the head weigh, and those do
    not scale with depth. Dividing would fold them into the per-layer figure
    and mis-size every hybrid by the same amount.
    """
    def N(L):
        return count_params(build_config(arch, d_model, n_heads, L, d_ffn,
                                         head_dim, vocab_size, seq_len, tie))
    return N(n_layers) - N(n_layers - 1)


def count_hybrid(arch, d_model, n_heads, n_layers, d_ffn, head_dim, vocab_size,
                 seq_len, tie) -> int:
    """Parameters of a mixed stack, composed from its two block types.

    There is no HF config that expresses "three of these then one of those",
    so the count is built rather than measured: take the linear arm's full
    model and swap the attention layers' cost in. Exact, because transformer
    blocks are uniform within a type -- which `_per_layer` gets by difference
    rather than by assuming it.
    """
    spec = ARCHS[arch]["hybrid"]
    every = int(spec["attn_every"])
    n_attn = sum(1 for i in range(n_layers) if (i + 1) % every == 0)

    linear_key = arch
    base = count_params(build_config(linear_key, d_model, n_heads, n_layers,
                                     d_ffn, head_dim, vocab_size, seq_len, tie))
    per_lin = _per_layer(linear_key, d_model, n_heads, n_layers, d_ffn,
                         head_dim, vocab_size, seq_len, tie)
    per_att = _per_layer("attn", d_model, n_heads, n_layers, d_ffn,
                         head_dim, vocab_size, seq_len, tie)

    # Qwen3-Next's output gate on the attention layers, which is what the
    # current hybrids use (Qwen3-Next, Kimi) and what the DENSE `attn` arm
    # deliberately does not: one hidden x (n_heads * head_dim) matrix per
    # attention layer, applied as a sigmoid to the attention output before
    # o_proj.
    #
    # Counted here rather than measured because `_per_layer` gets the
    # attention block's cost from the PLAIN arm by difference, and the plain
    # arm has no gate to measure. About 1.3% of total N at 302M, absorbed by
    # the d_ffn match -- so the hybrids stay parameter-matched to attention,
    # with a slightly narrower MLP paying for the gate.
    if spec.get("attn_output_gate"):
        per_att += d_model * n_heads * head_dim

    return base + n_attn * (per_att - per_lin)


def n_params_of(arch, d_model, n_heads, n_layers, d_ffn, head_dim, vocab_size,
                seq_len, tie) -> int:
    if "hybrid" in ARCHS[arch]:
        return count_hybrid(arch, d_model, n_heads, n_layers, d_ffn, head_dim,
                            vocab_size, seq_len, tie)
    return count_params(build_config(arch, d_model, n_heads, n_layers, d_ffn,
                                     head_dim, vocab_size, seq_len, tie))


def match_ffn(arch, target_N, d_model, n_heads, n_layers, head_dim,
              vocab_size, seq_len, tie, multiple=64):
    """Smallest-error ``intermediate_size`` (a multiple of ``multiple``)."""
    def N_of(ffn):
        return n_params_of(arch, d_model, n_heads, n_layers, ffn, head_dim,
                           vocab_size, seq_len, tie)

    lo, hi = multiple, 64 * multiple
    while N_of(hi) < target_N:
        hi *= 2
    while lo < hi:
        mid = ((lo + hi) // 2 // multiple) * multiple
        if mid <= lo:
            break
        if N_of(mid) < target_N:
            lo = mid
        else:
            hi = mid
    best = min([lo, hi], key=lambda f: abs(N_of(f) - target_N))
    return best, N_of(best)


