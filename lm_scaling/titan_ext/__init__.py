"""Register an ``fla``-backed model with torchtitan / titan-oellm.

Importing this module registers the train spec ``fla_custom``, which is
titan-oellm's ``qwen3_custom`` with every block's attention replaced by an
``fla`` linear-attention layer (``kda``, ``complex-kda`` or ``gdn``).

Why a swap rather than a fork of the model file: everything torchtitan gives us
-- FSDP2 sharding, activation checkpointing, compile, the loss and metrics
pipeline -- lives in ``parallelize_qwen3_custom`` and the surrounding train
spec, and none of it needs to know what the mixer is. Copying 650 lines of
model code to change one attribute would mean maintaining a divergent copy for
no benefit, and would make the "same layer under two backends" claim harder to
defend. The swap touches exactly one attribute name (``block.attention``).

Usage (inside a torchtitan run):

    import lm_scaling.titan_ext            # registers "fla_custom"
    # then set model.name = "fla_custom" and model.mixer = "complex-kda"

Requires ``titan_oellm`` and ``torchtitan`` on PYTHONPATH; both live in the
titan container, not in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .compile_relax import relaxing
from .mixer import MIXERS, FLAMixer
from .optimizer import build_optimizers_matching_lm

__all__ = ["MIXERS", "FLAMixer", "FLAModelArgs", "build_optimizers_matching_lm",
           "ladder_flavors", "register", "relaxing", "swap_in_fla_mixers"]


@dataclass
class _FLAExtras:
    """Fields ``qwen3_custom``'s args do not have. Merged in by ``register``."""

    mixer: str = "complex-kda"
    allow_neg_eigval: bool = True
    lower_bound: float = -5.0
    gate: str = "signed_sigmoid2"
    gate_init_style: str = "spread"
    # "lowrank" (Kimi Linear's factored output gate) or "linear" (Kimi K3's
    # full-rank one). The field exists here so the FLAVOR can say which, since
    # the layer has a default and a flavor that cannot express the difference
    # would build the wrong model under the right name.
    output_gate: str = "lowrank"
    # Beta's own init. Negative eigenvalues come from beta in (0,2) as much as
    # from the decay gate, so this is a second init axis, not a detail.
    beta_init_style: str = "standard"
    # Drop the silu on q/k/v. It is bounded below at ~-0.278 and pushes them
    # toward the non-negative orthant, which constrains k -- the direction of
    # the rank-1 update (I - beta k k^T). Directly relevant to how much
    # negative structure the recurrence can express, and never varied here.
    drop_silu: bool = False
    # The same idea at two granularities, and both have to be here or a
    # flavor that sets them builds the default model on torchtitan while the
    # `lm/` build of the same arm honours them:
    #   conv_silu      which of q/k/v keep silu on the SHORT CONV
    #   drop_key_silu  drop it from the key PROJECTION and its conv
    conv_silu: str = "qkv"
    drop_key_silu: bool = False
    # GatedDeltaNet defaults expand_v to 2.0. Every arm here runs 1, and left
    # to the layer's default gdn would carry a doubled value stream against a
    # parameter count matched for a single one.
    expand_v: float = 1.0
    # A HYBRID stack: layer i keeps torchtitan's own attention when
    # (i + 1) % attn_every == 0, and gets the mixer otherwise. 0 means every
    # layer is the mixer, which is every arm here but the two hybrids.
    #
    # Counted from 1 to match Megatron's --linear-attention-freq, which
    # expands as `0 if (i + 1) % freq == 0 else 1`. Not because Megatron is
    # authoritative, but because the two backends have to agree on WHICH
    # layers are attention, and an off-by-one there builds two different
    # models under one name.
    attn_every: int = 0
    # What a hybrid's ATTENTION layers are, as opposed to its mixer layers.
    # Both default to torchtitan's classical behaviour, so the dense `attn`
    # arm is unaffected and only an arm that asks gets the modern variant.
    #
    #   attn_output_gate  Qwen3-Next's sigmoid gate on the attention output,
    #                     before o_proj. Costs one hidden x hidden matrix a
    #                     layer and IS parameter-matched -- d_ffn pays for it.
    #   attn_use_rope     False is NoPE, as in Kimi's hybrid: the linear
    #                     layers carry position, so the attention layers need
    #                     not. Parameter-free.
    attn_output_gate: bool = False
    attn_use_rope: bool = True


def hybrid_layers(n_layers: int, attn_every: int) -> list[int]:
    """Which layer indices keep full attention.

    `attn_every` counts from 1: layer i is attention when
    (i + 1) % attn_every == 0. At 24 layers and every 4 that is
    [3, 7, 11, 15, 19, 23] -- six attention, eighteen linear, Kimi Linear's
    published 3:1.
    """
    if not attn_every:
        return []
    return [i for i in range(n_layers) if (i + 1) % attn_every == 0]


def swap_in_fla_mixers(model, args) -> int:
    """Replace blocks' ``attention`` with an :class:`FLAMixer`.

    Returns the number of blocks swapped, so a caller can assert it is neither
    zero nor all-of-them-when-a-hybrid-was-asked-for. A silent no-op here
    would train a plain transformer under the name ``complex-kda``; a silent
    swap of every layer would train a pure linear model under the name
    ``ckda-hybrid``. Both are the same failure and neither raises on its own.
    """
    mixer = getattr(args, "mixer", "complex-kda")
    if mixer not in MIXERS:
        raise ValueError(f"model.mixer must be one of {MIXERS}, got {mixer!r}")

    layers = getattr(model, "layers", None)
    if layers is None:
        raise RuntimeError(f"{type(model).__name__} has no .layers to swap")

    # torchtitan keeps `layers` as a ModuleDict keyed by str(layer_id) in recent
    # versions and a ModuleList in older ones; handle both.
    items = list(layers.items() if hasattr(layers, "items") else enumerate(layers))
    attn_every = int(getattr(args, "attn_every", 0) or 0)
    keep = set(hybrid_layers(len(items), attn_every))

    # A hybrid's attention layers are REPLACED too when the arm asks for a
    # gate or for NoPE, because torchtitan's own attention has neither. They
    # then run fla's Attention -- the same class `lm/` uses and the same one
    # the parameter match is computed from -- so the two backends build the
    # same hybrid by construction.
    #
    # Left alone when the arm asks for neither, which is what the dense `attn`
    # arm wants: torchtitan's attention, RoPE, no gate.
    gated = bool(getattr(args, "attn_output_gate", False))
    nope = not bool(getattr(args, "attn_use_rope", True))
    replace_attn = bool(keep) and (gated or nope)

    n = n_attn = 0
    for key, block in items:
        if not hasattr(block, "attention"):
            continue
        if int(key) in keep:
            if replace_attn:
                block.attention = FLAMixer("attn", args, layer_idx=int(key))
                n_attn += 1
            continue          # a full-attention layer of a hybrid stack
        block.attention = FLAMixer(mixer, args, layer_idx=int(key))
        n += 1

    if replace_attn and n_attn != len(keep):
        raise RuntimeError(
            f"hybrid asked for {len(keep)} gated/NoPE attention layers and "
            f"got {n_attn}; the rest would silently keep torchtitan's RoPE "
            "attention and the two backends would build different models")

    if attn_every and n != len(items) - len(keep):
        raise RuntimeError(
            f"hybrid asked for {len(items) - len(keep)} mixer layers of "
            f"{len(items)} and got {n}")

    # Printed once per run, because the settings that separate the arms --
    # gate, gate_init_style, output_gate -- do NOT change the parameter count,
    # so the count guard cannot tell `kda` from `complex-kda`. If they ever
    # arrive equal, the log is the only place it is visible, and a comparison
    # of two identical models is worse than a failed run.
    # What the LAYER is given, not what the flavor holds. The two are not the
    # same: a flavor field the mixer branch does not forward, or forwards under
    # a different default, shows as correct here while the model is not. That
    # is exactly how the anchor's Complex-KDA arm once built 331,103,056
    # parameters with a flavor whose fields all read right.
    from .mixer import fla_layer_kwargs

    cls, kw = fla_layer_kwargs(mixer, args)
    settings = {k: v for k, v in kw.items()
                if k not in ("hidden_size", "head_dim", "num_heads",
                             "norm_eps", "layer_idx")}
    print(f"[fla] swapped {n} blocks to {cls.__name__} "
          f"(mixer={mixer!r}) {settings}", flush=True)

    if n == 0:
        raise RuntimeError(
            "no transformer blocks were swapped; the model would train as plain "
            "attention while being labelled " + mixer
        )
    return n


def ladder_flavors(args_cls):
    """Our scaling rungs as torchtitan flavors, keyed ``"<arch>-<rung>"``.

    torchtitan picks geometry from a named flavor, and qwen3_custom ships only
    Qwen shapes -- none of which are on our ladder. Rather than hand-copy dims
    into a TOML per run, the flavors are generated from the same
    ``ladder_matched.json`` that the fla/`lm/` trainer and the Megatron probe
    read. That file is what makes the arms parameter-matched; a second, typed
    copy of the numbers is exactly how three "identical" runs stop being
    identical.

    The flavor carries the arch because ``d_ffn`` is per-arch: matching an
    ``fla`` arm to the dense baseline is done by resizing the MLP, so
    ``complex-kda-302M`` and ``attn-302M`` differ in ``hidden_dim`` and in
    nothing else.
    """
    import sys
    from pathlib import Path

    # The study's rungs AND the fixed points beside them, which are different
    # experiments. 340M is what the PAPER trained -- widths from
    # lm/configs/*.json, four separate layers, arms deliberately not
    # parameter-matched (kda 347.6M against complex-kda 344.9M) -- and is the
    # target a reproduction is measured against. 350M is the REDO: same
    # geometry, corpus and schedule, arms as the study now defines them, so
    # kda and complex-kda are one layer at one width and differ by the gate.
    # 1.3B is the geometry other groups publish 100BT results at
    # (replication_1p3B.py).
    #
    # Read through `geometries`, which the config's citations use too, so a
    # geometry the config can name is one torchtitan has a flavor for.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import geometries

    matched = geometries.table()
    takes_mixer = "mixer" in getattr(args_cls, "__dataclass_fields__", {})
    out = {}
    for rung, r in matched.items():
        for arch, a in r["archs"].items():
            # Gate on the LAYER the entry names, not on the arch's name. The
            # two stopped being the same thing when one layer started carrying
            # several arms: `ckda-p25` and `kda-neg` are both mixer
            # "complex-kda", and testing the name skipped every one of them, so
            # their flavors were never registered and eleven jobs died on
            #     KeyError: 'ckda-p10-47M'
            # a minute after starting. `attn` has no mixer at all -- it is the
            # dense baseline and belongs on the base spec, which does no
            # swapping.
            layer = a.get("mixer")
            if takes_mixer and (layer is None or layer not in MIXERS):
                continue
            kw = dict(
                dim=r["d_model"],
                n_layers=r["n_layers"],
                n_heads=r["n_heads"],
                n_kv_heads=r["n_heads"],
                head_dim=r["head_dim"],
                hidden_dim=a["d_ffn"],
                # The TOML's model.vocab_size replaces this in a real run --
                # update_from_config copies it unconditionally -- so it only
                # has to be right for a flavor built without one. A fixed
                # point with its own vocabulary says so; the ladder's rungs
                # all share ladder.yaml's.
                vocab_size=r.get("vocab_size", 50304),
                max_seq_len=2048,
                norm_eps=1e-6,
                rope_theta=10000,
                qk_norm=False,
                depth_init=False,
            )
            if takes_mixer:
                # `mixer` names the LAYER, which is no longer the arch: `kda`
                # and `complex-kda` are one layer differing in gate and init.
                # param_match writes both into ladder_matched.json so this
                # reads the definition instead of restating it.
                kw["mixer"] = layer
                fields = getattr(args_cls, "__dataclass_fields__", {})
                for name, value in _flavor_kwargs_or_raise(
                        arch, rung, a, args_cls).items():
                    if name not in fields:
                        raise TypeError(
                            f"{arch}-{rung}: ladder_matched.json asks for "
                            f"{name}={value!r} but {args_cls.__name__} has no "
                            f"such field, so torchtitan would drop it and "
                            f"build a different model than `lm/` does")
                    kw[name] = value
                # The stack's shape, which is not a mixer kwarg: it decides
                # which layers get a mixer at all. Absent for every pure arm.
                hybrid = a.get("hybrid")
                if hybrid:
                    if "attn_every" not in fields:
                        raise TypeError(
                            f"{arch}-{rung} is a hybrid but "
                            f"{args_cls.__name__} has no attn_every, so every "
                            "layer would get a mixer and the run would be a "
                            "pure linear model under a hybrid's name")
                    kw["attn_every"] = int(hybrid["attn_every"])
                    # Everything else the hybrid block declares -- what its
                    # ATTENTION layers are, as opposed to how many there are.
                    # Refused rather than dropped when the args class has no
                    # such field, for the same reason a mixer kwarg is: a
                    # dropped setting builds a different model under the same
                    # name, and the parameter count does not notice a missing
                    # RoPE at all.
                    for name, value in hybrid.items():
                        if name == "attn_every":
                            continue
                        if name not in fields:
                            raise TypeError(
                                f"{arch}-{rung} asks for {name}={value!r} but "
                                f"{args_cls.__name__} has no such field, so "
                                "torchtitan would build a hybrid whose "
                                "attention differs from the one `lm/` builds")
                        kw[name] = value
            out[f"{arch}-{rung}"] = args_cls(**kw)
            # The same arm without SiLU on q/k/v, under its OWN flavor name
            # rather than as a changed definition. A running job re-reads its
            # flavor from this table at every restart, and SiLU has no
            # parameters, so a checkpoint loads into either model without a
            # word: flipping `drop_silu` on the existing names would change the
            # model under every run in flight. A campaign opts in through
            # aux.drop_silu (resolvers.flavor).
            if (takes_mixer and layer in geometries.DROPS_SILU
                    and "drop_silu" in fields):
                out[f"{arch}-{rung}{geometries.NOSILU}"] = args_cls(
                    **{**kw, "drop_silu": True})
    return out


def _flavor_kwargs_or_raise(arch, rung, entry, args_cls) -> dict:
    """The arch table's layer settings, or a loud failure.

    Not optional. Without them the flavor falls back to the shared args
    dataclass's defaults, which are the KDA family's -- so a table entry
    written before a field existed silently gets the wrong value for it.
    The anchor's Complex-KDA arm picked up output_gate="lowrank" that way and
    built 331,103,056 parameters against the paper's 344,865,616, with every
    field on the flavor reading correctly.
    """
    kwargs = entry.get("kwargs")
    if kwargs is None:
        raise ValueError(
            f"{arch}-{rung}: the arch table entry has no 'kwargs', so the "
            f"flavor would take {getattr(args_cls, '__name__', args_cls)}'s "
            f"defaults for every layer setting. Regenerate it -- "
            f"param_match.py, anchor_redo.py and anchor_paper.json write them.")
    return kwargs


_REGISTERED = False


def register(data: str = "default") -> None:
    """Register the ``fla_custom`` train spec with torchtitan.

    ``data`` selects the dataloader:

    ``"default"``
        titan-oellm's own, for a normal run.
    ``"synthetic"``
        random token ids, so a throughput measurement never waits on GPFS
        (``synthetic.py``). Speed probes only -- the loss is meaningless.
    ``"megatron"``
        the same ``.bin``/``.idx`` corpus `lm/` and Megatron read
        (``megatron_data.py``). This is what loss matching needs: a difference
        in the data is indistinguishable from a difference in the model once
        you are comparing two curves.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import copy

    import titan_oellm.models  # noqa: F401  (registers qwen3_custom)
    from torchtitan.protocols.train_spec import get_train_spec, register_train_spec

    base = get_train_spec("qwen3_custom")

    base_args_cls = base.model_args[next(iter(base.model_args))].__class__

    @dataclass
    class FLAModelArgs(base_args_cls, _FLAExtras):  # type: ignore[misc, valid-type]
        pass

    configs = {}
    for name, cfg in base.model_args.items():
        new = FLAModelArgs(**{f.name: getattr(cfg, f.name)
                              for f in cfg.__dataclass_fields__.values()})
        configs[name] = new
    # Our rungs, on top of qwen3_custom's own flavors (kept so the debugmodel
    # stays available for a smoke run).
    configs.update(ladder_flavors(FLAModelArgs))

    # The dense baseline runs on qwen3_custom's own spec -- same blocks, same
    # parallelize function, no swap -- so it needs our geometries
    # registered there too. Mutating titan-oellm's flavor dict is deliberate:
    # the alternative is a second spec that differs from theirs only in a
    # dictionary, which is a worse thing to have to trust.
    base.model_args.update(ladder_flavors(base_args_cls))

    # Both specs initialize like an fla/HF model: every Linear and Embedding
    # outside the mixers from N(0, 0.02), as `lm/` and flame (fla-org's
    # reference trainer) do, instead of torchtitan's N(0, 1) embedding and
    # depth-scaled projections. The mixers take fla's HF init in
    # FLAMixer.init_weights.
    from .hf_init import with_hf_init, with_linear_bias

    # Biases first, so `init_weights` (which zeroes any bias it finds) runs
    # after they exist. TITAN_EXT_LINEAR_BIAS gates them; off, this is a no-op
    # wrapper and every finished campaign is unaffected.
    base_model_cls = with_hf_init(with_linear_bias(base.model_cls))
    base.model_cls = base_model_cls

    class FLAModel(base_model_cls):  # type: ignore[misc, valid-type]
        def __init__(self, model_args):
            super().__init__(model_args)
            swap_in_fla_mixers(self, model_args)

    spec = copy.copy(base)
    spec.model_cls = FLAModel
    spec.model_args = configs
    # Only the fla spec: the dense baseline runs on titan-oellm's own
    # parallelize function untouched, so its numbers stay comparable to any
    # other torchtitan run.
    spec.parallelize_fn = relaxing(base.parallelize_fn)
    # `lm/` keeps norms and biases out of weight decay; torchtitan decays
    # everything. At wd 0.1 that pulls every RMSNorm gain -- and the tied
    # embedding -- down throughout training, so the two backends are not
    # optimizing the same objective. Applied to both specs: the dense baseline
    # is the arm every other arm is compared against.
    spec.build_optimizers_fn = build_optimizers_matching_lm
    base.build_optimizers_fn = build_optimizers_matching_lm
    if data != "default":
        if data == "synthetic":
            from .synthetic import build_synthetic_dataloader as loader
        elif data == "megatron":
            from .megatron_data import build_megatron_dataloader as loader
        else:
            raise ValueError(f"data must be default/synthetic/megatron, got {data!r}")
        spec.build_dataloader_fn = loader
        base.build_dataloader_fn = loader
        if data == "megatron":
            # Validation does NOT go through build_dataloader_fn -- torchtitan's
            # Validator constructs the HuggingFace loader itself -- so setting
            # only the line above would train on the paper's corpus and report
            # validation loss on c4. See megatron_validate for the swap.
            from .megatron_validate import build_megatron_validator
            spec.build_validator_fn = build_megatron_validator
            base.build_validator_fn = build_megatron_validator
        # No tokenizer either: both loaders yield token ids directly, and
        # titan-oellm's builder wants a path to a real tokenizer that neither
        # a throughput measurement nor a pre-tokenized corpus has any use for.
        spec.build_tokenizer_fn = None
        base.build_tokenizer_fn = None
    # WSD's stable phase is re-usable by several cooldowns only if the
    # scheduler's checkpoint carries WHERE it is and not WHAT it is. It
    # carries both, so a cooldown loads the stable run's shape over its own
    # and never decays. Both specs get the fix: `attn` runs on qwen3_custom
    # and every other arm on ours, and a chained ladder that decays for five
    # arms and not the sixth is worse than one that decays for none.
    from .lr_state import install as _install_lr_state

    def _build_lr_schedulers(*args, **kwargs):
        return _install_lr_state(_base_build_lr(*args, **kwargs))

    _base_build_lr = base.build_lr_schedulers_fn
    spec.build_lr_schedulers_fn = _build_lr_schedulers
    base.build_lr_schedulers_fn = _build_lr_schedulers

    # Every step line says how long the run is -- `step: 6922/45824` -- so an
    # ETA is two timestamps rather than a lookup in the campaign plan. Both
    # specs, for the same reason as the scheduler fix above. A spec without its
    # own builder uses torchtitan's default, so that is what gets wrapped.
    from torchtitan.components.metrics import build_metrics_processor

    from .step_total import with_step_total

    _metrics = with_step_total(base.build_metrics_processor_fn
                               or build_metrics_processor)
    spec.build_metrics_processor_fn = _metrics
    base.build_metrics_processor_fn = _metrics

    # One checkpoint just before the job's wall clock as well as the ones on the
    # interval, so a restart resumes from minutes before the wall rather than
    # up to an interval before it. torchtitan's Trainer builds its
    # CheckpointManager itself, so this patches the class and covers both
    # specs at once. Off without a Slurm end time; see wall.py.
    from .wall import install as _install_wall

    _install_wall()

    # register_train_spec takes (name, spec) and raises if the name is already
    # taken, which is why this function is guarded above: torchtitan imports
    # the custom_import module once, but a test or a driver may call register()
    # again.
    register_train_spec("fla_custom", spec)
    _REGISTERED = True


# Exported for type annotations / config construction outside register().
FLAModelArgs = _FLAExtras


# torchtitan's `--experimental.custom_import=<module>` only imports the module;
# it never calls into it. Registering at import time is what makes that hook
# work -- and makes this file's own docstring true, which it was not.
#
# Guarded on torchtitan's presence rather than wrapped in a bare except: the
# unit tests import this module without torchtitan installed, but once
# torchtitan IS there a failure inside register() is a real breakage and must
# not be swallowed.
try:
    import torchtitan  # noqa: F401
except ImportError:
    pass
else:
    import os

    # The probe launchers set this before importing; a normal run does not.
    register(data=os.environ.get("TITAN_EXT_DATA", "default"))
