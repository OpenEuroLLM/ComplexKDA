"""Initialize the torchtitan model the way an fla/HF model is initialized.

HF's post_init draws every Linear and Embedding from N(0, initializer_range),
zeroes biases and leaves norms at 1, with no per-layer or depth scaling. That
is how `lm/` and flame (fla-org's reference trainer) start every model.
torchtitan's Qwen3 init differs on every weight outside the mixer:

    embedding   N(0, 1); tied, then overwritten by the head's truncated
                N(0, dim^-0.5)
    MLP         gate 0.02; up and down 0.02 / sqrt(2 n_layers)
    attention   q, k, v 0.02; out 0.02 / sqrt(2 n_layers)   (the dense arm)

`with_hf_init` wraps a model class so that its `init_weights` runs
torchtitan's own first -- buffers such as the RoPE cache, the norms, and each
FLAMixer, which takes fla's HF init itself (mixer.hf_init_) -- and then
redraws every Linear and Embedding outside the mixers from N(0, 0.02).
Weight tying is architecture, not init, and is left as the flavor sets it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mixer import INITIALIZER_RANGE, FLAMixer


#: Projections that write back into the residual stream. Megatron draws these
#: from N(0, std / sqrt(2 n_layers)) and everything else from N(0, std) --
#: `scaled_init_method_normal`, the GPT-2 convention for keeping the residual
#: stream from growing with depth at initialization. fla and HF do not, which
#: is why `init_like_hf_` is flat by default.
_RESIDUAL_OUT = ("wo", "w2")


def init_like_hf_(model: nn.Module, std: float = INITIALIZER_RANGE,
                  out_std: float | None = None) -> nn.Module:
    """Redraw every Linear and Embedding outside an FLAMixer from N(0, std).

    Counted by tensor, not by module, so a tied weight -- the head's IS the
    embedding's once torchtitan's parallelize has run -- is drawn exactly once.
    Under FSDP the weights are DTensor shards; nn.init draws them through the
    distributed RNG tracker, as torchtitan's own init does.

    `out_std` gives the residual output projections a different draw. Left
    None this is exactly the flat fla/HF init the ladder has always run; set,
    it is Megatron's, which the reference study uses and which our dense
    baseline has been missing -- measured, our `wo` and `w2` start at 0.02
    where theirs start at 0.02/sqrt(2 n_layers), five times smaller at 12
    layers.
    """
    seen: set[int] = set()
    mixers = [n for n, m in model.named_modules() if isinstance(m, FLAMixer)]

    def inside_mixer(name: str) -> bool:
        return any(name == p or name.startswith(p + ".") for p in mixers)

    with torch.no_grad():
        for name, m in model.named_modules():
            if not isinstance(m, (nn.Linear, nn.Embedding)) or inside_mixer(name):
                continue
            if id(m.weight) not in seen:
                seen.add(id(m.weight))
                leaf = name.rsplit(".", 1)[-1]
                draw = out_std if (out_std is not None
                                   and leaf in _RESIDUAL_OUT) else std
                nn.init.normal_(m.weight, mean=0.0, std=draw)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    return model


def with_hf_init(model_cls: type) -> type:
    """`model_cls` with `init_like_hf_` run after its own `init_weights`."""

    class HFInit(model_cls):  # type: ignore[misc, valid-type]
        def init_weights(self, *args, **kwargs):
            out = super().init_weights(*args, **kwargs)
            # OPT-IN, because it changes what every finished campaign would
            # produce and those runs are in the harvest.
            import os
            scaled = os.environ.get("TITAN_EXT_SCALED_OUT_INIT", "").strip()
            out_std = None
            if scaled not in ("", "0"):
                n_layers = int(getattr(self.model_args, "n_layers", 0) or 0)
                if n_layers <= 0:
                    raise ValueError(
                        "TITAN_EXT_SCALED_OUT_INIT needs n_layers on model_args "
                        "to compute Megatron's scaled init")
                out_std = INITIALIZER_RANGE / (2 * n_layers) ** 0.5
            init_like_hf_(self, out_std=out_std)
            return out

    HFInit.__name__ = model_cls.__name__
    HFInit.__qualname__ = model_cls.__qualname__
    return HFInit

#: The seven projections Megatron gives a bias. NOT the output head: with
#: `untie_embeddings_and_output_weights: False` it shares the embedding and
#: carries no bias of its own, and adding one there would break the tie.
_BIASED = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


def add_linear_biases_(model: nn.Module) -> nn.Module:
    """Give the attention and MLP projections a zero bias, as Megatron does.

    The reference sets `add_qkv_bias: true` and inherits Megatron's
    `add_bias_linear`, which defaults ON -- so every projection in their model
    has a bias. torchtitan's Qwen3 hard-codes `bias=False` in all seven
    Linears, which is the last architectural difference between our dense
    baseline and theirs that we can reach at all.

    A bias is ATTACHED rather than the module rebuilt: `nn.Linear.forward` is
    `F.linear(x, weight, bias)` and reads `self.bias` at call time, so setting
    the attribute is enough and nothing else about the module -- its weight,
    its place in the tree, the tie on the output head -- is disturbed. Done at
    construction, before parallelize shards anything.

    Zeros, because that is what both Megatron and HF's `_init_weights` do.
    """
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.bias is None \
                and name.rsplit(".", 1)[-1] in _BIASED:
            m.bias = nn.Parameter(torch.zeros(
                m.out_features, dtype=m.weight.dtype, device=m.weight.device))
            n += 1
    if n == 0:
        raise ValueError(
            "TITAN_EXT_LINEAR_BIAS is set but no projection was given a bias; "
            f"looked for {_BIASED} and found none without one")
    return model


def with_linear_bias(model_cls: type) -> type:
    """`model_cls` with Megatron's projection biases, when asked for."""

    class Biased(model_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import os
            if os.environ.get("TITAN_EXT_LINEAR_BIAS", "").strip() not in ("", "0"):
                add_linear_biases_(self)

    Biased.__name__ = model_cls.__name__
    Biased.__qualname__ = model_cls.__qualname__
    return Biased
