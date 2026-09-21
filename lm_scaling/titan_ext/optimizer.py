"""Exclude norms and biases from weight decay, as `lm/` does.

torchtitan builds one parameter group and applies ``weight_decay`` to all of
it (``components/optimizer.py``: a single ``optimizer_kwargs`` dict). `lm/`'s
``get_param_groups`` puts anything whose name contains ``bias`` or ``norm``, or
carries ``_no_weight_decay``, into a second group at ``weight_decay=0.0``.

At wd 0.1 that is not a detail. It decays every RMSNorm gain and -- with tied
embeddings -- the embedding table too, pulling the residual scale down
throughout training. Two backends that disagree about it do not train the same
model, however identical their parameter counts.

This wraps torchtitan's builder rather than replacing it: the container it
returns knows about FSDP, DTensor and the checkpointing hooks, and
reimplementing that to change one hyperparameter would be trading a small
mismatch for a large one. The regrouping happens before the first step, so no
optimizer state has been created yet.
"""

from __future__ import annotations

import os

# The opt-out, for a recipe that decays EVERY parameter: Gated DeltaNet's
# pretrain.py hands model.parameters() to AdamW as one group, and fwedu_1p3B
# follows it. An environment variable because torchtitan's [optimizer] section
# refuses keys it does not declare; the experiment sets it in `slurm.env`, so
# it is in the config archived beside the run.
DECAY_ALL = "TITAN_EXT_DECAY_ALL"


def _no_decay(name: str, param) -> bool:
    # Mirrors the reference trainer's `get_param_groups` exactly, including
    # the `_no_weight_decay` attribute fla sets on things like A_log.
    return (getattr(param, "_no_weight_decay", False)
            or "bias" in name or "norm" in name)


def build_optimizers_matching_lm(model_parts, optimizer_config, parallel_dims,
                                 ft_manager=None):
    """Drop-in for a train spec's ``build_optimizers_fn``."""
    from torchtitan.components.optimizer import build_optimizers

    container = build_optimizers(model_parts, optimizer_config, parallel_dims,
                                 ft_manager)

    if os.environ.get(DECAY_ALL) == "1":
        # torchtitan's own single group already decays everything. Printed,
        # because two runs differing in this alone differ in nothing a
        # parameter count or a config diff of [optimizer] would show.
        print(f"[titan_ext] {DECAY_ALL}=1: weight decay on every parameter, "
              "norms, biases and A_log included", flush=True)
        return container

    excluded = {id(p) for m in model_parts
                for n, p in m.named_parameters() if _no_decay(n, p)}
    if not excluded:
        raise RuntimeError(
            "no parameter matched the no-decay rule; either the model has no "
            "norms (it does) or the naming changed, and every norm gain is "
            "silently being decayed"
        )

    for opt in container.optimizers:
        regrouped = []
        for group in opt.param_groups:
            decay = [p for p in group["params"] if id(p) not in excluded]
            keep = [p for p in group["params"] if id(p) in excluded]
            if decay:
                regrouped.append({**group, "params": decay})
            if keep:
                regrouped.append({**group, "params": keep, "weight_decay": 0.0})
        opt.param_groups = regrouped

    return container
