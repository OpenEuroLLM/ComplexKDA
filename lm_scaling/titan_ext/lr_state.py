"""Make a WSD stable phase re-usable by several cooldowns.

The point of WSD on a scaling ladder is that the constant-LR phase is shared:
one stable run to 0.8*D_max, then a short cooldown branching off it for each
budget. With Ajroldi's intermediate decays at 6, 12, 20, 30, 50, 80, 120 and
200BT that is 263.6BT of training per cell against 518BT run separately --
1.97x, and the runs are otherwise identical.

It did not work, and this is why.

`UniversalLR` defines no `state_dict`, so it inherits `LRScheduler`'s, which
serialises `self.__dict__` minus the optimizer -- INCLUDING the shape of the
schedule: `total_steps`, `warm_steps`, `main_steps`, `cooldown_steps`. A
cooldown job builds a scheduler whose shape says "cool down over the next
2,560 steps", then loads the branch checkpoint, and `load_state_dict` does
`self.__dict__.update(...)` and replaces that shape with the stable run's --
in which `cooldown_steps` is 0. The learning rate then holds constant for the
whole window, and the job produces a constant-LR endpoint labelled as a WSD
result.

Measured on JUWELS at 47M attn, three runs to the same 3072 steps:

    constant-LR (no cooldown)   4.0433
    chained cooldown            4.0417   +0.0016
    single-stage WSD            3.9332   +0.1101

The cooldown is worth 0.11 nats and the chained scheme delivered 0.002 of it.

With this in place the two agree. Measured on develbooster, one shared stable
phase against the same three budgets trained in one piece:

    budget    chained   single-stage
     0.2BT     3.9099       3.9134
     0.3BT     3.7382       3.7543
     0.4BT     3.6404       3.6495

Within 0.016 nats, and the chain slightly ahead at every budget. Measured on
torchtitan; the run logs behind it are not part of this release, and the
Megatron ladder re-measures the same question as `ladder_chaincheck` in
`megatron_ext/submit_ladder.py`, whose comment records what it is for.

Excluding the scheduler from the load instead gives the right shape with the
counter reset to 0, which never reaches a cooldown beginning at step 2560 --
the other way to get a constant-LR run wearing a WSD label.

THE FIX. A scheduler's state is WHERE IT IS, not WHAT IT IS. The shape belongs
to the job's own config, which built it a moment ago from `cooldown_steps` and
`warm_steps`; only the step counter has to survive. So this carries
`last_epoch` and nothing else.

Two details make that sufficient rather than merely smaller:

`_main_end_lrs` is deliberately NOT carried. It caches the learning rate at
the end of the main phase so the cooldown starts smoothly from it, and it is
populated by `get_lr` on the single step that crosses the boundary -- a step a
resumed job never executes. Its absence is handled by the scheduler's own
fallback, `lr_start = base_lr`, which is EXACT here: `main_decay_type` is
`const`, so the main phase ends at base_lr by construction. Carrying it would
be pinning a value the resumed job can already derive.

The LRs are applied at load rather than at the next `step()`. Without that the
first optimizer step after a resume runs at whatever the fresh construction
left in the param groups -- step 0 of the warmup, which is near zero. One step
is not much, but the ladder branches on purpose and this is exactly the step
it branches at.
"""

from __future__ import annotations

from typing import Any

try:  # torchtitan's logger inside the container, stdlib's outside it.
    from torchtitan.tools.logging import logger
except ImportError:  # so the module -- and its tests -- import anywhere
    import logging

    logger = logging.getLogger(__name__)


def install(container) -> object:
    """Give `container` a state dict that carries the counter and not the shape.

    Patched on the instance rather than the class: `UniversalLR` comes from
    the container image, and a repo that reaches in and rewrites a vendored
    class changes behaviour for anything else importing it -- including a
    torchtitan whose own schedulers are fine.
    """
    schedulers = list(getattr(container, "schedulers", []) or [])
    if not schedulers:
        raise RuntimeError(
            "the LR scheduler container exposes no `schedulers`; titan_oellm's "
            "shape changed and this patch would silently do nothing")

    def state_dict() -> dict[str, Any]:
        return {"last_epoch": [int(s.last_epoch) for s in schedulers]}

    def load_state_dict(sd: dict[str, Any]) -> None:
        # Older checkpoints carry titan_oellm's own format. Read the counter
        # out of it and drop the rest, which is the whole point.
        if "last_epoch" in sd:
            steps = sd["last_epoch"]
        elif "scheduler_states" in sd:
            steps = [int(s.get("last_epoch", -1)) for s in sd["scheduler_states"]]
        else:
            raise KeyError(
                f"LR scheduler checkpoint has neither `last_epoch` nor "
                f"`scheduler_states` (has {sorted(sd)})")
        if len(steps) != len(schedulers):
            raise ValueError(
                f"checkpoint holds {len(steps)} scheduler counters but this "
                f"job built {len(schedulers)}")
        for sched, step in zip(schedulers, steps):
            sched.last_epoch = int(step)
            _apply(sched)
        logger.info(
            "Resumed %d LR scheduler(s) at step %s, keeping THIS job's shape "
            "(warm %s / main %s / cooldown %s)", len(schedulers), steps[0],
            schedulers[0].warm_steps, schedulers[0].main_steps,
            schedulers[0].cooldown_steps)

    container.state_dict = state_dict
    container.load_state_dict = load_state_dict
    return container


def _apply(sched) -> None:
    """Write the LR for the restored step into the optimizer, as `step()` does.

    `get_lr` warns when called outside a step, and the flag is how
    `LRScheduler.step` suppresses it.
    """
    sched._get_lr_called_within_step = True
    try:
        lrs = sched.get_lr()
    finally:
        sched._get_lr_called_within_step = False
    for group, lr in zip(sched.optimizer.param_groups, lrs):
        group["lr"] = lr
    sched._last_lr = list(lrs)
