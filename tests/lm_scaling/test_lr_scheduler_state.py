"""A WSD stable phase must be re-usable by several cooldowns.

That is the whole economy of WSD on a scaling ladder: one constant-LR run to
0.8*D_max, then a short cooldown branching off it per budget. With Ajroldi's
intermediate decays at 6/12/20/30/50/80/120/200BT it is 263.6BT a cell against
518BT run separately.

It did not work, and the reason was one missing method. `UniversalLR` defines
no `state_dict`, so it inherits `LRScheduler`'s -- which serialises
`self.__dict__` minus the optimizer, INCLUDING `warm_steps`, `main_steps` and
`cooldown_steps`. A cooldown built its own shape, loaded the branch, and
`load_state_dict` replaced that shape with the stable run's, in which
`cooldown_steps` is 0. Measured at 47M attn to 3072 steps: a chained cooldown
gained 0.0016 nats where the same budget trained in one piece gained 0.1101.

A scheduler's state is WHERE IT IS, not WHAT IT IS.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))
sys.path.insert(0, str(_ROOT))

from lm_scaling.titan_ext.lr_state import install  # noqa: E402


class _Sched:
    """The parts of an LRScheduler this patch touches."""

    def __init__(self, cooldown_steps):
        self.last_epoch = 0
        self.warm_steps, self.main_steps = 100, 2358
        self.cooldown_steps = cooldown_steps
        self.optimizer = type("O", (), {"param_groups": [{"lr": 0.0}]})()
        self._get_lr_called_within_step = False
        self.saw_flag = None

    def get_lr(self):
        # The real one warns unless this is set; recording it pins that the
        # patch suppresses the warning the same way `step()` does.
        self.saw_flag = self._get_lr_called_within_step
        return [0.002 if self.last_epoch < self.warm_steps + self.main_steps
                else 0.0001]


class _Box:
    def __init__(self, scheds):
        self.schedulers = scheds


def test_the_checkpoint_carries_the_counter_and_not_the_shape():
    """The one property that makes a branch re-usable."""
    s = _Sched(cooldown_steps=0)
    s.last_epoch = 2458
    sd = install(_Box([s])).state_dict()
    assert sd == {"last_epoch": [2458]}
    for forbidden in ("cooldown_steps", "warm_steps", "main_steps", "total_steps"):
        assert forbidden not in sd


def test_a_cooldown_keeps_its_own_shape_across_the_load():
    """The failure itself: the stable run has cooldown_steps 0, and loading it
    used to overwrite the cooldown's 614 with that."""
    stable = _Sched(cooldown_steps=0)
    stable.last_epoch = 2458
    sd = install(_Box([stable])).state_dict()

    cooldown = _Sched(cooldown_steps=614)
    install(_Box([cooldown])).load_state_dict(sd)

    assert cooldown.last_epoch == 2458, "the counter must come from the branch"
    assert cooldown.cooldown_steps == 614, "the shape must NOT"


def test_the_learning_rate_is_applied_at_load():
    """Otherwise the first optimizer step after a resume runs at whatever the
    fresh construction left in the param groups -- step 0 of the warmup, which
    is near zero. The ladder branches on purpose, and this is the step it
    branches at."""
    # Resuming inside the constant phase: the branch's own LR, not step 0's.
    s = _Sched(cooldown_steps=614)
    install(_Box([s])).load_state_dict({"last_epoch": [2400]})
    assert s.optimizer.param_groups[0]["lr"] == 0.002
    assert s._last_lr == [0.002]
    assert s.saw_flag is True, "get_lr must be called the way step() calls it"
    assert s._get_lr_called_within_step is False, "and the flag reset after"

    # And resuming exactly ON the boundary -- warm 100 + main 2358 -- takes
    # the first cooldown step, which is where every branch of a chained
    # ladder lands.
    b = _Sched(cooldown_steps=614)
    install(_Box([b])).load_state_dict({"last_epoch": [2458]})
    assert b.optimizer.param_groups[0]["lr"] == 0.0001


def test_titan_oellms_own_checkpoint_format_is_still_readable():
    """Runs already on disk carry the old shape. Read the counter out of it
    and drop the rest, which is the point."""
    s = _Sched(cooldown_steps=614)
    install(_Box([s])).load_state_dict(
        {"num_schedulers": 1,
         "scheduler_states": [{"last_epoch": 2458, "cooldown_steps": 0,
                               "warm_steps": 100, "total_steps": 3072}]})
    assert s.last_epoch == 2458 and s.cooldown_steps == 614


def test_a_container_with_no_schedulers_is_refused():
    """titan_oellm's shape could change under us, and a patch that silently
    does nothing would put the ladder back to constant-LR runs wearing WSD
    labels -- which is how this cost two campaigns."""
    with pytest.raises(RuntimeError, match="exposes no"):
        install(_Box([]))


def test_a_checkpoint_from_a_different_world_is_refused():
    with pytest.raises(ValueError, match="counters"):
        install(_Box([_Sched(0), _Sched(0)])).load_state_dict({"last_epoch": [5]})


def test_an_unrecognisable_checkpoint_raises():
    with pytest.raises(KeyError, match="last_epoch"):
        install(_Box([_Sched(0)])).load_state_dict({"something_else": 1})
