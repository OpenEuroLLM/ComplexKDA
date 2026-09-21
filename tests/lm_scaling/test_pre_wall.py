"""One checkpoint just before the wall, taken the same way on every rank.

titan_ext/wall.py wraps torchtitan's CheckpointManager.save. What has to hold:

  * it arms only from a real end time, and otherwise stays off and says why;
  * every rank reaches the collective on the same steps or none does -- whether
    it is reached depends on the step and the checkpoint config, never a clock;
  * the save fires once, early enough that the next check could not be late;
  * the save is torchtitan's own interval save: never `last_step`, and the
    forced `_should_save` does not outlive the call;
  * nothing about installing it can raise into a job's startup. A restarted
    job re-imports titan_ext, and the monitor restarts on a failure, so a crash
    here would be a restart loop.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "lm_scaling"))

from lm_scaling.titan_ext import wall as W  # noqa: E402


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class Votes:
    """all_reduce(MAX) with the other ranks' vote fixed; records every call."""

    def __init__(self, others=0):
        self.others = others
        self.calls = []

    def __call__(self, vote):
        self.calls.append(vote)
        return max(vote, self.others)


class FakeManager:
    """`save` and `_should_save` shaped as torchtitan's CheckpointManager has them."""

    def __init__(self, interval=100, enable=True, load_only=False):
        self.interval, self.enable, self.load_only = interval, enable, load_only
        self.saved = []

    def _should_save(self, curr_step, last_step=False):
        if not self.enable or self.load_only:
            return False
        return last_step or curr_step % self.interval == 0

    def save(self, curr_step, last_step=False):
        if not self._should_save(curr_step, last_step):
            return
        self.saved.append((curr_step, last_step))


def fresh_cls():
    # A new subclass per test, so a patched `save` cannot leak between tests.
    return type("Manager", (FakeManager,), {})


def armed(cls, end=1000, margin=100, **kw):
    return W.install(cls, env={W.END: str(end), W.MARGIN: str(margin)}, **kw)


@pytest.mark.parametrize("env, why", [
    ({}, "is not set"),
    ({W.END: ""}, "is not set"),
    ({W.END: "Unknown"}, "is not a time"),
    ({W.END: "0"}, "is not a time"),
    ({W.END: "1757599720", W.SWITCH: "0"}, "TITAN_EXT_PRE_WALL=0"),
])
def test_it_stays_off_without_a_real_end_time(env, why, capsys):
    cls = fresh_cls()
    original = cls.save
    assert W.install(cls, env=env) is None
    assert cls.save is original
    assert why in capsys.readouterr().out


@pytest.mark.parametrize("raw, want, refused", [
    ("", 180.0, False), ("45", 45.0, False),
    ("-3", 180.0, True), ("soon", 180.0, True), ("nan", 180.0, True),
])
def test_the_margin(raw, want, refused):
    margin, note = W.margin_from_env({W.MARGIN: raw})
    assert margin == want
    assert bool(note) is refused


def test_it_fires_once_when_the_next_check_could_be_late():
    clock, votes = Clock(0.0), Votes()
    wall = W.Wall(deadline=1000.0, margin=100.0, every=20, clock=clock, reduce=votes)
    fired = []
    for step in range(1, 2001):
        clock.t = step * 0.5                    # 10 s between checks
        if wall.due(step):
            fired.append((step, clock.t))
    # The first check with t + 10 + 100 >= 1000 is t = 890, step 1780.
    assert fired == [(1780, 890.0)]
    # One collective per check up to the one that fired, and none after it.
    assert len(votes.calls) == 1780 // 20


def test_a_slow_step_cannot_carry_it_past_the_window():
    # 60 s between checks against a 100 s margin.
    clock = Clock(0.0)
    wall = W.Wall(deadline=1000.0, margin=100.0, clock=clock, reduce=Votes())
    step = 0
    while not wall.due(step := step + 1):
        clock.t = step * 3.0
        assert clock.t < 1000.0, "never fired"
    assert clock.t + 100.0 <= 1000.0            # the whole margin still left


def test_any_rank_voting_yes_saves_on_every_rank():
    wall = W.Wall(deadline=1e12, clock=Clock(), reduce=Votes(others=1))
    assert wall.due(20) is True                 # far from the end here, not elsewhere
    assert wall.due(40) is False                # and only once


def test_off_the_check_steps_there_is_no_collective():
    votes = Votes()
    wall = W.Wall(deadline=1e12, clock=Clock(), reduce=votes)
    assert not any(wall.due(step) for step in range(1, 20))
    assert votes.calls == []


def test_the_forced_save_is_an_interval_save_and_does_not_linger(capsys):
    cls, clock = fresh_cls(), Clock(0.0)
    armed(cls, clock=clock, reduce=Votes())
    m = cls()
    for step in range(1, 201):
        clock.t = step * 5.0                    # 100 s between checks
        m.save(step, last_step=(step == 200))
    # The interval's own save at 100, the forced one where t + 100 + 100 first
    # reaches 1000 (t = 800, step 160), and the last step as torchtitan saves it.
    assert m.saved == [(100, False), (160, False), (200, True)]
    assert "_should_save" not in vars(m)
    assert "pre-wall checkpoint at step 160, 200 s before" in capsys.readouterr().out


def test_a_forced_step_that_is_also_an_interval_step_saves_once():
    cls, clock = fresh_cls(), Clock(0.0)
    armed(cls, clock=clock, reduce=Votes())
    m = cls(interval=20)
    for step in range(1, 201):
        clock.t = step * 5.0
        m.save(step, last_step=(step == 200))
    # 160 is the forced step and an interval step: one save into step-160.
    assert m.saved.count((160, False)) == 1
    assert [s for s, _ in m.saved] == list(range(20, 201, 20))


@pytest.mark.parametrize("kw", [dict(enable=False), dict(load_only=True)])
def test_no_checkpointing_means_no_collective(kw):
    cls, votes, clock = fresh_cls(), Votes(others=1), Clock()
    armed(cls, clock=clock, reduce=votes)
    m = cls(**kw)
    for step in range(1, 201):
        m.save(step, last_step=(step == 200))
    assert m.saved == [] and votes.calls == []


def test_the_last_step_and_the_seed_checkpoint_pass_straight_through():
    cls, votes = fresh_cls(), Votes(others=1)   # would fire on any check
    armed(cls, clock=Clock(), reduce=votes)
    m = cls()
    m.save(curr_step=0, last_step=True)         # torchtitan's seed checkpoint
    m.save(20, last_step=True)                  # a last step that is a check step
    assert m.saved == [(0, True), (20, True)]
    assert votes.calls == []


def test_a_failing_save_still_restores_should_save():
    cls = fresh_cls()

    def broken(self, curr_step, last_step=False):
        assert self._should_save(curr_step, last_step)   # the forced one
        raise OSError("disk full")

    cls.save = broken
    armed(cls, clock=Clock(), reduce=Votes(others=1))
    m = cls()
    with pytest.raises(OSError):
        m.save(20)
    assert "_should_save" not in vars(m)


def test_installing_twice_wraps_once():
    cls = fresh_cls()
    original = cls.save
    armed(cls, clock=Clock(), reduce=Votes())
    armed(cls, clock=Clock(), reduce=Votes())
    assert cls.save.__wrapped__ is original


def test_installing_cannot_raise_into_startup(capsys):
    class NoSave:
        pass

    assert W.install(NoSave, env={W.END: "1000"}) is None
    assert "no pre-wall checkpoint: AttributeError" in capsys.readouterr().out


def test_only_rank_zero_speaks(capsys):
    cls = fresh_cls()
    W.install(cls, env={W.END: "1000", "RANK": "3"}, clock=Clock(), reduce=Votes(others=1))
    cls().save(20)
    assert capsys.readouterr().out == ""


def test_the_real_reducer_is_the_vote_itself_without_a_process_group():
    pytest.importorskip("torch")
    assert W.all_reduce_max(0) == 0
    assert W.all_reduce_max(1) == 1
