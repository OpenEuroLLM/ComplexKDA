"""Save one checkpoint just before the job's wall clock, not only on the interval.

A campaign job runs in 12-hour allocations and resumes from its newest
checkpoint. The interval alone -- 9,728 steps on fwedu_1p3B, 1.5 to 1.65 hours
-- threw away whatever had trained since the last one at every wall: up to 1.6
hours of 32 GPUs, eight times over a four-arm campaign.

THE DEADLINE is SLURM_JOB_END_TIME, which Slurm (25.05 on JUPITER) sets for the
job and srun, apptainer and torchrun pass through to every rank -- the same way
TITAN_EXT_DECAY_ALL reaches them. Not a signal: `--signal` would have to cross
srun, apptainer and torchrun, and a SIGUSR1 that torchrun does not handle kills
it outright.

THE RANKS HAVE TO AGREE. A checkpoint is a collective save, so every rank must
enter it at the same step or the run hangs, and each rank reads its own clock.
Every CHECK_EVERY steps they all_reduce one vote with MAX and save together
when any of them votes yes. Every 20 steps rather than every step, because
reading the result back makes the host wait for the GPU.

THE VOTE is yes when the NEXT check could come too late:

    now + (time since the last check) + margin >= deadline

so a slow step cannot carry the run past the window. The margin covers the
save itself -- 1,336 ladder saves took 0.3 to 3.2 s -- with room for the
controller's clock. 180 s unless TITAN_EXT_PRE_WALL_MARGIN_S says otherwise.

AFTER THE SAVE NOTHING CHANGES: training carries on until Slurm ends the job.
Exiting cleanly would end it COMPLETED, which the monitor reads as finished and
never restarts; exiting non-zero would be a failure. Carrying on keeps the job
a TIMEOUT, restarted exactly as before, at the cost of the margin. A regular
save the wall cuts off is harmless: torchtitan resumes from the newest step
directory that has a `.metadata`.

THE SAVE IS TORCHTITAN'S OWN. `_should_save` is forced for that one call and
`save()` runs as it does on an interval step -- the same states, the same
dcp_save, the same `step-N` directory. Never `last_step=True`, which takes
`_save_last_step` and can write a model-only checkpoint.

WHAT IT CANNOT DO: create the directory the monitor reads as finished (that is
`step-<training.steps>`, and the last step is never forced), push an older
checkpoint out (keep_latest_k is 0, which keeps all of them), or change what a
chained cooldown loads (an explicit `step-${branch_step}`).

WHY THE CLASS. torchtitan's Trainer builds its CheckpointManager itself; the
train spec has no builder to wrap, unlike the metrics processor step_total.py
patches on the instance. A training process builds exactly one
CheckpointManager, so patching the class changes that one object and nothing
else.

IT MUST NOT BE ABLE TO STOP A RUN at import: a restarted job re-imports
titan_ext, and a crash here would be a restart loop. Anything it cannot read
leaves it off and says why. Off on purpose with TITAN_EXT_PRE_WALL=0.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

SWITCH = "TITAN_EXT_PRE_WALL"
MARGIN = "TITAN_EXT_PRE_WALL_MARGIN_S"
END = "SLURM_JOB_END_TIME"
DEFAULT_MARGIN_S = 180.0
CHECK_EVERY = 20


def _say(msg: str, loud: bool = True) -> None:
    # Once per job rather than once per rank: 32 copies bury the line.
    if loud:
        print(f"[titan_ext] {msg}", flush=True)


def deadline_from_env(env: Mapping[str, str] | None = None) -> tuple[float | None, str]:
    """The job's end as a unix time, or None and the reason it stays off."""
    env = os.environ if env is None else env
    if env.get(SWITCH, "1").strip() == "0":
        return None, f"{SWITCH}=0"
    raw = env.get(END, "").strip()
    if not raw:
        return None, f"{END} is not set"
    try:
        end = float(raw)
    except ValueError:
        end = float("nan")
    if not 0 < end < float("inf"):
        return None, f"{END}={raw!r} is not a time"
    return end, ""


def margin_from_env(env: Mapping[str, str] | None = None) -> tuple[float, str]:
    """Seconds before the end the save has to start by, and a note if refused."""
    env = os.environ if env is None else env
    raw = env.get(MARGIN, "").strip()
    if not raw:
        return DEFAULT_MARGIN_S, ""
    try:
        margin = float(raw)
    except ValueError:
        margin = float("nan")
    if not 0 <= margin < float("inf"):
        return DEFAULT_MARGIN_S, (f"{MARGIN}={raw!r} is not a number of "
                                  f"seconds; using {DEFAULT_MARGIN_S:g}")
    return margin, ""


def all_reduce_max(vote: int) -> int:
    """The largest vote over every rank; the vote itself outside a process group."""
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return vote
    # A CUDA tensor, because torchtitan's default group is NCCL, which has no
    # CPU path.
    device = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))
    flag = torch.tensor([vote], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item())


@dataclass
class Wall:
    """When to take the pre-wall checkpoint -- the same answer on every rank."""

    deadline: float
    margin: float = DEFAULT_MARGIN_S
    every: int = CHECK_EVERY
    clock: Callable[[], float] = time.time
    reduce: Callable[[int], int] = all_reduce_max
    loud: bool = True
    last_check: float | None = None
    fired: bool = False

    def __post_init__(self) -> None:
        # The first interval runs from here, so it includes building and
        # compiling the model: too long, which can only make it fire early.
        if self.last_check is None:
            self.last_check = self.clock()

    def vote(self, now: float) -> int:
        """1 if waiting for the next check could leave less than the margin."""
        interval = now - self.last_check
        self.last_check = now
        return int(now + interval + self.margin >= self.deadline)

    def due(self, step: int) -> bool:
        """True on at most one step. Every rank must call it with the same step."""
        if self.fired or step % self.every:
            return False
        self.fired = bool(self.reduce(self.vote(self.clock())))
        return self.fired


def _forced(*args: Any, **kwargs: Any) -> bool:
    return True


def wrap_save(save: Callable, wall: Wall) -> Callable:
    """`CheckpointManager.save`, plus the one forced save when the wall is near."""

    def pre_wall_save(self, curr_step: int, last_step: bool = False, *args, **kwargs):
        # Everything this reads is the same on every rank -- the step,
        # `last_step`, the checkpoint config -- so either all of them reach the
        # collective in `due()` or none does.
        armed = (not last_step
                 and bool(getattr(self, "enable", False))
                 and not getattr(self, "load_only", False))
        if not (armed and wall.due(curr_step)):
            return save(self, curr_step, last_step, *args, **kwargs)
        _say(f"pre-wall checkpoint at step {curr_step}, "
             f"{wall.deadline - wall.clock():.0f} s before the job ends", wall.loud)
        # Shadowed on the instance for this one call, so torchtitan's save()
        # runs exactly as it does on an interval step.
        self._should_save = _forced
        try:
            return save(self, curr_step, last_step, *args, **kwargs)
        finally:
            del self._should_save

    pre_wall_save.__wrapped__ = save
    pre_wall_save._pre_wall = True
    return pre_wall_save


def install(manager_cls: Any = None, env: Mapping[str, str] | None = None,
            **wall_kwargs: Any) -> Wall | None:
    """Arm from the environment and wrap `manager_cls.save`. Never raises.

    `manager_cls` is torchtitan's CheckpointManager unless a test passes its
    own, and `wall_kwargs` reach the Wall (a test's clock and reducer).
    Returns the Wall, or None when it stays off.
    """
    env = os.environ if env is None else env
    loud = env.get("RANK", "0") == "0"
    try:
        end, why = deadline_from_env(env)
        if end is None:
            _say(f"no pre-wall checkpoint: {why}", loud)
            return None
        margin, note = margin_from_env(env)
        if manager_cls is None:
            from torchtitan.components.checkpoint import CheckpointManager
            manager_cls = CheckpointManager
        original = manager_cls.save
        if getattr(original, "_pre_wall", False):
            original = original.__wrapped__
        wall = Wall(deadline=end, margin=margin, loud=loud, **wall_kwargs)
        manager_cls.save = wrap_save(original, wall)
    except Exception as e:  # noqa: BLE001 -- see IT MUST NOT BE ABLE TO STOP A RUN
        _say(f"no pre-wall checkpoint: {type(e).__name__}: {e}", loud)
        return None
    if note:
        _say(note, loud)
    ends = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
    _say(f"pre-wall checkpoint armed: the job ends {ends}; one save once fewer "
         f"than {margin:g} s plus a check interval are left, checked every "
         f"{wall.every} steps", loud)
    return wall
