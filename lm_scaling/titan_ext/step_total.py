"""Put the run's length on torchtitan's step line: `step: 6922/45824`.

    [titan] ... - root - INFO - step: 6922/45824  loss:  3.3222  grad_norm: ...

A line that says how far along it is makes an ETA two timestamps. Without the
total every reader has to look up how long the run was meant to be -- watch.py
reads the campaign plan for exactly that -- and a log copied off the cluster
has lost the plan.

HOW. torchtitan builds the line as one f-string inside `MetricsProcessor.log`,
so there is no field to pass. Copying `log` into this repo to add one would
mean owning a vendored function that moves under us with every image. Instead
the processor the train spec builds gets its `log` wrapped: for the duration of
that one call a logging.Filter on torchtitan's logger rewrites the leading
`step: N` to `step: N/TOTAL`. The filter exists only inside the call and only
matches a message that STARTS with `step:`, so nothing else torchtitan logs --
`validate step:` included -- can be touched. The logger is the root logger in
this image, which is why both limits matter.

WHY `N/TOTAL` AND NOT A NEW FIELD. The number stays directly after `step:`,
where every parser here looks for it: the monitor's stall detector
(`\\bstep:\\s*([0-9]+)`), watch.py, the probe collectors. The one parser that
required whitespace straight after the number, loss_match.py, now accepts the
suffix; tests/lm/test_step_total.py runs every titan-line regex in the repo
against the new format.

IT MUST NOT BE ABLE TO STOP A RUN. A restarted campaign job re-imports
titan_ext, so a bug here would kill every requeue -- and the monitor restarts
on a non-zero exit, so it would kill them in a loop. Every step is guarded: a
processor that cannot be patched, a config without `training.steps`, a message
that does not match -- each leaves torchtitan's own line exactly as it was.

`training.steps` is the ABSOLUTE end step. A chained cooldown resumes at its
branch, so it reads `step: 36865/45824` -- where it is in the schedule, not in
this job's share of it.
"""

from __future__ import annotations

import contextlib
import logging
import re
import sys
from collections.abc import Callable
from typing import Any

# The training line only: optional ANSI colour codes, then `step:` at the very
# start. `validate step:` does not start with `step:`, so it cannot match.
_LEAD = re.compile(r"^((?:\x1b\[[0-9;]*m)*)step:(\s*)(\d+)(?=\s)")


def rewrite(msg: str, total: int) -> str:
    """`step: N  loss: ...` -> `step: N/TOTAL  loss: ...`; anything else as is."""
    return _LEAD.sub(
        lambda m: f"{m.group(1)}step:{m.group(2)}{m.group(3)}/{total}", msg,
        count=1)


class _StepTotal(logging.Filter):
    def __init__(self, total: int):
        super().__init__()
        self.total = total

    def filter(self, record: logging.LogRecord) -> bool:
        # A filter that raises takes the logging call -- and the training step
        # that made it -- down with it. Never.
        try:
            if isinstance(record.msg, str) and not record.args:
                record.msg = rewrite(record.msg, self.total)
        except Exception:
            pass
        return True


def patch_processor(proc: Any, total: int) -> Any:
    """Wrap THIS processor's `log` so its step line carries `total`.

    On the instance, not the class -- the same rule titan_ext/lr_state.py
    follows: a vendored class rewritten from this repo changes behaviour for
    everything else that imports it.
    """
    try:
        total = int(total)
        if total <= 0:
            return proc
        original = proc.log
        # The logger the line goes through, taken from the module that defines
        # the processor -- torchtitan.components.metrics keeps it as `logger`.
        mod = sys.modules.get(type(proc).__module__)
        target = getattr(mod, "logger", None) or logging.getLogger()
    except Exception:
        return proc

    flt = _StepTotal(total)

    def log(*args, **kwargs):
        try:
            target.addFilter(flt)
        except Exception:
            return original(*args, **kwargs)
        try:
            return original(*args, **kwargs)
        finally:
            with contextlib.suppress(Exception):
                target.removeFilter(flt)

    with contextlib.suppress(Exception):
        proc.log = log
    return proc


def with_step_total(build_fn: Callable) -> Callable:
    """A `build_metrics_processor_fn` whose processors report the run's length.

    torchtitan calls it as `(job_config, parallel_dims, model_args, tag)`.
    """
    def build(*args, **kwargs):
        proc = build_fn(*args, **kwargs)
        try:
            job_config = kwargs.get("job_config", args[0] if args else None)
            total = int(job_config.training.steps)
        except Exception:
            return proc
        return patch_processor(proc, total)

    build.__wrapped__ = build_fn
    return build
