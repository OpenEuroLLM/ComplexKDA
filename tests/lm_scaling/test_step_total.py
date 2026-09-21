"""`step: 6922/45824` on torchtitan's metrics line, and nothing else changed.

The processor torchtitan builds gets its `log` wrapped so the line carries the
run's length (titan_ext/step_total.py). Three things have to hold:

  * the rewrite happens on the training line and only there -- torchtitan logs
    through the ROOT logger, so a filter left installed would see everything;
  * every parser in the repo still reads the line, the monitor's stall
    detector first among them, because it decides when a job is restarted;
  * nothing about it can raise into a training step. A restarted campaign job
    re-imports titan_ext, and the monitor restarts on a non-zero exit, so a
    crash here would be a restart loop.
"""

import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "lm_scaling"))

from lm_scaling.titan_ext.step_total import patch_processor, rewrite, with_step_total  # noqa: E402

# What the processor module exposes, the way torchtitan.components.metrics does.
logger = logging.getLogger()

LINE = ("step: 6922  loss:  3.3222  grad_norm:  0.9790  memory: 12.25GiB(12.90%)"
        "  tps: 500,705  tflops: 256.52  mfu: 25.94%")
NEW = LINE.replace("step: 6922", "step: 6922/45824", 1)
PREFIX = "[titan] 2026-09-11 10:06:39,645 - root - INFO - "


class FakeProcessor:
    """Logs the way MetricsProcessor.log does: one f-string, through `logger`."""

    def log(self, step, loss):
        logger.info(f"step: {step:2}  loss: {loss:7.4f}  grad_norm:  0.9790")


def test_the_training_line_gets_the_total():
    assert rewrite(LINE, 45824) == NEW
    # Colour codes lead the line when colour printing is on.
    red = "\x1b[31m"
    assert rewrite(f"{red}step:  7  loss: 1.0", 12) == f"{red}step:  7/12  loss: 1.0"


@pytest.mark.parametrize("msg", [
    "validate step: 5728  loss:  3.3253",
    "[rank0]: step: 7 is not how torchtitan starts its line",
    "Training starts at step 1",
    "step:  7/12  loss: 1.0",          # already carries a total
])
def test_nothing_else_is_touched(msg):
    assert rewrite(msg, 45824) == msg


def test_the_filter_lives_only_inside_the_call(caplog):
    proc = patch_processor(FakeProcessor(), 12)
    with caplog.at_level(logging.INFO):
        proc.log(7, 1.0)
        # The same text logged OUTSIDE the processor's call must be left alone:
        # a filter left on the root logger would rewrite everything.
        logger.info("step:  8  loss:  1.0000")
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs[0].startswith("step:  7/12  loss:"), msgs
    assert msgs[1] == "step:  8  loss:  1.0000", msgs
    assert not logger.filters, "the filter outlived the call"


def test_the_builder_reads_the_total_from_the_job_config(caplog):
    cfg = SimpleNamespace(training=SimpleNamespace(steps=45824))
    build = with_step_total(lambda job_config, parallel_dims, *a: FakeProcessor())
    proc = build(cfg, None)
    with caplog.at_level(logging.INFO):
        proc.log(6922, 3.3222)
    assert caplog.records[0].getMessage().startswith("step: 6922/45824  loss:")
    # Keyword form, too.
    proc = build(job_config=cfg, parallel_dims=None)
    assert proc.log.__name__ == "log"


@pytest.mark.parametrize("cfg", [
    None,
    SimpleNamespace(),                                  # no `training`
    SimpleNamespace(training=SimpleNamespace()),        # no `steps`
    SimpleNamespace(training=SimpleNamespace(steps="not a number")),
    SimpleNamespace(training=SimpleNamespace(steps=0)),
])
def test_a_config_it_cannot_read_leaves_the_line_as_it_was(cfg, caplog):
    build = with_step_total(lambda *a, **k: FakeProcessor())
    proc = build(cfg, None)
    with caplog.at_level(logging.INFO):
        proc.log(7, 1.0)
    assert caplog.records[0].getMessage().startswith("step:  7  loss:")


def test_a_processor_it_cannot_patch_still_trains():
    class Slotted:
        __slots__ = ()

        def log(self, step):
            return step

    proc = patch_processor(Slotted(), 12)
    assert proc.log(3) == 3


def test_a_broken_filter_cannot_raise_into_the_step(monkeypatch, caplog):
    import lm_scaling.titan_ext.step_total as st

    monkeypatch.setattr(st, "rewrite", lambda *a: 1 / 0)
    proc = patch_processor(FakeProcessor(), 12)
    with caplog.at_level(logging.INFO):
        proc.log(7, 1.0)                                # must not raise
    assert caplog.records[0].getMessage().startswith("step:  7  loss:")


def _stall_pattern() -> re.Pattern:
    spec = yaml.safe_load((_ROOT / "lm_scaling/config/job/auto_restart.yaml").read_text())
    (event,) = [e for e in spec["log_events"] if e["name"] == "stalled"]
    return re.compile(event["pattern"])


def test_every_parser_in_the_repo_reads_the_new_line():
    """The number has to stay right after `step:` for all of them.

    The monitor's stall detector matters most: it decides when a job is
    restarted, and a pattern that stopped matching would read every job as
    stalled after thirty minutes and restart it.
    """
    import titan_mbs_probe
    import watch

    line = PREFIX + NEW
    assert _stall_pattern().search(line).group(1) == "6922"
    m = watch._STEP.search(line)
    assert m and m.group(1) == "6922" and m.group(2) == "3.3222" and m.group(5) == "25.94"
    m = titan_mbs_probe._METRICS.search(line)
    assert m and m.group(1) == "6922" and m.group(4) == "256.52"
    assert re.search(r"\bstep:\s*\d+", line)            # the probe's "reached a step"

    # And the old format still reads, for every log already on disk.
    old = PREFIX + LINE
    assert _stall_pattern().search(old).group(1) == "6922"
    assert watch._STEP.search(old).group(1) == "6922"
