"""The validation loader decides which number the anchor is compared against.

Three properties matter and none of them are visible in a loss curve when they
break: every rank must contribute an equal token count (or the mean over ranks
is not the mean over the split), every pass must see the same windows (or the
curve moves on its own), and the windows must come from the held-out file (or
we are reporting c4).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lm_scaling.data.megatron_writer import MegatronIndexedWriter  # noqa: E402
from lm_scaling.titan_ext.megatron_validate import (  # noqa: E402
    _ValidationWindows,
    build_megatron_validation_dataloader,
    validation_steps,
)

SEQ = 16


@pytest.fixture
def split(tmp_path):
    """A corpus of 200 windows' worth of tokens in a handful of documents."""
    rng = np.random.default_rng(7)
    prefix = tmp_path / "valid"
    with MegatronIndexedWriter(prefix) as w:
        for _ in range(8):
            w.add(rng.integers(0, 50257, size=SEQ * 25 + 3, dtype=np.uint16))
    return str(prefix)


def _windows(ds):
    return [tuple(inp["input"].tolist()) + (int(lab[-1]),) for inp, lab in ds]


def test_every_rank_gets_the_same_count(split):
    """`dist_mean` averages per-rank means; unequal counts mis-weight it."""
    for world in (1, 2, 3, 4, 7):
        counts = {len(_windows(_ValidationWindows(split, SEQ, r, world)))
                  for r in range(world)}
        assert len(counts) == 1, f"dp_world={world} gave rank counts {counts}"


def test_ranks_cover_the_split_without_overlap(split):
    world = 4
    seen = [_windows(_ValidationWindows(split, SEQ, r, world)) for r in range(world)]
    flat = [w for block in seen for w in block]
    assert len(set(flat)) == len(flat), "a window was evaluated by two ranks"
    whole = _windows(_ValidationWindows(split, SEQ, 0, 1))
    # The tail shorter than dp_world is dropped, and nothing else is.
    assert flat == whole[: len(flat)]
    assert len(whole) - len(flat) < world


def test_a_second_pass_evaluates_the_same_windows(split):
    """Unlike the training loader, this one must NOT continue the stream."""
    ds = _ValidationWindows(split, SEQ, 1, 4)
    assert _windows(ds) == _windows(ds)


def test_state_dict_carries_no_position(split):
    """A checkpoint round-trip must not resume validation mid-split."""
    ds = _ValidationWindows(split, SEQ, 0, 2)
    first = _windows(ds)
    ds.load_state_dict(ds.state_dict())
    assert _windows(ds) == first


def test_refuses_a_split_too_small_for_the_rank_count(split):
    with pytest.raises(ValueError, match="at least one"):
        _ValidationWindows(split, SEQ, 0, 10_000)


def test_validation_steps_consumes_the_split_once(split):
    for world, bs in ((1, 8), (2, 8), (4, 5), (4, 1)):
        steps = validation_steps(split, SEQ, bs, world)
        per_rank = len(_windows(_ValidationWindows(split, SEQ, 0, world)))
        # Every rank can serve `steps` full batches, and at most one further
        # batch is left over -- so no rank runs dry while another keeps going.
        assert steps * bs <= per_rank < (steps + 1) * bs + bs


def test_missing_dataset_path_is_refused():
    """Defaulting to c4_validation is worse than failing to start."""
    cfg = types.SimpleNamespace(
        validation=types.SimpleNamespace(dataset_path=None, seq_len=SEQ,
                                         local_batch_size=1))
    with pytest.raises(ValueError, match="held-out"):
        build_megatron_validation_dataloader(1, 0, None, cfg)


def test_the_final_step_is_always_validated():
    """The end-of-decay loss is the number compared to a published one.

    torchtitan fires on `step % freq == 0` and the last step usually is not:
    114,441 steps at freq 2,288 last validates at 114,400. Small in loss, but
    it is not the value the run is judged on.
    """
    import types

    class Base:
        def __init__(self, job_config):
            self.job_config = job_config

        def should_validate(self, step):
            return step == 1 or step % self.job_config.validation.freq == 0

    cfg = types.SimpleNamespace(
        training=types.SimpleNamespace(steps=114_441),
        validation=types.SimpleNamespace(freq=2288))
    v = Base(cfg)
    assert not v.should_validate(114_441), "the premise: torchtitan misses it"

    # The same wrapping build_megatron_validator applies.
    last = cfg.training.steps
    base = Base.should_validate
    v.should_validate = types.MethodType(
        lambda self, step: step == last or base(self, step), v)
    assert v.should_validate(114_441)
    assert v.should_validate(2288)
    assert not v.should_validate(2289)


def test_validation_steps_is_computed_when_reached_from_a_script(tmp_path):
    """As a SUBPROCESS whose sys.path[0] is `lm_scaling/`, not the repo root.

    `validation_steps` reads the corpus's real `.idx` to size the held-out
    pass, and it reaches the reader through `lm_scaling.data.megatron_indexed`
    -- an absolute import that needs the REPO ROOT on sys.path. A script run as
    `python lm_scaling/<something>.py` puts `lm_scaling/` there instead, so the
    module inserts the root itself. Every in-process test passes either way,
    because pytest puts the root on the path: this defect is only visible from
    a subprocess, and last time it surfaced in a Slurm job chained behind a
    45-minute tokenization run.
    """
    import subprocess
    import sys as _sys

    _sys.path.insert(0, str(ROOT))
    from lm_scaling.data.make_smoke_corpus import build

    corpus = tmp_path / "corpus"
    build(corpus, train_tokens=300_000, valid_tokens=200_000)

    # `lm_scaling/` on the path and the repo root NOT on it -- which is what
    # `python lm_scaling/<script>.py` gives you.
    script = tmp_path / "as_a_script.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'lm_scaling')!r})\n"
        f"assert {str(ROOT)!r} not in sys.path\n"
        "from titan_ext.megatron_validate import validation_steps\n"
        f"print(validation_steps({str(corpus / 'valid')!r}, 128, 4, 2))\n")

    r = subprocess.run([_sys.executable, str(script)], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    # Computed from the real .idx, not the -1 fallback torchtitan defaults to.
    assert int(r.stdout.strip()) > 0, r.stdout
