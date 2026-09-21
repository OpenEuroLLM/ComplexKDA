"""The ceiling probe: what it searches, and what it refuses to conclude.

A micro-batch ceiling is the one number on this ladder that cannot be derived
-- it has to be measured, in the stack that will run, on the card that will
run it. These tests pin the three properties that make the measurement worth
trusting.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))

pytest.importorskip("hydra_staged_sweep", reason="probe plans through the config layer")
pytest.importorskip("monitor", reason="config layer needs monitor")

import titan_mbs_probe as M  # noqa: E402


def test_candidates_are_powers_of_two_descending():
    """The paper's batch grid is 2^4..2^10, so a global batch over 4 GPUs
    leaves a power of two per GPU and only a power-of-two micro-batch divides
    it with integral gradient accumulation. A ceiling of 12 or 20 could not be
    used even if it fit."""
    assert M.candidates(16) == [16, 8, 4, 2, 1]
    assert M.candidates(8) == [8, 4, 2, 1]
    assert M.candidates(1) == [1]


def test_candidates_never_exceed_the_per_gpu_share():
    """A micro-batch above it cannot be used however well it fits, because
    gradient accumulation would have to be fractional."""
    assert max(M.candidates(6)) == 4
    assert max(M.candidates(31)) == 16


def test_the_cap_bounds_the_cost_at_the_upper_rungs():
    """`--max-mbs`. At 1.7B everything above a small N is a certain OOM, and
    a queue slot spent proving it is a queue slot."""
    assert M.candidates(32, 8) == [8, 4, 2, 1]
    assert M.candidates(2, 8) == [2, 1], "the per-GPU share still applies"


@pytest.mark.parametrize("got,want", [
    ({1: "fits", 2: "oom"}, 1),
    ({1: "fits", 2: "fits", 4: "oom"}, 2),
    ({1: "fits", 2: "fits", 4: "fits", 8: "oom"}, 4),
])
def test_a_ceiling_needs_an_oom_above_it(tmp_path, got, want):
    _write(tmp_path, "kda", "302M", got)
    best, problems = M.collect(tmp_path)
    assert best[("kda", "302M")] == want
    assert not problems


def test_a_probe_that_never_ran_is_not_a_ceiling(tmp_path):
    """The failure this guards against is silent and permanent: a job still
    queued, one killed by the wall clock, or one that died in setup looks
    exactly like an OOM to anything that only asks "did it reach a step".
    A ceiling that is too low costs gradient accumulation on every cell that
    reads it, forever, and comes back as a complaint rather than a number.
    """
    _write(tmp_path, "kda", "302M", {1: "fits", 2: "fits", 4: "unresolved"})
    best, problems = M.collect(tmp_path)
    assert ("kda", "302M") not in best
    assert any("not an OOM" in p for p in problems)


def test_the_largest_probed_is_reported_as_a_floor(tmp_path):
    """Nothing was tested above it, so it bounds nothing -- it is only the
    largest value known to work. 47M is exactly this case: 16 is all the
    ladder can use there, so nothing above it was probed."""
    _write(tmp_path, "attn", "47M", {8: "fits", 16: "fits"})
    best, problems = M.collect(tmp_path)
    assert best[("attn", "47M")] == 16
    assert any("floor, not a ceiling" in p for p in problems)


def test_reaching_a_step_is_the_evidence_not_starting(tmp_path):
    """torchtitan builds the model and the optimizer before the first
    forward, so a job that dies in the backward has still printed a good deal
    of encouraging log."""
    d = tmp_path / "mbs" / "kda_302M_m8"
    d.mkdir(parents=True)
    (d / "slurm-1.log").write_text(
        "Building qwen3_custom\nApplied FSDP to the model\n"
        "Training starts at step 1\ntorch.OutOfMemoryError: CUDA out of memory\n")
    best, problems = M.collect(tmp_path)
    assert not best, "a run that only STARTED must not count as fitting"


def _write(root: Path, arch: str, rung: str, verdicts: dict[int, str]) -> None:
    bodies = {
        "fits": "[titan] step: 12  loss: 7.6  tps: 57,944  tflops: 156.03  mfu: 50.01%\n",
        "oom": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n",
        "unresolved": "[titan] Training starts at step 1\n",
    }
    for mbs, verdict in verdicts.items():
        d = root / "mbs" / f"{arch}_{rung}_m{mbs}"
        d.mkdir(parents=True)
        (d / "slurm-1.log").write_text(bodies[verdict])


def test_a_larger_micro_batch_that_buys_nothing_is_refused(tmp_path):
    """Same speed, nearly twice the memory, is not an improvement.

    complex-kda at 1.7B on 8 GPUs runs at 117.5 TFLOP/s at mbs 2 and 122.3 at
    mbs 1, using 34.8 GiB against 19.9. Memory held for no throughput is pure
    risk over a run that lasts weeks: fragmentation grows and validation
    allocates on top of the training peak.
    """
    _write_speed(tmp_path, "complex-kda", "1.7B", {1: 122.3, 2: 117.5})
    best, problems = M.collect(tmp_path)
    assert best[("complex-kda", "1.7B")] == 1
    assert any("does not buy throughput" in p for p in problems)


def test_noise_does_not_shrink_a_micro_batch(tmp_path):
    """Twelve-step probes, so a percent or two is noise and must not cost a
    micro-batch that is genuinely as fast."""
    _write_speed(tmp_path, "attn", "47M", {8: 123.0, 16: 124.0})
    best, _ = M.collect(tmp_path)
    assert best[("attn", "47M")] == 8, "within 3%: take the cheaper one"

    _write_speed(tmp_path, "gdn", "47M", {8: 100.0, 16: 124.0})
    best, _ = M.collect(tmp_path)
    assert best[("gdn", "47M")] == 16, "a real gain keeps the larger batch"


def test_a_ceiling_that_costs_throughput_is_refused(tmp_path):
    """More micro-batch must never buy LESS throughput.

    Near the top of the card the allocator spends the step defragmenting and
    retrying rather than training, and the run survives -- it reaches a step,
    so every "did it fit" test says yes. Measured on JUWELS: gdn at 983M
    reaches a step at mbs 4 using 38.2 GiB of 39.5, at 1.1 TFLOP/s, where
    mbs 2 runs at 167.5. Taking the larger value would have made that cell
    150x slower with nothing in the log saying so.
    """
    _write_speed(tmp_path, "gdn", "983M", {1: 131.9, 2: 167.5, 4: 1.1})
    best, problems = M.collect(tmp_path)
    assert best[("gdn", "983M")] == 2, "the fastest candidate, not the largest"
    assert any("does not buy throughput" in p for p in problems)


def test_a_normal_throughput_curve_keeps_the_largest(tmp_path):
    """The guard must not fire on the usual shape, where throughput rises
    with the micro-batch and flattens."""
    _write_speed(tmp_path, "attn", "1.7B", {1: 144.8, 2: 169.1})
    best, problems = M.collect(tmp_path)
    assert best[("attn", "1.7B")] == 2
    assert not any("does not buy throughput" in p for p in problems)


def _write_speed(root: Path, arch: str, rung: str, tflops: dict[int, float]) -> None:
    for mbs, tf in tflops.items():
        d = root / "mbs" / f"{arch}_{rung}_m{mbs}"
        d.mkdir(parents=True)
        (d / "slurm-1.log").write_text(
            f"[titan] step: 12  loss: 7.6  memory: 20.0GiB(50.0%)  "
            f"tps: 5,000  tflops: {tf:.2f}  mfu: 40.00%\n")
