"""A wall-clock estimate must not round-trip through FLOPs.

`throughput.yaml` records TFLOP/s. Turning that back into tokens per second
needs a FLOPs-per-token convention, and it has to be the SAME one the rate was
measured under. It was not:

    torchtitan   6*(nparams - nparams_embedding)
                   + 6*n_layers*n_heads*head_dims*seq_len
    throughput   6*N_non_embedding

The attention term the second one drops is not small at 4096, and it is
dropped for every arm -- so the whole campaign priced roughly 2x optimistic,
with the error largest at the small rungs where the term dominates. Nothing
was inconsistent internally, which is why it survived a hand check: the hand
check used the same convention as the code.

`token_rate.yaml` records torchtitan's own `tps:` field, so pricing needs no
denominator anyone can disagree about -- and `gpu_hours` has no fallback
through the FLOPs table at all: an unmeasured cell prices as None.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))

import throughput as T  # noqa: E402

REGIME = "gh200_seq4096_titan_g4"


def test_wall_clock_comes_from_a_measured_token_rate():
    """The number that prices a run is read, not derived."""
    assert T.token_rate("kda-sig-lowrank", "1.3B", 4, REGIME) == 34210.0
    # The campaign's own cell: 190,976 steps of 128 sequences at 4096, on 32
    # GPUs. gpu_hours returns GPU-hours, so the world size cancels out of the
    # arithmetic and only the per-GPU rate is used.
    hours = T.gpu_hours("kda-sig-lowrank", "1.3B", 190_976, 128, 4096, gpus=32,
                        gpu="gh200", mbs=4, prefer=REGIME)
    assert hours == pytest.approx(190_976 * 128 * 4096 / 34210.0 / 3600, rel=1e-9)


def test_the_two_tables_cover_the_same_cells():
    """A rate present in one and missing from the other is a cell that prices
    one way today and another way after an edit."""
    import yaml

    rates = yaml.safe_load((_ROOT / "lm_scaling" / "token_rate.yaml").read_text())
    flops = yaml.safe_load((_ROOT / "lm_scaling" / "throughput.yaml").read_text())
    for section, cells in rates.items():
        assert section in flops, section
        assert set(cells) == set(flops[section]), section
        for key, entry in cells.items():
            assert set(entry) == set(flops[section][key]), f"{section}/{key}"


def test_every_measured_rate_rises_with_the_micro_batch():
    """`token_rate` rounds a request DOWN on the ground that the lower
    neighbour under-promises. A non-monotone series breaks that silently --
    which is why the emitter drops points slower than a smaller micro-batch.
    """
    import yaml

    rates = yaml.safe_load((_ROOT / "lm_scaling" / "token_rate.yaml").read_text())
    for section, cells in rates.items():
        for key, entry in cells.items():
            series = [entry[m] for m in sorted(entry)]
            assert series == sorted(series), f"{section}/{key}: {entry}"


def test_an_unmeasured_cell_prices_as_None_rather_than_by_arithmetic():
    """There is deliberately NO fallback through throughput.yaml. Every cell
    this release runs was probed at its own geometry, so an unpriced cell is
    an unmeasured one -- and the honest answer to that is "no estimate", not a
    number reconstructed under a FLOPs convention that may not match."""
    assert T.token_rate("not-an-arm", "1.3B", 4, REGIME) is None
    assert T.gpu_hours("not-an-arm", "1.3B", 1000, 128, 4096, gpus=32,
                       gpu="gh200", mbs=4, prefer=REGIME) is None
