# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from pathlib import Path

import pytest

from evidence.aggregate_results import aggregate_audio, aggregate_state
from evidence.verify_archive import verify
from evidence.verify_sign_gauge import compare_case


def test_audio_aggregate_retains_rows_and_full_precision() -> None:
    rows = [
        {"model": "kda", "seed": 0, "metrics": {"16": {"waveform_mse": 0.123456789012345}}},
        {"model": "kda", "seed": 1, "metrics": {"16": {"waveform_mse": 0.223456789012345}}},
    ]
    result = aggregate_audio(rows)["models"]["kda"]
    assert result["per_seed"] == rows
    assert result["aggregate"]["16"]["waveform_mse"]["mean"] == pytest.approx(0.173456789012345)


def test_state_aggregate_reports_distribution_and_best_seed() -> None:
    rows = [
        {"task": "s3", "gate": "signed", "allow_neg_eigval": True, "seed": 0, "chance": 0.25, "acc": {"8": 0.4}},
        {"task": "s3", "gate": "signed", "allow_neg_eigval": True, "seed": 1, "chance": 0.25, "acc": {"8": 0.7}},
    ]
    setting = next(iter(aggregate_state(rows)["settings"].values()))
    assert setting["seeds"] == [0, 1]
    assert setting["aggregate"]["8"]["best_seed_scaled_accuracy"] == 0.6
    assert setting["aggregate"]["8"]["scaled_accuracy"]["count"] == 2


def test_sign_gauge_matches_output_state_and_backward() -> None:
    result = compare_case(seed=0, length=8, sign_mode="random")
    assert result["output"]["max_absolute"] == 0
    assert result["final_state"]["max_absolute"] == 0
    assert all(value["max_absolute"] == 0 for value in result["backward"].values())


def test_archived_audio_arrays_match_hashes_and_metrics() -> None:
    result = verify(Path("evidence/results/periodic_waveform"), metric_rtol=2e-4)
    assert result["status"] == "pass"
    assert result["arrays_checked"] == 6
