# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Aggregate raw audio or state-tracking JSONL without rounding or dropping seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def aggregate_audio(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    output = {"kind": "periodic_waveform", "models": {}}
    for model, model_rows in sorted(grouped.items()):
        lengths = sorted({int(length) for row in model_rows for length in row["metrics"]})
        metrics = {}
        for length in lengths:
            names = sorted(model_rows[0]["metrics"][str(length)])
            metrics[str(length)] = {
                name: summarize([row["metrics"][str(length)][name] for row in model_rows])
                for name in names
            }
        output["models"][model] = {
            "seeds": [row["seed"] for row in model_rows],
            "per_seed": model_rows,
            "aggregate": metrics,
        }
    return output


def aggregate_state(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        key = (row["task"], row.get("variant", row.get("gate")), row.get("allow_neg_eigval"))
        grouped[key].append(row)
    output = {"kind": "state_tracking", "settings": {}}
    for (task, variant, allow_neg_eigval), setting_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        lengths = sorted({int(length) for row in setting_rows for length in row["acc"]})
        chance = setting_rows[0]["chance"]
        per_length = {}
        for length in lengths:
            raw = [row["acc"][str(length)] for row in setting_rows]
            scaled = [max(0.0, min(1.0, (value - chance) / (1.0 - chance))) for value in raw]
            per_length[str(length)] = {
                "accuracy": summarize(raw),
                "scaled_accuracy": summarize(scaled),
                "best_seed_scaled_accuracy": max(scaled),
            }
        name = f"{task}:{variant}:allow_neg_eigval={str(allow_neg_eigval).lower()}"
        output["settings"][name] = {
            "seeds": [row["seed"] for row in setting_rows],
            "selection_rule_in_paper": "maximum scaled accuracy across seeds",
            "per_seed": setting_rows,
            "aggregate": per_length,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("audio", "state"), required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not rows:
        raise ValueError("no result rows found")
    aggregate = aggregate_audio(rows) if args.kind == "audio" else aggregate_state(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
