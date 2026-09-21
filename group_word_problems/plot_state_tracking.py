# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Plot length generalization from the state-tracking reproduction driver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402,F401

TASKS = ("s3", "s4", "a5")
TASK_LABELS = {"s3": r"$S_3$", "s4": r"$S_4$", "a5": r"$A_5$"}
SETTINGS = (
    ("a11_b02", r"KDA $\alpha\in[-1,1]$, $\beta\in[0,2]$", "#3778C8", "-"),
    ("a01_b02", r"KDA $\alpha\in[0,1]$, $\beta\in[0,2]$", "#E66A3A", "-"),
    ("a11_b01", r"KDA $\alpha\in[-1,1]$, $\beta\in[0,1]$", "#27AE83", "-"),
    ("a01_b01", r"KDA $\alpha\in[0,1]$, $\beta\in[0,1]$", "#E5A100", "-"),
    ("deltaproduct2", r"DeltaProduct$_2$", "#8E5EA2", (0, (5, 2))),
    ("theory_init", "KDA theory init", "#8C564B", "-"),
)


def load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no completed runs in {path}")
    lengths = np.asarray(sorted(int(length) for length in rows[0]["acc"]))
    chance = rows[0]["chance"]
    accuracy = np.asarray([[row["acc"][str(length)] for length in lengths] for row in rows])
    scaled = np.clip((accuracy - chance) / (1 - chance), 0, 1)
    return lengths, scaled.max(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    available_tasks = [
        task for task in TASKS
        if any((args.results_dir / f"{task}_{name}.jsonl").exists() for name, *_ in SETTINGS)
    ]
    if not available_tasks:
        raise FileNotFoundError(f"no state-tracking results in {args.results_dir}")

    plt.style.use(["science", "no-latex", "light"])
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Times New Roman"
    plt.rcParams["mathtext.it"] = "Times New Roman:italic"
    plt.rcParams.update({"axes.linewidth": 1.0, "font.size": 11, "legend.fontsize": 9})
    figure, axes = plt.subplots(1, len(available_tasks), figsize=(3.5 * len(available_tasks), 2.65), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, task in zip(axes, available_tasks):
        for name, label, color, linestyle in SETTINGS:
            path = args.results_dir / f"{task}_{name}.jsonl"
            if not path.exists():
                continue
            lengths, values = load_curve(path)
            axis.plot(lengths, values, color=color, linestyle=linestyle, linewidth=1.8, label=label)
        axis.axvline(32, color="0.55", linestyle="--", linewidth=0.9)
        axis.set_title(TASK_LABELS[task])
        axis.set_xlim(1, max(line.get_xdata()[-1] for line in axis.lines))
        axis.set_ylim(-0.02, 1.04)
        axis.set_xlabel("Sequence length")
        axis.grid(True, linestyle="--", linewidth=0.5, color="0.88")
    axes[0].set_ylabel("Scaled accuracy")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6), frameon=False)
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.29, top=0.86, wspace=0.08)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
