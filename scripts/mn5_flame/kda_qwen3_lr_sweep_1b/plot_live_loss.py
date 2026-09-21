# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

UPPER_LEARNING_RATES = (0.010, 0.014, 0.020, 0.028, 0.040)
LOWER_LEARNING_RATES = (0.0025, 0.0035, 0.0050, 0.0070, 0.0090)
METRIC = re.compile(r"step:\s*(\d+)\s+.*?loss:\s*([0-9.]+)")
TOKENS_PER_STEP = 98_304


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: dict[int, float] = {}
    for step, loss in METRIC.findall(path.read_text(errors="ignore")):
        values.setdefault(int(step), float(loss))
    if not values:
        raise ValueError(f"No loss metrics found in {path}")
    steps = np.asarray(sorted(values))
    return steps, np.asarray([values[step] for step in steps])


def configure_style() -> None:
    try:
        plt.style.use(["science", "no-latex", "light"])
    except OSError:
        plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", default="45575737")
    parser.add_argument("--lower-job-id")
    args = parser.parse_args()

    specs = [(args.job_id, task, rate) for task, rate in enumerate(UPPER_LEARNING_RATES)]
    if args.lower_job_id is not None:
        specs = [(args.lower_job_id, task, rate) for task, rate in enumerate(LOWER_LEARNING_RATES)] + specs
    available_specs = []
    curves = []
    for job_id, task, learning_rate in specs:
        path = args.log_dir / f"slurm-{job_id}_{task}.err"
        try:
            curve = parse_log(path)
        except (FileNotFoundError, ValueError):
            continue
        available_specs.append((job_id, task, learning_rate))
        curves.append(curve)
    if not curves:
        raise ValueError("No learning-rate metrics are available")
    learning_rates = tuple(spec[-1] for spec in available_specs)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.9, len(curves)))
    shared_step = min(int(steps[-1]) for steps, _ in curves)
    shared_tokens = shared_step * TOKENS_PER_STEP / 1e9

    configure_style()
    figure, (curve_axis, rate_axis) = plt.subplots(
        1,
        2,
        figsize=(7.7, 3.0),
        gridspec_kw={"width_ratios": (1.65, 1), "wspace": 0.3},
    )

    matched_losses = []
    for learning_rate, color, (steps, losses) in zip(learning_rates, colors, curves):
        matched = steps <= shared_step
        curve_axis.plot(
            steps[matched] * TOKENS_PER_STEP / 1e9,
            losses[matched],
            color=color,
            linewidth=1.35,
            label=f"{learning_rate:.4g}",
        )
        later = steps >= shared_step
        if later.sum() > 1:
            curve_axis.plot(steps[later] * TOKENS_PER_STEP / 1e9, losses[later], color=color, linewidth=1.0, alpha=0.28)

        eligible = np.flatnonzero(matched)
        window = eligible[-min(10, len(eligible)):]
        matched_losses.append(float(losses[window].mean()))

    curve_axis.axvline(shared_tokens, color="0.45", linestyle=":", linewidth=0.8)
    if max(steps[-1] for steps, _ in curves) >= 2000:
        curve_axis.axvline(2000 * TOKENS_PER_STEP / 1e9, color="0.6", linestyle="--", linewidth=0.8)
    curve_axis.set(xlabel="Tokens seen (B)", ylabel="Training cross-entropy loss", title="(a) Live learning-rate sweep")
    curve_axis.set_yscale("log")
    curve_axis.set_ylim(top=4.0)
    curve_axis.set_xlim(left=0)
    curve_axis.grid(True, alpha=0.22, linewidth=0.5)
    curve_axis.legend(title="Muon LR", frameon=False, ncol=1)

    order = np.argsort(learning_rates)
    sorted_rates = np.asarray(learning_rates)[order]
    sorted_losses = np.asarray(matched_losses)[order]
    sorted_colors = np.asarray(colors)[order]
    rate_positions = np.arange(len(sorted_rates))
    rate_axis.plot(rate_positions, sorted_losses, color="0.25", linewidth=0.8, alpha=0.6)
    rate_axis.scatter(rate_positions, sorted_losses, c=sorted_colors, s=32, edgecolors="white", linewidths=0.6, zorder=3)
    rate_axis.set(
        xlabel="Muon peak learning rate",
        ylabel="Matched training loss",
        title=f"(b) Mean ending at {shared_tokens:.3f}B tokens",
    )
    rate_axis.set_xticks(rate_positions, [f"{learning_rate:.4g}" for learning_rate in sorted_rates], rotation=40)
    rate_axis.grid(True, alpha=0.22, linewidth=0.5)

    figure.text(
        0.995,
        0.008,
        "Curves are fully opaque through the shared horizon and faint thereafter.",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="0.4",
    )
    figure.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.22)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
