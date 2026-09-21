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

VARIANTS = (
    ("base", 0, r"$\alpha[0,1],\ \beta[0,1]$ — std.", "#0072B2", "-"),
    ("base", 1, r"$\alpha[-1,1],\ \beta[0,1]$ — std.", "#009E73", "-"),
    ("base", 2, r"$\alpha[0,1],\ \beta[0,2]$ — std.", "#E69F00", "-"),
    ("base", 3, r"$\alpha[-1,1],\ \beta[0,2]$ — std.", "#D55E00", "-"),
    ("base", 4, r"$\alpha[-1,1],\ \beta[0,1]$ — gate spread", "#009E73", "--"),
    ("base", 5, r"$\alpha[0,1],\ \beta[0,2]$ — $\beta$ spread", "#E69F00", "--"),
    ("base", 6, r"$\alpha[-1,1],\ \beta[0,2]$ — both spread", "#D55E00", "--"),
    ("extra", 0, r"$\alpha[-1,1],\ \beta[0,2]$ — gate spread", "#CC79A7", "-."),
    ("extra", 1, r"$\alpha[-1,1],\ \beta[0,2]$ — $\beta$ spread", "#56B4E9", "-."),
)
METRIC = re.compile(r"step:\s*(\d+)\s+.*?loss:\s*([0-9.]+)")
TOKENS_PER_STEP = 98_304


def parse_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: dict[int, float] = {}
    for match in METRIC.finditer(path.read_text(errors="ignore")):
        values.setdefault(int(match.group(1)), float(match.group(2)))
    if not values:
        raise ValueError(f"No loss metrics found in {path}")
    steps = np.asarray(sorted(values))
    losses = np.asarray([values[step] for step in steps])
    return steps, losses


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
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", default="45568662")
    parser.add_argument("--extra-job-id")
    parser.add_argument("--combined-job-id")
    parser.add_argument("--condition-label")
    parser.add_argument("--loss-top", type=float, default=4.0)
    parser.add_argument("--expected-arms", type=int, default=9)
    args = parser.parse_args()

    if args.combined_job_id is not None:
        specs = [(variant, args.combined_job_id, task) for task, variant in enumerate(VARIANTS)]
    else:
        job_ids = {"base": args.job_id, "extra": args.extra_job_id}
        specs = [(variant, job_ids[variant[0]], variant[1]) for variant in VARIANTS if job_ids[variant[0]] is not None]

    loaded = []
    for variant, job_id, task in specs:
        try:
            curve = parse_log(args.log_dir / f"slurm-{job_id}_{task}.err")
        except (FileNotFoundError, ValueError):
            continue
        loaded.append((variant, curve))
    if not loaded:
        raise ValueError("No training metrics are available")
    selected_variants, curves = zip(*loaded)
    base_curves = [curve for variant, curve in zip(selected_variants, curves) if variant[0] == "base"]
    if not base_curves or selected_variants[0][1] != 0:
        raise ValueError("The standard KDA baseline is required")
    common_step = min(int(steps[-1]) for steps, _ in base_curves)
    common_tokens = common_step * TOKENS_PER_STEP / 1e9

    configure_style()
    figure, (loss_axis, delta_axis) = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.55),
        gridspec_kw={"width_ratios": (1.55, 1)},
    )

    for (source, _, label, color, line_style), (steps, losses) in zip(selected_variants, curves):
        comparable = steps <= common_step if source == "base" else np.ones_like(steps, dtype=bool)
        loss_axis.plot(
            steps[comparable] * TOKENS_PER_STEP / 1e9,
            losses[comparable],
            color=color,
            linestyle=line_style,
            linewidth=1.35,
            label=label,
        )
        tail = steps >= common_step if source == "base" else np.zeros_like(steps, dtype=bool)
        if tail.sum() > 1:
            loss_axis.plot(
                steps[tail] * TOKENS_PER_STEP / 1e9,
                losses[tail],
                color=color,
                linestyle=line_style,
                linewidth=1.0,
                alpha=0.32,
            )

    baseline_steps, baseline_losses = curves[0]
    baseline = dict(zip(baseline_steps.tolist(), baseline_losses.tolist()))
    for (_, _, label, color, line_style), (steps, losses) in zip(selected_variants[1:], curves[1:]):
        aligned_steps = np.asarray([step for step in steps if step <= common_step and step in baseline])
        aligned_loss = np.asarray([losses[np.where(steps == step)[0][0]] for step in aligned_steps])
        delta = aligned_loss - np.asarray([baseline[step] for step in aligned_steps])
        delta_axis.plot(
            aligned_steps * TOKENS_PER_STEP / 1e9,
            delta,
            color=color,
            linestyle=line_style,
            linewidth=1.25,
            label=label,
        )

    for axis in (loss_axis, delta_axis):
        axis.grid(True, alpha=0.22, linewidth=0.5)
        axis.axvline(common_tokens, color="0.45", linestyle=":", linewidth=0.8)
        axis.set_xlabel("Tokens seen (B)")

    warmup_tokens = 2000 * TOKENS_PER_STEP / 1e9
    loss_axis.axvline(warmup_tokens, color="0.6", linestyle="--", linewidth=0.8)
    loss_axis.text(warmup_tokens, args.loss_top * 0.995, "warm-up end", rotation=90, va="top", ha="right", color="0.4")
    loss_axis.text(
        common_tokens,
        args.loss_top * 0.995,
        "shared horizon",
        rotation=90,
        va="top",
        ha="right",
        color="0.4",
    )
    loss_axis.set_ylabel("Training cross-entropy loss")
    loss_axis.set_yscale("log")
    loss_axis.set_ylim(top=args.loss_top)
    condition = f" ({args.condition_label})" if args.condition_label else ""
    loss_axis.set_title(f"(a) Live loss curves{condition}")
    loss_axis.set_xlim(left=0)

    delta_axis.axhline(0, color="0.25", linewidth=0.75)
    delta_axis.set_ylabel(r"Loss difference from $\alpha\in[0,1],\ \beta\in[0,1]$ (std.)")
    delta_axis.set_title("(b) Matched-step difference")
    delta_axis.set_xlim(0, common_tokens)

    handles, labels = loss_axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    if args.combined_job_id is None:
        note = (
            f"Original seven: full opacity through {common_tokens:.2f}B shared tokens, faint thereafter; "
            "two new arms started later."
        )
    else:
        note = (
            f"{len(curves)}/{args.expected_arms} arms available; full opacity through {common_tokens:.2f}B shared tokens, "
            "faint thereafter."
        )
    figure.text(
        0.995,
        0.012,
        note,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color="0.4",
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.34, wspace=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
