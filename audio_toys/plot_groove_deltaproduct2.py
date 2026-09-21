# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Plot DeltaProduct-2 controls against the main waveform-continuation baselines."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402,F401

from audio_toys.plot_groove_waveform_comparison import load_model, parse_lengths  # noqa: E402
from audio_toys.train_groove_waveform import evaluate, make_waveform_pattern  # noqa: E402

plt.style.use(["science", "no-latex", "light"])
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams.update({
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "font.size": 8.0,
})


def read_rows(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", default="audio_toys/results/groove_waveform_d128_h8_hd16_all_core_cosine.jsonl")
    parser.add_argument("--default-result", default="audio_toys/results/groove_waveform_deltaproduct2.jsonl")
    parser.add_argument("--ungated-result", default="audio_toys/results/groove_waveform_deltaproduct2_ungated.jsonl")
    parser.add_argument("--reflection-result", default="audio_toys/results/groove_waveform_deltaproduct2_reflection.jsonl")
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("16,24,32,40,56,72,88,104,120,136,152,168,184,200,216,232,248,264"),
    )
    parser.add_argument("--output-json", default="audio_toys/results/groove_waveform_deltaproduct2_dense.json")
    parser.add_argument("--output", default="figures/groove_waveform_deltaproduct2.pdf")
    args = parser.parse_args()

    baselines = {row["model"]: row for row in read_rows(args.baseline_results)}
    experiments = (
        ("KDA, signed ranges", baselines["a11_b02"], "#A3333D", "D"),
        ("GRU", baselines["gru"], "#4B7B63", "v"),
        ("Causal Transformer", baselines["transformer"], "#27232A", "X"),
        ("Gated DeltaProduct-2, default init", read_rows(args.default_result)[0], "#8A8175", "o"),
        ("DeltaProduct-2, gate off", read_rows(args.ungated_result)[0], "#B7A98C", "s"),
        ("Gated DeltaProduct-2, reflection init", read_rows(args.reflection_result)[0], "#386C8C", "P"),
    )
    reference = experiments[0][1]
    pattern = make_waveform_pattern(reference["frame_size"], reference["source_sample_rate"], bpm=124, seed=2026)
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.01)

    metrics = {}
    figure, axis = plt.subplots(figsize=(5.5, 2.8))
    for label, row, color, marker in experiments:
        model = load_model(row)
        result = {
            str(length): evaluate(model, (pattern - mean) / std, mean, std, length, row["prefix_steps"]).__dict__
            for length in args.lengths
        }
        metrics[label] = result
        axis.plot(
            args.lengths,
            [result[str(length)]["waveform_mse"] for length in args.lengths],
            color=color,
            label=label,
            linewidth=1.8 if "reflection" in label or label in {"KDA, signed ranges", "GRU"} else 1.1,
            marker=marker,
            markersize=3.7,
            markevery=2,
        )

    axis.axvline(136, color="0.35", linestyle="--", linewidth=0.8)
    axis.set_yscale("log")
    axis.set_xticks([16, 72, 136, 200, 264])
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Waveform MSE")
    axis.grid(axis="y", which="major", color="0.84", linewidth=0.6)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2, fontsize=7.0, frameon=False)

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output)
    figure.savefig(os.path.splitext(args.output)[0] + ".png", dpi=300)


if __name__ == "__main__":
    main()
