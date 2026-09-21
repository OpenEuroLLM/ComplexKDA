# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Plot compact waveform continuations for the matched groove models."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import scienceplots  # noqa: E402,F401
import torch  # noqa: E402

from audio_toys.train_groove_waveform import (  # noqa: E402
    WaveformGRU,
    WaveformModel,
    WaveformTransformer,
    make_waveform_pattern,
    predict_frames,
)

plt.style.use(["science", "no-latex", "light"])
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"
plt.rcParams.update({
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "font.size": 11.5,
})

MODEL_ORDER = (
    "a01_b01",
    "a01_b02",
    "a11_b01",
    "a11_b02",
    "gru",
    "transformer",
)
LABELS = {
    "a01_b01": r"KDA: $\alpha\in[0,1],\ \beta\in[0,1]$",
    "a01_b02": r"KDA: $\alpha\in[0,1],\ \beta\in[0,2]$",
    "a11_b01": r"KDA: $\alpha\in[-1,1],\ \beta\in[0,1]$",
    "a11_b02": r"Complex-KDA: $\alpha\in[-1,1],\ \beta\in[0,2]$",
    "gru": "GRU",
    "transformer": "Causal Transformer",
    "transformer_noncausal": "Non-causal Transformer",
}
COLORS = {
    "a01_b01": "#24445C",
    "a01_b02": "#6F8795",
    "a11_b01": "#C08A5B",
    "a11_b02": "#A3333D",
    "gru": "#4B7B63",
    "transformer": "#27232A",
    "transformer_noncausal": "#62537B",
}
CUE_COLOR = "#C59A45"
TARGET_COLOR = "#B8B5AE"


def waveform_envelope(
    frames: np.ndarray,
    start: int,
    stop: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    waveform = frames[start:stop].reshape(-1)
    boundaries = np.linspace(0, waveform.shape[0], bins + 1, dtype=int)
    minimum = np.array([
        waveform[left:right].min(initial=0)
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ])
    maximum = np.array([
        waveform[left:right].max(initial=0)
        for left, right in zip(boundaries[:-1], boundaries[1:])
    ])
    positions = start + (np.arange(bins) + 0.5) * (stop - start) / bins
    return positions, minimum, maximum


def load_prediction(row: dict, total_steps: int, bpm: float) -> tuple[np.ndarray, np.ndarray]:
    pattern = make_waveform_pattern(row["frame_size"], row["source_sample_rate"], bpm, seed=2026)
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.01)
    if row["model"] == "gru":
        model = WaveformGRU(row["frame_size"], row["d_model"])
    elif row["model"].startswith("transformer"):
        model = WaveformTransformer(
            row["frame_size"],
            row["d_model"],
            row["heads"],
            causal=row["model"] != "transformer_noncausal",
            position_encoding=row.get("position_encoding", row.get("init", "sinusoidal")),
        )
    else:
        model = WaveformModel(
            frame_size=row["frame_size"],
            d_model=row["d_model"],
            n_heads=row["heads"],
            head_dim=row["head_dim"],
            backend="naive_recurrent",
            variant=row["model"],
            gate_init_style=row["gate_init_style"],
            beta_init_style=row["beta_init_style"],
        )
    model.load_state_dict(torch.load(row["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    return predict_frames(model, pattern, mean, std, total_steps, row["prefix_steps"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        default=["audio_toys/results/groove_waveform_d128_h8_hd16_all_core_cosine.jsonl"],
    )
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--immediate-stop", type=int, default=24)
    parser.add_argument("--far-start", type=int, default=232)
    parser.add_argument("--total-steps", type=int, default=264)
    parser.add_argument("--bpm", type=float, default=124)
    parser.add_argument("--output", default="figures/groove_waveform_continuations.pdf")
    args = parser.parse_args()

    rows = {}
    for path in args.results:
        with open(path) as handle:
            rows.update({row["model"]: row for row in map(json.loads, handle)})
    gap_start = args.immediate_stop + 6
    far_width = args.total_steps - args.far_start
    display_stop = gap_start + far_width

    figure, axes = plt.subplots(3, 2, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for axis, model_name in zip(axes.flat, MODEL_ORDER):
        target, prediction = load_prediction(rows[model_name], args.total_steps, args.bpm)
        immediate_x, immediate_target_minimum, immediate_target_maximum = waveform_envelope(
            target, args.prefix_steps, args.immediate_stop, bins=210,
        )
        _, immediate_prediction_minimum, immediate_prediction_maximum = waveform_envelope(
            prediction, args.prefix_steps, args.immediate_stop, bins=210,
        )
        far_x, far_target_minimum, far_target_maximum = waveform_envelope(
            target, args.far_start, args.total_steps, bins=330,
        )
        _, far_prediction_minimum, far_prediction_maximum = waveform_envelope(
            prediction, args.far_start, args.total_steps, bins=330,
        )
        far_x = far_x - args.far_start + gap_start
        cue_x, cue_minimum, cue_maximum = waveform_envelope(target, 0, args.prefix_steps, bins=105)

        axis.fill_between(
            immediate_x,
            immediate_target_minimum,
            immediate_target_maximum,
            color=TARGET_COLOR,
            alpha=0.55,
            linewidth=0,
        )
        axis.fill_between(
            immediate_x,
            immediate_prediction_minimum,
            immediate_prediction_maximum,
            color=COLORS[model_name],
            alpha=0.72,
            linewidth=0,
        )
        axis.fill_between(
            far_x,
            far_target_minimum,
            far_target_maximum,
            color=TARGET_COLOR,
            alpha=0.55,
            linewidth=0,
        )
        axis.fill_between(
            far_x,
            far_prediction_minimum,
            far_prediction_maximum,
            color=COLORS[model_name],
            alpha=0.72,
            linewidth=0,
        )
        axis.fill_between(
            cue_x,
            cue_minimum,
            cue_maximum,
            color=CUE_COLOR,
            alpha=0.9,
            linewidth=0,
        )
        axis.axvline(args.prefix_steps, color=CUE_COLOR, linewidth=0.9)
        axis.axhline(0, color="0.55", linewidth=0.35)
        axis.text((args.immediate_stop + gap_start) / 2, 0, "//", ha="center", va="center", fontsize=12)
        axis.text(0.02, 0.86, LABELS[model_name], transform=axis.transAxes, ha="left", va="top", fontsize=8.7)
        axis.set_xlim(0, display_stop)
        axis.set_ylim(-1.0, 1.0)
        axis.set_yticks([])
        axis.set_xticks(
            [0, args.prefix_steps, args.immediate_stop, gap_start, display_stop],
            ["0", str(args.prefix_steps), str(args.immediate_stop), str(args.far_start), str(args.total_steps)],
        )
        axis.tick_params(axis="x", labelsize=7.5, labelbottom=True)

    key_axis = axes.flat[-1]
    key_axis.axis("off")
    key_axis.text(0.03, 0.68, "Gold: observed context", color=CUE_COLOR, fontsize=9.0, transform=key_axis.transAxes)
    key_axis.text(0.03, 0.40, "Grey: target waveform", color=TARGET_COLOR, fontsize=9.0, transform=key_axis.transAxes)
    key_axis.text(0.03, 0.12, "Color: model continuation", color="#27232A", fontsize=9.0, transform=key_axis.transAxes)
    figure.supxlabel("Sequence step (broken axis)", fontsize=8.8)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output)
    png_output = os.path.splitext(args.output)[0] + ".png"
    figure.savefig(png_output, dpi=300)
    plt.close(figure)
    print(args.output)
    print(png_output)


if __name__ == "__main__":
    main()
