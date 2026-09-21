# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare positional encodings in the matched waveform Transformer."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scienceplots  # noqa: E402,F401
import torch  # noqa: E402

from audio_toys.plot_groove_waveform_comparison import load_model  # noqa: E402
from audio_toys.train_groove_waveform import (  # noqa: E402
    evaluate,
    frames_to_audio,
    make_waveform_pattern,
    predict_frames,
    write_wav,
)

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
    "font.size": 10.0,
})

MODEL_ORDER = (
    "a11_b02",
    "transformer",
    "transformer_rope",
    "transformer_rope_500k",
    "transformer_selective_rope",
    "transformer_no_pos",
)
LABELS = {
    "a11_b02": r"KDA, $\alpha\in[-1,1]$, $\beta\in[0,2]$",
    "transformer": "Transformer (sinusoidal)",
    "transformer_rope": r"Transformer (RoPE, $10^4$)",
    "transformer_rope_500k": r"Transformer (RoPE, $5{\times}10^5$)",
    "transformer_selective_rope": "Transformer (Selective RoPE)",
    "transformer_no_pos": "Transformer (no positions)",
}
COLORS = {
    "a11_b02": "#A3333D",
    "transformer": "#27232A",
    "transformer_rope": "#366A8C",
    "transformer_rope_500k": "#6C87A0",
    "transformer_selective_rope": "#5A3A7A",
    "transformer_no_pos": "#B78035",
}
MARKERS = {
    "a11_b02": "D",
    "transformer": "X",
    "transformer_rope": "o",
    "transformer_rope_500k": "^",
    "transformer_selective_rope": "P",
    "transformer_no_pos": "s",
}


def parse_lengths(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        default=[
            "audio_toys/results/groove_waveform_d128_h8_hd16_all_core_cosine.jsonl",
            "audio_toys/results/groove_waveform_transformer_rope.jsonl",
            "audio_toys/results/groove_waveform_transformer_rope_500k.jsonl",
            "audio_toys/results/groove_waveform_transformer_selective_rope.jsonl",
            "audio_toys/results/groove_waveform_transformer_no_pos.jsonl",
        ],
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("16,24,32,40,56,72,88,104,120,136,152,168,184,200,216,232,248,264"),
    )
    parser.add_argument("--bpm", type=float, default=124)
    parser.add_argument("--audio-sample-rate", type=int, default=8000)
    parser.add_argument("--output-json", default="audio_toys/results/groove_waveform_transformer_positions_dense.json")
    parser.add_argument("--audio-dir", default="audio_toys/audio/groove_waveform_transformer_positions")
    parser.add_argument("--output", default="figures/groove_waveform_transformer_positions.pdf")
    args = parser.parse_args()

    rows = {}
    for path in args.results:
        with open(path) as handle:
            rows.update({row["model"]: row for row in map(json.loads, handle)})
    missing = set(MODEL_ORDER) - rows.keys()
    if missing:
        raise ValueError(f"missing models: {sorted(missing)}")

    reference = rows[MODEL_ORDER[0]]
    pattern = make_waveform_pattern(
        reference["frame_size"],
        reference["source_sample_rate"],
        args.bpm,
        seed=2026,
    )
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.01)
    normalized_pattern = (pattern - mean) / std
    dense_metrics = {}
    predictions = {}
    for model_name in MODEL_ORDER:
        model = load_model(rows[model_name])
        dense_metrics[model_name] = {
            str(length): evaluate(
                model,
                normalized_pattern,
                mean,
                std,
                length,
                rows[model_name]["prefix_steps"],
            ).__dict__
            for length in args.lengths
        }
        _, predictions[model_name] = predict_frames(
            model,
            pattern,
            mean,
            std,
            max(args.lengths),
            rows[model_name]["prefix_steps"],
        )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(dense_metrics, handle, indent=2)
        handle.write("\n")

    figure, axis = plt.subplots(figsize=(7.0, 2.8))
    for model_name in MODEL_ORDER:
        errors = [dense_metrics[model_name][str(length)]["waveform_mse"] for length in args.lengths]
        axis.plot(
            args.lengths,
            errors,
            color=COLORS[model_name],
            label=LABELS[model_name],
            linewidth=1.8,
            marker=MARKERS[model_name],
            markersize=4.2,
            markevery=2,
        )
    axis.axvline(136, color="0.35", linestyle="--", linewidth=0.8)
    axis.set_yscale("log")
    axis.set_xticks([16, 40, 72, 104, 136, 168, 200, 232, 264])
    axis.set_xlabel("Sequence length (sixteenth-note frames)")
    axis.set_ylabel("Waveform MSE")
    axis.grid(axis="y", which="major", color="0.84", linewidth=0.6)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3, fontsize=7.4, frameon=False)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output)
    png_output = os.path.splitext(args.output)[0] + ".png"
    figure.savefig(png_output, dpi=300)
    plt.close(figure)

    target_frames = pattern[torch.arange(max(args.lengths)) % pattern.shape[0]].numpy()
    target_audio = frames_to_audio(target_frames, args.bpm, args.audio_sample_rate)
    rendered = {
        model_name: frames_to_audio(prediction, args.bpm, args.audio_sample_rate)
        for model_name, prediction in predictions.items()
    }
    peak = max(float(np.abs(target_audio).max()), *(float(np.abs(audio).max()) for audio in rendered.values()))
    gain = 0.94 / max(peak, 0.94)
    os.makedirs(args.audio_dir, exist_ok=True)
    write_wav(os.path.join(args.audio_dir, "target.wav"), gain * target_audio, args.audio_sample_rate)
    for model_name, audio in rendered.items():
        write_wav(os.path.join(args.audio_dir, f"{model_name}.wav"), gain * audio, args.audio_sample_rate)
        comparison = np.column_stack((target_audio, audio))
        write_wav(
            os.path.join(args.audio_dir, f"{model_name}_target-left_model-right.wav"),
            gain * comparison,
            args.audio_sample_rate,
        )

    print(args.output)
    print(png_output)
    print(args.output_json)
    print(args.audio_dir)


if __name__ == "__main__":
    main()
