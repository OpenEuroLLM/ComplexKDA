# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Plot matched KDA, GRU, and Transformer groove-continuation errors."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt  # noqa: E402
import scienceplots  # noqa: E402,F401
import torch  # noqa: E402

from audio_toys.train_groove_waveform import (  # noqa: E402
    WaveformDeltaProduct2,
    WaveformGRU,
    WaveformModel,
    WaveformTransformer,
    evaluate,
    make_waveform_pattern,
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
    "a01_b01": r"KDA, $\alpha\in[0,1]$, $\beta\in[0,1]$",
    "a01_b02": r"KDA, $\alpha\in[0,1]$, $\beta\in[0,2]$",
    "a11_b01": r"KDA, $\alpha\in[-1,1]$, $\beta\in[0,1]$",
    "a11_b02": r"Complex-KDA, $\alpha\in[-1,1]$, $\beta\in[0,2]$",
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
MARKERS = {
    "a01_b01": "o",
    "a01_b02": "s",
    "a11_b01": "^",
    "a11_b02": "D",
    "gru": "v",
    "transformer": "X",
    "transformer_noncausal": "P",
}


def parse_lengths(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def load_model(row: dict) -> torch.nn.Module:
    if row["model"] == "gru":
        model = WaveformGRU(row["frame_size"], row["d_model"])
    elif row["model"].startswith("deltaproduct2"):
        model = WaveformDeltaProduct2(
            row["frame_size"],
            row["d_model"],
            row["heads"],
            row["head_dim"],
            use_forget_gate=row.get("use_forget_gate", row["model"] == "deltaproduct2"),
        )
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
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        default=["audio_toys/results/groove_waveform_d128_h8_hd16_all_core_cosine.jsonl"],
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("16,24,32,40,56,72,88,104,120,136,152,168,184,200,216,232,248,264"),
    )
    parser.add_argument("--output-json", default="audio_toys/results/groove_waveform_d128_h8_hd16_dense_eval.json")
    parser.add_argument("--output", default="figures/groove_waveform_comparison.pdf")
    args = parser.parse_args()

    rows = {}
    for path in args.results:
        with open(path) as handle:
            rows.update({row["model"]: row for row in map(json.loads, handle)})
    missing = set(MODEL_ORDER) - rows.keys()
    if missing:
        raise ValueError(f"missing models: {sorted(missing)}")

    reference_row = rows[MODEL_ORDER[0]]
    pattern = make_waveform_pattern(
        reference_row["frame_size"],
        reference_row["source_sample_rate"],
        bpm=124,
        seed=2026,
    )
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.01)
    dense_metrics = {}
    for model_name in MODEL_ORDER:
        model = load_model(rows[model_name])
        dense_metrics[model_name] = {
            str(length): evaluate(model, (pattern - mean) / std, mean, std, length, rows[model_name]["prefix_steps"]).__dict__
            for length in args.lengths
        }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(dense_metrics, handle, indent=2)
        handle.write("\n")

    figure, axis = plt.subplots(figsize=(7.0, 3.7))
    for model_name in MODEL_ORDER:
        errors = [dense_metrics[model_name][str(length)]["waveform_mse"] for length in args.lengths]
        emphasized = model_name in {"a11_b02", "transformer", "transformer_noncausal"}
        axis.plot(
            args.lengths,
            errors,
            color=COLORS[model_name],
            label=LABELS[model_name],
            linewidth=2.1 if emphasized else 1.25,
            marker=MARKERS[model_name],
            markersize=6.2 if emphasized else 4.8,
            zorder=3 if emphasized else 2,
        )

    axis.axvline(136, color="0.35", linestyle="--", linewidth=0.9, zorder=1)
    axis.set_yscale("log")
    axis.set_xticks([16, 40, 72, 104, 136, 168, 200, 232, 264])
    axis.set_xlabel("Sequence length (sixteenth-note frames)")
    axis.set_ylabel("Waveform mean-squared error")
    axis.grid(axis="y", which="major", color="0.84", linewidth=0.7)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=3,
        fontsize=8.4,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.2,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output)
    png_output = os.path.splitext(args.output)[0] + ".png"
    figure.savefig(png_output, dpi=300)
    plt.close(figure)
    print(args.output)
    print(png_output)


if __name__ == "__main__":
    main()
