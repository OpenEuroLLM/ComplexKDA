# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Build the main-paper and appendix groove-waveform figures."""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from audio_toys.plot_groove_waveform_comparison import MARKERS  # noqa: E402
from audio_toys.plot_groove_waveform_continuations import (  # noqa: E402
    COLORS,
    CUE_COLOR,
    LABELS,
    MODEL_ORDER,
    TARGET_COLOR,
    load_prediction,
    waveform_envelope,
)

MAIN_MODELS = ("a01_b01", "a11_b02", "transformer", "gru")
APPENDIX_MODELS = ("a01_b02", "a11_b01")
PAPER_MODELS = MODEL_ORDER
TEXT_WIDTH_INCHES = 5.5
TEXT_SIZE = 10.0
LEGEND_SIZE = 8.0
PANEL_LABEL_SIZE = 8.0
AXIS_LABEL_SIZE = 8.0
BROKEN_AXIS_GAP = 10
MAIN_WAVEFORM_COLOR = "#344A5E"
MAIN_PANEL_LABELS = {
    "a01_b01": "KDA",
    "a11_b02": "KDA",
    "transformer": "Causal Transformer",
    "gru": "GRU",
}
MAIN_PANEL_SUBLABELS = {
    "a01_b01": r"$\alpha\in[0,1],\ \beta\in[0,1]$",
    "a11_b02": r"$\alpha\in[-1,1],\ \beta\in[0,2]$",
}
SHORT_LABELS = {
    "a01_b01": r"KDA: $\alpha\in[0,1],\ \beta\in[0,1]$",
    "a01_b02": r"KDA: $\alpha\in[0,1],\ \beta\in[0,2]$",
    "a11_b01": r"KDA: $\alpha\in[-1,1],\ \beta\in[0,1]$",
    "a11_b02": r"KDA: $\alpha\in[-1,1],\ \beta\in[0,2]$",
    "gru": "GRU",
    "transformer": "Causal Transformer",
}
PAPER_LEGEND_LABELS = {
    "a01_b01": r"KDA: $\alpha\in[0,1],\ \beta\in[0,1]$",
    "a01_b02": r"KDA: $\alpha\in[0,1],\ \beta\in[0,2]$",
    "a11_b01": r"KDA: $\alpha\in[-1,1],\ \beta\in[0,1]$",
    "a11_b02": r"KDA: $\alpha\in[-1,1],\ \beta\in[0,2]$",
    "gru": "GRU",
    "transformer": "Causal Transformer",
}


def load_rows(paths: list[str]) -> dict[str, dict]:
    rows = {}
    for path in paths:
        with open(path) as handle:
            rows.update({row["model"]: row for row in map(json.loads, handle)})
    missing = set(PAPER_MODELS) - rows.keys()
    if missing:
        raise ValueError(f"missing models: {sorted(missing)}")
    return rows


def prepare_envelopes(
    rows: dict[str, dict],
    total_steps: int,
    bpm: float,
    prefix_steps: int,
    immediate_stop: int,
    far_start: int,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    prepared = {}
    gap_start = immediate_stop + BROKEN_AXIS_GAP
    for model_name in PAPER_MODELS:
        target, prediction = load_prediction(rows[model_name], total_steps, bpm)
        cue = waveform_envelope(target, 0, prefix_steps, bins=105)
        immediate_target = waveform_envelope(target, prefix_steps, immediate_stop, bins=210)
        immediate_prediction = waveform_envelope(prediction, prefix_steps, immediate_stop, bins=210)
        far_target = waveform_envelope(target, far_start, total_steps, bins=330)
        far_prediction = waveform_envelope(prediction, far_start, total_steps, bins=330)
        prepared[model_name] = {
            "cue": cue,
            "immediate_target": immediate_target,
            "immediate_prediction": immediate_prediction,
            "far_target": (far_target[0] - far_start + gap_start, far_target[1], far_target[2]),
            "far_prediction": (far_prediction[0] - far_start + gap_start, far_prediction[1], far_prediction[2]),
        }
    return prepared


def fill_envelope(axis: plt.Axes, envelope: tuple[np.ndarray, np.ndarray, np.ndarray], color: str, alpha: float) -> None:
    axis.fill_between(*envelope, color=color, alpha=alpha, linewidth=0)


def plot_waveform_axis(
    axis: plt.Axes,
    model_name: str,
    envelopes: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    prefix_steps: int,
    immediate_stop: int,
    far_start: int,
    total_steps: int,
    show_ticks: bool,
    far_display_width: float,
    label_size: float = 7.1,
    tick_size: float = 6.5,
    panel_label: str | None = None,
    panel_sublabel: str | None = None,
    sublabel_size: float = 7.0,
    prediction_color: str | None = None,
    compact_ticks: bool = False,
    show_origin_label: bool = True,
    label_outside: bool = False,
) -> None:
    gap_start = immediate_stop + BROKEN_AXIS_GAP
    natural_far_width = total_steps - far_start
    display_stop = gap_start + far_display_width
    data = envelopes[model_name]

    def compress_far(
        envelope: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, lower, upper = envelope
        compressed_x = gap_start + (x - gap_start) * far_display_width / natural_far_width
        return compressed_x, lower, upper

    fill_envelope(axis, data["immediate_target"], TARGET_COLOR, 0.52)
    model_color = prediction_color or COLORS[model_name]
    fill_envelope(axis, data["immediate_prediction"], model_color, 0.76)
    fill_envelope(axis, compress_far(data["far_target"]), TARGET_COLOR, 0.52)
    fill_envelope(axis, compress_far(data["far_prediction"]), model_color, 0.76)
    fill_envelope(axis, data["cue"], CUE_COLOR, 0.92)
    axis.axvline(prefix_steps, color=CUE_COLOR, linewidth=0.85)
    axis.axhline(0, color="0.55", linewidth=0.32)
    axis.text((immediate_stop + gap_start) / 2, 0, "//", ha="center", va="center", fontsize=label_size)
    label_y = 1.04 if label_outside else 0.86
    label_vertical_alignment = "bottom" if label_outside else "top"
    axis.text(
        0.025,
        label_y,
        panel_label or LABELS[model_name],
        transform=axis.transAxes,
        ha="left",
        va=label_vertical_alignment,
        fontsize=label_size,
        clip_on=False,
    )
    if panel_sublabel:
        axis.text(
            0.23 if label_outside else 0.27,
            label_y if label_outside else 0.82,
            panel_sublabel,
            transform=axis.transAxes,
            ha="left",
            va=label_vertical_alignment,
            fontsize=sublabel_size,
            bbox=None if label_outside else {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.25},
            clip_on=False,
        )
    axis.set_xlim(0, display_stop)
    axis.set_ylim(-1, 1)
    axis.set_yticks([])
    if compact_ticks:
        axis.set_xticks(
            [0, prefix_steps, immediate_stop, display_stop],
            ["0" if show_origin_label else "", str(prefix_steps), str(immediate_stop), str(total_steps)],
        )
    else:
        axis.set_xticks(
            [0, prefix_steps, immediate_stop, gap_start, display_stop],
            ["0", str(prefix_steps), str(immediate_stop), str(far_start), str(total_steps)],
        )
    axis.tick_params(axis="x", labelsize=tick_size, labelbottom=show_ticks)


def plot_error_axis(
    axis: plt.Axes,
    dense_metrics: dict[str, dict[str, dict[str, float]]],
    model_names: tuple[str, ...],
    complete: bool,
    font_size: float = 7.0,
    show_legend: bool = True,
    show_title: bool = True,
    axis_label_size: float | None = None,
) -> None:
    lengths = [int(length) for length in next(iter(dense_metrics.values()))]
    plotted_errors = []
    for model_name in model_names:
        errors = [dense_metrics[model_name][str(length)]["waveform_mse"] for length in lengths]
        plotted_errors.extend(errors)
        emphasized = model_name in {"a11_b02", "transformer", "gru"}
        axis.plot(
            lengths,
            errors,
            color=COLORS[model_name],
            label=SHORT_LABELS[model_name],
            linewidth=1.7 if emphasized else 1.15,
            marker="o" if model_name.startswith("a") else MARKERS[model_name],
            markersize=4.2 if emphasized else 3.4,
            markevery=2,
            zorder=3 if emphasized else 2,
        )
    axis.axvline(136, color="0.35", linestyle="--", linewidth=0.8, zorder=1)
    axis.set_yscale("log")
    lower_limit = 10 ** np.floor(np.log10(min(plotted_errors)))
    upper_limit = 10 ** np.ceil(np.log10(max(plotted_errors)))
    if lower_limit == upper_limit:
        upper_limit *= 10
    axis.set_ylim(lower_limit, upper_limit)
    axis.yaxis.set_major_locator(LogLocator(base=10))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xticks([16, 72, 136, 200, 264])
    axis.tick_params(axis="both", labelsize=font_size)
    label_size = axis_label_size or font_size
    axis.set_xlabel("Sequence length", fontsize=label_size)
    axis.set_ylabel("Waveform MSE", fontsize=label_size)
    if show_title:
        axis.set_title("(b) Numerical comparison", loc="left", fontsize=font_size)
    axis.grid(axis="y", which="major", color="0.84", linewidth=0.6)
    if show_legend:
        axis.legend(
            loc="upper center" if complete else "center left",
            bbox_to_anchor=(0.5, -0.18) if complete else None,
            fontsize=font_size,
            frameon=False,
            ncols=2 if complete else 1,
            handlelength=1.8,
            borderaxespad=0.2,
            columnspacing=0.8,
        )


def waveform_legend(axis: plt.Axes) -> None:
    handles = [
        Patch(facecolor=CUE_COLOR, alpha=0.92, label="Observed cue"),
        Patch(facecolor=TARGET_COLOR, alpha=0.52, label="Target"),
        Patch(facecolor="#444444", alpha=0.76, label="Model continuation"),
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        ncols=3,
        frameon=True,
        framealpha=0.82,
        facecolor="white",
        edgecolor="none",
        fontsize=5.8,
        columnspacing=0.7,
        handlelength=1.2,
        borderaxespad=0.3,
    )


def save_figure(figure: plt.Figure, output: str, tight: bool = True) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if tight:
        figure.savefig(output)
        figure.savefig(os.path.splitext(output)[0] + ".png", dpi=300)
    else:
        figure.set_layout_engine("none")
        with plt.rc_context({"figure.constrained_layout.use": False, "savefig.bbox": None}):
            figure.savefig(output, bbox_inches=None)
            figure.savefig(os.path.splitext(output)[0] + ".png", bbox_inches=None, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        default=["audio_toys/results/groove_waveform_d128_h8_hd16_all_core_cosine.jsonl"],
    )
    parser.add_argument(
        "--metrics",
        default="audio_toys/results/groove_waveform_d128_h8_hd16_dense_eval.json",
    )
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--immediate-stop", type=int, default=24)
    parser.add_argument("--far-start", type=int, default=232)
    parser.add_argument("--total-steps", type=int, default=264)
    parser.add_argument(
        "--far-display-width",
        type=float,
        default=18,
        help="Displayed width of the far-horizon segment; its original step labels are retained.",
    )
    parser.add_argument("--bpm", type=float, default=124)
    parser.add_argument("--main-output", default="figures/groove_waveform_main.pdf")
    parser.add_argument("--appendix-output", default="figures/groove_waveform_appendix.pdf")
    args = parser.parse_args()

    rows = load_rows(args.results)
    with open(args.metrics) as handle:
        dense_metrics = json.load(handle)
    envelopes = prepare_envelopes(
        rows,
        args.total_steps,
        args.bpm,
        args.prefix_steps,
        args.immediate_stop,
        args.far_start,
    )

    main_figure = plt.figure(figsize=(TEXT_WIDTH_INCHES, 1.90), constrained_layout=False)
    main_grid = main_figure.add_gridspec(
        1,
        2,
        left=0.01,
        right=0.99,
        top=0.82,
        bottom=0.435,
        width_ratios=(1.45, 1),
        wspace=0.30,
    )
    main_waveform_grid = main_grid[0, 0].subgridspec(2, 2, wspace=0.06, hspace=0.68)
    main_waveform_axes = []
    for index, model_name in enumerate(MAIN_MODELS):
        axis = main_figure.add_subplot(
            main_waveform_grid[index // 2, index % 2],
            sharex=main_waveform_axes[0] if main_waveform_axes else None,
        )
        main_waveform_axes.append(axis)
        plot_waveform_axis(
            axis,
            model_name,
            envelopes,
            args.prefix_steps,
            args.immediate_stop,
            args.far_start,
            args.total_steps,
            show_ticks=index >= 2,
            far_display_width=args.far_display_width,
            label_size=PANEL_LABEL_SIZE,
            tick_size=TEXT_SIZE,
            panel_label=MAIN_PANEL_LABELS[model_name],
            panel_sublabel=MAIN_PANEL_SUBLABELS.get(model_name),
            prediction_color=MAIN_WAVEFORM_COLOR,
            compact_ticks=True,
            show_origin_label=index % 2 == 0,
            label_outside=True,
        )
    main_figure.text(
        main_waveform_axes[0].get_position().x0,
        0.98,
        "(a) Waveform continuations",
        ha="left",
        va="top",
        fontsize=TEXT_SIZE,
    )
    main_waveform_label_axis = main_figure.add_subplot(main_grid[0, 0], frameon=False)
    main_waveform_label_axis.set_xticks([])
    main_waveform_label_axis.set_yticks([])
    main_waveform_label_axis.set_xlabel("Sequence step (broken axis)", fontsize=AXIS_LABEL_SIZE, labelpad=17)
    main_waveform_label_axis.patch.set_visible(False)
    main_error_axis = main_figure.add_subplot(main_grid[0, 1])
    plot_error_axis(
        main_error_axis,
        dense_metrics,
        PAPER_MODELS,
        complete=True,
        font_size=TEXT_SIZE,
        show_legend=False,
        show_title=False,
        axis_label_size=AXIS_LABEL_SIZE,
    )
    main_error_axis.yaxis.set_label_coords(-0.09, 0.5)
    main_figure.text(
        main_error_axis.get_position().x0,
        0.98,
        "(b) Numerical comparison",
        ha="left",
        va="top",
        fontsize=TEXT_SIZE,
    )
    handles, _ = main_error_axis.get_legend_handles_labels()
    main_figure.legend(
        handles,
        [PAPER_LEGEND_LABELS[name] for name in PAPER_MODELS],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        fontsize=LEGEND_SIZE,
        frameon=False,
        ncols=3,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.65,
        borderaxespad=0,
    )
    save_figure(main_figure, args.main_output, tight=False)

    appendix_figure = plt.figure(figsize=(7.0, 3.45), constrained_layout=True)
    appendix_grid = appendix_figure.add_gridspec(
        len(APPENDIX_MODELS),
        2,
        width_ratios=(1.22, 1),
        wspace=0.12,
        hspace=0.05,
    )
    appendix_waveform_axes = []
    for index, model_name in enumerate(APPENDIX_MODELS):
        axis = appendix_figure.add_subplot(
            appendix_grid[index, 0],
            sharex=appendix_waveform_axes[0] if appendix_waveform_axes else None,
        )
        appendix_waveform_axes.append(axis)
        plot_waveform_axis(
            axis,
            model_name,
            envelopes,
            args.prefix_steps,
            args.immediate_stop,
            args.far_start,
            args.total_steps,
            show_ticks=index == len(APPENDIX_MODELS) - 1,
            far_display_width=args.far_display_width,
        )
    appendix_waveform_axes[0].set_title("(a) Additional continuations", loc="left", fontsize=8.5)
    waveform_legend(appendix_waveform_axes[0])
    appendix_waveform_axes[-1].set_xlabel("Sequence step (broken axis)", fontsize=7.5)
    appendix_error_axis = appendix_figure.add_subplot(appendix_grid[:, 1])
    plot_error_axis(appendix_error_axis, dense_metrics, PAPER_MODELS, complete=True)
    save_figure(appendix_figure, args.appendix_output)

    print(args.main_output)
    print(args.appendix_output)


if __name__ == "__main__":
    main()
