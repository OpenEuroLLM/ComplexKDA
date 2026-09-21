# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fla.layers.complex_kda_layer import compute_gate
from fla.models.complex_kda import ComplexKDAConfig, ComplexKDAModel

VARIANTS = (
    (r"$\alpha\in[0,1],\ \beta\in[0,1]$ — std.", "sigmoid", "shipped", False, "standard", "#0072B2"),
    (r"$\alpha\in[-1,1],\ \beta\in[0,1]$ — std.", "signed_sigmoid2", "shipped", False, "standard", "#009E73"),
    (r"$\alpha\in[0,1],\ \beta\in[0,2]$ — std.", "sigmoid", "shipped", True, "standard", "#E69F00"),
    (r"$\alpha\in[-1,1],\ \beta\in[0,2]$ — std.", "signed_sigmoid2", "shipped", True, "standard", "#D55E00"),
    (r"$\alpha\in[-1,1],\ \beta\in[0,1]$ — gate spread", "signed_sigmoid2", "spread", False, "standard", "#009E73"),
    (r"$\alpha\in[0,1],\ \beta\in[0,2]$ — $\beta$ spread", "sigmoid", "shipped", True, "spread", "#E69F00"),
    (r"$\alpha\in[-1,1],\ \beta\in[0,2]$ — both spread", "signed_sigmoid2", "spread", True, "spread", "#D55E00"),
    (r"$\alpha\in[-1,1],\ \beta\in[0,2]$ — gate spread", "signed_sigmoid2", "spread", True, "standard", "#CC79A7"),
    (r"$\alpha\in[-1,1],\ \beta\in[0,2]$ — $\beta$ spread", "signed_sigmoid2", "shipped", True, "spread", "#56B4E9"),
)


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
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )


def smooth(values: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    radius = int(4 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


@torch.no_grad()
def sample_initialization(
    gate: str,
    gate_init_style: str,
    allow_neg_eigval: bool,
    beta_init_style: str,
    num_tokens: int,
    chunk_size: int,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    torch.manual_seed(42)
    config = ComplexKDAConfig(
        hidden_size=768,
        head_dim=64,
        num_heads=12,
        num_hidden_layers=1,
        intermediate_size=2816,
        hidden_ratio=None,
        vocab_size=128,
        use_short_conv=True,
        allow_neg_eigval=allow_neg_eigval,
        gate=gate,
        gate_init_style=gate_init_style,
        beta_init_style=beta_init_style,
    )
    attention = ComplexKDAModel(config).layers[0].attn.eval()
    alpha_counts = torch.zeros(num_bins, dtype=torch.float64)
    beta_counts = torch.zeros(num_bins, dtype=torch.float64)
    alpha_sum = alpha_square_sum = alpha_negative = 0.0
    beta_sum = beta_square_sum = 0.0
    alpha_n = beta_n = 0
    generator = torch.Generator().manual_seed(31415)

    for start in range(0, num_tokens, chunk_size):
        size = min(chunk_size, num_tokens - start)
        hidden = torch.randn(size, 768, generator=generator)
        hidden *= torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True))

        raw_alpha = attention.f_proj(hidden).reshape(size, 12, 64)
        sign, log_magnitude = compute_gate(
            attention.gate,
            raw_alpha,
            attention.A_log,
            attention.dt_bias,
            attention.lower_bound,
        )
        alpha = log_magnitude.exp()
        if sign is not None:
            alpha *= sign
        beta = torch.sigmoid(attention.b_proj(hidden)) * (2.0 if allow_neg_eigval else 1.0)

        alpha_flat = alpha.flatten().double()
        beta_flat = beta.flatten().double()
        alpha_counts += torch.histc(alpha_flat, bins=num_bins, min=-1.0, max=1.0)
        beta_counts += torch.histc(beta_flat, bins=num_bins, min=0.0, max=2.0)
        alpha_sum += alpha_flat.sum().item()
        alpha_square_sum += alpha_flat.square().sum().item()
        alpha_negative += (alpha_flat < 0).sum().item()
        beta_sum += beta_flat.sum().item()
        beta_square_sum += beta_flat.square().sum().item()
        alpha_n += alpha_flat.numel()
        beta_n += beta_flat.numel()

    alpha_density = alpha_counts.numpy() / (alpha_n * (2.0 / num_bins))
    beta_density = beta_counts.numpy() / (beta_n * (2.0 / num_bins))
    statistics = {
        "alpha_mean": alpha_sum / alpha_n,
        "alpha_std": max(alpha_square_sum / alpha_n - (alpha_sum / alpha_n) ** 2, 0.0) ** 0.5,
        "alpha_negative": alpha_negative / alpha_n,
        "beta_mean": beta_sum / beta_n,
        "beta_std": max(beta_square_sum / beta_n - (beta_sum / beta_n) ** 2, 0.0) ** 0.5,
    }
    del attention
    gc.collect()
    return smooth(alpha_density), smooth(beta_density), statistics


def draw_ridgelines(
    axis: plt.Axes,
    centers: np.ndarray,
    densities: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    title: str,
    xlabel: str,
) -> None:
    scale = 0.78 / max(float(density.max()) for density in densities)
    rows = np.arange(len(densities))[::-1]
    for row, density, color in zip(rows, densities, colors):
        height = density * scale
        axis.fill_between(centers, row, row + height, color=color, alpha=0.28, linewidth=0)
        axis.plot(centers, row + height, color=color, linewidth=1.15)
        axis.axhline(row, color="0.86", linewidth=0.45, zorder=0)
    axis.set_yticks(rows, labels)
    axis.set_ylim(-0.18, len(densities) - 0.02)
    axis.set_xlim(centers[0], centers[-1])
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=0.18, linewidth=0.5)
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.tick_params(axis="y", length=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("figures/kda_9arm_initializations.png"))
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--num-bins", type=int, default=240)
    parser.add_argument("--output-data", type=Path)
    args = parser.parse_args()

    alpha_densities = []
    beta_densities = []
    statistics = []
    for label, gate, gate_init_style, allow_neg_eigval, beta_init_style, _ in VARIANTS:
        alpha, beta, stats = sample_initialization(
            gate,
            gate_init_style,
            allow_neg_eigval,
            beta_init_style,
            args.num_tokens,
            args.chunk_size,
            args.num_bins,
        )
        alpha_densities.append(alpha)
        beta_densities.append(beta)
        statistics.append((label, stats))

    configure_style()
    figure, (alpha_axis, beta_axis) = plt.subplots(1, 2, figsize=(9.1, 5.15), gridspec_kw={"wspace": 0.12})
    labels = [variant[0] for variant in VARIANTS]
    colors = [variant[-1] for variant in VARIANTS]
    alpha_centers = np.linspace(-1.0, 1.0, args.num_bins, endpoint=False) + 1.0 / args.num_bins
    beta_centers = np.linspace(0.0, 2.0, args.num_bins, endpoint=False) + 1.0 / args.num_bins

    draw_ridgelines(alpha_axis, alpha_centers, alpha_densities, labels, colors, "(a) Gate initialization", r"$\alpha$")
    draw_ridgelines(beta_axis, beta_centers, beta_densities, [""] * len(labels), colors, "(b) Rate initialization", r"$\beta$")
    alpha_axis.axvline(0.0, color="0.4", linestyle=":", linewidth=0.75)
    beta_axis.axvline(1.0, color="0.4", linestyle=":", linewidth=0.75)
    figure.subplots_adjust(left=0.32, right=0.99, top=0.91, bottom=0.13)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    output_data = args.output_data or args.output.with_name(f"{args.output.stem}_data.npz")
    output_data.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_data,
        labels=np.asarray(labels),
        alpha_centers=alpha_centers,
        beta_centers=beta_centers,
        alpha_densities=np.stack(alpha_densities),
        beta_densities=np.stack(beta_densities),
        statistics_json=np.asarray(json.dumps(dict(statistics))),
        num_tokens=np.asarray(args.num_tokens),
        random_seeds=np.asarray([42, 31415]),
    )

    for label, stats in statistics:
        print(
            f"{label}: alpha mean={stats['alpha_mean']:+.4f}, std={stats['alpha_std']:.4f}, "
            f"negative={stats['alpha_negative']:.3f}; beta mean={stats['beta_mean']:.4f}, "
            f"std={stats['beta_std']:.4f}"
        )


if __name__ == "__main__":
    main()
