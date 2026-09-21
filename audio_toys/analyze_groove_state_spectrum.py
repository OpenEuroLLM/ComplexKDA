# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Measure the blank-input state-transition spectrum of trained groove models."""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from audio_toys.train_groove_waveform import WaveformModel, make_waveform_pattern
from fla.layers.complex_kda_layer import compute_gate


def read_jsonl(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def project_step(model: WaveformModel, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
    layer = model.layer
    q = rearrange(layer.act(layer.q_proj(hidden)), "b (h d) -> b h d", d=layer.head_k_dim)
    k = rearrange(layer.k_act(layer.k_proj(hidden)), "b (h d) -> b h d", d=layer.head_k_dim)
    v = rearrange(layer.act(layer.v_proj(hidden)), "b (h d) -> b h d", d=layer.head_v_dim)
    q = F.normalize(q.float(), dim=-1, eps=1e-6)
    k = F.normalize(k.float(), dim=-1, eps=1e-6)
    raw_gate = rearrange(layer.f_proj(hidden), "b (h d) -> b h d", d=layer.head_k_dim)
    sign, log_magnitude = compute_gate(
        layer.gate,
        raw_gate,
        layer.A_log,
        layer.dt_bias,
        layer.lower_bound,
    )
    alpha = log_magnitude.exp()
    if sign is not None:
        alpha = alpha * sign
    beta_scale = 2.0 if layer.allow_neg_eigval else 1.0
    beta = torch.sigmoid(layer.b_proj(hidden).float()) * beta_scale
    output_gate = rearrange(layer.g_proj(hidden), "b (h d) -> b h d", d=layer.head_v_dim)
    return q, k, v, alpha, beta, output_gate


def recurrent_step(
    model: WaveformModel,
    hidden: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = model.layer
    q, k, v, alpha, beta, output_gate = project_step(model, hidden)
    state = state * alpha[..., None]
    residual = v - torch.einsum("bhk,bhkv->bhv", k, state)
    state = state + beta[..., None, None] * k[..., None] * residual[..., None, :]
    output = torch.einsum("bhk,bhkv->bhv", q * layer.head_k_dim**-0.5, state)
    output = layer.o_norm(output, output_gate)
    recurrent = layer.o_proj(rearrange(output, "b h d -> b (h d)"))
    return state, model.readout(hidden + recurrent)


def select_modes(
    state: torch.Tensor,
    transitions: list[torch.Tensor],
    complex_tolerance: float,
    keep_complex: bool,
) -> torch.Tensor:
    selected_heads = []
    for head, transition in enumerate(transitions):
        eigenvalues, eigenvectors = torch.linalg.eig(transition)
        coefficients = torch.linalg.solve(eigenvectors, state[:, head].double().to(torch.complex128))
        complex_mask = eigenvalues.imag.abs() > complex_tolerance
        keep = complex_mask if keep_complex else ~complex_mask
        coefficients[:, ~keep, :] = 0
        selected = torch.einsum("ij,bjv->biv", eigenvectors, coefficients).real.float()
        selected_heads.append(selected)
    return torch.stack(selected_heads, dim=1)


@torch.no_grad()
def measure_mode_ablation(
    model: WaveformModel,
    row: dict,
    transitions: list[torch.Tensor],
    complex_tolerance: float,
    bpm: float,
) -> dict:
    pattern = make_waveform_pattern(
        row["frame_size"],
        row["source_sample_rate"],
        bpm,
        seed=2026,
    )
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.01)
    length = 136
    prefix_steps = row["prefix_steps"]
    phases = torch.arange(pattern.shape[0])
    steps = torch.arange(length)
    target = (pattern[(phases[:, None] + steps[None]) % pattern.shape[0]] - mean) / std
    inputs = torch.zeros_like(target)
    inputs[:, :prefix_steps] = target[:, :prefix_steps]
    hidden = model.embed(inputs)

    state = hidden.new_zeros(
        hidden.shape[0],
        model.layer.num_v_heads,
        model.layer.head_k_dim,
        model.layer.head_v_dim,
    )
    for step in range(prefix_steps):
        state, _ = recurrent_step(model, hidden[:, step], state)
    complex_state = select_modes(state, transitions, complex_tolerance, keep_complex=True)
    real_state = select_modes(state, transitions, complex_tolerance, keep_complex=False)
    reconstruction_error = (state - complex_state - real_state).abs().max().item()

    states = {
        "baseline": state.clone(),
        "remove_complex": real_state,
        "complex_only": complex_state,
    }
    predictions = {name: [] for name in states}
    for step in range(prefix_steps, length):
        for name in states:
            states[name], prediction = recurrent_step(model, hidden[:, step], states[name])
            predictions[name].append(prediction)
    prediction_tensors = {name: torch.stack(value, dim=1) for name, value in predictions.items()}
    continuation_target = target[:, prefix_steps:]
    losses = {
        name: F.mse_loss(prediction, continuation_target).item()
        for name, prediction in prediction_tensors.items()
    }

    direct_prediction = model(inputs)[:, prefix_steps:]
    direct_manual_max_abs = (direct_prediction - prediction_tensors["baseline"]).abs().max().item()
    state_energy = state.square().sum().clamp_min(1e-20)
    return {
        "baseline_normalized_mse": losses["baseline"],
        "remove_complex_normalized_mse": losses["remove_complex"],
        "complex_only_normalized_mse": losses["complex_only"],
        "remove_complex_mse_ratio": losses["remove_complex"] / losses["baseline"],
        "complex_state_squared_norm_fraction": (complex_state.square().sum() / state_energy).item(),
        "modal_reconstruction_max_abs": reconstruction_error,
        "manual_vs_layer_prediction_max_abs": direct_manual_max_abs,
    }


@torch.no_grad()
def analyze_row(row: dict, complex_tolerance: float, bpm: float) -> dict:
    variant = row.get("model", row.get("variant"))
    model = WaveformModel(
        frame_size=row["frame_size"],
        d_model=row["d_model"],
        n_heads=row["heads"],
        head_dim=row["head_dim"],
        backend="naive_recurrent",
        variant=variant,
        gate_init_style=row["gate_init_style"],
        beta_init_style=row["beta_init_style"],
    )
    state_dict = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    blank = torch.zeros(1, 1, row["frame_size"])
    hidden = model.embed(blank)
    layer = model.layer
    k = layer.k_act(layer.k_proj(hidden))
    k = rearrange(k, "b t (h d) -> b t h d", d=layer.head_k_dim)
    k = F.normalize(k.float(), dim=-1, eps=1e-6)[0, 0]
    raw_gate = rearrange(layer.f_proj(hidden), "b t (h d) -> b t h d", d=layer.head_k_dim)
    sign, log_magnitude = compute_gate(
        layer.gate,
        raw_gate,
        layer.A_log,
        layer.dt_bias,
        layer.lower_bound,
    )
    alpha = log_magnitude.exp()[0, 0]
    if sign is not None:
        alpha = alpha * sign[0, 0]
    beta_scale = 2.0 if layer.allow_neg_eigval else 1.0
    beta = torch.sigmoid(layer.b_proj(hidden).float())[0, 0] * beta_scale
    blank_value = layer.act(layer.v_proj(hidden))[0, 0]

    heads = []
    transitions = []
    for head in range(layer.num_v_heads):
        direction = k[head].double()
        decay = torch.diag(alpha[head].double())
        delta = torch.eye(layer.head_k_dim, dtype=torch.float64) - beta[head].double() * torch.outer(
            direction,
            direction,
        )
        transition = delta @ decay
        transitions.append(transition)
        eigenvalues = torch.linalg.eigvals(transition).cpu().numpy()
        entries = []
        for eigenvalue in eigenvalues:
            angle = float(np.angle(eigenvalue))
            is_complex = abs(eigenvalue.imag) > complex_tolerance
            entries.append({
                "real": float(eigenvalue.real),
                "imag": float(eigenvalue.imag),
                "magnitude": float(abs(eigenvalue)),
                "angle_radians": angle,
                "period_steps": 2 * math.pi / abs(angle) if is_complex else None,
            })
        heads.append({
            "head": head,
            "alpha": alpha[head].tolist(),
            "beta": beta[head].item(),
            "spectral_radius": float(np.max(np.abs(eigenvalues))),
            "complex_eigenvalues": int(np.sum(np.abs(eigenvalues.imag) > complex_tolerance)),
            "eigenvalues": entries,
        })

    complex_per_head = sum(head["complex_eigenvalues"] for head in heads)
    analysis = {
        "model": variant,
        "seed": row["seed"],
        "checkpoint": row["checkpoint"],
        "blank_value_max_abs": blank_value.abs().max().item(),
        "per_head_state_shape": [layer.head_k_dim, layer.head_v_dim],
        "full_state_shape": [layer.num_v_heads, layer.head_k_dim, layer.head_v_dim],
        "value_dimension_multiplicity": layer.head_v_dim,
        "complex_eigenvalues_per_k_spectrum": complex_per_head,
        "complex_eigenvalues_in_full_state": complex_per_head * layer.head_v_dim,
        "heads": heads,
    }
    analysis["mode_ablation"] = measure_mode_ablation(
        model,
        row,
        transitions,
        complex_tolerance,
        bpm,
    )
    return analysis


def plot_spectra(analyses: list[dict], output: str) -> None:
    figure, axes = plt.subplots(1, len(analyses), figsize=(4.2 * len(analyses), 4.2), constrained_layout=True)
    if len(analyses) == 1:
        axes = [axes]
    theta = np.linspace(0, 2 * math.pi, 512)
    for axis, analysis in zip(axes, analyses):
        axis.plot(np.cos(theta), np.sin(theta), color="0.65", linewidth=1, linestyle="--")
        color_map = plt.get_cmap("tab10").resampled(len(analysis["heads"]))
        colors = [color_map(index) for index in range(len(analysis["heads"]))]
        for head, color in zip(analysis["heads"], colors):
            values = head["eigenvalues"]
            axis.scatter(
                [value["real"] for value in values],
                [value["imag"] for value in values],
                s=30,
                color=color,
                label=f"head {head['head']}",
            )
        axis.axhline(0, color="0.82", linewidth=0.8)
        axis.axvline(0, color="0.82", linewidth=0.8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-1.08, 1.08)
        axis.set_ylim(-1.08, 1.08)
        axis.set_xlabel("Real part")
        axis.set_ylabel("Imaginary part")
        total_eigenvalues = sum(len(head["eigenvalues"]) for head in analysis["heads"])
        axis.set_title(
            f"Seed {analysis['seed']} · "
            f"{analysis['complex_eigenvalues_per_k_spectrum']}/{total_eigenvalues} complex",
            loc="left",
            fontweight="bold",
        )
        axis.grid(color="0.90", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle("Complex-KDA transition spectrum", fontweight="bold", fontsize=14)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    figure.savefig(output, dpi=300)
    figure.savefig(os.path.splitext(output)[0] + ".pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="audio_toys/results/groove_waveform_d128_h8_hd16_all_core_cosine.jsonl",
    )
    parser.add_argument("--model", default="a11_b02")
    parser.add_argument(
        "--output-json",
        default="audio_toys/results/groove_waveform_d128_h8_hd16_all_core_cosine_spectrum.json",
    )
    parser.add_argument(
        "--output-figure",
        default="audio_toys/figures/groove_waveform_d128_h8_hd16_all_core_cosine_spectrum.png",
    )
    parser.add_argument("--bpm", type=float, default=124)
    parser.add_argument("--complex-tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    rows = [row for row in read_jsonl(args.results) if row.get("model", row.get("variant")) == args.model]
    if not rows:
        raise ValueError(f"model {args.model!r} not found in {args.results}")
    analyses = [analyze_row(row, args.complex_tolerance, args.bpm) for row in rows]
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(analyses, handle, indent=2)
        handle.write("\n")
    plot_spectra(analyses, args.output_figure)
    for analysis in analyses:
        radii = [head["spectral_radius"] for head in analysis["heads"]]
        total_eigenvalues = sum(len(head["eigenvalues"]) for head in analysis["heads"])
        total_full_state = total_eigenvalues * analysis["value_dimension_multiplicity"]
        print(
            f"seed={analysis['seed']} complex={analysis['complex_eigenvalues_per_k_spectrum']}/{total_eigenvalues} "
            f"full_complex={analysis['complex_eigenvalues_in_full_state']}/{total_full_state} "
            f"spectral_radius={max(radii):.8f}",
        )
    print(args.output_json)
    print(args.output_figure)


if __name__ == "__main__":
    main()
