# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Measure the zero-input transition spectrum of a trained waveform DeltaProduct-2 model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from audio_toys.plot_groove_waveform_comparison import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with open(args.result) as handle:
        row = json.loads(handle.readline())
    model = load_model(row)
    layer = model.layer
    with torch.inference_mode():
        hidden = model.embed(torch.zeros(1, 1, row["frame_size"]))
        key = layer.k_proj(hidden).view(1, 1, layer.num_householder, layer.num_heads, layer.head_dim)
        key = F.normalize(key.float(), dim=-1)[0, 0]
        beta = (2 * torch.sigmoid(layer.b_proj(hidden).float())).view(
            1,
            1,
            layer.num_householder,
            layer.num_heads,
        )[0, 0]
        if layer.use_forget_gate:
            log_decay = -layer.A_log.float().exp() * F.softplus(layer.a_proj(hidden).float() + layer.dt_bias)
            decay = log_decay.exp()[0, 0]
        else:
            decay = torch.ones(layer.num_heads)
        generator = torch.Generator().manual_seed(2026)
        causal_input = torch.randn(2, 64, row["frame_size"], generator=generator)
        full_prediction = model(causal_input)
        prefix_prediction = model(causal_input[:, :31])
        causal_difference = (full_prediction[:, :31] - prefix_prediction).abs().max()

    identity = torch.eye(layer.head_dim)
    head_rows = []
    all_eigenvalues = []
    for head in range(layer.num_heads):
        transition = identity.clone()
        for factor in range(layer.num_householder):
            direction = key[factor, head]
            householder = identity - beta[factor, head] * torch.outer(direction, direction)
            transition = householder @ transition
        transition *= decay[head]
        eigenvalues = torch.linalg.eigvals(transition)
        all_eigenvalues.append(eigenvalues)
        complex_values = eigenvalues[eigenvalues.imag.abs() > 1e-5]
        phases = torch.angle(complex_values).abs()
        nonzero_phases = phases[phases > 1e-5]
        period = float(2 * math.pi / nonzero_phases.min()) if nonzero_phases.numel() else None
        head_rows.append({
            "head": head,
            "decay": float(decay[head]),
            "beta": [float(value) for value in beta[:, head]],
            "spectral_radius": float(eigenvalues.abs().max()),
            "complex_eigenvalues": int(complex_values.numel()),
            "longest_complex_period_steps": period,
        })

    all_eigenvalues = torch.cat(all_eigenvalues)
    report = {
        "result": args.result,
        "complex_eigenvalue_fraction": float((all_eigenvalues.imag.abs() > 1e-5).float().mean()),
        "maximum_spectral_radius": float(all_eigenvalues.abs().max()),
        "minimum_spectral_radius": float(all_eigenvalues.abs().min()),
        "causal_prefix_max_abs_difference": float(causal_difference),
        "heads": head_rows,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
