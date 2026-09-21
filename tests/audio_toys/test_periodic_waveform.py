# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from audio_toys.train_groove import make_batch
from audio_toys.train_groove_waveform import (
    WaveformDeltaProduct2,
    WaveformModel,
    initialize_deltaproduct2_reflections,
    predict_all_phases,
)


def test_waveform_batch_contains_only_the_cue() -> None:
    pattern = torch.arange(32 * 4, dtype=torch.float32).view(32, 4)
    inputs, targets = make_batch(pattern, 5, 40, 8, torch.Generator().manual_seed(0))
    torch.testing.assert_close(inputs[:, :8], targets[:, :8])
    torch.testing.assert_close(inputs[:, 8:], torch.zeros_like(inputs[:, 8:]))
    torch.testing.assert_close(targets[:, 32:], targets[:, :8])


def test_complex_kda_waveform_model_is_strictly_causal() -> None:
    torch.manual_seed(0)
    model = WaveformModel(
        frame_size=4,
        d_model=16,
        n_heads=2,
        head_dim=8,
        backend="naive_recurrent",
        variant="a11_b02",
    ).eval()
    inputs = torch.randn(2, 12, 4)
    changed = inputs.clone()
    changed[:, 7:] = torch.randn_like(changed[:, 7:])
    with torch.inference_mode():
        baseline = model(inputs)
        perturbed = model(changed)
    torch.testing.assert_close(baseline[:, :7], perturbed[:, :7], rtol=0, atol=0)


def test_deltaproduct_reflection_initialization_sets_persistent_factors() -> None:
    model = WaveformDeltaProduct2(frame_size=4, d_model=16, n_heads=2, head_dim=8, use_forget_gate=True)
    initialize_deltaproduct2_reflections(model, gate_magnitude=0.9999)
    hidden = model.embed(torch.zeros(1, 1, 4))
    with torch.inference_mode():
        beta = 2 * torch.sigmoid(model.layer.b_proj(hidden))
        log_decay = -model.layer.A_log.exp() * torch.nn.functional.softplus(
            model.layer.a_proj(hidden) + model.layer.dt_bias,
        )
    torch.testing.assert_close(beta, torch.full_like(beta, 1.998), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(log_decay.exp(), torch.full_like(log_decay, 0.9999), rtol=1e-5, atol=1e-6)


def test_prediction_export_enumerates_every_phase() -> None:
    model = WaveformModel(
        frame_size=4,
        d_model=16,
        n_heads=2,
        head_dim=8,
        backend="naive_recurrent",
        variant="a11_b02",
    ).eval()
    pattern = torch.randn(6, 4)
    mean = pattern.mean(0)
    std = pattern.std(0).clamp_min(0.01)
    phases, inputs, targets, predictions = predict_all_phases(model, pattern, mean, std, length=10, prefix_steps=3)
    assert phases.tolist() == list(range(6))
    assert inputs.shape == targets.shape == predictions.shape == (6, 10, 4)
    torch.testing.assert_close(torch.from_numpy(inputs[:, 3:]), torch.zeros(6, 7, 4))
