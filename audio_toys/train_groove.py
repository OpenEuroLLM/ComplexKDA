# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Train one KDA layer to continue an original four-on-the-floor groove."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_toys.kda_variants import VARIANTS, parse_ints
from audio_toys.make_techno_loop import (
    BASS_PATTERN,
    PULSE_PATTERN,
    add_event,
    clap,
    hat,
    kick,
    lowpass,
    midi_frequency,
    pan_mono,
    synth_note,
    write_wav,
)
from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
from group_word_problems.train_wordproblem import build_optimizer

FEATURE_NAMES = (
    "kick",
    "clap",
    "closed_hat",
    "open_hat",
    "bass_velocity",
    "bass_pitch",
    "bass_gate",
    "pulse_velocity",
    "pulse_pitch",
)


@dataclass
class Metrics:
    normalized_mse: float
    tail_normalized_mse: float
    control_mae: float
    event_f1: float


def groove_controls() -> torch.Tensor:
    period = len(BASS_PATTERN)
    controls = torch.zeros(period, len(FEATURE_NAMES))
    for step in range(period):
        controls[step, 0] = float(step % 4 == 0)
        controls[step, 1] = float(step % 16 in (4, 12))
        if step % 2 == 0:
            eighth = step // 2
            is_offbeat = eighth % 2 == 1
            is_open = is_offbeat and eighth % 4 == 1
            controls[step, 2] = 0 if is_open else (0.62 if is_offbeat else 0.20)
            controls[step, 3] = 0.62 if is_open else 0

        bass_event = BASS_PATTERN[step]
        if bass_event is not None:
            note, velocity, gate = bass_event
            controls[step, 4] = velocity
            controls[step, 5] = velocity * (note - 42) / 12
            controls[step, 6] = gate

        pulse_note = PULSE_PATTERN[step % len(PULSE_PATTERN)]
        if pulse_note is not None:
            velocity = 0.28
            controls[step, 7] = velocity
            controls[step, 8] = velocity * (pulse_note - 67) / 12
    return controls


class GrooveModel(nn.Module):
    """A fixed control embedding, one KDA layer, and an MLP readout."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        head_dim: int,
        backend: str,
        variant: str,
        gate_init_style: str | None = None,
        beta_init_style: str | None = None,
    ) -> None:
        super().__init__()
        if d_model < n_features + 1:
            raise ValueError(f"d_model must be at least n_features + 1 ({n_features + 1})")
        self.d_model = d_model
        config = VARIANTS[variant]
        gate_init_style = gate_init_style or config["gate_init_style"]
        beta_init_style = beta_init_style or config["beta_init_style"]
        self.layer = ComplexKimiDeltaAttention(
            hidden_size=d_model,
            head_dim=head_dim,
            num_heads=n_heads,
            gate=config["gate"],
            allow_neg_eigval=config["allow_neg_eigval"],
            gate_init_style=gate_init_style,
            beta_init_style=beta_init_style,
            backend=backend,
            lower_bound=-5.0,
            use_short_conv=False,
            drop_silu=True,
            layer_idx=0,
        )
        with torch.no_grad():
            self.layer.v_proj.weight[:, 0] = 0
        self.layer.v_proj.weight.register_hook(self._mask_control_value_gradient)
        self.readout = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, n_features),
        )

    @staticmethod
    def _mask_control_value_gradient(gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.clone()
        gradient[:, 0] = 0
        return gradient

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.new_zeros(*inputs.shape[:-1], self.d_model)
        hidden[..., 0] = 1
        hidden[..., 1:inputs.shape[-1] + 1] = inputs
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(inputs)
        recurrent, _, _ = self.layer(hidden)
        return self.readout(hidden + recurrent)


@torch.no_grad()
def initialize_transition(
    model: GrooveModel,
    variant: str,
    period: int,
    gate_magnitude: float,
    beta_fraction: float,
    harmonic: bool,
    freeze: bool,
) -> None:
    if not 0 < gate_magnitude < 1 or not 0 < beta_fraction < 1:
        raise ValueError("initialization values must lie in (0, 1)")
    layer = model.layer
    config = VARIANTS[variant]
    layer.A_log.zero_()
    layer.f_proj[-1].weight.zero_()
    if config["gate"].startswith("signed_"):
        epsilon = math.exp(layer.lower_bound)
        target = (gate_magnitude - epsilon) / (1 - epsilon)
        magnitude_logit = 2 * math.atanh(target)
        signs = torch.ones_like(layer.dt_bias)
        signs.view(layer.num_v_heads, layer.head_k_dim)[:, :layer.head_k_dim // 2] = -1
        layer.dt_bias.copy_(signs * magnitude_logit)
    else:
        probability = math.log(gate_magnitude) / layer.lower_bound
        magnitude_logit = math.log(probability) - math.log1p(-probability)
        layer.dt_bias.fill_(magnitude_logit)

    raw_beta = math.log(beta_fraction) - math.log1p(-beta_fraction)
    layer.b_proj.weight.zero_()
    layer.b_proj.weight[:, 0] = raw_beta
    if layer.b_proj.bias is not None:
        layer.b_proj.bias.zero_()

    if harmonic:
        layer.k_proj.weight.zero_()
        positive_coordinate = layer.head_k_dim // 2
        for head in range(layer.num_heads):
            angular_frequency = 2 * math.pi * (head + 1) / period
            offset = head * layer.head_k_dim
            layer.k_proj.weight[offset, 0] = math.cos(angular_frequency / 2)
            layer.k_proj.weight[offset + positive_coordinate, 0] = math.sin(angular_frequency / 2)

    if freeze:
        for module in (layer.k_proj, layer.f_proj, layer.b_proj):
            module.requires_grad_(False)
        layer.A_log.requires_grad_(False)
        layer.dt_bias.requires_grad_(False)


def make_batch(
    normalized_pattern: torch.Tensor,
    batch_size: int,
    length: int,
    prefix_steps: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    period = normalized_pattern.shape[0]
    phases = torch.randint(period, (batch_size,), generator=generator, device="cpu").to(normalized_pattern.device)
    steps = torch.arange(length, device=normalized_pattern.device)
    indices = (phases[:, None] + steps[None]) % period
    target = normalized_pattern[indices]
    inputs = torch.zeros_like(target)
    inputs[:, :prefix_steps] = target[:, :prefix_steps]
    return inputs, target


def continuation_loss(prediction: torch.Tensor, target: torch.Tensor, prefix_steps: int) -> torch.Tensor:
    return F.mse_loss(prediction[:, prefix_steps:], target[:, prefix_steps:])


@torch.no_grad()
def evaluate(
    model: GrooveModel,
    normalized_pattern: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    length: int,
    prefix_steps: int,
) -> Metrics:
    model.eval()
    period = normalized_pattern.shape[0]
    phases = torch.arange(period, device=normalized_pattern.device)
    steps = torch.arange(length, device=normalized_pattern.device)
    indices = (phases[:, None] + steps[None]) % period
    target = normalized_pattern[indices]
    inputs = torch.zeros_like(target)
    inputs[:, :prefix_steps] = target[:, :prefix_steps]
    prediction = model(inputs)
    tail_start = max(3 * length // 4, prefix_steps)
    raw_prediction = prediction * std + mean
    raw_target = target * std + mean
    event_features = (0, 1, 2, 3, 4, 7)
    pred_events = raw_prediction[..., event_features] > 0.12
    target_events = raw_target[..., event_features] > 0.05
    continuation = slice(prefix_steps, None)
    true_positive = (pred_events[:, continuation] & target_events[:, continuation]).sum().float()
    false_positive = (pred_events[:, continuation] & ~target_events[:, continuation]).sum().float()
    false_negative = (~pred_events[:, continuation] & target_events[:, continuation]).sum().float()
    event_f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)
    metrics = Metrics(
        normalized_mse=F.mse_loss(prediction[:, continuation], target[:, continuation]).item(),
        tail_normalized_mse=F.mse_loss(prediction[:, tail_start:], target[:, tail_start:]).item(),
        control_mae=F.l1_loss(raw_prediction[:, continuation], raw_target[:, continuation]).item(),
        event_f1=event_f1.item(),
    )
    model.train()
    return metrics


def render_controls(
    controls: np.ndarray,
    sample_rate: int,
    bpm: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beat_seconds = 60 / bpm
    step_seconds = beat_seconds / 4
    n_samples = round(len(controls) * step_seconds * sample_rate)
    drums = np.zeros((n_samples, 2), dtype=np.float64)
    bass_mono = np.zeros(n_samples, dtype=np.float64)
    pulse_left = np.zeros(n_samples, dtype=np.float64)
    pulse_right = np.zeros(n_samples, dtype=np.float64)
    for step, control in enumerate(controls):
        swing_offset = 0.08 * step_seconds if step % 2 else 0
        start = round((step * step_seconds + swing_offset) * sample_rate)
        kick_level = float(np.clip(control[0], 0, 1.2))
        clap_level = float(np.clip(control[1], 0, 1.2))
        closed_hat_level = float(np.clip(control[2], 0, 0.8))
        open_hat_level = float(np.clip(control[3], 0, 0.8))
        if kick_level > 0.03:
            add_event(drums, pan_mono(kick_level * kick(sample_rate, rng), 0), start)
        if clap_level > 0.03:
            add_event(drums, pan_mono(clap_level * clap(sample_rate, rng), 0), start)
        if closed_hat_level > 0.02:
            pan = -0.32 if step % 4 < 2 else 0.32
            add_event(drums, pan_mono(closed_hat_level * hat(sample_rate, rng, False), pan), start)
        if open_hat_level > 0.02:
            pan = -0.32 if step % 4 < 2 else 0.32
            add_event(drums, pan_mono(open_hat_level * hat(sample_rate, rng, True), pan), start)

        bass_velocity = float(np.clip(control[4], 0, 1.1))
        if bass_velocity > 0.04:
            pitch = 42 + 12 * float(control[5]) / max(bass_velocity, 0.1)
            pitch = float(np.clip(pitch, 30, 54))
            gate = float(np.clip(control[6], 0.25, 1.8))
            sound = synth_note(midi_frequency(pitch), gate * step_seconds, sample_rate, bass_velocity)
            add_event(bass_mono, sound, start)

        pulse_velocity = float(np.clip(control[7], 0, 0.5))
        if pulse_velocity > 0.025:
            pitch = 67 + 12 * float(control[8]) / max(pulse_velocity, 0.05)
            pitch = float(np.clip(pitch, 56, 78))
            sound = synth_note(midi_frequency(pitch), 0.55 * step_seconds, sample_rate, pulse_velocity)
            target = pulse_left if step % 4 < 2 else pulse_right
            add_event(target, sound, start)

    phase_in_step = np.mod(np.arange(n_samples) / sample_rate, step_seconds) / step_seconds
    cutoff = 230 + 1120 * np.exp(-4.8 * phase_in_step)
    bass_mono = lowpass(bass_mono, cutoff, sample_rate)
    bass_track = 0.82 * pan_mono(np.tanh(1.35 * bass_mono), 0)
    pulse_track = np.column_stack((0.10 * pulse_left, 0.10 * pulse_right))
    phase_in_beat = np.mod(np.arange(n_samples) / sample_rate, beat_seconds)
    ducking = 0.32 + 0.68 * (1 - np.exp(-8.5 * phase_in_beat))
    mix = drums + (bass_track + pulse_track) * ducking[:, None]
    return np.tanh(1.22 * mix) / np.tanh(1.22)


@torch.no_grad()
def predict_controls(
    model: GrooveModel,
    pattern: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    length: int,
    prefix_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = torch.arange(length, device=pattern.device) % pattern.shape[0]
    target = pattern[indices]
    normalized = (target - mean) / std
    inputs = torch.zeros_like(normalized).unsqueeze(0)
    inputs[:, :prefix_steps] = normalized[:prefix_steps]
    prediction = model(inputs)[0] * std + mean
    prediction[:prefix_steps] = target[:prefix_steps]
    return target.cpu().numpy(), prediction.cpu().numpy()


def run(
    variant: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    pattern: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[dict, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    normalized_pattern = (pattern - mean) / std
    model = GrooveModel(
        n_features=pattern.shape[-1],
        d_model=args.d_model,
        n_heads=args.heads,
        head_dim=args.head_dim,
        backend=args.backend,
        variant=variant,
        gate_init_style=args.gate_init_style,
        beta_init_style=args.beta_init_style,
    ).to(device)
    if args.init != "default":
        initialize_transition(
            model,
            variant,
            period=pattern.shape[0],
            gate_magnitude=args.gate_magnitude_init,
            beta_fraction=args.beta_fraction_init,
            harmonic=args.init == "harmonic",
            freeze=args.freeze_transition,
        )

    optimizer, max_lr = build_optimizer(model, vars(args))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=args.steps,
        pct_start=0.1,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 4242)
    curriculum_stage = 0
    curriculum_streak = 0
    curriculum_history = []
    start_time = time.time()
    loss = torch.tensor(float("nan"))
    for step in range(args.steps):
        length = args.curriculum_lengths[curriculum_stage]
        inputs, target = make_batch(normalized_pattern, args.batch, length, args.prefix_steps, generator)
        prediction = model(inputs)
        loss = continuation_loss(prediction, target, args.prefix_steps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        stage_metrics = None
        if (step + 1) % args.curriculum_check_every == 0:
            stage_metrics = evaluate(model, normalized_pattern, mean, std, length, args.prefix_steps)
            if stage_metrics.tail_normalized_mse <= args.curriculum_threshold:
                curriculum_streak += 1
            else:
                curriculum_streak = 0
            if curriculum_streak >= args.curriculum_passes and curriculum_stage < len(args.curriculum_lengths) - 1:
                curriculum_history.append({
                    "length": length,
                    "step": step + 1,
                    "tail_normalized_mse": round(stage_metrics.tail_normalized_mse, 6),
                })
                curriculum_stage += 1
                curriculum_streak = 0
                print(
                    f"      curriculum: L{length} learned at step {step + 1}; "
                    f"advancing to L{args.curriculum_lengths[curriculum_stage]}",
                    flush=True,
                )
        if args.log_every and (step + 1) % args.log_every == 0:
            if stage_metrics is None:
                stage_metrics = evaluate(model, normalized_pattern, mean, std, length, args.prefix_steps)
            print(
                f"      {step + 1:5d} L{length:<4d} loss={loss.item():.6f} "
                f"tail={stage_metrics.tail_normalized_mse:.6f} F1={stage_metrics.event_f1:.3f}",
                flush=True,
            )

    metrics = {
        str(length): asdict(evaluate(model, normalized_pattern, mean, std, length, args.prefix_steps))
        for length in args.eval_lengths
    }
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(args.checkpoint_dir, f"groove-{variant}-seed{seed}.pt")
    torch.save(model.state_dict(), checkpoint)
    target_controls, prediction_controls = predict_controls(
        model,
        pattern,
        mean,
        std,
        args.audio_steps,
        args.prefix_steps,
    )
    config = VARIANTS[variant]
    row = {
        "task": "four_on_floor_control_continuation",
        "variant": variant,
        "seed": seed,
        **config,
        "gate_init_style": model.layer.gate_init_style,
        "beta_init_style": model.layer.beta_init_style,
        "period_steps": pattern.shape[0],
        "features": FEATURE_NAMES,
        "prefix_steps": args.prefix_steps,
        "steps": args.steps,
        "batch": args.batch,
        "d_model": args.d_model,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "backend": model.layer.backend,
        "device": str(device),
        "lr": args.lr,
        "optimizer": args.optimizer,
        "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
        "muon_scope": args.muon_scope if args.optimizer == "muon" else None,
        "init": args.init,
        "freeze_transition": args.freeze_transition,
        "gate_magnitude_init": args.gate_magnitude_init,
        "beta_fraction_init": args.beta_fraction_init,
        "curriculum_lengths": args.curriculum_lengths,
        "curriculum_threshold": args.curriculum_threshold,
        "curriculum_final_length": args.curriculum_lengths[curriculum_stage],
        "curriculum_history": curriculum_history,
        "final_loss": loss.item(),
        "metrics": metrics,
        "checkpoint": checkpoint,
        "seconds": round(time.time() - start_time, 1),
    }
    return row, target_controls, prediction_controls


def write_audio_comparison(
    target_controls: np.ndarray,
    predictions: dict[str, np.ndarray],
    output_dir: str,
    sample_rate: int,
    bpm: float,
    seed: int,
) -> dict[str, dict[str, str] | str]:
    os.makedirs(output_dir, exist_ok=True)
    target_audio = render_controls(target_controls, sample_rate, bpm, seed)
    rendered = {variant: render_controls(controls, sample_rate, bpm, seed) for variant, controls in predictions.items()}
    peak = max(float(np.max(np.abs(target_audio))), *(float(np.max(np.abs(audio))) for audio in rendered.values()))
    gain = 0.94 / max(peak, 0.94)
    target_path = os.path.join(output_dir, "target.wav")
    write_wav(target_path, gain * target_audio, sample_rate)
    paths: dict[str, dict[str, str] | str] = {"target": target_path}
    for variant, audio in rendered.items():
        mono_target = target_audio.mean(axis=1)
        mono_model = audio.mean(axis=1)
        stereo = np.column_stack((mono_target, mono_model))
        model_path = os.path.join(output_dir, f"{variant}.wav")
        comparison_path = os.path.join(output_dir, f"{variant}_target-left_model-right.wav")
        write_wav(model_path, gain * audio, sample_rate)
        write_wav(comparison_path, gain * stereo, sample_rate)
        paths[variant] = {"model": model_path, "comparison": comparison_path}
    return paths


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-12)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-scope", choices=("hidden", "all2d"), default="hidden")
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--curriculum-lengths", type=parse_ints, default=parse_ints("12,16,24,40,72,136"))
    parser.add_argument("--curriculum-threshold", type=float, default=0.05)
    parser.add_argument("--curriculum-check-every", type=int, default=100)
    parser.add_argument("--curriculum-passes", type=int, default=1)
    parser.add_argument("--eval-lengths", type=parse_ints, default=parse_ints("40,136,264"))
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--init", choices=("default", "endpoint", "harmonic"), default="harmonic")
    parser.add_argument("--gate-init-style", choices=("shipped", "spread"))
    parser.add_argument("--beta-init-style", choices=("standard", "spread"))
    parser.add_argument("--freeze-transition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gate-magnitude-init", type=float, default=0.9999)
    parser.add_argument("--beta-fraction-init", type=float, default=0.999)
    parser.add_argument("--audio-steps", type=int, default=128)
    parser.add_argument("--audio-sample-rate", type=int, default=44100)
    parser.add_argument("--bpm", type=float, default=124)
    parser.add_argument("--checkpoint-dir", default="audio_toys/checkpoints/groove_harmonic_seed0")
    parser.add_argument("--audio-dir", default="audio_toys/audio/groove_kda_seed0")
    parser.add_argument("--out", default="audio_toys/results/groove_harmonic_seed0.jsonl")
    args = parser.parse_args()
    if args.steps < 20 or args.seeds < 1 or args.batch < 1:
        parser.error("steps >= 20, seeds >= 1, and batch >= 1 are required")
    if args.prefix_steps >= min(args.curriculum_lengths):
        parser.error("prefix-steps must be shorter than every curriculum length")

    device = choose_device(args.device)
    pattern = groove_controls().to(device)
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.05)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    rows = []
    audio_by_seed = {}
    print(f"device={device} variants={','.join(variants)} init={args.init} freeze={args.freeze_transition}", flush=True)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        predictions = {}
        target_controls = None
        for variant in variants:
            print(f"  seed={seed} variant={variant}", flush=True)
            row, target_controls, prediction = run(variant, seed, args, device, pattern, mean, std)
            rows.append(row)
            predictions[variant] = prediction
            print(json.dumps(row), flush=True)
        if target_controls is not None:
            seed_audio_dir = args.audio_dir if args.seeds == 1 else os.path.join(args.audio_dir, f"seed{seed}")
            audio_by_seed[str(seed)] = write_audio_comparison(
                target_controls,
                predictions,
                seed_audio_dir,
                args.audio_sample_rate,
                args.bpm,
                seed + 2026,
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"results": args.out, "audio": audio_by_seed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
