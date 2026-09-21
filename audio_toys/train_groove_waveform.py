# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Train one KDA layer to continue low-rate waveform frames from the groove."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import wave
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_toys.kda_variants import VARIANTS, parse_ints
from audio_toys.train_groove import (
    choose_device,
    continuation_loss,
    groove_controls,
    initialize_transition,
    make_batch,
    render_controls,
)
from fla.layers.gated_deltaproduct import GatedDeltaProduct
from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
from group_word_problems.train_wordproblem import Muon, build_optimizer

TRANSFORMER_VARIANTS = {
    "transformer": {"causal": True, "position_encoding": "sinusoidal"},
    "transformer_rope": {"causal": True, "position_encoding": "rope"},
    "transformer_rope_500k": {"causal": True, "position_encoding": "rope_500k"},
    "transformer_selective_rope": {"causal": True, "position_encoding": "selective_rope"},
    "transformer_no_pos": {"causal": True, "position_encoding": "none"},
    "transformer_noncausal": {"causal": False, "position_encoding": "sinusoidal"},
}
PAPER_VARIANTS = (*VARIANTS, "gru", "transformer")


@dataclass
class Metrics:
    normalized_mse: float
    tail_normalized_mse: float
    waveform_mse: float
    waveform_snr_db: float


class WaveformModel(nn.Module):
    """A learned frame projection, one KDA layer, and an MLP frame decoder."""

    def __init__(
        self,
        frame_size: int,
        d_model: int,
        n_heads: int,
        head_dim: int,
        backend: str,
        variant: str,
        gate_init_style: str | None = None,
        beta_init_style: str | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        config = VARIANTS[variant]
        gate_init_style = gate_init_style or config["gate_init_style"]
        beta_init_style = beta_init_style or config["beta_init_style"]
        self.input_proj = nn.Linear(frame_size, d_model - 1, bias=False)
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
            nn.Linear(4 * d_model, frame_size),
        )

    @staticmethod
    def _mask_control_value_gradient(gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.clone()
        gradient[:, 0] = 0
        return gradient

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.new_zeros(*inputs.shape[:-1], self.d_model)
        hidden[..., 0] = 1
        hidden[..., 1:] = self.input_proj(inputs)
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(inputs)
        recurrent, _, _ = self.layer(hidden)
        return self.readout(hidden + recurrent)


class WaveformDeltaProduct2(nn.Module):
    """A learned frame projection, one DeltaProduct-2 layer, and an MLP frame decoder."""

    def __init__(
        self,
        frame_size: int,
        d_model: int,
        n_heads: int,
        head_dim: int,
        use_forget_gate: bool,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(frame_size, d_model - 1, bias=False)
        self.layer = GatedDeltaProduct(
            hidden_size=d_model,
            expand_v=1,
            head_dim=head_dim,
            num_heads=n_heads,
            mode="chunk",
            use_output_gate=True,
            use_short_conv=False,
            use_forget_gate=use_forget_gate,
            allow_neg_eigval=True,
            num_householder=2,
            no_conv_silu="",
            layer_idx=0,
        )
        with torch.no_grad():
            self.layer.v_proj.weight[:, 0] = 0
        self.layer.v_proj.weight.register_hook(self._mask_control_value_gradient)
        self.readout = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, frame_size),
        )

    @staticmethod
    def _mask_control_value_gradient(gradient: torch.Tensor) -> torch.Tensor:
        gradient = gradient.clone()
        gradient[:, 0] = 0
        return gradient

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.new_zeros(*inputs.shape[:-1], self.d_model)
        hidden[..., 0] = 1
        hidden[..., 1:] = self.input_proj(inputs)
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(inputs)
        recurrent, _, _ = self.layer(hidden)
        return self.readout(hidden + recurrent)


def initialize_deltaproduct2_reflections(model: WaveformDeltaProduct2, gate_magnitude: float) -> None:
    """Initialize the two rank-one factors near norm-preserving reflections."""
    beta_fraction = torch.tensor(0.999, device=model.layer.b_proj.weight.device)
    beta_logit = torch.logit(beta_fraction)
    with torch.no_grad():
        model.layer.b_proj.weight.zero_()
        model.layer.b_proj.weight[:, 0] = beta_logit
        if model.layer.use_forget_gate:
            model.layer.a_proj.weight.zero_()
            rate = -math.log(gate_magnitude)
            dt = F.softplus(model.layer.dt_bias.float())
            model.layer.A_log.copy_(torch.log(torch.full_like(dt, rate) / dt))


class WaveformGRU(nn.Module):
    """A learned frame projection, one GRU layer, and an MLP frame decoder."""

    def __init__(self, frame_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(frame_size, d_model - 1, bias=False)
        self.recurrent = nn.GRU(d_model, d_model, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, frame_size),
        )

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.new_zeros(*inputs.shape[:-1], self.d_model)
        hidden[..., 0] = 1
        hidden[..., 1:] = self.input_proj(inputs)
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.recurrent(self.embed(inputs))
        return self.readout(recurrent)


class SelectiveRotaryEmbedding(nn.Module):
    """Input-dependent rotary embedding using the released Selective RoPE configuration."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        projection_rank: int = 32,
        convolution_width: int = 4,
        rope_base: float = 500000.0,
    ) -> None:
        super().__init__()
        phase_dim = n_heads * head_dim // 2
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.phase_dim = head_dim // 2
        self.convolution_width = convolution_width
        self.phase_in = nn.Linear(d_model, projection_rank, bias=False)
        self.phase_out = nn.Linear(projection_rank, phase_dim, bias=False)
        self.phase_conv = nn.Conv1d(
            phase_dim,
            phase_dim,
            kernel_size=convolution_width,
            groups=phase_dim,
            bias=False,
        )
        temperature = rope_base ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer("temperature", temperature.view(1, 1, 1, self.phase_dim))

    def rotate(self, inputs: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        first = inputs[..., :self.phase_dim]
        second = inputs[..., self.phase_dim:]
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        return torch.cat((first * cosine - second * sine, first * sine + second * cosine), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden_states.shape
        head_states = hidden_states.view(batch, length, self.n_heads, self.head_dim)
        normalized_states = F.normalize(head_states, dim=-1).flatten(-2)
        phase = self.phase_out(self.phase_in(normalized_states)).transpose(1, 2)
        phase = F.conv1d(
            F.pad(phase, (self.convolution_width - 1, 0)),
            self.phase_conv.weight,
            groups=phase.shape[1],
        )
        phase = phase.transpose(1, 2).view(batch, length, self.n_heads, self.phase_dim)
        angles = torch.cumsum(phase, dim=1) * self.temperature
        query = self.rotate(query.transpose(1, 2), angles).transpose(1, 2)
        key = self.rotate(key.transpose(1, 2), angles).transpose(1, 2)
        return query, key


class WaveformTransformer(nn.Module):
    """A learned frame projection, one Transformer layer, and an MLP frame decoder."""

    def __init__(
        self,
        frame_size: int,
        d_model: int,
        n_heads: int,
        causal: bool = True,
        position_encoding: str = "sinusoidal",
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if position_encoding not in {"sinusoidal", "rope", "rope_500k", "selective_rope", "none"}:
            raise ValueError(f"unknown position encoding: {position_encoding}")
        if position_encoding in {"rope", "rope_500k", "selective_rope"} and (d_model // n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.position_encoding = position_encoding
        self.input_proj = nn.Linear(frame_size, d_model - 1, bias=False)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.selective_rope = (
            SelectiveRotaryEmbedding(d_model, n_heads, self.head_dim)
            if position_encoding == "selective_rope"
            else None
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.readout = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, frame_size),
        )

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs.new_zeros(*inputs.shape[:-1], self.d_model)
        hidden[..., 0] = 1
        hidden[..., 1:] = self.input_proj(inputs)
        if self.position_encoding != "sinusoidal":
            return hidden
        positions = torch.arange(inputs.shape[1], device=inputs.device, dtype=inputs.dtype).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.d_model, 2, device=inputs.device, dtype=inputs.dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = positions * frequencies.unsqueeze(0)
        position_encoding = torch.zeros(inputs.shape[1], self.d_model, device=inputs.device, dtype=inputs.dtype)
        position_encoding[:, 0::2] = torch.sin(angles)
        position_encoding[:, 1::2] = torch.cos(angles[:, :position_encoding[:, 1::2].shape[1]])
        return hidden + position_encoding.unsqueeze(0)

    def apply_rope(self, inputs: torch.Tensor) -> torch.Tensor:
        rope_base = 500000.0 if self.position_encoding == "rope_500k" else 10000.0
        positions = torch.arange(inputs.shape[-2], device=inputs.device, dtype=inputs.dtype)
        frequencies = torch.exp(
            torch.arange(0, self.head_dim, 2, device=inputs.device, dtype=inputs.dtype)
            * (-math.log(rope_base) / self.head_dim)
        )
        angles = positions[:, None] * frequencies[None, :]
        cosine = torch.cos(angles)[None, None]
        sine = torch.sin(angles)[None, None]
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

    def rotary_attention(self, normalized: torch.Tensor) -> torch.Tensor:
        query, key, value = F.linear(
            normalized,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
        ).chunk(3, dim=-1)
        batch, length, _ = query.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        if self.position_encoding in {"rope", "rope_500k"}:
            query = self.apply_rope(query)
            key = self.apply_rope(key)
        else:
            assert self.selective_rope is not None
            query, key = self.selective_rope(query, key, normalized)
        value = split_heads(value)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.ones(length, length, device=normalized.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attention = (weights @ value).transpose(1, 2).contiguous().view(batch, length, self.d_model)
        return F.linear(attention, self.attention.out_proj.weight, self.attention.out_proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(inputs)
        normalized = self.attention_norm(hidden)
        if self.position_encoding in {"rope", "rope_500k", "selective_rope"}:
            attention = self.rotary_attention(normalized)
        else:
            attention_mask = None
            if self.causal:
                attention_mask = torch.ones(
                    inputs.shape[1],
                    inputs.shape[1],
                    device=inputs.device,
                    dtype=torch.bool,
                ).triu(1)
            attention, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=attention_mask,
                need_weights=False,
            )
        hidden = hidden + attention
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return self.readout(hidden)


def make_waveform_pattern(frame_size: int, source_sample_rate: int, bpm: float, seed: int) -> torch.Tensor:
    controls = groove_controls().numpy()
    audio = render_controls(controls, source_sample_rate, bpm, seed).mean(axis=1)
    step_seconds = 60 / bpm / 4
    frames = []
    for step in range(len(controls)):
        start = round(step * step_seconds * source_sample_rate)
        stop = round((step + 1) * step_seconds * source_sample_rate)
        source = audio[start:stop]
        source_positions = np.linspace(0, 1, source.shape[0], endpoint=False)
        target_positions = np.linspace(0, 1, frame_size, endpoint=False)
        frames.append(np.interp(target_positions, source_positions, source))
    return torch.tensor(np.stack(frames), dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module,
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
    raw_error = raw_prediction[:, prefix_steps:] - raw_target[:, prefix_steps:]
    waveform_mse = raw_error.square().mean()
    signal_power = raw_target[:, prefix_steps:].square().mean()
    metrics = Metrics(
        normalized_mse=F.mse_loss(prediction[:, prefix_steps:], target[:, prefix_steps:]).item(),
        tail_normalized_mse=F.mse_loss(prediction[:, tail_start:], target[:, tail_start:]).item(),
        waveform_mse=waveform_mse.item(),
        waveform_snr_db=(10 * torch.log10(signal_power / waveform_mse.clamp_min(1e-20))).item(),
    )
    model.train()
    return metrics


@torch.no_grad()
def predict_frames(
    model: nn.Module,
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


@torch.no_grad()
def predict_all_phases(
    model: nn.Module,
    pattern: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    length: int,
    prefix_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return inputs, targets, and predictions for every circular phase."""
    phases = torch.arange(pattern.shape[0], device=pattern.device)
    steps = torch.arange(length, device=pattern.device)
    indices = (phases[:, None] + steps[None]) % pattern.shape[0]
    target = pattern[indices]
    normalized_target = (target - mean) / std
    inputs = torch.zeros_like(normalized_target)
    inputs[:, :prefix_steps] = normalized_target[:, :prefix_steps]
    prediction = model(inputs) * std + mean
    return (
        phases.cpu().numpy(),
        inputs.cpu().numpy(),
        target.cpu().numpy(),
        prediction.cpu().numpy(),
    )


def frames_to_audio(frames: np.ndarray, bpm: float, sample_rate: int) -> np.ndarray:
    waveform = frames.reshape(-1)
    duration = frames.shape[0] * 60 / bpm / 4
    output_size = round(duration * sample_rate)
    source_positions = np.linspace(0, duration, waveform.shape[0], endpoint=False)
    target_positions = np.linspace(0, duration, output_size, endpoint=False)
    return np.interp(target_positions, source_positions, waveform).astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    pcm = np.round(np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1 if audio.ndim == 1 else audio.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float | list[float],
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LRScheduler:
    if args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=args.steps,
            pct_start=0.1,
        )

    warmup_steps = min(args.warmup_steps, args.steps - 1)

    def warmup_cosine(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps - 1)
        cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)


def build_audio_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, float | list[float]]:
    if args.optimizer != "muon" or args.muon_scope != "core":
        return build_optimizer(model, vars(args))

    adapter_modules = [model.input_proj, model.readout]
    selective_rope = getattr(model, "selective_rope", None)
    if selective_rope is not None:
        adapter_modules.append(selective_rope)
    adapter_parameters = {
        id(parameter)
        for module in adapter_modules
        for parameter in module.parameters()
    }
    muon_parameters = []
    adamw_parameters = []
    for parameter in model.parameters():
        if parameter.requires_grad:
            target = muon_parameters if parameter.ndim == 2 and id(parameter) not in adapter_parameters else adamw_parameters
            target.append(parameter)
    print(
        f"      muon core: {len(muon_parameters)} 2D tensors "
        f"({sum(parameter.numel() for parameter in muon_parameters):,} params) | "
        f"adamw adapters/other: {len(adamw_parameters)} tensors "
        f"({sum(parameter.numel() for parameter in adamw_parameters):,} params)",
        flush=True,
    )
    groups = [
        {
            "params": muon_parameters,
            "use_muon": True,
            "lr": args.muon_lr,
            "momentum": args.muon_momentum,
            "nesterov": True,
            "ns_steps": args.muon_ns_steps,
            "weight_decay": args.weight_decay,
        },
        {
            "params": adamw_parameters,
            "use_muon": False,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    ]
    optimizer = Muon(
        groups,
        lr=args.muon_lr,
        adamw_lr=args.lr,
        momentum=args.muon_momentum,
        ns_steps=args.muon_ns_steps,
        weight_decay=args.weight_decay,
    )
    return optimizer, [group["lr"] for group in optimizer.param_groups]


def run(
    model_name: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    pattern: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[dict, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    torch.manual_seed(seed)
    normalized_pattern = (pattern - mean) / std
    if model_name == "gru":
        model = WaveformGRU(pattern.shape[-1], args.d_model).to(device)
    elif model_name in {"deltaproduct2", "deltaproduct2_ungated"}:
        model = WaveformDeltaProduct2(
            pattern.shape[-1],
            args.d_model,
            args.heads,
            args.head_dim,
            use_forget_gate=model_name == "deltaproduct2",
        ).to(device)
        if args.dp_init == "reflection":
            initialize_deltaproduct2_reflections(model, args.gate_magnitude_init)
    elif model_name in TRANSFORMER_VARIANTS:
        transformer_config = TRANSFORMER_VARIANTS[model_name]
        model = WaveformTransformer(
            pattern.shape[-1],
            args.d_model,
            args.heads,
            causal=transformer_config["causal"],
            position_encoding=transformer_config["position_encoding"],
        ).to(device)
    else:
        model = WaveformModel(
            frame_size=pattern.shape[-1],
            d_model=args.d_model,
            n_heads=args.heads,
            head_dim=args.head_dim,
            backend=args.backend,
            variant=model_name,
            gate_init_style=args.gate_init_style,
            beta_init_style=args.beta_init_style,
        ).to(device)
        if args.init != "default":
            initialize_transition(
                model,
                model_name,
                period=pattern.shape[0],
                gate_magnitude=args.gate_magnitude_init,
                beta_fraction=args.beta_fraction_init,
                harmonic=args.init == "harmonic",
                freeze=args.freeze_transition,
            )
    optimizer, max_lr = build_audio_optimizer(model, args)
    scheduler = build_scheduler(optimizer, max_lr, args)
    generator = torch.Generator(device="cpu").manual_seed(seed + 8181)
    curriculum_stage = 0
    curriculum_streak = 0
    curriculum_history = []
    training_log = []
    start_time = time.time()
    loss = torch.tensor(float("nan"))
    for step in range(args.steps):
        if args.curriculum_mode == "scheduled":
            curriculum_span = max(len(args.curriculum_lengths), round(args.steps * args.curriculum_fraction))
            scheduled_stage = min(
                step * len(args.curriculum_lengths) // curriculum_span,
                len(args.curriculum_lengths) - 1,
            )
            if scheduled_stage > curriculum_stage:
                curriculum_stage = scheduled_stage
                curriculum_history.append({
                    "length": args.curriculum_lengths[curriculum_stage],
                    "step": step + 1,
                    "reason": "schedule",
                })
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
            if args.curriculum_mode == "adaptive":
                if stage_metrics.tail_normalized_mse <= args.curriculum_threshold:
                    curriculum_streak += 1
                else:
                    curriculum_streak = 0
                if curriculum_streak >= args.curriculum_passes and curriculum_stage < len(args.curriculum_lengths) - 1:
                    curriculum_history.append({
                        "length": length,
                        "step": step + 1,
                        "tail_normalized_mse": stage_metrics.tail_normalized_mse,
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
                f"tail={stage_metrics.tail_normalized_mse:.6f} SNR={stage_metrics.waveform_snr_db:.2f} dB",
                flush=True,
            )
            training_log.append({
                "step": step + 1,
                "length": length,
                "training_loss": loss.item(),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                **asdict(stage_metrics),
            })

    metrics = {
        str(length): asdict(evaluate(model, normalized_pattern, mean, std, length, args.prefix_steps))
        for length in args.eval_lengths
    }
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(args.checkpoint_dir, f"groove-waveform-{model_name}-seed{seed}.pt")
    torch.save(model.state_dict(), checkpoint)
    target_frames, prediction_frames = predict_frames(
        model,
        pattern,
        mean,
        std,
        args.audio_steps,
        args.prefix_steps,
    )
    phases, phase_inputs, phase_targets, phase_predictions = predict_all_phases(
        model,
        pattern,
        mean,
        std,
        args.audio_steps,
        args.prefix_steps,
    )
    row = {
        "task": "four_on_floor_waveform_frame_continuation",
        "model": model_name,
        "seed": seed,
        **VARIANTS.get(model_name, {}),
        "gate_init_style": getattr(getattr(model, "layer", None), "gate_init_style", None),
        "beta_init_style": getattr(getattr(model, "layer", None), "beta_init_style", None),
        "period_steps": pattern.shape[0],
        "frame_size": pattern.shape[-1],
        "source_sample_rate": args.source_sample_rate,
        "prefix_steps": args.prefix_steps,
        "steps": args.steps,
        "batch": args.batch,
        "d_model": args.d_model,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "warmup_steps": args.warmup_steps if args.scheduler == "warmup_cosine" else None,
        "min_lr_ratio": args.min_lr_ratio if args.scheduler == "warmup_cosine" else None,
        "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
        "muon_momentum": args.muon_momentum if args.optimizer == "muon" else None,
        "muon_ns_steps": args.muon_ns_steps if args.optimizer == "muon" else None,
        "muon_scope": args.muon_scope if args.optimizer == "muon" else None,
        "selective_rope_optimizer": (
            "adamw"
            if getattr(model, "selective_rope", None) is not None and args.optimizer == "muon"
            else None
        ),
        "position_encoding": getattr(model, "position_encoding", None),
        "num_householder": getattr(getattr(model, "layer", None), "num_householder", None),
        "allow_neg_eigval": getattr(getattr(model, "layer", None), "allow_neg_eigval", None),
        "expand_v": getattr(getattr(model, "layer", None), "expand_v", None),
        "use_forget_gate": getattr(getattr(model, "layer", None), "use_forget_gate", None),
        "dp_init": args.dp_init if model_name.startswith("deltaproduct2") else None,
        "init": (
            args.init
            if model_name in VARIANTS
            else (
                args.dp_init
                if model_name.startswith("deltaproduct2")
                else (getattr(model, "position_encoding", None) or "pytorch_default")
            )
        ),
        "freeze_transition": args.freeze_transition if model_name in VARIANTS else False,
        "curriculum_final_length": args.curriculum_lengths[curriculum_stage],
        "curriculum_mode": args.curriculum_mode,
        "curriculum_fraction": args.curriculum_fraction if args.curriculum_mode == "scheduled" else None,
        "curriculum_history": curriculum_history,
        "training_log": training_log,
        "final_training_loss": loss.item(),
        "metrics": metrics,
        "checkpoint": checkpoint,
        "seconds": time.time() - start_time,
    }
    arrays = {
        "phases": phases,
        "normalized_inputs": phase_inputs,
        "targets": phase_targets,
        "predictions": phase_predictions,
        "pattern": pattern.cpu().numpy(),
        "normalization_mean": mean.cpu().numpy(),
        "normalization_std": std.cpu().numpy(),
    }
    return row, target_frames, prediction_frames, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=(*VARIANTS, "gru", "deltaproduct2", "deltaproduct2_ungated", *TRANSFORMER_VARIANTS, "paper", "all"),
        default="all",
    )
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument("--source-sample-rate", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-12)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--scheduler", choices=("onecycle", "warmup_cosine"), default="onecycle")
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--muon-lr", type=float, default=2e-2)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-scope", choices=("core", "hidden", "all2d"), default="hidden")
    parser.add_argument("--prefix-steps", type=int, default=8)
    parser.add_argument("--curriculum-lengths", type=parse_ints, default=parse_ints("12,16,24,40,72,136"))
    parser.add_argument("--curriculum-threshold", type=float, default=0.08)
    parser.add_argument("--curriculum-check-every", type=int, default=100)
    parser.add_argument("--curriculum-passes", type=int, default=1)
    parser.add_argument("--curriculum-mode", choices=("adaptive", "scheduled"), default="adaptive")
    parser.add_argument("--curriculum-fraction", type=float, default=0.4)
    parser.add_argument("--eval-lengths", type=parse_ints, default=parse_ints("40,136,264"))
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--init", choices=("default", "endpoint", "harmonic"), default="endpoint")
    parser.add_argument("--gate-init-style", choices=("shipped", "spread"))
    parser.add_argument("--beta-init-style", choices=("standard", "spread"))
    parser.add_argument("--freeze-transition", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gate-magnitude-init", type=float, default=0.9999)
    parser.add_argument("--beta-fraction-init", type=float, default=0.999)
    parser.add_argument("--dp-init", choices=("default", "reflection"), default="default")
    parser.add_argument("--audio-steps", type=int, default=128)
    parser.add_argument("--audio-sample-rate", type=int, default=8000)
    parser.add_argument("--bpm", type=float, default=124)
    parser.add_argument("--checkpoint-dir", default="audio_toys/checkpoints/groove_waveform")
    parser.add_argument("--audio-dir", default="audio_toys/audio/groove_waveform")
    parser.add_argument("--array-dir", default="audio_toys/arrays/groove_waveform")
    parser.add_argument("--out", default="audio_toys/results/groove_waveform.jsonl")
    args = parser.parse_args()
    if args.steps < 20 or args.seeds < 1 or args.batch < 1:
        parser.error("steps >= 20, seeds >= 1, and batch >= 1 are required")
    if not 0 < args.min_lr_ratio <= 1 or not 0 < args.curriculum_fraction <= 1:
        parser.error("min-lr-ratio and curriculum-fraction must lie in (0, 1]")

    device = choose_device(args.device)
    pattern = make_waveform_pattern(args.frame_size, args.source_sample_rate, args.bpm, seed=2026).to(device)
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.01)
    if args.variant == "all":
        variants = (*VARIANTS, "gru", "deltaproduct2", "deltaproduct2_ungated", *TRANSFORMER_VARIANTS)
    elif args.variant == "paper":
        variants = PAPER_VARIANTS
    else:
        variants = (args.variant,)
    rows = []
    predictions = {}
    target_frames = None
    os.makedirs(args.array_dir, exist_ok=True)
    print(f"device={device} variants={','.join(variants)} waveform_frames={args.frame_size}", flush=True)
    for seed in range(args.seeds):
        for variant in variants:
            print(f"  seed={seed} variant={variant}", flush=True)
            row, target_frames, prediction, arrays = run(variant, seed, args, device, pattern, mean, std)
            array_path = os.path.join(args.array_dir, f"{variant}-seed{seed}.npz")
            np.savez_compressed(array_path, **arrays)
            row["prediction_arrays"] = array_path
            rows.append(row)
            if seed == 0:
                predictions[variant] = prediction
            print(json.dumps(row), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if target_frames is not None:
        os.makedirs(args.audio_dir, exist_ok=True)
        target_audio = frames_to_audio(target_frames, args.bpm, args.audio_sample_rate)
        rendered = {
            variant: frames_to_audio(frames, args.bpm, args.audio_sample_rate)
            for variant, frames in predictions.items()
        }
        peak = max(float(np.max(np.abs(target_audio))), *(float(np.max(np.abs(audio))) for audio in rendered.values()))
        gain = 0.94 / max(peak, 0.94)
        write_wav(os.path.join(args.audio_dir, "target.wav"), gain * target_audio, args.audio_sample_rate)
        for variant, audio in rendered.items():
            write_wav(os.path.join(args.audio_dir, f"{variant}.wav"), gain * audio, args.audio_sample_rate)
            comparison = np.column_stack((target_audio, audio))
            path = os.path.join(args.audio_dir, f"{variant}_target-left_model-right.wav")
            write_wav(path, gain * comparison, args.audio_sample_rate)
    print(json.dumps({"results": args.out, "audio_dir": args.audio_dir}, indent=2), flush=True)


if __name__ == "__main__":
    main()
