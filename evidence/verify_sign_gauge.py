# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Compare a direct signed KDA recurrence with the sign-gauged recurrence on CPU."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch


def direct_recurrence(q, k, v, log_magnitude, beta, sign, initial_state):
    state = initial_state
    outputs = []
    scale = q.shape[-1] ** -0.5
    for step in range(q.shape[1]):
        alpha = sign[:, step].to(q) * log_magnitude[:, step].exp()
        state = state * alpha[..., None]
        residual = v[:, step] - torch.einsum("bhk,bhkv->bhv", k[:, step], state)
        state = state + beta[:, step, :, None, None] * k[:, step, :, :, None] * residual[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, step] * scale, state))
    return torch.stack(outputs, dim=1), state


def gauged_recurrence(q, k, v, log_magnitude, beta, sign, initial_state):
    parity = sign.cumprod(dim=1).to(q)
    output, state = direct_recurrence(
        q * parity,
        k * parity,
        v,
        log_magnitude,
        beta,
        torch.ones_like(sign),
        initial_state,
    )
    return output, state * parity[:, -1, :, :, None]


def discrepancy(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    absolute = (reference - candidate).abs()
    denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).tiny)
    return {
        "max_absolute": absolute.max().item(),
        "max_relative": (absolute / denominator).max().item(),
    }


def compare_case(seed: int, length: int, sign_mode: str) -> dict:
    generator = torch.Generator().manual_seed(seed)
    shape = (2, length, 3, 7)
    q = torch.randn(shape, generator=generator, dtype=torch.float64)
    k = torch.randn(shape, generator=generator, dtype=torch.float64)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    v = torch.randn(2, length, 3, 5, generator=generator, dtype=torch.float64)
    log_magnitude = -torch.rand(shape, generator=generator, dtype=torch.float64) * 5
    beta = torch.rand(2, length, 3, generator=generator, dtype=torch.float64) * 2
    initial_state = torch.randn(2, 3, 7, 5, generator=generator, dtype=torch.float64)
    if sign_mode == "alternating":
        steps = torch.arange(length).view(1, length, 1, 1)
        sign = torch.where(steps % 2 == 0, 1, -1).expand(shape).to(torch.int8)
    elif sign_mode == "random":
        sign = torch.where(torch.rand(shape, generator=generator) < 0.5, -1, 1).to(torch.int8)
    else:
        raise ValueError(sign_mode)
    direct_inputs = [tensor.detach().requires_grad_(True) for tensor in (q, k, v, log_magnitude, beta, initial_state)]
    gauged_inputs = [tensor.detach().requires_grad_(True) for tensor in (q, k, v, log_magnitude, beta, initial_state)]
    direct_output, direct_state = direct_recurrence(*direct_inputs[:5], sign, direct_inputs[5])
    gauged_output, gauged_state = gauged_recurrence(*gauged_inputs[:5], sign, gauged_inputs[5])
    output_weight = torch.randn(direct_output.shape, generator=generator, dtype=torch.float64)
    state_weight = torch.randn(direct_state.shape, generator=generator, dtype=torch.float64)
    (direct_output.mul(output_weight).sum() + direct_state.mul(state_weight).sum()).backward()
    (gauged_output.mul(output_weight).sum() + gauged_state.mul(state_weight).sum()).backward()
    names = ("q", "k", "v", "log_magnitude", "beta", "initial_state")
    return {
        "seed": seed,
        "length": length,
        "sign_mode": sign_mode,
        "output": discrepancy(direct_output, gauged_output),
        "final_state": discrepancy(direct_state, gauged_state),
        "backward": {
            name: discrepancy(direct.grad, gauged.grad)
            for name, direct, gauged in zip(names, direct_inputs, gauged_inputs)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "algebraic CPU reference; optimized CUDA comparisons are covered by tests/ops/test_complex_kda.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "dtype": "float64",
        },
        "cases": [
            compare_case(seed, length, sign_mode)
            for seed, length in ((0, 8), (1, 17), (2, 64))
            for sign_mode in ("alternating", "random")
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
