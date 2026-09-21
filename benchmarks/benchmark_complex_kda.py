# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Record absolute KDA training and prefill timings with fused and PyTorch sign gauges."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fla.ops.kda import chunk_kda

IMPLEMENTATIONS = ("unsigned", "signed_fused", "signed_pytorch")


def git_revision() -> str:
    try:
        return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def running_sign(sign: torch.Tensor) -> torch.Tensor:
    parity = (sign < 0).to(torch.int32).cumsum(dim=1)
    return torch.where(parity & 1 == 1, -1, 1).to(torch.int8)


def call_kda(inputs: dict[str, torch.Tensor], implementation: str):
    q, k = inputs["q"], inputs["k"]
    sign = inputs["sign"] if implementation == "signed_fused" else None
    internal_norm = implementation != "signed_pytorch"
    parity = None
    if implementation == "signed_pytorch":
        parity = running_sign(inputs["sign"])
        q = F.normalize(q.float(), dim=-1).to(q) * parity.to(q)
        k = F.normalize(k.float(), dim=-1).to(k) * parity.to(k)
    output, state = chunk_kda(
        q=q,
        k=k,
        v=inputs["v"],
        g=inputs["g"],
        beta=inputs["beta"],
        sign=sign,
        output_final_state=True,
        use_qk_l2norm_in_kernel=internal_norm,
        safe_gate=True,
        lower_bound=-5.0,
    )
    if parity is not None:
        state = state * parity[:, -1].to(state).unsqueeze(-1)
    return output, state


def make_inputs(batch: int, length: int, heads: int, head_dim: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    shape = (batch, length, heads, head_dim)
    sign = torch.where(torch.rand(shape, device="cuda") < 0.5, -1, 1).to(torch.int8)
    return {
        "q": torch.randn(shape, device="cuda", dtype=dtype),
        "k": torch.randn(shape, device="cuda", dtype=dtype),
        "v": torch.randn(shape, device="cuda", dtype=dtype),
        "g": -5 * torch.rand(shape, device="cuda", dtype=torch.float32),
        "beta": torch.rand(batch, length, heads, device="cuda", dtype=dtype),
        "sign": sign,
    }


def timed_sample(inputs: dict[str, torch.Tensor], implementation: str, mode: str) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    output, state = call_kda(inputs, implementation)
    if mode == "training":
        (output.float().square().mean() + state.float().square().mean()).backward()
        for tensor in inputs.values():
            if tensor.grad is not None:
                tensor.grad = None
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - started)


def measure(
    implementation: str,
    mode: str,
    batch: int,
    length: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    repeats: int,
) -> dict:
    inputs = make_inputs(batch, length, heads, head_dim, dtype)
    if mode == "training":
        for name in ("q", "k", "v", "g", "beta"):
            inputs[name].requires_grad_(True)
    for _ in range(warmup):
        timed_sample(inputs, implementation, mode)
    samples = [timed_sample(inputs, implementation, mode) for _ in range(repeats)]
    median_ms = statistics.median(samples)
    return {
        "implementation": implementation,
        "mode": mode,
        "batch": batch,
        "sequence_length": length,
        "tokens": batch * length,
        "raw_milliseconds": samples,
        "median_milliseconds": median_ms,
        "mean_milliseconds": statistics.fmean(samples),
        "sample_std_milliseconds": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "tokens_per_second_from_median": batch * length * 1000 / median_ms,
    }


def parse_lengths(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("512,1024,2048,4096"))
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("this benchmark requires CUDA")
    dtype = torch.bfloat16
    results = []
    for mode in ("training", "prefill"):
        for length in args.lengths:
            if args.tokens_per_step % length:
                parser.error(f"tokens-per-step must be divisible by sequence length {length}")
            batch = args.tokens_per_step // length
            for implementation in IMPLEMENTATIONS:
                results.append(measure(
                    implementation,
                    mode,
                    batch,
                    length,
                    args.heads,
                    args.head_dim,
                    dtype,
                    args.warmup,
                    args.repeats,
                ))
    for row in results:
        baseline = next(
            candidate for candidate in results
            if candidate["implementation"] == "unsigned"
            and candidate["mode"] == row["mode"]
            and candidate["sequence_length"] == row["sequence_length"]
        )
        row["relative_throughput_to_unsigned"] = (
            row["tokens_per_second_from_median"] / baseline["tokens_per_second_from_median"]
        )
    report = {
        "scope": (
            "chunk_kda operator; prefill includes output and final-state construction; "
            "training includes forward and backward"
        ),
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize before and after every sample",
            "warmup_samples": args.warmup,
            "measured_samples": args.repeats,
            "dtype": str(dtype),
            "safe_gate": True,
            "lower_bound": -5.0,
        },
        "environment": {
            "git_revision": git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
