# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Positive-control phase interventions on a learned S3 Complex-KDA head."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from plot_complex_kda_interpretability import _load_model, _read_last_row
from train_wordproblem import make_batch, perm_group

from fla.layers.complex_kda_layer import compute_gate


def modify_blocks(transition: torch.Tensor, mode: str) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eig(transition.double())
    positive = eigenvalues.imag > 1e-5
    valid = positive.any(-1)
    index = eigenvalues.abs().masked_fill(~positive, -1).argmax(-1)
    row = torch.arange(len(transition))
    eigenvalue = eigenvalues[row, index]
    vector = eigenvectors[row, :, index]
    plane, _ = torch.linalg.qr(torch.stack((vector.real, vector.imag), dim=-1))
    block = plane.mT @ transition.double() @ plane
    if mode == "radial":
        target = eigenvalue.abs()[:, None, None] * torch.eye(2, dtype=torch.float64)
    elif mode == "reverse":
        target = block.mT
    else:
        raise ValueError(f"Unknown phase intervention {mode!r}")
    delta = plane @ (target - block) @ plane.mT
    return torch.where(valid[:, None, None], transition.double() + delta, transition.double()).float()


@contextmanager
def replace_head_phase(model, head: int, mode: str):
    attention = model.layer
    captured: dict[str, torch.Tensor] = {}

    def capture(name):
        def hook(_module, _args, output):
            captured[name] = output

        return hook

    def replace(_module, args):
        q = rearrange(attention.act(captured["query"]), "b t (h d) -> b t h d", d=attention.head_k_dim)
        k = rearrange(attention.k_act(captured["key"]), "b t (h d) -> b t h d", d=attention.head_k_dim)
        v = rearrange(attention.act(captured["value"]), "b t (h d) -> b t h d", d=attention.head_v_dim)
        q = F.normalize(q.float(), dim=-1)[:, :, head]
        k = F.normalize(k.float(), dim=-1)[:, :, head]
        v = v.float()[:, :, head]
        gate = rearrange(captured["gate"], "b t (h d) -> b t h d", d=attention.head_k_dim)
        sign, log_abs_alpha = compute_gate(
            attention.gate,
            gate,
            attention.A_log,
            attention.dt_bias,
            attention.lower_bound,
        )
        if sign is None:
            sign = torch.ones_like(log_abs_alpha, dtype=torch.int8)
        alpha = (sign.float() * log_abs_alpha.exp())[:, :, head]
        beta = 2 * torch.sigmoid(captured["beta"].float()[:, :, head])

        dimension = attention.head_k_dim
        eye = torch.eye(dimension, device=k.device)
        transition = (eye - beta[..., None, None] * k[..., :, None] * k[..., None, :]) * alpha[..., None, :]
        if mode != "manual":
            shape = transition.shape
            flat = transition.reshape(-1, dimension, dimension).cpu()
            unique, inverse = torch.unique(flat, dim=0, return_inverse=True)
            transition = modify_blocks(unique, mode)[inverse].to(k.device).reshape(shape)
        state = torch.zeros(
            len(k),
            dimension,
            attention.head_v_dim,
            device=k.device,
            dtype=torch.float32,
        )
        outputs = []
        for position in range(k.shape[1]):
            state = transition[:, position] @ state + (
                beta[:, position, None, None]
                * k[:, position, :, None]
                * v[:, position, None, :]
            )
            outputs.append(attention.head_k_dim**-0.5 * torch.einsum("bd,bdv->bv", q[:, position], state))
        replacement = torch.stack(outputs, dim=1)
        raw = args[0].clone()
        raw[:, :, head] = replacement.to(raw)
        return (raw, *args[1:])

    handles = [
        attention.q_proj.register_forward_hook(capture("query")),
        attention.k_proj.register_forward_hook(capture("key")),
        attention.v_proj.register_forward_hook(capture("value")),
        attention.f_proj.register_forward_hook(capture("gate")),
        attention.b_proj.register_forward_hook(capture("beta")),
        attention.o_norm.register_forward_pre_hook(replace),
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def accuracy(model, x: torch.Tensor, y: torch.Tensor) -> float:
    quarter = max(x.shape[1] // 4, 1)
    predictions = model(x).argmax(-1)
    return float((predictions[:, -quarter:] == y[:, -quarter:]).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", default="11", help="Comma-separated heads to intervene on jointly")
    parser.add_argument("--lengths", default="32,128")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    row = _read_last_row(args.run)
    if row["task"] != "s3":
        raise ValueError("This positive control expects an S3 run")
    model, elements = _load_model(row)
    model = model.to(args.device)
    heads = [int(value) for value in args.heads.split(",")]
    _, table, identity = perm_group("s3")
    results = {}
    for length in (int(value) for value in args.lengths.split(",")):
        x, y = make_batch(
            table,
            identity,
            args.batch,
            length,
            np.random.default_rng(1729 + length),
            torch.device(args.device),
            bos_id=len(elements),
        )
        length_result = {"baseline": accuracy(model, x, y)}
        for mode in ("manual", "radial", "reverse"):
            with ExitStack() as stack:
                for head in heads:
                    stack.enter_context(replace_head_phase(model, head, mode))
                length_result[mode] = accuracy(model, x, y)
        results[str(length)] = length_result
        print(length, length_result, flush=True)

    report = {
        "run": str(args.run),
        "checkpoint": row["ckpt"],
        "heads": heads,
        "batch": args.batch,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
