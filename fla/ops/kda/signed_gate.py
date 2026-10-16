# -*- coding: utf-8 -*-

"""The decay gates of `fla.layers.complex_kda_layer.compute_gate` -- the signed one of Complex
KDA and the unsigned softplus / sigmoid ones (Triadic KDA) -- as one Triton kernel each way.

    u0 = z + dt_bias[h, k],  A = exp(A_log)[h]
    signed:   a = tanh(s A u0) (s = 1/2: "sigmoid2", 1: "tanh"), sign = -1 if a < 0 else 1
              (int8), g = log(eps + (1 - eps) |a|), eps = exp(lower_bound)
    softplus: g = -A softplus(u0)
    sigmoid:  g = lower_bound * sigmoid(A u0)

Eager, that is 5-10 fp32 passes over [B, T, H, K] forward and as many backward -- the glue
that made Complex KDA ~3 ms per layer slower than KDA, whose gate FLA computes inside its
kernels, and that Triadic KDA pays as well. Here the forward reads z once and writes g and the sign; the
backward recomputes the forward from z and writes dz plus per-row-block partial sums for
dt_bias and A_log (reduced in torch). tanh is libdevice's accurate one: the sign is decided
near u = 0.
"""

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice  # noqa: F401  (tl.extra.libdevice: also resolvable when Inductor re-emits the kernels)


@triton.jit
def _signed_gate_fwd_kernel(z, dt_bias, A_log, g, sign, N, C, K: tl.constexpr, SCALE: tl.constexpr,
                            EPS: tl.constexpr, HAS_DT: tl.constexpr, HAS_A: tl.constexpr,
                            MODE: tl.constexpr, LB: tl.constexpr, BR: tl.constexpr,
                            BC: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    rm, cm = rows < N, cols < C
    m = rm[:, None] & cm[None, :]
    off = rows.to(tl.int64)[:, None] * C + cols[None, :]
    u = tl.load(z + off, mask=m, other=0.0).to(tl.float32)
    if HAS_DT:
        u += tl.load(dt_bias + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
    A = tl.full((BC,), 1.0, tl.float32)
    if HAS_A:
        A = tl.exp(tl.load(A_log + cols // K, mask=cm, other=0.0).to(tl.float32))
    if MODE == 0:  # signed: sign(a), log(eps + (1 - eps)|a|), a = tanh(s A u)
        a = tl.extra.libdevice.tanh(SCALE * (u * A[None, :]))
        tl.store(g + off, tl.log(EPS + (1.0 - EPS) * tl.abs(a)), mask=m)
        tl.store(sign + off, tl.where(a < 0, -1, 1).to(tl.int8), mask=m)
    elif MODE == 1:  # softplus: -A softplus(u), the overflow-free form
        sp = tl.maximum(u, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(u)))  # no libdevice log1p on older Triton
        tl.store(g + off, -A[None, :] * sp, mask=m)
    else:  # sigmoid: lower_bound * sigmoid(A u)
        tl.store(g + off, LB * tl.sigmoid(u * A[None, :]), mask=m)


@triton.jit
def _signed_gate_bwd_kernel(z, dt_bias, A_log, dg, dz, ddt_part, dA_part, N, C,
                            K: tl.constexpr, SCALE: tl.constexpr, EPS: tl.constexpr,
                            HAS_DT: tl.constexpr, HAS_A: tl.constexpr, MODE: tl.constexpr,
                            LB: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    pr = tl.program_id(0)
    rows = pr * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    rm, cm = rows < N, cols < C
    m = rm[:, None] & cm[None, :]
    off = rows.to(tl.int64)[:, None] * C + cols[None, :]
    u0 = tl.load(z + off, mask=m, other=0.0).to(tl.float32)
    if HAS_DT:
        u0 += tl.load(dt_bias + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
    A = tl.full((BC,), 1.0, tl.float32)
    if HAS_A:
        A = tl.exp(tl.load(A_log + cols // K, mask=cm, other=0.0).to(tl.float32))
    dgv = tl.load(dg + off, mask=m, other=0.0).to(tl.float32)
    if MODE == 0:
        a = tl.extra.libdevice.tanh(SCALE * (u0 * A[None, :]))
        # d|a|/da = sign(a) with 0 at a = 0 (torch's abs backward)
        sgn = tl.where(a > 0, 1.0, tl.where(a < 0, -1.0, 0.0))
        # du: d/d(A u0); dz = du * A, dA = du * u0
        du = dgv * (1.0 - EPS) * sgn / (EPS + (1.0 - EPS) * tl.abs(a)) * SCALE * (1.0 - a * a)
        dzv = du * A[None, :]
        dAv = du * u0
    elif MODE == 1:
        sp = tl.maximum(u0, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(u0)))
        dzv = -dgv * A[None, :] * tl.sigmoid(u0)
        dAv = -dgv * sp
    else:
        sg = tl.sigmoid(u0 * A[None, :])
        du = dgv * LB * sg * (1.0 - sg)
        dzv = du * A[None, :]
        dAv = du * u0
    dzv = tl.where(m, dzv, 0.0)
    tl.store(dz + off, dzv.to(dz.dtype.element_ty), mask=m)
    if HAS_DT:
        tl.store(ddt_part + pr * C + cols, tl.sum(dzv, axis=0), mask=cm)
    if HAS_A:
        tl.store(dA_part + pr * C + cols, tl.sum(tl.where(m, dAv, 0.0), axis=0) * A, mask=cm)


_BR, _BC = 32, 128


class _SignedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, dt_bias, A_log, lower_bound, scale, mode):
        shape = z.shape
        H, K = shape[-2], shape[-1]
        zc = z.contiguous().view(-1, H * K)
        N, C = zc.shape
        eps = math.exp(lower_bound)
        g = torch.empty((N, C), dtype=torch.float32, device=z.device)
        # (unsigned gates: a 1-element placeholder, never written)
        sign = torch.empty((N, C) if mode == 0 else (1, 1), dtype=torch.int8, device=z.device)
        dt = dt_bias.reshape(-1).contiguous() if dt_bias is not None else zc
        A = A_log.reshape(-1).contiguous() if A_log is not None else zc
        grid = (triton.cdiv(N, _BR), triton.cdiv(C, _BC))
        _signed_gate_fwd_kernel[grid](zc, dt, A, g, sign, N, C, K=K, SCALE=scale, EPS=eps,
                                      HAS_DT=dt_bias is not None, HAS_A=A_log is not None,
                                      MODE=mode, LB=float(lower_bound), BR=_BR, BC=_BC)
        ctx.save_for_backward(zc, dt_bias, A_log)
        ctx.shape, ctx.K, ctx.eps, ctx.scale, ctx.mode, ctx.lb = shape, K, eps, scale, mode, lower_bound
        ctx.mark_non_differentiable(sign)
        return (sign.view(shape) if mode == 0 else sign), g.view(shape)

    @staticmethod
    def backward(ctx, _dsign, dg):
        zc, dt_bias, A_log = ctx.saved_tensors
        N, C = zc.shape
        nr = triton.cdiv(N, _BR)
        dz = torch.empty_like(zc)
        ddt = torch.empty((nr, C), dtype=torch.float32, device=zc.device) if dt_bias is not None else zc
        dA = torch.empty((nr, C), dtype=torch.float32, device=zc.device) if A_log is not None else zc
        dt = dt_bias.reshape(-1).contiguous() if dt_bias is not None else zc
        A = A_log.reshape(-1).contiguous() if A_log is not None else zc
        _signed_gate_bwd_kernel[(nr, triton.cdiv(C, _BC))](
            zc, dt, A, dg.contiguous().view(N, C), dz, ddt, dA, N, C, K=ctx.K, SCALE=ctx.scale,
            EPS=ctx.eps, HAS_DT=dt_bias is not None, HAS_A=A_log is not None, MODE=ctx.mode,
            LB=float(ctx.lb), BR=_BR, BC=_BC)
        d_dt = ddt.sum(0).view_as(dt_bias).to(dt_bias.dtype) if dt_bias is not None else None
        d_A = None
        if A_log is not None:
            d_A = dA.sum(0).view(-1, ctx.K).sum(-1).view_as(A_log).to(A_log.dtype)
        return dz.view(ctx.shape), d_dt, d_A, None, None, None


def fused_signed_gate(z, A_log=None, dt_bias=None, lower_bound: float = -5.0,
                      activation: str = "sigmoid2"):
    """(sign int8, log|alpha| fp32) of `fla.layers.complex_kda_layer.signed_gate`, fused.

    z [..., H, K]; dt_bias [H, K] (or H*K); A_log [H] (or [H, 1]).
    """
    if activation not in ("sigmoid2", "tanh"):
        raise ValueError(f"activation must be 'sigmoid2' or 'tanh', got {activation!r}")
    scale = 0.5 if activation == "sigmoid2" else 1.0
    return _SignedGate.apply(z, dt_bias, A_log, lower_bound, scale, 0)


def fused_gate(gate: str, z, A_log=None, dt_bias=None, lower_bound: float = -5.0):
    """`fla.layers.complex_kda_layer.compute_gate`, fused: (sign or None, log alpha fp32) for
    the gates "softplus" (-exp(A_log) softplus(z + dt_bias)), "sigmoid"
    (lower_bound sigmoid(exp(A_log) (z + dt_bias))) and "signed_sigmoid2" / "signed_tanh"."""
    if gate.startswith("signed_"):
        return fused_signed_gate(z, A_log, dt_bias, lower_bound, activation=gate[len("signed_"):])
    mode = {"softplus": 1, "sigmoid": 2}.get(gate)
    if mode is None:
        raise ValueError(f"unknown gate {gate!r}")
    return None, _SignedGate.apply(z, dt_bias, A_log, lower_bound, 1.0, mode)[1]
