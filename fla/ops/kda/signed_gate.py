# -*- coding: utf-8 -*-

"""The signed decay gate of Complex KDA (`fla.layers.complex_kda_layer.signed_gate`) as one
Triton kernel each way.

    u = exp(A_log)[h] * (z + dt_bias[h, k]),  a = tanh(s * u)  (s = 1/2: "sigmoid2", 1: "tanh")
    sign = -1 if a < 0 else 1 (int8),  g = log(eps + (1 - eps) * |a|),  eps = exp(lower_bound)

Eager, that is about ten fp32 passes over [B, T, H, K] forward and as many backward -- the
glue that made Complex KDA ~3 ms per layer slower than KDA, whose (unsigned) gate FLA
computes inside its kernels. Here the forward reads z once and writes g and the sign; the
backward recomputes the forward from z and writes dz plus per-row-block partial sums for
dt_bias and A_log (reduced in torch). tanh is libdevice's accurate one: the sign is decided
near u = 0.
"""

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


@triton.jit
def _signed_gate_fwd_kernel(z, dt_bias, A_log, g, sign, N, C, K: tl.constexpr, SCALE: tl.constexpr,
                            EPS: tl.constexpr, HAS_DT: tl.constexpr, HAS_A: tl.constexpr,
                            BR: tl.constexpr, BC: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    rm, cm = rows < N, cols < C
    m = rm[:, None] & cm[None, :]
    off = rows.to(tl.int64)[:, None] * C + cols[None, :]
    u = tl.load(z + off, mask=m, other=0.0).to(tl.float32)
    if HAS_DT:
        u += tl.load(dt_bias + cols, mask=cm, other=0.0).to(tl.float32)[None, :]
    if HAS_A:
        u *= tl.exp(tl.load(A_log + cols // K, mask=cm, other=0.0).to(tl.float32))[None, :]
    a = tldevice.tanh(SCALE * u)
    tl.store(g + off, tl.log(EPS + (1.0 - EPS) * tl.abs(a)), mask=m)
    tl.store(sign + off, tl.where(a < 0, -1, 1).to(tl.int8), mask=m)


@triton.jit
def _signed_gate_bwd_kernel(z, dt_bias, A_log, dg, dz, ddt_part, dA_part, N, C,
                            K: tl.constexpr, SCALE: tl.constexpr, EPS: tl.constexpr,
                            HAS_DT: tl.constexpr, HAS_A: tl.constexpr,
                            BR: tl.constexpr, BC: tl.constexpr):
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
    a = tldevice.tanh(SCALE * (u0 * A[None, :]))
    # d|a|/da = sign(a) with 0 at a = 0 (torch's abs backward)
    sgn = tl.where(a > 0, 1.0, tl.where(a < 0, -1.0, 0.0))
    du = tl.load(dg + off, mask=m, other=0.0).to(tl.float32) * (1.0 - EPS) * sgn \
        / (EPS + (1.0 - EPS) * tl.abs(a)) * SCALE * (1.0 - a * a)
    du = tl.where(m, du, 0.0)
    tl.store(dz + off, (du * A[None, :]).to(dz.dtype.element_ty), mask=m)
    if HAS_DT:
        tl.store(ddt_part + pr * C + cols, tl.sum(du * A[None, :], axis=0), mask=cm)
    if HAS_A:
        tl.store(dA_part + pr * C + cols, tl.sum(du * u0, axis=0) * A, mask=cm)


_BR, _BC = 32, 128


class _SignedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, dt_bias, A_log, lower_bound, scale):
        shape = z.shape
        H, K = shape[-2], shape[-1]
        zc = z.contiguous().view(-1, H * K)
        N, C = zc.shape
        eps = math.exp(lower_bound)
        g = torch.empty((N, C), dtype=torch.float32, device=z.device)
        sign = torch.empty((N, C), dtype=torch.int8, device=z.device)
        dt = dt_bias.reshape(-1).contiguous() if dt_bias is not None else zc
        A = A_log.reshape(-1).contiguous() if A_log is not None else zc
        grid = (triton.cdiv(N, _BR), triton.cdiv(C, _BC))
        _signed_gate_fwd_kernel[grid](zc, dt, A, g, sign, N, C, K=K, SCALE=scale, EPS=eps,
                                      HAS_DT=dt_bias is not None, HAS_A=A_log is not None,
                                      BR=_BR, BC=_BC)
        ctx.save_for_backward(zc, dt_bias, A_log)
        ctx.shape, ctx.K, ctx.eps, ctx.scale = shape, K, eps, scale
        ctx.mark_non_differentiable(sign)
        return sign.view(shape), g.view(shape)

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
            EPS=ctx.eps, HAS_DT=dt_bias is not None, HAS_A=A_log is not None, BR=_BR, BC=_BC)
        d_dt = ddt.sum(0).view_as(dt_bias).to(dt_bias.dtype) if dt_bias is not None else None
        d_A = None
        if A_log is not None:
            d_A = dA.sum(0).view(-1, ctx.K).sum(-1).view_as(A_log).to(A_log.dtype)
        return dz.view(ctx.shape), d_dt, d_A, None, None


def fused_signed_gate(z, A_log=None, dt_bias=None, lower_bound: float = -5.0,
                      activation: str = "sigmoid2"):
    """(sign int8, log|alpha| fp32) of `fla.layers.complex_kda_layer.signed_gate`, fused.

    z [..., H, K]; dt_bias [H, K] (or H*K); A_log [H] (or [H, 1]).
    """
    if activation not in ("sigmoid2", "tanh"):
        raise ValueError(f"activation must be 'sigmoid2' or 'tanh', got {activation!r}")
    scale = 0.5 if activation == "sigmoid2" else 1.0
    return _SignedGate.apply(z, dt_bias, A_log, lower_bound, scale)
