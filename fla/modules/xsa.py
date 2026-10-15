# Copyright (c) 2026, OpenEuroLLM.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Exclusive self attention (XSA) as one fused Triton pass.

Zhai, "Exclusive Self Attention" (arXiv:2603.09078): the attention output of a token is
made orthogonal to its own value vector, removing the "attention similarity bias"
(the self-copy component) so attention carries context only. Per token and query head
``h`` of key/value group ``g(h)``::

    c = <o_h, v_g> / (|v_g|^2 + eps),    o_h <- o_h - c v_g

optionally followed by the sigmoid output gate of gated attention, fused into the same
pass (``o_h <- o_h * sigmoid(gate_h)``). One program per (token, kv head) handles the
query heads of that group, so the backward accumulates ``dv`` over them in registers
(no atomics). :func:`xsa_reference` is the torch definition the kernel is tested against.

For linear attention (``xsa=True`` on the GDN / KDA / Complex KDA layers) the same
projection removes the self-copy part of the output: a token's own write
``beta k v^T`` returns about ``beta (q . k) v`` at its read.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

EPS = 1e-6


def xsa_reference(
    o: torch.Tensor, v: torch.Tensor, gate: Optional[torch.Tensor] = None, eps: float = EPS
) -> torch.Tensor:
    """``o`` [T, Hq, D], ``v`` [T, Hkv, D], ``gate`` [T, Hq, D] or None -> [T, Hq, D]."""
    hq, hkv = o.shape[1], v.shape[1]
    vf = v.float().repeat_interleave(hq // hkv, dim=1)
    of = o.float()
    c = (of * vf).sum(-1, keepdim=True) / (vf.pow(2).sum(-1, keepdim=True) + eps)
    out = of - c * vf
    if gate is not None:
        out = out * torch.sigmoid(gate.float())
    return out.to(o.dtype)


if HAVE_TRITON:

    @triton.jit
    def _xsa_fwd(
        o_ptr, v_ptr, g_ptr, y_ptr, HQ, HKV, D, eps,
        HAS_GATE: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        kh = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        m = d < D
        v = tl.load(v_ptr + (t * HKV + kh) * D + d, mask=m, other=0.0).to(tl.float32)
        n = tl.sum(v * v, axis=0) + eps
        group = HQ // HKV
        for i in range(0, group):
            qh = kh * group + i
            base = (t * HQ + qh) * D
            o = tl.load(o_ptr + base + d, mask=m, other=0.0).to(tl.float32)
            c = tl.sum(o * v, axis=0) / n
            y = o - c * v
            if HAS_GATE:
                g = tl.load(g_ptr + base + d, mask=m, other=0.0).to(tl.float32)
                y = y * tl.sigmoid(g)
            tl.store(y_ptr + base + d, y.to(y_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _xsa_bwd(
        dy_ptr, o_ptr, v_ptr, g_ptr, do_ptr, dv_ptr, dg_ptr, HQ, HKV, D, eps,
        HAS_GATE: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        kh = tl.program_id(1)
        d = tl.arange(0, BLOCK_D)
        m = d < D
        v = tl.load(v_ptr + (t * HKV + kh) * D + d, mask=m, other=0.0).to(tl.float32)
        n = tl.sum(v * v, axis=0) + eps
        dv = tl.zeros([BLOCK_D], dtype=tl.float32)
        group = HQ // HKV
        for i in range(0, group):
            qh = kh * group + i
            base = (t * HQ + qh) * D
            o = tl.load(o_ptr + base + d, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(dy_ptr + base + d, mask=m, other=0.0).to(tl.float32)
            c = tl.sum(o * v, axis=0) / n
            if HAS_GATE:
                g = tl.load(g_ptr + base + d, mask=m, other=0.0).to(tl.float32)
                s = tl.sigmoid(g)
                x = o - c * v
                tl.store(dg_ptr + base + d, (dy * x * s * (1.0 - s)).to(dg_ptr.dtype.element_ty), mask=m)
                dx = dy * s
            else:
                dx = dy
            a = tl.sum(dx * v, axis=0)
            tl.store(do_ptr + base + d, (dx - (a / n) * v).to(do_ptr.dtype.element_ty), mask=m)
            dv += -c * dx - (a / n) * (o - 2.0 * c * v)
        tl.store(dv_ptr + (t * HKV + kh) * D + d, dv.to(dv_ptr.dtype.element_ty), mask=m)

    class _XSA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, o, v, gate, eps):
            o, v = o.contiguous(), v.contiguous()
            gate = gate.contiguous() if gate is not None else None
            t, hq, dim = o.shape
            hkv = v.shape[1]
            y = torch.empty_like(o)
            block = triton.next_power_of_2(dim)
            _xsa_fwd[(t, hkv)](
                o, v, gate if gate is not None else o, y, hq, hkv, dim, eps,
                HAS_GATE=gate is not None, BLOCK_D=block,
            )
            ctx.save_for_backward(o, v, gate)
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, dy):
            o, v, gate = ctx.saved_tensors
            t, hq, dim = o.shape
            hkv = v.shape[1]
            do, dv = torch.empty_like(o), torch.empty_like(v)
            dg = torch.empty_like(gate) if gate is not None else None
            _xsa_bwd[(t, hkv)](
                dy.contiguous(), o, v, gate if gate is not None else o, do, dv,
                dg if dg is not None else do, hq, hkv, dim, ctx.eps,
                HAS_GATE=gate is not None, BLOCK_D=triton.next_power_of_2(dim),
            )
            return do, dv, dg, None


def exclusive_self_attention(
    o: torch.Tensor, v: torch.Tensor, gate: Optional[torch.Tensor] = None, eps: float = EPS
) -> torch.Tensor:
    """XSA (and the optional output gate) over ``o`` [T, Hq, D] with own values ``v``
    [T, Hkv, D] (Hq a multiple of Hkv, GQA)."""
    if o.shape[1] % v.shape[1]:
        raise ValueError(f"query heads ({o.shape[1]}) not a multiple of kv heads ({v.shape[1]})")
    if HAVE_TRITON and o.is_cuda:
        return _XSA.apply(o, v, gate, eps)
    return xsa_reference(o, v, gate, eps)
