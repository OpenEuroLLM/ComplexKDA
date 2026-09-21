# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Sign gauge for KDA with a signed decay `alpha` in `[-1, 1]`.
#
# The kernels only ever see `|alpha|`. The sign is carried as a running parity
# `P_t = prod_{u<=t} s_u` pushed onto `q` and `k`, which is exact: every use of
# `q`/`k` in the recurrence is bilinear against the state, so a pair `(t, s)`
# picks up `P_t P_s = prod_{u=s+1..t} s_u`, the sign the pair should carry.
# The state is left in the gauged frame and un-gauged once at the boundary.
#
# `P` is applied in an l2norm epilogue rather than in a pass of its own -- the
# norm already materialises the `q`/`k` the chunk kernels read -- and it commutes
# with the norm exactly (`l2norm(P q) = P l2norm(q)`, since the norm is computed
# from squares). The norm used for that is KDA's own copy below, so nothing in
# `fla/modules/l2norm.py` changes for its other callers.

import torch
import triton
import triton.language as tl

from fla.ops.backends import dispatch
from fla.ops.utils.cache import fla_cache_autotune
from fla.utils import autotune_cache_kwargs, input_guard


@fla_cache_autotune(
    configs=[
        triton.Config({'BD': BD, 'BT': BT}, num_warps=num_warps)
        for BD in [32, 64, 128, 256]
        for BT in [16, 32, 64, 128]
        for num_warps in [1, 2, 4]
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def kda_sign_cumprod_kernel(
    s,
    p,
    cu_seqlens,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1).to(tl.int64)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T

    o_d = i_d * BD + tl.arange(0, BD)
    m_d = o_d < D
    # parity carried across tiles as a count; only its low bit matters
    b_carry = tl.zeros([BD], dtype=tl.int32)
    for i_t in range(bos, eos, BT):
        o_t = i_t + tl.arange(0, BT)
        m_s = (o_t < eos)[:, None] & m_d[None, :]
        p_s = s + o_t[:, None] * D + o_d[None, :]
        b_neg = (tl.load(p_s, mask=m_s, other=1) < 0).to(tl.int32)
        b_cnt = b_carry[None, :] + tl.cumsum(b_neg, 0)
        b_carry += tl.sum(b_neg, 0)
        p_p = p + o_t[:, None] * D + o_d[None, :]
        tl.store(p_p, tl.where(b_cnt & 1 == 1, -1, 1).to(p.dtype.element_ty), mask=m_s)


@dispatch('kda')
@input_guard
def kda_sign_cumprod(
    sign: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Running product of the gate signs along the time axis, `P_t = prod_{u<=t} s_u`.

    Integer parity prefix sum: exact at any length, and reset at sequence starts,
    so a call both begins and ends in the true frame.

    Args:
        sign (torch.Tensor):
            Gate signs in `{-1, +1}` of shape `[B, T, HV, K]`. Only the sign bit is read.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training.

    Returns:
        `int8` tensor in `{-1, +1}` of shape `[B, T, HV, K]`.
    """
    B, T, HV, K = sign.shape
    D = HV * K
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    p = torch.empty(B, T, HV, K, device=sign.device, dtype=torch.int8)

    def grid(meta):
        return (triton.cdiv(D, meta['BD']), N)

    kda_sign_cumprod_kernel[grid](
        s=sign,
        p=p,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        IS_VARLEN=cu_seqlens is not None,
    )
    return p


def gauge_last(
    p: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """`P_T` per sequence, of shape `[N, HV, K]`."""
    return p[:, -1] if cu_seqlens is None else p[0, cu_seqlens[1:] - 1]


def gauge_state(
    state: torch.Tensor,
    p_last: torch.Tensor,
    state_v_first: bool,
) -> torch.Tensor:
    """`Diag(P_T) S`, on whichever axis holds K.

    Its own inverse, so it both un-gauges the final state in the forward and
    re-gauges the incoming `dht` in the backward.
    """
    p_last = p_last.to(state.dtype)
    return state * (p_last.unsqueeze(-2) if state_v_first else p_last.unsqueeze(-1))


# -- l2norm with the gauge folded into its epilogue -------------------------
#
# A KDA-local copy of `fla/modules/l2norm.py`'s BT-tiled kernels, differing only
# in the `* P` on the store. It is duplicated rather than added as a flag on the
# shared op on purpose: `l2norm` has ten callers across fla plus an NPU backend,
# and a new constexpr there would change every one of their compile keys for a
# feature only KDA uses. The unsigned path still calls the shared op, unchanged.
#
# Only the `D <= 512` path is mirrored -- `chunk_kda` asserts `K <= 256`.

@fla_cache_autotune(
    configs=[triton.Config({'BT': BT}, num_warps=num_warps)
             for num_warps in [1, 2, 4, 8, 16] for BT in [8, 16, 32, 64, 128]],
    key=['D', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def l2norm_gauge_fwd_kernel(
    x,
    y,
    rstd,
    p,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    o_x = o_t[:, None] * D + o_d[None, :]

    b_x = tl.load(x + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    # the norm is sign-blind, so the gauge commutes: l2norm(P x) == P l2norm(x)
    b_y = b_x * b_rstd[:, None] * tl.load(p + o_x, mask=m_x, other=1).to(tl.float32)

    tl.store(y + o_x, b_y.to(y.dtype.element_ty), mask=m_x)
    tl.store(rstd + o_t, b_rstd.to(rstd.dtype.element_ty), mask=m_t)


@fla_cache_autotune(
    configs=[triton.Config({'BT': BT}, num_warps=num_warps)
             for num_warps in [1, 2, 4, 8, 16] for BT in [8, 16, 32, 64, 128]],
    key=['D', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def l2norm_gauge_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    p,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    o_x = o_t[:, None] * D + o_d[None, :]

    b_y = tl.load(y + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + o_t, mask=m_t, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    # `y` is stored gauged, so `sum(dy * y)` is already gauge-invariant and only
    # the leading factor is left to undo
    b_dx *= tl.load(p + o_x, mask=m_x, other=1).to(tl.float32)
    tl.store(dx + o_x, b_dx.to(dx.dtype.element_ty), mask=m_x)


@fla_cache_autotune(
    configs=[triton.Config({'BT': BT}, num_warps=num_warps)
             for num_warps in [1, 2, 4, 8, 16] for BT in [8, 16, 32, 64, 128]],
    key=['D', 'NB', 'HAS_PARITY'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def l2norm_gauge_pair_fwd_kernel(
    xq,
    xk,
    yq,
    yk,
    rstd_q,
    rstd_k,
    p,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
    HAS_PARITY: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    o_x = o_t[:, None] * D + o_d[None, :]

    b_xq = tl.load(xq + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_xk = tl.load(xk + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd_q = 1 / tl.sqrt(tl.sum(b_xq * b_xq, 1) + eps)
    b_rstd_k = 1 / tl.sqrt(tl.sum(b_xk * b_xk, 1) + eps)
    b_p = 1.0
    if HAS_PARITY:
        b_p = tl.load(p + o_x, mask=m_x, other=1).to(tl.float32)

    tl.store(yq + o_x, (b_xq * b_rstd_q[:, None] * b_p).to(yq.dtype.element_ty), mask=m_x)
    tl.store(yk + o_x, (b_xk * b_rstd_k[:, None] * b_p).to(yk.dtype.element_ty), mask=m_x)
    tl.store(rstd_q + o_t, b_rstd_q.to(rstd_q.dtype.element_ty), mask=m_t)
    tl.store(rstd_k + o_t, b_rstd_k.to(rstd_k.dtype.element_ty), mask=m_t)


@fla_cache_autotune(
    configs=[triton.Config({'BT': BT}, num_warps=num_warps)
             for num_warps in [1, 2, 4, 8, 16] for BT in [8, 16, 32, 64, 128]],
    key=['D', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def l2norm_gauge_pair_bwd_kernel(
    yq,
    yk,
    rstd_q,
    rstd_k,
    dyq,
    dyk,
    dxq,
    dxk,
    p,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    o_x = o_t[:, None] * D + o_d[None, :]

    b_yq = tl.load(yq + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_yk = tl.load(yk + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_dyq = tl.load(dyq + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_dyk = tl.load(dyk + o_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd_q = tl.load(rstd_q + o_t, mask=m_t, other=0.0).to(tl.float32)
    b_rstd_k = tl.load(rstd_k + o_t, mask=m_t, other=0.0).to(tl.float32)
    b_p = tl.load(p + o_x, mask=m_x, other=1).to(tl.float32)

    b_dxq = b_dyq * b_rstd_q[:, None] - tl.sum(b_dyq * b_yq, 1)[:, None] * b_yq * b_rstd_q[:, None]
    b_dxk = b_dyk * b_rstd_k[:, None] - tl.sum(b_dyk * b_yk, 1)[:, None] * b_yk * b_rstd_k[:, None]
    tl.store(dxq + o_x, (b_dxq * b_p).to(dxq.dtype.element_ty), mask=m_x)
    tl.store(dxk + o_x, (b_dxk * b_p).to(dxk.dtype.element_ty), mask=m_x)


def _grid(T):
    def grid(meta):
        return (triton.cdiv(T, meta['BT']),)
    return grid


@input_guard
def l2norm_gauge_fwd(x: torch.Tensor, p: torch.Tensor, eps: float = 1e-6):
    """`P * l2norm(x)`, in one pass. Mirrors `l2norm_fwd`'s return contract."""
    shape = x.shape
    x, p = x.view(-1, shape[-1]), p.view(-1, shape[-1])
    T, D = x.shape
    if D > 512:
        raise NotImplementedError(f"the gauged l2norm covers feature dim <= 512, got {D}.")
    y = torch.empty_like(x)
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    l2norm_gauge_fwd_kernel[_grid(T)](
        x=x, y=y, rstd=rstd, p=p, eps=eps, T=T, D=D,
        BD=triton.next_power_of_2(D), NB=triton.cdiv(T, 2048 * 32),
    )
    return y.view(shape), rstd.view(shape[:-1])


@input_guard
def l2norm_gauge_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor,
                     p: torch.Tensor, eps: float = 1e-6):
    """`y` must be the gauged output of `l2norm_gauge_fwd` and `p` the same gauge."""
    shape = y.shape
    y, dy, p = (t.view(-1, shape[-1]) for t in (y, dy, p))
    T, D = y.shape
    if D > 512:
        raise NotImplementedError(f"the gauged l2norm covers feature dim <= 512, got {D}.")
    dx = torch.empty_like(y)
    l2norm_gauge_bwd_kernel[_grid(T)](
        y=y, rstd=rstd, dy=dy, dx=dx, p=p, eps=eps, T=T, D=D,
        BD=triton.next_power_of_2(D), NB=triton.cdiv(T, 2048 * 32),
    )
    return dx.view(shape)


@dispatch('kda')
@input_guard
def l2norm_gauge_pair_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize q and k together, optionally applying a shared sign gauge."""
    shape = q.shape
    q, k = (x.view(-1, shape[-1]) for x in (q, k))
    p = p.view(-1, shape[-1]) if p is not None else None
    T, D = q.shape
    if D > 512:
        raise NotImplementedError(f"the paired gauged l2norm covers feature dim <= 512, got {D}.")
    yq, yk = torch.empty_like(q), torch.empty_like(k)
    rstd_q = torch.empty((T,), dtype=torch.float32, device=q.device)
    rstd_k = torch.empty((T,), dtype=torch.float32, device=k.device)
    l2norm_gauge_pair_fwd_kernel[_grid(T)](
        xq=q,
        xk=k,
        yq=yq,
        yk=yk,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        p=p,
        eps=eps,
        T=T,
        D=D,
        BD=triton.next_power_of_2(D),
        NB=triton.cdiv(T, 2048 * 32),
        HAS_PARITY=p is not None,
    )
    return yq.view(shape), rstd_q.view(shape[:-1]), yk.view(shape), rstd_k.view(shape[:-1])


@dispatch('kda')
@input_guard
def l2norm_gauge_pair_bwd(
    yq: torch.Tensor,
    rstd_q: torch.Tensor,
    dyq: torch.Tensor,
    yk: torch.Tensor,
    rstd_k: torch.Tensor,
    dyk: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate paired gauged q/k normalization in one launch."""
    shape = yq.shape
    yq, dyq, yk, dyk, p = (x.view(-1, shape[-1]) for x in (yq, dyq, yk, dyk, p))
    T, D = yq.shape
    if D > 512:
        raise NotImplementedError(f"the paired gauged l2norm covers feature dim <= 512, got {D}.")
    dxq, dxk = torch.empty_like(yq), torch.empty_like(yk)
    l2norm_gauge_pair_bwd_kernel[_grid(T)](
        yq=yq,
        yk=yk,
        rstd_q=rstd_q,
        rstd_k=rstd_k,
        dyq=dyq,
        dyk=dyk,
        dxq=dxq,
        dxk=dxk,
        p=p,
        eps=eps,
        T=T,
        D=D,
        BD=triton.next_power_of_2(D),
        NB=triton.cdiv(T, 2048 * 32),
    )
    return dxq.view(shape), dxk.view(shape)
