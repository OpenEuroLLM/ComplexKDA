# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li; wide-key kernels (c) 2026 OpenEuroLLM
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Wide-key state recurrence for the delta-rule chunk kernels (K > 256).

``chunk_gated_delta_rule_fwd_h`` / ``bwd_dhu`` keep a head's whole ``K x BV`` state
tile in registers as at most four 64-wide tiles, hence ``K <= 256``. Wide keys -- the
joint key of triadic linear attention is ``E * D`` wide, up to 2048 -- run here: the
running state lives in an fp32 buffer in global memory (per head ``K * V * 4`` bytes,
L2-resident at training sizes) and every chunk step loops over 64-wide key tiles: read
the state tiles for ``v_new = u - w h`` (storing the per-chunk copy the backward needs),
then update them, ``h <- decay h + k^T v_new``. Each program owns its value columns of
the state, so the only synchronisation is a barrier between the two phases. K-major
state layout only.
"""

import torch
import triton
import triton.language as tl

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.op import exp2

BK = tl.constexpr(64)  # key-tile width of the state loops


@triton.jit(do_not_specialize=["T"])
def _wide_fwd_h(
    k, v, w, v_new, g, gk, h, hs, h0, ht, cu_seqlens, chunk_offsets, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, SAVE_NEW_VALUE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    NK = tl.cdiv(K, BK)
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    hs += i_nh * K * V
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    # running fp32 state <- initial state (or zeros)
    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        m = (o_k < K)[:, None] & m_v[None, :]
        b = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_INITIAL_STATE:
            b += tl.load(h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=m, other=0.0)
        tl.store(hs + o_k[:, None] * V + o_v[None, :], b, mask=m)
    tl.debug_barrier()

    for i_t in range(NT):
        i_t64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        # phase 1: the per-chunk state copy, and v_new = u - w h
        b_v = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m = m_k[:, None] & m_v[None, :]
            b_h = tl.load(hs + o_k[:, None] * V + o_v[None, :], mask=m, other=0.0)
            tl.store(h + i_t64 * HV * K * V + o_k[:, None] * V + o_v[None, :],
                     b_h.to(h.dtype.element_ty), mask=m)
            b_w = tl.load(w + o_t[:, None] * (HV * K) + o_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_v += tl.dot(b_w, b_h.to(b_w.dtype))
        b_v = tl.load(v + o_t[:, None] * (HV * V) + o_v[None, :],
                      mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v
        if SAVE_NEW_VALUE:
            tl.store(v_new + o_t[:, None] * (HV * V) + o_v[None, :],
                     b_v.to(v_new.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        last_idx = min((i_t + 1) * BT, T) - 1
        b_g_last = 0.0
        if USE_G:
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(tl.float32)
            b_g = tl.load(g + (bos * HV + i_h).to(tl.int64) + o_t * HV, mask=m_t, other=0.0)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g.to(tl.float32)), 0)[:, None]
        b_v = b_v.to(k.dtype.element_ty)
        tl.debug_barrier()
        # phase 2: h <- decay h + k^T v_new
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m = m_k[:, None] & m_v[None, :]
            p_hs = hs + o_k[:, None] * V + o_v[None, :]
            b_h = tl.load(p_hs, mask=m, other=0.0)
            if USE_G:
                b_h *= exp2(b_g_last)
            if USE_GK:
                b_gk = tl.load(gk + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_h *= exp2(b_gk)[:, None]
            b_k = tl.load(k + o_k[:, None] + o_t[None, :] * (H * K),
                          mask=m_k[:, None] & m_t[None, :], other=0.0)
            b_h += tl.dot(b_k, b_v)
            tl.store(p_hs, b_h, mask=m)
        tl.debug_barrier()

    if STORE_FINAL_STATE:
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m = (o_k < K)[:, None] & m_v[None, :]
            b_h = tl.load(hs + o_k[:, None] * V + o_v[None, :], mask=m, other=0.0)
            tl.store(ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :], b_h, mask=m)

@triton.jit(do_not_specialize=["T"])
def _wide_bwd_dhu(
    q, k, w, g, gk, dht, dh0, do, dh, dhs, dv, dv2, cu_seqlens, chunk_offsets, scale, T,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BV: tl.constexpr,
    USE_G: tl.constexpr, USE_GK: tl.constexpr, USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v, i_nh = pid % NV, (pid // NV).to(tl.int64)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int64)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT
    NK = tl.cdiv(K, BK)
    q += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    do += (bos * HV + i_h).to(tl.int64) * V
    dv += (bos * HV + i_h).to(tl.int64) * V
    dv2 += (bos * HV + i_h).to(tl.int64) * V
    dh += (boh * HV + i_h).to(tl.int64) * K * V
    dhs += i_nh * K * V
    if USE_GK:
        gk += (bos * HV + i_h).to(tl.int64) * K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    for i_k in range(NK):
        o_k = i_k * BK + tl.arange(0, BK)
        m = (o_k < K)[:, None] & m_v[None, :]
        b = tl.zeros([BK, BV], dtype=tl.float32)
        if USE_FINAL_STATE_GRADIENT:
            b += tl.load(dht + i_nh * K * V + o_k[:, None] * V + o_v[None, :], mask=m, other=0.0)
        tl.store(dhs + o_k[:, None] * V + o_v[None, :], b, mask=m)
    tl.debug_barrier()

    for i_t in range(NT - 1, -1, -1):
        i_t64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        last_idx = min((i_t + 1) * BT, T) - 1
        # phase 1: the per-chunk dh copy, and dv = k dh (+ the incoming dv)
        b_dv = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m = m_k[:, None] & m_v[None, :]
            b_dh = tl.load(dhs + o_k[:, None] * V + o_v[None, :], mask=m, other=0.0)
            tl.store(dh + i_t64 * HV * K * V + o_k[:, None] * V + o_v[None, :],
                     b_dh.to(dh.dtype.element_ty), mask=m)
            b_k = tl.load(k + o_t[:, None] * (H * K) + o_k[None, :],
                          mask=m_t[:, None] & m_k[None, :], other=0.0)
            b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))
        bg_last = 0.0
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * HV + i_h).to(tl.float32)
            b_g = tl.load(g + bos * HV + i_h + o_t * HV, mask=m_t, other=0.0).to(tl.float32)
            b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(dv + o_t[:, None] * (HV * V) + o_v[None, :],
                        mask=m_t[:, None] & m_v[None, :], other=0.0)
        tl.store(dv2 + o_t[:, None] * (HV * V) + o_v[None, :],
                 b_dv.to(dv2.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])
        b_do = tl.load(do + o_t[:, None] * (HV * V) + o_v[None, :],
                       mask=m_t[:, None] & m_v[None, :], other=0.0)
        tl.debug_barrier()
        # phase 2: dh <- decay dh + scale q^T do - w^T dv
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m = m_k[:, None] & m_v[None, :]
            p_dhs = dhs + o_k[:, None] * V + o_v[None, :]
            b_dh = tl.load(p_dhs, mask=m, other=0.0)
            b_q = tl.load(q + o_k[:, None] + o_t[None, :] * (H * K),
                          mask=m_k[:, None] & m_t[None, :], other=0.0)
            b_w = tl.load(w + o_k[:, None] + o_t[None, :] * (HV * K),
                          mask=m_k[:, None] & m_t[None, :], other=0.0)
            if USE_G:
                b_dh *= exp2(bg_last)
                b_q = b_q * exp2(b_g)[None, :]
            if USE_GK:
                b_gk = tl.load(gk + last_idx * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)
                b_dh *= exp2(b_gk)[:, None]
            b_dh += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale \
                - tl.dot(b_w, b_dv.to(b_w.dtype))
            tl.store(p_dhs, b_dh, mask=m)
        tl.debug_barrier()

    if USE_INITIAL_STATE:
        for i_k in range(NK):
            o_k = i_k * BK + tl.arange(0, BK)
            m = (o_k < K)[:, None] & m_v[None, :]
            b_dh = tl.load(dhs + o_k[:, None] * V + o_v[None, :], mask=m, other=0.0)
            tl.store(dh0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :], b_dh, mask=m)


def _layout(k, cu_seqlens, chunk_size, chunk_indices):
    B, T = k.shape[:2]
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        return B, triton.cdiv(T, chunk_size), None
    return len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, chunk_size)


def wide_fwd_h(k, w, u, g=None, gk=None, initial_state=None, output_final_state=False,
               chunk_size=64, save_new_value=True, state_v_first=False, cu_seqlens=None,
               cu_seqlens_cpu=None, chunk_indices=None, BV=32, num_warps=4):
    """``chunk_gated_delta_rule_fwd_h`` for any K (K-major state)."""
    if state_v_first:
        raise NotImplementedError("the wide-key kernels use the K-major state layout")
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    N, NT, chunk_offsets = _layout(k, cu_seqlens, chunk_size, chunk_indices)
    h = k.new_empty(B, NT, HV, K, V)
    hs = torch.empty(N * HV, K, V, device=k.device, dtype=torch.float32)
    final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None
    grid = (triton.cdiv(V, BV) * N * HV,)
    _wide_fwd_h[grid](
        k, u, w, v_new, g, gk, h, hs, initial_state, final_state, cu_seqlens, chunk_offsets, T,
        H=H, HV=HV, K=K, V=V, BT=chunk_size, BV=BV,
        USE_G=g is not None, USE_GK=gk is not None, USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None, SAVE_NEW_VALUE=v_new is not None,
        IS_VARLEN=cu_seqlens is not None, num_warps=num_warps,
    )
    return h, v_new, final_state


def wide_bwd_dhu(q, k, w, do, dv, g=None, gk=None, h0=None, dht=None, scale=None,
                 state_v_first=False, cu_seqlens=None, chunk_size=64, chunk_indices=None,
                 BV=32, num_warps=4):
    """``chunk_gated_delta_rule_bwd_dhu`` for any K (K-major state)."""
    if state_v_first:
        raise NotImplementedError("the wide-key kernels use the K-major state layout")
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    N, NT, chunk_offsets = _layout(q, cu_seqlens, chunk_size, chunk_indices)
    dh = q.new_empty(B, NT, HV, K, V)
    dhs = torch.empty(N * HV, K, V, device=q.device, dtype=torch.float32)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)
    grid = (triton.cdiv(V, BV) * N * HV,)
    _wide_bwd_dhu[grid](
        q, k, w, g, gk, dht, dh0, do, dh, dhs, dv, dv2, cu_seqlens, chunk_offsets,
        scale if scale is not None else K**-0.5, T,
        H=H, HV=HV, K=K, V=V, BT=chunk_size, BV=BV,
        USE_G=g is not None, USE_GK=gk is not None, USE_INITIAL_STATE=h0 is not None,
        USE_FINAL_STATE_GRADIENT=dht is not None, IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps,
    )
    return dh, dh0, dv2
