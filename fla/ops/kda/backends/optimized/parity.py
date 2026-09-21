# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def sign_local_scan_kernel(
    sign,
    parity,
    block_sign,
    T,
    D: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = (o_t[:, None] < T) & (o_d[None, :] < D)
    offsets = (i_b * T + o_t[:, None]) * D + o_d[None, :]
    negative = (tl.load(sign + offsets, mask=mask, other=1) < 0).to(tl.int32)
    count = tl.cumsum(negative, axis=0)
    tl.store(parity + offsets, tl.where(count & 1 == 1, -1, 1).to(tl.int8), mask=mask)

    block_offsets = (i_b * NT + i_t) * D + o_d
    block_negative = tl.sum(negative, axis=0) & 1
    tl.store(block_sign + block_offsets, tl.where(block_negative == 1, -1, 1).to(tl.int8), mask=o_d < D)


@triton.jit
def sign_block_scan_kernel(
    block_sign,
    block_prefix,
    D: tl.constexpr,
    NT: tl.constexpr,
    BNT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_b = tl.program_id(1).to(tl.int64)
    o_t = tl.arange(0, BNT)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = (o_t[:, None] < NT) & (o_d[None, :] < D)
    offsets = (i_b * NT + o_t[:, None]) * D + o_d[None, :]
    negative = (tl.load(block_sign + offsets, mask=mask, other=1) < 0).to(tl.int32)
    inclusive = tl.cumsum(negative, axis=0)
    exclusive = inclusive - negative
    tl.store(block_prefix + offsets, tl.where(exclusive & 1 == 1, -1, 1).to(tl.int8), mask=mask)


@triton.jit(do_not_specialize=["T"])
def sign_apply_prefix_kernel(
    parity,
    block_prefix,
    T,
    D: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_d = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_b = tl.program_id(2).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = (o_t[:, None] < T) & (o_d[None, :] < D)
    offsets = (i_b * T + o_t[:, None]) * D + o_d[None, :]
    prefix_offsets = (i_b * NT + i_t) * D + o_d
    prefix = tl.load(block_prefix + prefix_offsets, mask=o_d < D, other=1).to(tl.int8)
    values = tl.load(parity + offsets, mask=mask, other=1).to(tl.int8)
    tl.store(parity + offsets, values * prefix[None, :], mask=mask)


def sign_cumprod_parallel(sign: torch.Tensor, block_t: int, block_d: int) -> torch.Tensor:
    B, T, H, K = sign.shape
    D = H * K
    NT = triton.cdiv(T, block_t)
    parity = torch.empty_like(sign, dtype=torch.int8)
    block_sign = torch.empty((B, NT, D), device=sign.device, dtype=torch.int8)
    block_prefix = torch.empty_like(block_sign)
    grid = (triton.cdiv(D, block_d), NT, B)
    sign_local_scan_kernel[grid](
        sign=sign,
        parity=parity,
        block_sign=block_sign,
        T=T,
        D=D,
        NT=NT,
        BT=block_t,
        BD=block_d,
    )
    sign_block_scan_kernel[(triton.cdiv(D, block_d), B)](
        block_sign=block_sign,
        block_prefix=block_prefix,
        D=D,
        NT=NT,
        BNT=triton.next_power_of_2(NT),
        BD=block_d,
    )
    sign_apply_prefix_kernel[grid](
        parity=parity,
        block_prefix=block_prefix,
        T=T,
        D=D,
        NT=NT,
        BT=block_t,
        BD=block_d,
    )
    return parity
