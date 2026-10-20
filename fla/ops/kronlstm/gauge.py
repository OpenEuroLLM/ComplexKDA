# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Sign gauge for KronLSTM with a signed forget gate `f in [-1, 1]`, as for Complex KDA
# (`fla/ops/kda/gauge.py`). The kernels only ever see `g = log|f|`. With the running parity
# `P_t = prod_{u<=t} sign(f_u)` per key dimension, the gauged state `S'_t = Diag(P_t) S_t` obeys
#
#     S'_t = Diag(|f_t|) S'_{t-1} + u_t (P_t k_t) v_t^T,
#     y_t  = (P_t k_t)^T Diag(|f_t|) S'_{t-1} v_t,     o_t = scale (P_t q_t)^T S'_t,
#
# so the unsigned kernels on `(P q, P k)` are exact. `P` is a diagonal +-1 map, so `||P k|| = ||k||`
# and the unit-norm key the rule assumes stays unit-norm. The parity resets at every sequence start,
# so a call begins and ends in the true frame: an incoming state is used as is,
# and the final state is un-gauged once, `S_T = Diag(P_T) S'_T`.

import torch

from fla.ops.kda.gauge import gauge_last, gauge_state, kda_sign_cumprod


def sign_parity(sign: torch.Tensor, K: int, cu_seqlens: torch.LongTensor | None = None,
                cp_context=None) -> torch.Tensor:
    """`P_t` of shape `[B, T, H, K]` in int8, from gate signs `[B, T, H]` (per-head) or `[B, T, H, K]`."""
    if sign.dim() == 3:
        sign = sign.unsqueeze(-1).expand(*sign.shape, K)
    p = kda_sign_cumprod(sign.to(torch.int8).contiguous(), cu_seqlens)
    if cp_context is not None:
        from fla.ops.cp.sign import cp_sign_prefix
        p = cp_sign_prefix(p, cp_context)
    return p


def gauge_qk(q: torch.Tensor, k: torch.Tensor, p: torch.Tensor):
    """`(P q, P k)`; differentiable, and its own adjoint."""
    p = p.to(q.dtype)
    return q * p, k * p


def ungauge_final_state(ht: torch.Tensor | None, p: torch.Tensor,
                        cu_seqlens: torch.LongTensor | None = None) -> torch.Tensor | None:
    """`Diag(P_T) S'_T` for the `[N, H, K, V]` state."""
    if ht is None:
        return None
    p_last = gauge_last(p, cu_seqlens)
    if cu_seqlens is not None:
        # an empty sequence has P_T = +1; `gauge_last` would read the previous sequence's last token
        p_last = torch.where((cu_seqlens[1:] > cu_seqlens[:-1]).view(-1, 1, 1), p_last, torch.ones_like(p_last))
    return gauge_state(ht, p_last, state_v_first=False)
