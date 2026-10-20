# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Context parallelism for KronLSTM: a state relay around the scans.
#
# FLA's KCP for GDN / KDA (`fla/ops/cp`) compresses a rank's effect on the incoming state into a
# K x K transition, because their chunk transition acts on the key axis alone. KronLSTM's acts on
# the whole K x V state: the score p_t = (gamma_t k_t)^T S_0 v_t contracts both axes, so the
# map S_0 -> S_C is a general (KV x KV) linear map and nothing of that size is worth forming.
# Only the scans need the incoming state, though: the Grams, the inverse and the decay-scaled
# keys (`chunk_kronlstm_fwd_intra`) are computed on every rank at once, then the forward scans
# run in rank order, each starting from the state the previous rank sends, and the read-out is
# parallel again. The backward mirrors it: the read-out Gram cotangent in parallel, the reverse
# scans in reverse rank order (each rank's dh0 is the previous rank's dht), the rest in parallel.
#
# Only the first local sequence can continue from earlier ranks and only the last can continue on
# (`FLACPContext.pre_num_ranks` / `post_num_ranks`, consistent across a boundary by construction),
# so each rank receives and sends at most one [H, K, V] state per pass, and a rank whose first
# sequence starts on it does not wait.

from __future__ import annotations

import torch
import torch.distributed as dist


def _peer(group, offset: int) -> int:
    return dist.get_global_rank(group, dist.get_rank(group) + offset)


def _p2p(op, x: torch.Tensor, peer: int, group) -> None:
    # gloo (the CPU test backend) has no CUDA point-to-point; stage through the host there
    if x.is_cuda and dist.get_backend(group) == 'gloo':
        host = x.cpu()
        op(host, peer, group=group)
        if op is dist.recv:
            x.copy_(host)
        return
    op(x, peer, group=group)


def cp_initial_state(cp_context, N: int, H: int, K: int, V: int, device) -> torch.Tensor:
    """The [N, H, K, V] fp32 initial states of the local sequences: the previous rank's final state
    for the first one when it continues from there, zeros otherwise."""
    h0 = torch.zeros(N, H, K, V, dtype=torch.float32, device=device)
    if cp_context.pre_num_ranks:
        buf = torch.empty(H, K, V, dtype=torch.float32, device=device)
        _p2p(dist.recv, buf, _peer(cp_context.group, -1), cp_context.group)
        h0[0] = buf
    return h0


def cp_send_final_state(cp_context, ht: torch.Tensor) -> None:
    """Hand the last local sequence's final state to the next rank when it continues there."""
    if cp_context.post_num_ranks:
        _p2p(dist.send, ht[-1].contiguous(), _peer(cp_context.group, 1), cp_context.group)


def cp_final_state_grad(cp_context, N: int, H: int, K: int, V: int, device) -> torch.Tensor:
    """The [N, H, K, V] fp32 final-state cotangents: the next rank's dh0 for the last local
    sequence when it continues there, zeros otherwise."""
    dht = torch.zeros(N, H, K, V, dtype=torch.float32, device=device)
    if cp_context.post_num_ranks:
        buf = torch.empty(H, K, V, dtype=torch.float32, device=device)
        _p2p(dist.recv, buf, _peer(cp_context.group, 1), cp_context.group)
        dht[-1] = buf
    return dht


def cp_send_initial_state_grad(cp_context, dh0: torch.Tensor) -> None:
    """Hand the first local sequence's dh0 to the previous rank when it continues from there."""
    if cp_context.pre_num_ranks:
        _p2p(dist.send, dh0[0].float().contiguous(), _peer(cp_context.group, -1), cp_context.group)
