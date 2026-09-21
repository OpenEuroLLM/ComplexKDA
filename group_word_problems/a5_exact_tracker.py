# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""An exact A_5 tracker in an UNMODIFIED ComplexKimiDeltaAttention, by weight-setting.

Per thm:a5-four, token g acts on a 4-dim head as A_g = (I - 2 a_g a_g^T)
Diag(-1,1,1,1), where a_g is g's lift into the binary icosahedral group 2I
(2I/{+-1} = A_5). That is exactly the KDA transition at beta=2, alpha=(-1,1,1,1),
so the recurrence composes the group for free.

Nothing the layer exposes as an argument is needed; the embedding carries it:

    dims [0:4]  a_g            -> k_proj
    dim  [4]    constant 1     -> q_proj (fixed query) and b_proj (beta=2; b_proj
                                  has no bias, so this IS the bias)
    dims [5:9]  prefix one-hot -> v_proj, so v=0 on group tokens

f_proj=0 makes alpha token-independent; dt_bias=+-20 saturates tanh so
|alpha| = 1 EXACTLY (at +-12 it is 0.9999878, which compounds to 0.9938 over 512
steps and puts the state 3e-2 off); identical rows across heads tie them.

The one thing weights cannot supply is the initial state: the recurrence starts
at S=0 and its update is invertible, hence rank-preserving, so rank(S) = number
of writes and the tracker needs rank = head_dim. Hence a head_dim-token prefix,
the only tokens allowed to write.

    python group_word_problems/a5_exact_tracker.py [--length 512] [--batch 64]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
from fla.models.utils import Cache

PHI = (1 + 5 ** 0.5) / 2


def binary_icosahedral():
    """The 120 unit quaternions of 2I, as (lifts, table, identity) for 2I/{+-1}."""
    def ham(a, b):
        aw, ax, ay, az = a.T
        bw, bx, by, bz = b.T
        return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                         aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], -1)
    els = [np.eye(4)[i] * s for i in range(4) for s in (1.0, -1.0)]
    els += [np.array(sg) for sg in itertools.product([.5, -.5], repeat=4)]
    base = [0.0, 1.0, 1 / PHI, PHI]
    for p in itertools.permutations(range(4)):
        if sum(p[i] > p[j] for i in range(4) for j in range(i+1, 4)) % 2:
            continue
        for sg in itertools.product([1.0, -1.0], repeat=3):
            w = np.empty(4)
            w[list(p)] = np.array([base[0], sg[0]*base[1], sg[1]*base[2], sg[2]*base[3]]) / 2
            els.append(w)
    Q = np.unique(np.round(np.array(els), 9), axis=0)
    assert len(Q) == 120
    idx = {tuple(np.round(r, 6)): i for i, r in enumerate(Q)}
    prod = np.array([[idx[tuple(np.round(ham(Q[i:i+1], Q[j:j+1])[0], 6))]
                      for j in range(120)] for i in range(120)])
    cls, rep = -np.ones(120, int), []
    for i in range(120):
        if cls[i] < 0:
            cls[i] = cls[idx[tuple(np.round(-Q[i], 6))]] = len(rep)
            rep.append(i)
    table = np.array([[cls[prod[rep[a], rep[b]]] for b in range(60)] for a in range(60)])
    return Q[np.array(rep)], table, cls[idx[(1.0, 0.0, 0.0, 0.0)]]


def isomorphism(sa, ia, sq, iq):
    """phi: perm_group('a5') -> 2I/{+-1}. BFS on words in a (2,3,5) generator
    pair, then verified as a bijection and a homomorphism on all 3600 pairs."""
    def order(t, e, g):
        o, x = 1, g
        while x != e:
            x, o = t[g, x], o + 1
        return o

    def pairs(t, e):
        o = {g: order(t, e, g) for g in range(60)}
        return [(r, s) for r in range(60) if o[r] == 5
                for s in range(60) if o[s] == 2 and o[t[r, s]] == 3]

    for ra, ssa in pairs(sa, ia):
        for rq, ssq in pairs(sq, iq):
            phi, frontier = {ia: iq}, [(ia, iq)]
            while frontier:
                nxt = []
                for a, q in frontier:
                    for ga, gq in ((ra, rq), (ssa, ssq)):
                        if sa[ga, a] not in phi:
                            phi[sa[ga, a]] = sq[gq, q]
                            nxt.append((sa[ga, a], sq[gq, q]))
                frontier = nxt
            if len(set(phi.values())) == 60 and all(
                    phi[sa[a, b]] == sq[phi[a], phi[b]] for a in range(60) for b in range(60)):
                return np.array([phi[g] for g in range(60)])
    raise RuntimeError("no isomorphism found")


def transitions(keys: torch.Tensor) -> torch.Tensor:
    """Four-dimensional signed-Householder representation of A5."""
    keys = F.normalize(keys, dim=-1)
    eye = torch.eye(4, dtype=keys.dtype, device=keys.device)
    diagonal = torch.diag(
        torch.tensor([-1.0, 1.0, 1.0, 1.0], dtype=keys.dtype, device=keys.device)
    )
    return (eye - 2.0 * keys.unsqueeze(-1) * keys.unsqueeze(-2)) @ diagonal


def f_plus(matrix: torch.Tensor) -> torch.Tensor:
    """Apply the explicit SO(4) to SO(3) decoder from the A5 construction."""
    scalar = matrix[..., :1, :1]
    row = matrix[..., :1, 1:]
    column = matrix[..., 1:, :1]
    block = matrix[..., 1:, 1:]
    x, y, z = row.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(*row.shape[:-2], 3, 3)
    return scalar * block - column @ row - block @ skew


def build(quats, heads=4, head_dim=4, d_model=64, b_sat=20.0, seed=0):
    """The layer with every weight set by hand, plus the matching embedding."""
    V, H, K = 60, heads, head_dim
    layer = ComplexKimiDeltaAttention(
        hidden_size=d_model, head_dim=K, num_heads=H, gate="signed_sigmoid2",
        allow_neg_eigval=True, use_short_conv=False, drop_silu=True,
        backend="naive_recurrent", layer_idx=0).eval()
    emb = nn.Embedding(V + K, d_model)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        emb.weight.zero_()
        emb.weight[:V, 0:4] = torch.tensor(quats, dtype=torch.float32)
        emb.weight[:, 4] = 1.0
        for i in range(K):
            emb.weight[V + i, 0:4] = F.one_hot(torch.tensor(i), 4).float()
            emb.weight[V + i, 5 + i] = 1.0
        layer.k_proj.weight.zero_()
        layer.q_proj.weight.zero_()
        for h in range(H):
            layer.k_proj.weight[K*h:K*h+K, 0:4] = torch.eye(4)
            layer.q_proj.weight[K*h:K*h+K, 4] = F.one_hot(torch.tensor(h), K).float()
        layer.v_proj.weight.zero_()
        layer.v_proj.weight[:, 5:5+K] = torch.randn(K*H, K, generator=g)
        layer.b_proj.weight.zero_()
        layer.b_proj.weight[:, 4] = 30.0
        for lin in layer.f_proj:
            lin.weight.zero_()
        layer.dt_bias.copy_(torch.tensor([-b_sat, b_sat, b_sat, b_sat] * H))
        layer.A_log.zero_()
        for lin in layer.g_proj:
            lin.weight.zero_()
        layer.g_proj[-1].bias.zero_()
        layer.o_norm.weight.fill_(1.0)
        layer.o_proj.weight.zero_()
        layer.o_proj.weight[K*H:2*K*H, :] = torch.eye(K*H)
    return layer, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    quats, q_table, q_ident = binary_icosahedral()
    sys.argv = ["x"]
    spec = importlib.util.spec_from_file_location(
        "twp", "group_word_problems/train_wordproblem.py")
    twp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twp)
    _, table, ident = twp.perm_group("a5")
    table = np.array(table)
    quats = quats[isomorphism(table, ident, q_table, q_ident)]
    print("isomorphism perm_group('a5') -> 2I/{+-1} verified on all 3600 pairs")

    layer, emb = build(quats, seed=a.seed)
    K = layer.head_k_dim
    print(f"layer: {sum(p.numel() for p in layer.parameters()):,} params, all hand-set, "
          f"0 trained | prefix: {K} tokens (= head_dim, the rank the state needs)")

    # A_5 word problems; same recursion the trainer uses: acc <- table[token, acc]
    rng = np.random.default_rng(a.seed)
    w = rng.integers(0, 60, size=(a.batch, a.length))
    y, acc = np.empty_like(w), np.full(a.batch, ident)
    for t in range(a.length):
        acc = table[w[:, t], acc]
        y[:, t] = acc
    x = torch.from_numpy(np.concatenate([np.tile(np.arange(60, 60 + K), (a.batch, 1)), w], 1))
    print(f"\nbatch: {tuple(x.shape)} tokens ({a.batch} x [{K} prefix + {a.length} A_5])")

    with torch.no_grad():
        cache = Cache()
        out, _, cache = layer(emb(x), use_cache=True, past_key_values=cache)
        pre = Cache()
        layer(emb(x[:, :K]), use_cache=True, past_key_values=pre)
    state = cache[0]["recurrent_state"]
    print(f"forward: output {tuple(out.shape)}, final state {tuple(state.shape)}")

    A = np.stack([(np.eye(4) - 2*np.outer(q, q)) @ np.diag([-1.0, 1.0, 1.0, 1.0])
                  for q in quats])
    S = pre[0]["recurrent_state"][:, 0].double().numpy()
    for t in range(a.length):
        S = A[w[:, t]] @ S
    print(f"\nstate == composed group action over {a.length} steps: "
          f"max abs err {np.abs(S - state[:, 0].double().numpy()).max():.2e}")
    print(f"state rank {np.linalg.matrix_rank(state[:, 0].numpy()).min()} of {K} "
          f"(one prefix token would give 1)")


if __name__ == "__main__":
    main()
