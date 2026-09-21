"""Group word problems for the signed KDA gate.

Uses ComplexKimiDeltaAttention itself -- the same layer as the real model, not a
reimplementation. The layer picks its own backend: the fla Triton kernel when
one can launch, and fla's OWN reference in fla.ops.kda.naive otherwise, so the
same script runs on a laptop and on a GPU box.

    python train_wordproblem.py --task s3                        # 4 gates x 1 seed
    python train_wordproblem.py --task s2 --no-neg-eigval --seeds 3
    python train_wordproblem.py --all --seeds 3 --out runs.jsonl
    python train_wordproblem.py --summary runs.jsonl

ON THE CPU BACKEND. The default fallback is fla's naive_recurrent_kda. Its
chunkwise sibling naive_chunk_kda is available via --backend naive_chunk but is
GUARDED: it evaluates exp(g_i - g_j) over a whole block and masks afterwards, so
once the exponent span |log alpha| * chunk exceeds ~88 the masked upper triangle
overflows fp32 to inf, the forward silently discards it, and the BACKWARD
computes 0 * inf = NaN. A signed gate drives |alpha| to the eps floor at every
sign crossing, so that regime is routine here rather than a corner. The layer
raises rather than returning NaN gradients. The Triton kernel is unaffected: it
rebases to the sub-block midpoint and masks at the exp. See
test_gate_grad_near_zero.py.

ON THE GATE. One name selects the whole parameterisation. The old trio
(allow_neg_gate x gate_activation x gate_form) had eight combinations but only
four distinct configurations, because gate_form applies solely to the unsigned
arm and gate_activation solely to the signed one.

    softplus         alpha in (0,1]   -exp(A_log)*softplus(u)     [shipped default]
    sigmoid          alpha in (0,1]   lower_bound*sigmoid(A*u)    [shipped safe_gate]
    signed_sigmoid2  alpha in [-1,1]  2*sigmoid(u)-1 == tanh(u/2)
    signed_tanh      alpha in [-1,1]  tanh(u)

ON --no-neg-eigval. Pins beta to (0,1], making the Householder a projection that
cannot itself supply a sign flip, so any state tracking is attributable to the
gate. Use it when the gate is what you are measuring; leave it off when you want
the full model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.gated_deltaproduct import GatedDeltaProduct
from fla.layers.complex_kda_layer import GATES, HAS_TRITON, ComplexKimiDeltaAttention, compute_gate, is_signed
from fla.ops.gated_delta_rule.gate import naive_gdn_gate

LB = -5.0
TASKS = ["s2", "z5", "s3", "a4", "s4", "a5", "s5"]


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #

def _parity(p):
    s, p = 1, list(p)
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            s = -s
    return s


def perm_group(name):
    """(elements, Cayley table, identity index). Non-solvable: a5, s5."""
    if name == "s2":
        els = [(0, 1), (1, 0)]
    elif name == "z5":
        els = [tuple((i + k) % 5 for i in range(5)) for k in range(5)]
    elif name == "s3":
        els = sorted(itertools.permutations(range(3)))
    elif name == "a4":
        els = [p for p in itertools.permutations(range(4)) if _parity(p) == 1]
    elif name == "s4":
        els = sorted(itertools.permutations(range(4)))
    elif name == "a5":
        els = [p for p in itertools.permutations(range(5)) if _parity(p) == 1]
    elif name == "s5":
        els = sorted(itertools.permutations(range(5)))
    else:
        raise ValueError(name)
    idx = {e: i for i, e in enumerate(els)}
    tab = np.zeros((len(els), len(els)), dtype=np.int64)
    for i, a in enumerate(els):
        for j, b in enumerate(els):
            tab[i, j] = idx[tuple(a[b[x]] for x in range(len(a)))]
    return els, tab, idx[tuple(range(len(els[0])))]


def transpositions(els):
    """Indices into `els` of the transpositions (swap exactly two positions,
    fix the rest) -- the generating set for --restrict-gens transpositions.
    The OUTPUT space (the Cayley table / label vocab) is untouched, still the
    full group; only the INPUT tokens fed to the model are restricted to this
    subset, so the task becomes "read off the group element as a word in
    swaps" instead of "read off a word in arbitrary elements"."""
    n = len(els[0])
    idx = []
    for i, p in enumerate(els):
        diff = [k for k in range(n) if p[k] != k]
        if len(diff) == 2:
            idx.append(i)
    if not idx:
        raise ValueError("this group has no transpositions to restrict to "
                         "(e.g. alternating groups a4/a5 contain no odd "
                         "permutations)")
    return idx


def cycle_lengths(p):
    """Lengths (>1) of the disjoint cycles of permutation `p`, sorted. A
    transposition is [2], a 3-cycle is [3], a double-transposition is [2, 2]."""
    n = len(p)
    seen = [False] * n
    lens = []
    for i in range(n):
        if seen[i]:
            continue
        length, j = 0, i
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        if length > 1:
            lens.append(length)
    return sorted(lens)


def cycle_curriculum_gens(els, cycle_lens):
    """Cumulative generator index sets, one per curriculum stage: stage i
    allows every permutation that is a SINGLE cycle of a length seen in
    cycle_lens[:i+1] (all other positions fixed). E.g. [2, 3, 4] on s4 gives
    stage 0 = transpositions only, stage 1 = + 3-cycles, stage 2 = + 4-cycles
    -- cumulative so earlier, easier generators stay in the mix rather than
    being swapped out (which risks the same shock/forgetting seen with the
    length curriculum's hard stage transitions)."""
    stages, allowed = [], set()
    for k in cycle_lens:
        allowed |= {i for i, p in enumerate(els) if cycle_lengths(p) == [k]}
        stages.append(sorted(allowed))
    return stages


def subgroup_chain(table, ident):
    """An increasing chain of subgroups {e} < H1 < ... < G, as index lists.

    Motivation is reachability, not difficulty. The A_5 tracker's solution has a
    wide basin (a ~38 degree cone around the true quaternions recovers to 1.000)
    but random init sits at 90 degrees and never enters it. Placing 60
    quaternions correctly by chance is hopeless; placing the 4 of V_4 is not,
    and each subgroup's solution is a SUB-STRUCTURE of the next one's, so every
    stage warm-starts the following one. For A_5 this gives V_4 (4) < A_4 (12)
    < A_5 (60).

    Greedy: repeatedly adjoin whichever outside element yields the smallest
    proper closure, so the chain climbs in the smallest available steps.
    """
    n = table.shape[0]

    def closure(gens):
        H = {ident}
        frontier = list(gens)
        while frontier:
            nxt = []
            for g in frontier:
                for h in list(H):
                    for p in (table[g, h], table[h, g]):
                        if p not in H:
                            H.add(p); nxt.append(p)
                if g not in H:
                    H.add(g); nxt.append(g)
            frontier = nxt
        return frozenset(H)

    chain, H = [], frozenset({ident})
    while len(H) < n:
        best = None
        for g in range(n):
            if g in H:
                continue
            c = closure(list(H) + [g])
            if best is None or len(c) < len(best):
                best = c
        H = best
        chain.append(sorted(H))
    return chain


def make_batch(table, ident, batch, length, rng, device, gen_idx=None, bos_id=None):
    """bos_id, if given, prepends a beginning-of-sequence token (id bos_id,
    outside the group-element vocab) to every sequence, with its label fixed
    to `ident` (no generator applied yet). Since it's a prefix, all downstream
    tail-anchored ("last q of L") accuracy windows are unaffected; only
    absolute-position code (evaluate_continuous) needs to skip column 0."""
    if gen_idx is None:
        x = rng.integers(0, table.shape[0], size=(batch, length))
    else:
        x = np.asarray(gen_idx, dtype=np.int64)[
            rng.integers(0, len(gen_idx), size=(batch, length))]
    y = np.empty_like(x)
    acc = np.full(batch, ident, dtype=np.int64)
    for t in range(length):
        acc = table[x[:, t], acc]
        y[:, t] = acc
    if bos_id is not None:
        bos_col = np.full((batch, 1), bos_id, dtype=np.int64)
        ident_col = np.full((batch, 1), ident, dtype=np.int64)
        x = np.concatenate([bos_col, x], axis=1)
        y = np.concatenate([ident_col, y], axis=1)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def initialize_a5_theory_heads(model, table, ident, heads=2, saturation=12.0):
    """Initialize A5 token coordinates and a few exact four-dimensional transitions."""
    from group_word_problems.a5_exact_tracker import binary_icosahedral, isomorphism

    layer = model.layer
    if layer.gate != "signed_sigmoid2" or not layer.allow_neg_eigval:
        raise ValueError("A5 theory initialization requires signed_sigmoid2 with beta in [0, 2]")
    if layer.act is F.silu:
        raise ValueError("A5 theory initialization requires --drop-silu")
    if layer.b_proj.bias is None:
        raise ValueError("A5 theory initialization requires --beta-init-style spread")
    if heads < 1 or heads > layer.num_heads:
        raise ValueError(f"A5 theory initialization needs 1 <= heads <= {layer.num_heads}")
    if layer.head_k_dim < 4 or model.emb.embedding_dim < 6:
        raise ValueError("A5 theory initialization requires head_dim >= 4 and d_model >= 6")

    quaternions, quaternion_table, quaternion_identity = binary_icosahedral()
    mapping = isomorphism(np.asarray(table), ident, quaternion_table, quaternion_identity)
    keys = torch.as_tensor(
        quaternions[mapping],
        dtype=model.emb.weight.dtype,
        device=model.emb.weight.device,
    )
    vocab = len(table)
    head_dim = layer.head_k_dim

    model.emb.weight[:vocab].zero_()
    model.emb.weight[:vocab, :4] = keys
    model.emb.weight[:vocab, 4] = 1.0
    model.emb.weight[vocab].zero_()
    model.emb.weight[vocab, 4:6] = 1.0

    for head in range(heads):
        start = head * head_dim
        stop = start + head_dim
        bos_key = layer.k_proj.weight[start:stop, 5].clone()
        layer.k_proj.weight[start:stop].zero_()
        layer.k_proj.weight[start:start + 4, :4] = torch.eye(
            4,
            dtype=layer.k_proj.weight.dtype,
            device=layer.k_proj.weight.device,
        )
        layer.k_proj.weight[start:stop, 5] = bos_key
        layer.f_proj[1].weight[start:stop].zero_()
        layer.dt_bias[start:stop] = saturation
        layer.dt_bias[start] = -saturation
        layer.A_log[head] = 0.0
        layer.b_proj.weight[head].zero_()
        layer.b_proj.bias[head] = saturation


def sample_train_len(base_len, rng, dist, alpha=2.0, max_len=None):
    """Length for ONE training step's batch (shared by every row in it).

    "fixed" reproduces the old single-length behaviour exactly, including not
    consuming any rng state, so old runs stay reproducible byte-for-byte.
    "pareto" draws a Lomax-distributed length with scale=base_len: minimum is
    base_len, most mass sits just above it, and a power-law tail occasionally
    reaches much longer sequences, capped at max_len.
    """
    if dist == "fixed":
        return base_len
    max_len = max_len or 4 * base_len
    L = base_len * (1.0 + rng.pareto(alpha))
    return int(min(max(round(L), 1), max_len))


# --------------------------------------------------------------------------- #
# Muon: orthogonalised-momentum SGD for 2D weights, AdamW for everything else.
# --------------------------------------------------------------------------- #

@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps: int = 5, eps: float = 1e-7):
    """Approximate the orthogonal polar factor of G via a quintic Newton-Schulz
    iteration -- i.e. replace G = U S V^T by U V^T without an SVD.

    The coefficients are Jordan's: they do NOT converge to the exact polar
    factor, they converge to something whose singular values all sit in roughly
    [0.7, 1.3]. That is deliberate and is why 5 steps suffice -- the update
    direction only needs to be approximately orthogonal, and chasing exactness
    costs iterations for no gain.

    Runs in float32 here rather than the bfloat16 of the reference
    implementation: this harness trains on CPU/MPS, where bfloat16 matmul is
    either unimplemented or slower than fp32, and the matrices are tiny
    (at most 512x128) so the memory saving is irrelevant.
    """
    assert G.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = G.size(-2) > G.size(-1)
    if transposed:                      # iterate on the wider orientation
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon on param groups flagged `use_muon=True`, AdamW on the rest.

    One optimizer rather than two so that a single LR scheduler drives both --
    torch's schedulers walk `optimizer.param_groups`, so a two-optimizer split
    would need two schedulers kept in lockstep by hand.

    Muon updates only make sense for matrices: the update is the orthogonal
    factor of the momentum buffer, and orthogonalising a vector just returns a
    unit vector, discarding its magnitude entirely. So biases, norm gains,
    A_log and dt_bias go to AdamW -- see `split_muon_params`.
    """

    def __init__(self, param_groups, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, adamw_lr=3e-3, adamw_betas=(0.9, 0.95),
                 adamw_eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, weight_decay=weight_decay)
        for g in param_groups:
            g.setdefault("use_muon", False)
            g.setdefault("lr", lr if g["use_muon"] else adamw_lr)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            wd, lr = group["weight_decay"], group["lr"]
            if group["use_muon"]:
                mom, nesterov, ns = (group["momentum"], group["nesterov"],
                                     group["ns_steps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(g)
                    buf = st["momentum_buffer"]
                    buf.mul_(mom).add_(g)
                    d = g.add(buf, alpha=mom) if nesterov else buf
                    d = zeropower_via_newtonschulz5(d, steps=ns)
                    # Aspect-ratio correction: the orthogonal factor has unit
                    # singular values, so its Frobenius norm grows like
                    # sqrt(min(m, n)) and a tall matrix would otherwise take a
                    # systematically smaller step per output unit than a square
                    # one at the same lr.
                    scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.add_(d.reshape(p.shape).to(p.dtype), alpha=-lr * scale)
            else:
                b1, b2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if "step" not in st:
                        st["step"] = 0
                        st["exp_avg"] = torch.zeros_like(g)
                        st["exp_avg_sq"] = torch.zeros_like(g)
                    st["step"] += 1
                    t = st["step"]
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    mh = m / (1 - b1 ** t)
                    vh = v / (1 - b2 ** t)
                    if wd:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(mh, vh.sqrt().add_(eps), value=-lr)
        return loss


def split_muon_params(model, scope="hidden"):
    """(muon_params, adamw_params, names) for `scope`.

    "hidden" (default): every 2D parameter EXCEPT the token embedding and the
        final classifier. Both are 2D but neither is a hidden transform -- their
        rows are per-token, so a row's magnitude carries how strongly that token
        is embedded/predicted, and orthogonalising throws exactly that away. It
        is also the split every published Muon result uses; putting the
        embedding under Muon is the standard way to make it underperform AdamW.
    "all2d": literally every 2D parameter, embedding and classifier included --
        available so the claim above can be tested rather than assumed.
    """
    if scope not in ("hidden", "all2d"):
        raise ValueError(f"muon scope must be 'hidden' or 'all2d', got {scope!r}")
    excluded = set()
    if scope == "hidden":
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding):
                excluded.add(f"{name}.weight" if name else "weight")
        # the classifier: last Linear of the readout MLP
        lin = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        if lin:
            excluded.add(f"{lin[-1]}.weight")
    muon, adamw, names = [], [], {"muon": [], "adamw": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and name not in excluded:
            muon.append(p); names["muon"].append(name)
        else:
            adamw.append(p); names["adamw"].append(name)
    return muon, adamw, names


def build_optimizer(model, cfg):
    """AdamW over everything (default), or Muon on 2D weights + AdamW on the
    rest. Returns (optimizer, per_group_max_lr) -- the schedulers need one
    max_lr per group, since the two groups run at very different scales."""
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        return opt, cfg["lr"]
    muon_p, adamw_p, names = split_muon_params(model, cfg["muon_scope"])
    print(f"      muon: {len(muon_p)} 2D tensors "
          f"({sum(p.numel() for p in muon_p):,} params) | adamw: {len(adamw_p)} "
          f"tensors ({sum(p.numel() for p in adamw_p):,} params)", flush=True)
    groups = [dict(params=muon_p, use_muon=True, lr=cfg["muon_lr"],
                   momentum=cfg["muon_momentum"], nesterov=True,
                   ns_steps=cfg["muon_ns_steps"], weight_decay=cfg["weight_decay"]),
              dict(params=adamw_p, use_muon=False, lr=cfg["lr"],
                   weight_decay=cfg["weight_decay"])]
    groups = [g for g in groups if g["params"]]
    opt = Muon(groups, lr=cfg["muon_lr"], adamw_lr=cfg["lr"],
               momentum=cfg["muon_momentum"], ns_steps=cfg["muon_ns_steps"],
               weight_decay=cfg["weight_decay"])
    return opt, [g["lr"] for g in opt.param_groups]


def curriculum_len(it, steps, lens):
    """Length for step `it` of `steps` total: split training into len(lens)
    equal-width stages, each training at one length from `lens` in order
    (e.g. [2, 4, 8, 16] eases into the task before the target length)."""
    stage = min(it * len(lens) // steps, len(lens) - 1)
    return lens[stage]


# --------------------------------------------------------------------------- #
# model: a real ComplexKimiDeltaAttention block
# --------------------------------------------------------------------------- #

@torch.no_grad()
def layer_flip_rate(layer, e):
    """Fraction of (token, channel) pairs where the gate SIGN changes. If this
    collapses to ~0 on a signed arm, the range extension went unused."""
    z = rearrange(layer.f_proj(e), "... (h d) -> ... h d", d=layer.head_k_dim)
    s, _ = compute_gate(layer.gate, z, layer.A_log, layer.dt_bias, layer.lower_bound)
    if s is None:
        return 0.0                       # unsigned gate: no sign to flip
    return (s[:, 1:] != s[:, :-1]).float().mean().item()


@torch.no_grad()
def layer_gate_saturation(layer, e):
    """Fraction of gate pre-activations in the dead tail, where the activation's
    derivative underflows to exactly 0 (fp32: |u| = 19 for sigmoid2, 10 for
    tanh). A channel that saturates can never flip sign again."""
    z = rearrange(layer.f_proj(e), "... (h d) -> ... h d", d=layer.head_k_dim)
    u = layer.A_log.float().exp().view(1, 1, -1, 1) * (
        z.float() + layer.dt_bias.view(1, 1, -1, z.shape[-1]))
    thresh = 19.0 if layer.gate == "signed_sigmoid2" else 10.0
    return (u.abs() > thresh).float().mean().item()


class MaxoutReadout(nn.Module):
    """score_g(o) = max over M codewords of <c, o>, then argmax over classes.

    This is the decoder of thm:a5-four eq:maxout-decoder, not an approximation
    of it. For a hidden group that covers the visible group m-to-one, a class's
    outputs form a codebook of m distinct vectors, and that codebook is
    centrally symmetric -- so it has mean zero, and NO affine classifier can be
    correct: averaging a strictly-correct class inequality over the codebook
    demands that class g's bias exceed every other bias, for every g at once.

    Max pooling over the codewords is what breaks that. M should be the size of
    the hidden fibre: 120 for the binary icosahedral cover of A_5 (|2I| = 120,
    covering |A_5| = 60 two-to-one, times the 60 right-factor states).

    A hard max routes gradient to exactly ONE codeword per class per sample, so
    with 120 codewords a class the other 119 sit dead at their random init and
    training crawls. logsumexp is the smooth relaxation -- and is not merely a
    trick: it is the log-likelihood of the class after MARGINALISING over the
    hidden fibre, which is exactly what the decoder is supposed to do. It also
    tends to max as the scores separate, so the trained solution is the same.
    """

    def __init__(self, d_model, vocab, codewords, agg="logsumexp"):
        super().__init__()
        self.vocab, self.codewords, self.agg = vocab, codewords, agg
        self.proj = nn.Linear(d_model, vocab * codewords)

    def forward(self, x):
        s = self.proj(x).view(*x.shape[:-1], self.vocab, self.codewords)
        return s.amax(-1) if self.agg == "max" else s.logsumexp(-1)


class Model(nn.Module):
    def __init__(self, vocab, d_model, n_heads, head_dim, gate,
                 allow_neg_eigval, backend, gate_init_style="shipped",
                 beta_init_style="standard",
                 vocab_in=None, readout="mlp", maxout_codewords=120,
                 maxout_agg="logsumexp", readout_hidden=None, drop_silu=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_in or vocab, d_model)
        self.layer = ComplexKimiDeltaAttention(
            hidden_size=d_model, head_dim=head_dim, num_heads=n_heads,
            gate=gate, allow_neg_eigval=allow_neg_eigval, backend=backend,
            lower_bound=LB,
            use_short_conv=False,          # the task has no local structure to
                                           # convolve; keeps the gate isolated
            gate_init_style=gate_init_style, beta_init_style=beta_init_style,
            drop_silu=drop_silu,
            layer_idx=0)
        self.norm = nn.LayerNorm(d_model)
        if readout == "maxout":
            self.mlp = MaxoutReadout(d_model, vocab, maxout_codewords, maxout_agg)
        elif readout == "mlp":
            # Width must scale with the number of REACHABLE STATES, not with
            # d_model: the readout has to memorise a partition of them.
            hid = readout_hidden or 4 * d_model
            self.mlp = nn.Sequential(nn.Linear(d_model, hid), nn.GELU(),
                                     nn.Linear(hid, vocab))
        else:
            raise ValueError(f"readout must be 'mlp' or 'maxout', got {readout!r}")

    def forward(self, x):
        e = self.emb(x)
        h, _, _ = self.layer(e)
        return self.mlp(self.norm(e + h))

    @torch.no_grad()
    def gate_stats(self, x):
        """Sign statistics straight off the layer's own gate."""
        e = self.emb(x)
        z = rearrange(self.layer.f_proj(e), "... (h d) -> ... h d",
                      d=self.layer.head_k_dim)
        s, g = compute_gate(self.layer.gate, z, self.layer.A_log,
                            self.layer.dt_bias, self.layer.lower_bound)
        alpha = (s.float() if s is not None else 1.0) * g.exp()
        beta_raw = self.layer.b_proj(e)
        beta = torch.sigmoid(beta_raw.float()) * (2.0 if self.layer.allow_neg_eigval else 1.0)
        neg = alpha < 0                   # [B, T, HV, K]
        householder_neg = beta > 1        # [B, T, HV]  -- (1 - beta) < 0
        both = neg & householder_neg.unsqueeze(-1)   # broadcast over K
        # per-head breakdown: is the "both negative" mechanism concentrated in
        # a few heads (one head suffices to solve the task) or spread evenly?
        both_per_head = both.float().mean(dim=(0, 1, 3))   # [HV]
        return dict(frac_alpha_neg=round(neg.float().mean().item(), 4),
                    flip_rate=round(layer_flip_rate(self.layer, e), 4),
                    saturated=round(layer_gate_saturation(self.layer, e), 4),
                    alpha_min=round(alpha.min().item(), 4),
                    alpha_max=round(alpha.max().item(), 4),
                    frac_beta_gt1=round(householder_neg.float().mean().item(), 4),
                    frac_both_neg=round(both.float().mean().item(), 4),
                    frac_both_neg_per_head=[round(v, 4) for v in both_per_head.tolist()],
                    beta_min=round(beta.min().item(), 4),
                    beta_max=round(beta.max().item(), 4))


# --------------------------------------------------------------------------- #
# model: the ORIGINAL hand-rolled recurrence (group_word_problems/complex_kda.py
# / group_wordproblem.py's GatedDeltaLayer), ported to accept the SAME
# --gates / --no-neg-eigval vocabulary as ComplexKimiDeltaAttention so it can
# be swapped in via --layer old with everything else (data gen, --restrict-gens,
# --eval-curve, ...) held identical -- isolating whether it's the from-scratch
# recurrence itself (no short conv, no output forget-gate, k_dim==v_dim) that
# explains the gap, separate from data, gate formula, or init.
# --------------------------------------------------------------------------- #

class OldGatedDeltaLayer(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, gate, allow_neg_eigval=True,
                 gate_init_style="spread", eps_floor=None, qk_scale=False,
                 gauge=False, log_space=False, fproj=False, qkv_silu=False,
                 fproj_rank=None, beta_nobias=False, per_head_norm=False):
        super().__init__()
        if gate not in ("sigmoid", "signed_sigmoid2", "signed_tanh"):
            raise ValueError(
                f"--layer old only supports gate in "
                f"('sigmoid', 'signed_sigmoid2', 'signed_tanh'), got {gate!r} -- "
                f"the old prototype has no softplus-equivalent formula")
        self.h, self.n = n_heads, head_dim
        self.gate, self.allow_neg_eigval = gate, allow_neg_eigval
        self.eps_floor = eps_floor        # ablation: replicate signed_gate's
                                          # sign(a)*(eps+(1-eps)*|a|) jump-at-0
                                          # discontinuity instead of the smooth
                                          # 2*sigmoid(a)-1 / tanh(a) pass-through
        self.qk_scale = head_dim ** -0.5 if qk_scale else 1.0
        # ablation: naive_recurrent_kda scales q by head_dim**-0.5 before the
        # readout (q^T S); this layer never did. Test if that alone matters.
        self.gauge = gauge
        # ablation: run the recurrence with |alpha| only and carry the +-1 part
        # as a running parity P_t pushed onto q/k -- exactly what
        # ComplexKimiDeltaAttention does (signed_gate + running_sign +
        # apply_sign). Provably output-identical to applying the signed alpha
        # directly (see _recurrence docstring), so any training difference is
        # attributable to the mechanism, not the function being computed.
        self.log_space = log_space
        # ablation: carry the gate as log|alpha| and exponentiate it back
        # inside the recurrence, which is what the kda path does --
        # signed_gate returns g = log(eps + (1-eps)|a|) and
        # naive_recurrent_kda applies `S = S * g.exp()`. Mathematically an
        # identity round trip, but a different gradient path (d log/d|a| =
        # 1/|a| blows up near the eps floor, cancelled by the exp on the way
        # back) and different float behaviour. Requires an eps floor, since
        # log(0) = -inf; defaults to exp(LB), matching the kda lower_bound.
        inner = n_heads * head_dim
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.fproj = fproj
        self.qkv_silu = qkv_silu
        # ablation: with use_short_conv=False the kda layer applies F.silu to
        # q/k/v (it stands in for the conv's activation). silu is bounded
        # below at ~-0.278, so it squashes the negative half-line and pushes
        # q/k/v toward the non-negative orthant. k is the DIRECTION of the
        # rank-1 update (I - beta k k^T), so this restricts the reachable
        # update directions -- plausibly fine for a 6-element group, not for
        # a 24-element one. The old layer applies no nonlinearity here.
        self.per_head_norm = per_head_norm
        # ablation: kda normalises the recurrence output per head (over V);
        # this layer L2-normalises across all heads jointly.
        # ablation: produce the gate logits the way ComplexKimiDeltaAttention
        # does -- a RANK-head_dim factored map (two Linears, no nonlinearity
        # between them) plus a separate per-channel dt_bias and a per-head
        # exp(A_log) rescale -- instead of one full-rank Linear+bias. This is
        # the last structural difference in the gate path: rank min(d_model,
        # head_dim) vs rank min(d_model, inner).
        if fproj:
            rank = fproj_rank or head_dim      # kda's bottleneck is head_v_dim
            self.f_proj = nn.Sequential(
                nn.Linear(d_model, rank, bias=False),
                nn.Linear(rank, inner, bias=False),
            )
            self.A_log = nn.Parameter(torch.zeros(n_heads, dtype=torch.float32))
            self.dt_bias = nn.Parameter(torch.zeros(inner, dtype=torch.float32))
        self.a = nn.Linear(d_model, inner)                         # decay logits
        self.b = nn.Linear(d_model, n_heads, bias=not beta_nobias)  # rate logits
        self.o = nn.Linear(inner, d_model, bias=False)
        with torch.no_grad():
            self.a.weight.mul_(0.1)
            if gate == "sigmoid":
                self.a.bias.fill_(2.0)                              # alpha ~ 0.88
            elif gate_init_style == "spread":
                # mixed signs at the same |bias| as the all-positive branch
                # below, so the two styles differ only in sign.
                self.a.bias.uniform_(1.0, 2.0).mul_(
                    torch.where(torch.rand(inner) < 0.5, -1.0, 1.0))
            else:
                self.a.bias.uniform_(1.0, 2.0)                      # all alpha > 0
            if self.b.bias is not None:
                self.b.bias.fill_(0.5)
            if fproj:                       # same init, f_proj parameterisation
                self.f_proj[-1].weight.mul_(0.1)
                if gate == "sigmoid":
                    self.dt_bias.fill_(2.0)
                elif gate_init_style == "spread":
                    self.dt_bias.uniform_(1.0, 2.0).mul_(
                        torch.where(torch.rand(inner) < 0.5, -1.0, 1.0))
                else:
                    self.dt_bias.uniform_(1.0, 2.0)

    def gates(self, x):
        B, T, _ = x.shape
        if self.fproj:
            # u = exp(A_log) * (f_proj(x) + dt_bias), exactly signed_gate's
            # pre-activation, vs the plain Linear+bias below.
            z = self.f_proj(x).view(B, T, self.h, self.n)
            a = z + self.dt_bias.view(1, 1, self.h, self.n)
            a = self.A_log.exp().view(1, 1, self.h, 1) * a
        else:
            a = self.a(x).view(B, T, self.h, self.n)
        b = self.b(x).view(B, T, self.h)
        beta = (2.0 if self.allow_neg_eigval else 1.0) * torch.sigmoid(b)
        if self.gate == "sigmoid":
            return torch.sigmoid(a), beta
        alpha = torch.tanh(0.5 * a) if self.gate == "signed_sigmoid2" else torch.tanh(a)
        if self.eps_floor is not None:
            eps = self.eps_floor
            alpha = torch.sign(alpha.detach()) * (eps + (1.0 - eps) * alpha.abs())
        return alpha, beta

    def forward(self, x, collect=False):
        B, T, _ = x.shape
        sh = (B, T, self.h, self.n)
        if self.qkv_silu:                  # kda's use_short_conv=False path
            qp, kp, vp = (F.silu(p(x)) for p in (self.q, self.k, self.v))
        else:
            qp, kp, vp = self.q(x), self.k(x), self.v(x)
        q = F.normalize(qp.view(sh), dim=-1) * self.qk_scale
        k = F.normalize(kp.view(sh), dim=-1)
        v = vp.view(sh)
        alpha, beta = self.gates(x)

        # alpha_used is what Diag(.) sees inside the loop. Under the gauge it
        # is |alpha|, with the sign carried on q/k instead.
        #
        #   alpha_t = s_t |alpha_t|,   P_t = prod_{u<=t} s_u
        #   Diag(alpha_t) H_{t-1} = Diag(P_t) Diag(|alpha_t|) Ht~_{t-1}
        #   => H_t = Diag(P_t) Ht~_t  and  o_t = q_t^T H_t = (P_t q_t)^T Ht~_t
        #
        # so gauging BOTH q and k by P_t and running with |alpha| is exactly
        # output-identical. Sign is detached and the parity scan is integer
        # (no autograd graph), matching signed_gate/running_sign; sign is
        # piecewise constant so this is still the exact a.e. gradient.
        # P commutes with the l2-norm above (it is +-1 per channel), so
        # applying it after normalize is equivalent to before.
        if self.log_space:
            # exactly signed_gate's return: s, g = log(eps + (1-eps)|a|),
            # then the recurrence exponentiates g back (naive_recurrent_kda's
            # `S = S * g.exp()`). NOTE gates() has already applied the floor
            # when eps_floor is set, so only floor here if it has not -- else
            # the floor lands twice and this stops being the same function.
            if self.eps_floor is None:
                eps = math.exp(LB)
                mag_in = eps + (1.0 - eps) * alpha.abs()
            else:
                mag_in = alpha.abs()
            g = mag_in.log()
            mag = g.exp()
        else:
            mag = alpha.abs()

        if self.gauge:
            s = torch.where(alpha.detach() < 0, -1, 1)
            par = (s < 0).to(torch.int32).cumsum(dim=1)
            P = torch.where(par & 1 == 1, -1.0, 1.0).to(alpha.dtype)
            q, k = q * P, k * P
            alpha_used = mag
        elif self.log_space:
            alpha_used = torch.where(alpha.detach() < 0, -1.0, 1.0) * mag
        else:
            alpha_used = alpha

        H = x.new_zeros(B, self.h, self.n, self.n)
        outs = []
        for t in range(T):
            at, kt, vt, bt = alpha_used[:, t], k[:, t], v[:, t], beta[:, t]
            S = at.unsqueeze(-1) * H                                # Diag(alpha) H
            kS = torch.einsum("bhn,bhnd->bhd", kt, S)               # k^T S
            w = bt.unsqueeze(-1) * (kS - vt)
            H = S - kt.unsqueeze(-1) * w.unsqueeze(-2)              # delta update
            outs.append(torch.einsum("bhn,bhnd->bhd", q[:, t], H))
        o = torch.stack(outs, 1)                                   # [B, T, h, n]
        if self.per_head_norm:                 # kda: normalise each head's V
            o = F.normalize(o, dim=-1).reshape(B, T, self.h * self.n)
        else:                                  # old: one norm across all heads
            o = F.normalize(o.reshape(B, T, self.h * self.n), dim=-1)
        out = self.o(o)
        # collect returns the TRUE signed alpha either way, so gate_stats'
        # frac_alpha_neg / flip_rate stay comparable across the two modes.
        return (out, alpha.detach(), k.detach(), beta.detach()) if collect else out


class OldModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, head_dim, gate,
                 allow_neg_eigval, gate_init_style="spread", eps_floor=None,
                 qk_scale=False, gauge=False, log_space=False, fproj=False,
                 qkv_silu=False, fproj_rank=None, beta_nobias=False,
                 per_head_norm=False, vocab_in=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_in or vocab, d_model)
        self.layer = OldGatedDeltaLayer(d_model, n_heads, head_dim, gate,
                                        allow_neg_eigval, gate_init_style, eps_floor,
                                        qk_scale, gauge, log_space, fproj,
                                        qkv_silu, fproj_rank, beta_nobias,
                                        per_head_norm)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, vocab))

    def forward(self, x):
        e = self.emb(x)
        return self.mlp(self.norm(e + self.layer(e)))

    @torch.no_grad()
    def gate_stats(self, x):
        e = self.emb(x)
        _, alpha, _, beta = self.layer(e, collect=True)
        neg = alpha < 0                   # [B, T, h, n]
        flip = ((neg[:, 1:] != neg[:, :-1]).float().mean().item()
                if alpha.shape[1] > 1 else 0.0)
        householder_neg = beta > 1        # [B, T, h]  -- (1 - beta) < 0
        both = neg & householder_neg.unsqueeze(-1)   # broadcast over head_dim
        # per-head breakdown: is the "both negative" mechanism concentrated in
        # a few heads (one head suffices to solve the task) or spread evenly?
        both_per_head = both.float().mean(dim=(0, 1, 3))   # [h]
        return dict(frac_alpha_neg=round(neg.float().mean().item(), 4),
                    flip_rate=round(flip, 4), saturated=0.0,        # not tracked
                    alpha_min=round(alpha.min().item(), 4),
                    alpha_max=round(alpha.max().item(), 4),
                    frac_beta_gt1=round(householder_neg.float().mean().item(), 4),
                    frac_both_neg=round(both.float().mean().item(), 4),
                    frac_both_neg_per_head=[round(v, 4) for v in both_per_head.tolist()],
                    beta_min=round(beta.min().item(), 4),
                    beta_max=round(beta.max().item(), 4))


# --------------------------------------------------------------------------- #
# model: GatedDeltaNet / GatedDeltaProduct (--layer gateddeltanet /
# deltaproduct). Neither has a signed-gate option -- the forget gate is
# always g = -A_log.exp()*softplus(a_proj(x)+dt_bias) <= 0, so alpha=exp(g)
# never goes negative; only beta (Householder) can, via allow_neg_eigval.
# Runs through the CPU/no-triton fallback added to fla.layers.gated_deltanet
# / gated_deltaproduct (naive_recurrent_gated_delta_rule / _product loops).
# --------------------------------------------------------------------------- #

class GDModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, head_dim, allow_neg_eigval,
                 layer_cls, vocab_in=None, **layer_kwargs):
        super().__init__()
        self.emb = nn.Embedding(vocab_in or vocab, d_model)
        self.layer = layer_cls(
            hidden_size=d_model, head_dim=head_dim, num_heads=n_heads,
            allow_neg_eigval=allow_neg_eigval, use_short_conv=False,
            layer_idx=0, **layer_kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, vocab))

    def forward(self, x):
        e = self.emb(x)
        h, _, _ = self.layer(e)
        return self.mlp(self.norm(e + h))

    @torch.no_grad()
    def gate_stats(self, x):
        e = self.emb(x)
        g = naive_gdn_gate(self.layer.a_proj(e), self.layer.A_log, self.layer.dt_bias)
        alpha = g.exp()                   # [..., h] -- always in (0, 1], never negative
        beta_raw = self.layer.b_proj(e)   # [..., h * num_householder]
        beta = torch.sigmoid(beta_raw.float()) * (2.0 if self.layer.allow_neg_eigval else 1.0)
        num_householder = getattr(self.layer, "num_householder", 1)
        if num_householder > 1:
            # decay is applied once per step (alpha, one value per head); beta
            # carries one value per Householder reflection composed within
            # that step -- broadcast alpha over the extra Householder axis.
            beta = rearrange(beta, "... (h n) -> ... h n", n=num_householder)
            neg = (alpha < 0).unsqueeze(-1)
        else:
            neg = alpha < 0                # trivially all False
        householder_neg = beta > 1
        both = neg & householder_neg
        # per-head breakdown: average over every dim except the head axis
        # (last axis normally; second-to-last when a Householder axis follows it)
        head_dim_idx = both.ndim - 2 if num_householder > 1 else both.ndim - 1
        reduce_dims = tuple(d for d in range(both.ndim) if d != head_dim_idx)
        both_per_head = both.float().mean(dim=reduce_dims)
        return dict(frac_alpha_neg=round(neg.float().mean().item(), 4),
                    flip_rate=0.0, saturated=0.0,
                    alpha_min=round(alpha.min().item(), 4),
                    alpha_max=round(alpha.max().item(), 4),
                    frac_beta_gt1=round(householder_neg.float().mean().item(), 4),
                    frac_both_neg=round(both.float().mean().item(), 4),
                    frac_both_neg_per_head=[round(v, 4) for v in both_per_head.tolist()],
                    beta_min=round(beta.min().item(), 4),
                    beta_max=round(beta.max().item(), 4))


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(model, table, ident, lengths, device, seed, batch=256, gen_idx=None, bos_id=None):
    """Accuracy on the final quarter: the part that needs carried state."""
    model.eval()
    out = {}
    for L in lengths:
        x, y = make_batch(table, ident, batch, L, np.random.default_rng(seed), device, gen_idx, bos_id)
        p = model(x).argmax(-1)
        q = max(L // 4, 1)
        out[L] = (p[:, -q:] == y[:, -q:]).float().mean().item()
    model.train()
    return out


@torch.no_grad()
def evaluate_continuous(model, table, ident, max_len, device, seed, batch=256, gen_idx=None, bos_id=None):
    """Accuracy at EVERY length 1..max_len, from ONE forward pass.

    The model is strictly causal (no positional embedding, no length-dependent
    branch on the naive backend -- verified: running the full max_len sequence
    and slicing prefix [:, :L] agrees with an independent forward pass on
    x[:, :L] alone to float32 rounding, ~2e-7). So position L-1's output from a
    single max_len forward equals what a standalone length-L call would give,
    and "final quarter of length L" can be read off by slicing one pass
    instead of running max_len separate batches.

    With bos_id set, column 0 of every batch is the BOS prefix (not a real
    generator token), so it's dropped before the length-indexed windowing --
    after that, column i (0-based) again means "after i+1 real tokens", same
    as the no-BOS case.
    """
    model.eval()
    x, y = make_batch(table, ident, batch, max_len, np.random.default_rng(seed), device, gen_idx, bos_id)
    p = model(x).argmax(-1)
    correct = (p == y)
    if bos_id is not None:
        correct = correct[:, 1:]
    out = {}
    for L in range(1, max_len + 1):
        q = max(L // 4, 1)
        out[L] = correct[:, L - q:L].float().mean().item()
    model.train()
    return out


@torch.no_grad()
def diagnostics(model, table, ident, device, seed, batch=128, L=32, gen_idx=None, bos_id=None):
    x, _ = make_batch(table, ident, batch, L, np.random.default_rng(seed), device, gen_idx, bos_id)
    return model.gate_stats(x)


def run(task, gate, cfg, seed, device, backend):
    torch.manual_seed(seed)
    els, table, ident = perm_group(task)
    V = len(els)
    bos_id = V           # one id past the group elements, added to the input vocab only
    stage_gens = None
    if cfg["restrict_gens"] == "transpositions":
        gen_idx = transpositions(els)
    elif cfg["restrict_gens"] == "subgroup_curriculum":
        stage_gens = subgroup_chain(table, ident)
        gen_idx = stage_gens[-1]
    elif cfg["restrict_gens"] == "cycle_curriculum":
        stage_gens = cycle_curriculum_gens(els, cfg["cycle_curriculum_lens"])
        gen_idx = stage_gens[-1]        # eval/diagnostics: the terminal (hardest) stage
    else:
        gen_idx = None
    if cfg["layer"] == "old":
        model = OldModel(V, cfg["d_model"], cfg["heads"], cfg["head_dim"], gate,
                         cfg["allow_neg_eigval"], cfg["gate_init_style"],
                         cfg["old_eps_floor"], cfg["old_qk_scale"],
                         cfg["old_gauge"], cfg["old_log_space"],
                         cfg["old_fproj"], cfg["old_qkv_silu"],
                         cfg["old_fproj_rank"], cfg["old_beta_nobias"],
                         cfg["old_per_head_norm"], vocab_in=V + 1).to(device)
    elif cfg["layer"] == "gateddeltanet":
        model = GDModel(V, cfg["d_model"], cfg["heads"], cfg["head_dim"],
                        cfg["allow_neg_eigval"], GatedDeltaNet,
                        vocab_in=V + 1).to(device)
    elif cfg["layer"] == "deltaproduct":
        model = GDModel(V, cfg["d_model"], cfg["heads"], cfg["head_dim"],
                        cfg["allow_neg_eigval"], GatedDeltaProduct,
                        vocab_in=V + 1,
                        num_householder=cfg["dp_num_householder"],
                        no_conv_silu=cfg["dp_no_conv_silu"]).to(device)
    else:
        model = Model(V, cfg["d_model"], cfg["heads"], cfg["head_dim"], gate,
                      cfg["allow_neg_eigval"], backend,
                      cfg["gate_init_style"],
                      cfg["beta_init_style"],
                      vocab_in=V + 1,
                      readout=cfg["readout"],
                      maxout_codewords=cfg["maxout_codewords"],
                      maxout_agg=cfg["maxout_agg"],
                      readout_hidden=cfg["readout_hidden"],
                      drop_silu=cfg["drop_silu"]).to(device)
        if cfg["freeze_a_log"]:
            model.layer.A_log.requires_grad_(False)   # ablation: A_log has no
                                                       # equivalent in --layer old
    if cfg["a5_theory_init"]:
        if task != "a5" or cfg["layer"] != "kda":
            raise ValueError("--a5-theory-init is only defined for --task a5 --layer kda")
        initialize_a5_theory_heads(
            model,
            table,
            ident,
            heads=cfg["a5_theory_heads"],
            saturation=cfg["a5_theory_saturation"],
        )
    opt, max_lr = build_optimizer(model, cfg)
    if cfg["lr_schedule"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["steps"])
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr,
                                                    total_steps=cfg["steps"], pct_start=0.1)
    rng = np.random.default_rng(seed + 1234)
    T, t0 = cfg["train_len"], time.time()          # base/min length for training
    E = cfg["eval_len"] or 4 * T                   # top eval length; decoupled from T
    adaptive_curriculum = cfg["len_dist"] == "performance_curriculum"
    curriculum_stage = 0
    curriculum_streak = 0
    curriculum_history = []
    training_log = []
    for it in range(cfg["steps"]):
        if cfg["len_dist"] == "curriculum":
            Lt = curriculum_len(it, cfg["steps"], cfg["curriculum_lens"])
        elif adaptive_curriculum:
            Lt = cfg["curriculum_lens"][curriculum_stage]
        else:
            Lt = sample_train_len(T, rng, cfg["len_dist"], cfg["pareto_alpha"],
                                  cfg["max_train_len"])
        if stage_gens is not None:
            n_stages = len(stage_gens)
            stage = min(it * n_stages // cfg["steps"], n_stages - 1)
            step_gen_idx = stage_gens[stage]
        else:
            step_gen_idx = gen_idx
        x, y = make_batch(table, ident, cfg["batch"], Lt, rng, device, step_gen_idx, bos_id)
        logits = model(x)
        loss = F.cross_entropy(logits[:, 1:].reshape(-1, V), y[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        stage_acc = None
        if adaptive_curriculum and (it + 1) % cfg["curriculum_check_every"] == 0:
            stage_acc = evaluate(
                model,
                table,
                ident,
                [Lt],
                device,
                999,
                cfg["curriculum_eval_batch"],
                step_gen_idx,
                bos_id,
            )[Lt]
            if stage_acc >= cfg["curriculum_threshold"]:
                curriculum_streak += 1
            else:
                curriculum_streak = 0
            if (curriculum_streak >= cfg["curriculum_passes"]
                    and curriculum_stage < len(cfg["curriculum_lens"]) - 1):
                next_length = cfg["curriculum_lens"][curriculum_stage + 1]
                curriculum_history.append({
                    "length": Lt,
                    "step": it + 1,
                    "accuracy": stage_acc,
                })
                curriculum_stage += 1
                curriculum_streak = 0
                print(f"      curriculum: L{Lt} mastered at step {it + 1}; "
                      f"advancing to L{next_length}", flush=True)
        if cfg["log_every"] and (it + 1) % cfg["log_every"] == 0:
            log_length = Lt if adaptive_curriculum else T
            if stage_acc is None:
                stage_acc = evaluate(
                    model, table, ident, [log_length], device, 999, 128,
                    step_gen_idx, bos_id,
                )[log_length]
            print(f"      {it+1:5d}  loss {loss.item():.4f}  "
                  f"acc@{log_length} {stage_acc:.3f}", flush=True)
            training_log.append({
                "step": it + 1,
                "length": Lt,
                "training_loss": loss.item(),
                "evaluation_length": log_length,
                "evaluation_accuracy": stage_acc,
                "learning_rates": [group["lr"] for group in opt.param_groups],
            })
    if cfg["eval_continuous"]:
        acc = evaluate_continuous(model, table, ident, E, device, 999,
                                  batch=cfg["eval_batch"], gen_idx=gen_idx, bos_id=bos_id)
    else:
        eval_lens = cfg["eval_curve_lens"] if cfg["eval_curve"] else [E // 4, E // 2, E]
        acc = evaluate(model, table, ident, eval_lens, device, 999,
                       batch=cfg["eval_batch"], gen_idx=gen_idx, bos_id=bos_id)
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    ckpt_path = os.path.join(
        cfg["ckpt_dir"],
        f"{task}-{gate}-{cfg['layer']}-h{cfg['heads']}-hd{cfg['head_dim']}-"
        f"b{cfg['batch']}-s{seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return dict(task=task, gate=gate, signed=is_signed(gate), seed=seed,
                layer=cfg["layer"],
                backend=getattr(model.layer, "backend", cfg["layer"]),
                allow_neg_eigval=cfg["allow_neg_eigval"],
                train_len=T, eval_len=E, len_dist=cfg["len_dist"],
                pareto_alpha=cfg["pareto_alpha"],
                curriculum_lens=cfg["curriculum_lens"],
                curriculum_threshold=cfg["curriculum_threshold"],
                curriculum_check_every=cfg["curriculum_check_every"],
                curriculum_passes=cfg["curriculum_passes"],
                curriculum_final_length=(cfg["curriculum_lens"][curriculum_stage]
                                         if adaptive_curriculum else None),
                curriculum_history=curriculum_history,
                restrict_gens=cfg["restrict_gens"],
                cycle_curriculum_lens=cfg["cycle_curriculum_lens"],
                gate_init_style=cfg["gate_init_style"],
                beta_init_style=cfg["beta_init_style"], drop_silu=cfg["drop_silu"],
                a5_theory_init=cfg["a5_theory_init"],
                a5_theory_heads=cfg["a5_theory_heads"],
                a5_theory_saturation=cfg["a5_theory_saturation"],
                readout=cfg["readout"], readout_hidden=cfg["readout_hidden"],
                heads=cfg["heads"], head_dim=cfg["head_dim"], batch=cfg["batch"],
                eval_batch=cfg["eval_batch"], lr=cfg["lr"], steps=cfg["steps"], d_model=cfg["d_model"],
                optimizer=cfg["optimizer"],
                **({} if cfg["optimizer"] == "adamw" else
                   dict(muon_lr=cfg["muon_lr"], muon_scope=cfg["muon_scope"],
                        muon_momentum=cfg["muon_momentum"], muon_ns_steps=cfg["muon_ns_steps"])),
                lr_schedule=cfg["lr_schedule"], weight_decay=cfg["weight_decay"],
                trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                total_parameters=sum(parameter.numel() for parameter in model.parameters()),
                chance=1.0 / V, loss=loss.item(),
                secs=time.time() - t0,
                training_log=training_log,
                acc={str(L): v for L, v in acc.items()},
                ckpt=ckpt_path,
                **diagnostics(model, table, ident, device, 999, gen_idx=gen_idx, bos_id=bos_id))


def summarise(rows):
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["gate"]), []).append(r)
    tasks = sorted({t for t, _ in by}, key=TASKS.index)
    for t in tasks:
        rs0 = next(v for (tt, _), v in by.items() if tt == t)
        E = rs0[0].get("eval_len") or 4 * rs0[0]["train_len"]
        L1, L4 = str(E // 4), str(E)
        print(f"\n=== {t}   chance {rs0[0]['chance']:.3f}   "
              f"allow_neg_eigval={rs0[0]['allow_neg_eigval']}   "
              f"backend={rs0[0]['backend']}")
        print(f"{'gate':<18}{'signed':>7}{'acc@' + L1:>14}{'acc@' + L4:>14}"
              f"{'frac a<0':>10}{'flip rate':>11}")
        print("-" * 74)
        for g in GATES:
            rs = by.get((t, g))
            if not rs:
                continue
            f = lambda key: (np.mean([r["acc"][key] for r in rs]),
                             np.std([r["acc"][key] for r in rs]))
            m1, s1 = f(L1)
            m4, s4 = f(L4)
            print(f"{g:<18}{str(rs[0]['signed']):>7}{m1:>9.3f}+-{s1:<4.2f}"
                  f"{m4:>9.3f}+-{s4:<4.2f}"
                  f"{np.mean([r['frac_alpha_neg'] for r in rs]):>10.2f}"
                  f"{np.mean([r['flip_rate'] for r in rs]):>11.2f}")
    print("\nflip rate: fraction of (token, channel) pairs where the gate sign "
          "changes.\n  If it is ~0 on a signed arm the range extension went unused "
          "and the arm\n  is really an unsigned one wearing a different name.")


# --------------------------------------------------------------------------- #

def pick_device(p):
    if p != "auto":
        return p
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", metavar="FILE", help="render a table from a .jsonl")
    ap.add_argument("--task", default="s3", choices=TASKS)
    ap.add_argument("--all", action="store_true", help="s2, z5, s3, s4")
    ap.add_argument("--gates", nargs="*", default=list(GATES), choices=list(GATES))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed to run; --seeds controls how many consecutive seeds")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--train-len", type=int, default=16)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=2048,
                    help="number of sequences in the eval batch (evaluate / "
                         "evaluate_continuous)")
    ap.add_argument("--ckpt-dir", default="checkpoints",
                    help="every run's trained model.state_dict() is saved here "
                         "as {task}-{gate}-{layer}-h{heads}-hd{head_dim}-"
                         "b{batch}-s{seed}.pt; path is recorded in the output "
                         "row's 'ckpt' field")
    ap.add_argument("--len-dist", default="fixed",
                    choices=["fixed", "pareto", "curriculum",
                             "performance_curriculum"],
                    help="fixed: every step trains at --train-len (default). "
                         "pareto: each step's batch samples one length from a "
                         "power-law with minimum --train-len, most mass just "
                         "above it, capped at --max-train-len. curriculum: "
                         "splits training into equal stages over "
                         "--curriculum-lens. performance_curriculum: advances "
                         "only after validation accuracy at the current length "
                         "passes --curriculum-threshold.")
    ap.add_argument("--pareto-alpha", type=float, default=2.0,
                    help="pareto shape; smaller = heavier tail")
    ap.add_argument("--max-train-len", type=int, default=None,
                    help="cap for --len-dist pareto; default 4x --train-len")
    ap.add_argument("--curriculum-lens", default="6,8,16",
                    help="comma-separated lengths for --len-dist curriculum, "
                         "trained in order over equal-width stages")
    ap.add_argument("--curriculum-threshold", type=float, default=0.99,
                    help="accuracy required to advance a performance curriculum")
    ap.add_argument("--curriculum-check-every", type=int, default=250,
                    help="steps between performance-curriculum evaluations")
    ap.add_argument("--curriculum-passes", type=int, default=2,
                    help="consecutive passing evaluations required to advance")
    ap.add_argument("--curriculum-eval-batch", type=int, default=256,
                    help="validation batch size used to advance the curriculum")
    ap.add_argument("--eval-len", type=int, default=None,
                    help="top eval length (eval always uses [E/4, E/2, E]); "
                         "default 4x --train-len. Decoupled from --train-len "
                         "so a short pareto floor can still be tested at a "
                         "fixed, larger eval length, e.g. --train-len 4 "
                         "--eval-len 64")
    ap.add_argument("--eval-continuous", action="store_true",
                    help="evaluate at EVERY integer length 1..--eval-len from "
                         "a single forward pass per batch, instead of a sparse "
                         "grid of separate forward passes. Overrides "
                         "--eval-curve.")
    ap.add_argument("--eval-curve", action="store_true",
                    help="evaluate at --eval-curve-lens instead of the usual "
                         "[E/4, E/2, E] three points, for plotting a full "
                         "accuracy-vs-length curve")
    ap.add_argument("--eval-curve-lens",
                    default="1,2,4,8,16,32,64,128,256",
                    help="comma-separated lengths for --eval-curve")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                    help="adamw (default): one AdamW over every parameter. "
                         "muon: orthogonalised-momentum SGD on the 2D weight "
                         "matrices, AdamW on everything else (biases, norm "
                         "gains, A_log, dt_bias, and -- under the default "
                         "--muon-scope hidden -- the embedding and classifier).")
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="lr for the Muon group. Muon's update is orthogonal, "
                         "so its scale is set by the lr alone rather than by "
                         "the gradient magnitude; it wants a much larger lr "
                         "than AdamW. --lr still drives the AdamW group.")
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--muon-ns-steps", type=int, default=5,
                    help="Newton-Schulz iterations per step.")
    ap.add_argument("--muon-scope", default="hidden", choices=["hidden", "all2d"],
                    help="hidden (default): 2D weights excluding the embedding "
                         "and the final classifier. all2d: every 2D weight.")
    ap.add_argument("--lr-schedule", default="onecycle", choices=["onecycle", "cosine"],
                    help="onecycle (default): warms up then decays, peaking at "
                         "--lr. cosine: starts at --lr, no warmup, cosine decay "
                         "to 0 over --steps.")
    ap.add_argument("--restrict-gens", default="none",
                    choices=["none", "transpositions", "cycle_curriculum",
                             "subgroup_curriculum"],
                    help="none (default): input tokens are drawn from the "
                         "whole group. transpositions: input tokens are "
                         "restricted to the transpositions (swaps) only. "
                         "cycle_curriculum: splits training into equal "
                         "stages over --cycle-curriculum-lens, cumulatively "
                         "adding single-cycle generators of each length in "
                         "turn (e.g. 2,3,4 = swaps, then +3-cycles, then "
                         "+4-cycles). In all cases the OUTPUT/label space is "
                         "still the full group; only the generators fed to "
                         "the model are restricted.")
    ap.add_argument("--cycle-curriculum-lens", default="2,3,4",
                    help="comma-separated cycle lengths for --restrict-gens "
                         "cycle_curriculum, trained cumulatively in order "
                         "over equal-width stages")
    ap.add_argument("--gate-init-style", default="shipped",
                    choices=["shipped", "spread"],
                    help="only affects signed gates. shipped (default): all "
                         "channels start with alpha > 0, and training must "
                         "discover the negative ones itself. spread: half the "
                         "channels start negative, at the same |alpha| ~ 0.99, "
                         "so the sign population costs no memory horizon. NOTE "
                         "spread was redefined -- it previously drew the bias "
                         "from Uniform(-1.5, 1.5), giving |alpha| ~ 0.36; runs "
                         "in s3_paper/ and s4_paper/ predate the change.")
    ap.add_argument("--beta-init-style", default="standard",
                    choices=["standard", "spread"],
                    help="only affects --layer kda. standard (default): keep "
                         "the bias-free beta projection. spread: initialize "
                         "half the heads around beta=0.5 and half around "
                         "beta=1.5; requires negative eigenvalues to be enabled")
    ap.add_argument("--a5-theory-init", action="store_true",
                    help="replace the A5 token embeddings by exact four-dimensional "
                         "coordinates and initialize a small number of heads with "
                         "the corresponding transitions; all remaining weights "
                         "retain their random initialization and every parameter "
                         "remains trainable")
    ap.add_argument("--a5-theory-heads", type=int, default=2,
                    help="number of heads initialized by --a5-theory-init")
    ap.add_argument("--a5-theory-saturation", type=float, default=12.0,
                    help="finite gate and beta logits for --a5-theory-init")
    ap.add_argument("--layer", default="kda",
                    choices=["kda", "old", "deltaproduct", "gateddeltanet"],
                    help="kda (default): ComplexKimiDeltaAttention (short conv "
                         "off, output forget-gate on). old: the original "
                         "hand-rolled GatedDeltaLayer prototype (no conv, no "
                         "output gate, k_dim==v_dim) -- same data/CLI otherwise, "
                         "for isolating architecture effects. Only supports "
                         "gate in (sigmoid, signed_sigmoid2, signed_tanh). "
                         "deltaproduct: GatedDeltaProduct (--dp-num-householder "
                         "Householders per step, unsigned forget gate, "
                         "--no-neg-eigval controls the Householder range). "
                         "gateddeltanet: GatedDeltaNet (single Householder, "
                         "unsigned forget gate, --no-neg-eigval controls the "
                         "Householder range). Both run via a pure-PyTorch CPU "
                         "fallback here (no launchable Triton) and ignore "
                         "--gates/--gate-init-style (no signed-gate option).")
    ap.add_argument("--dp-num-householder", type=int, default=2,
                    help="only affects --layer deltaproduct: Householder "
                         "reflections composed per step.")
    ap.add_argument("--dp-no-conv-silu", default="qkv",
                    help="only affects --layer deltaproduct: which of q/k/v "
                         "get F.silu (default 'qkv', shipped). 'qv' drops it "
                         "from the key only -- k is the Householder update "
                         "direction, silu's floor at ~-0.278 squashes it. "
                         "'' drops it entirely.")
    ap.add_argument("--old-eps-floor", type=float, default=None,
                    help="only affects --layer old with a signed gate. If set "
                         "(e.g. to exp(-5) ~= 0.0067, matching --layer kda's "
                         "lower_bound=-5), replaces the smooth "
                         "2*sigmoid(a)-1 / tanh(a) pass-through-zero with "
                         "sign(a)*(eps+(1-eps)*|a|) -- the same jump-at-zero "
                         "construction signed_gate() uses. Ablation to test "
                         "whether that discontinuity, not the gauge/detach "
                         "mechanism, is why --layer kda fails on hard tasks.")
    ap.add_argument("--old-qk-scale", action="store_true",
                    help="only affects --layer old. Scale q by head_dim**-0.5 "
                         "after L2-norm, matching naive_recurrent_kda's "
                         "convention (this layer never scaled q before).")
    ap.add_argument("--old-gauge", action="store_true",
                    help="only affects --layer old with a signed gate. Carry "
                         "the +-1 part of alpha as a running parity pushed "
                         "onto q/k and run the recurrence with |alpha| only -- "
                         "ComplexKimiDeltaAttention's gauge mechanism, dropped "
                         "into the otherwise-working old layer. Provably "
                         "output-identical; isolates the mechanism itself.")
    ap.add_argument("--old-log-space", action="store_true",
                    help="only affects --layer old with a signed gate. Carry "
                         "the gate as log|alpha| and exponentiate it back "
                         "inside the recurrence, as signed_gate + "
                         "naive_recurrent_kda do. Implies an eps floor "
                         "(exp(-5) unless --old-eps-floor is given).")
    ap.add_argument("--old-fproj", action="store_true",
                    help="only affects --layer old. Produce the gate logits "
                         "the kda way: a rank-head_dim factored f_proj (two "
                         "Linears, no nonlinearity between) plus dt_bias and "
                         "a per-head exp(A_log) rescale, instead of one "
                         "full-rank Linear+bias.")
    ap.add_argument("--old-qkv-silu", action="store_true",
                    help="only affects --layer old. Apply F.silu to q/k/v "
                         "before normalising, as the kda layer does when "
                         "use_short_conv=False. silu bounds below at ~-0.278, "
                         "pushing q/k/v toward the non-negative orthant and "
                         "restricting the directions k can take in the "
                         "rank-1 update (I - beta k k^T).")
    ap.add_argument("--old-fproj-rank", type=int, default=None,
                    help="bottleneck width for --old-fproj, independent of "
                         "--head-dim (default: head_dim, matching kda). Set "
                         ">= d_model to make the factored gate map full rank "
                         "and isolate rank from the factored structure.")
    ap.add_argument("--old-beta-nobias", action="store_true",
                    help="only affects --layer old. Drop the bias on the beta "
                         "projection (and its 0.5 init), matching kda's "
                         "b_proj(bias=False).")
    ap.add_argument("--old-per-head-norm", action="store_true",
                    help="only affects --layer old. Normalise the recurrence "
                         "output per head, as kda's o_norm does, instead of "
                         "once across all heads jointly.")
    ap.add_argument("--drop-silu", action="store_true",
                    help="replace the q/k/v silu with the identity, in both the "
                         "short-conv and no-conv branches. silu is bounded below "
                         "at ~-0.278 and pushes q/k/v toward the non-negative "
                         "orthant, which restricts k -- the direction of the "
                         "rank-1 update (I - beta k k^T).")
    ap.add_argument("--readout", default="mlp", choices=["mlp", "maxout"],
                    help="mlp (default): LayerNorm -> Linear -> GELU -> Linear. "
                         "maxout: max over --maxout-codewords linear scores per "
                         "class. For a hidden group covering the visible group "
                         "many-to-one the per-class codebook is centrally "
                         "symmetric, so no affine classifier is correct and the "
                         "max is required, not merely helpful.")
    ap.add_argument("--readout-hidden", type=int, default=None,
                    help="hidden width of the --readout mlp (default 4*d_model). "
                         "Must scale with the number of REACHABLE STATES, not "
                         "with d_model: A_5 has 7200 of them behind 60 labels.")
    ap.add_argument("--maxout-agg", default="logsumexp", choices=["max", "logsumexp"],
                    help="how --readout maxout pools codewords. logsumexp "
                         "(default) gives every codeword gradient; max gives it "
                         "to one, leaving the rest at their random init.")
    ap.add_argument("--maxout-codewords", type=int, default=120,
                    help="codewords per class for --readout maxout (120 = the "
                         "binary icosahedral cover of A_5).")
    ap.add_argument("--weight-decay", type=float, default=1e-12,
                    help="AdamW weight decay (default 1e-12, i.e. effectively "
                         "off). The loss curves show the old layer groks -- a "
                         "sharp loss collapse ~step 1500 -- while kda plateaus; "
                         "weight decay is the classic lever for inducing that "
                         "transition. The original standalone script used 1e-4.")
    ap.add_argument("--freeze-a-log", action="store_true",
                    help="only affects --layer kda. Freeze A_log at its init "
                         "value (0), removing the extra learnable per-head "
                         "multiplicative rescale of the gate pre-activation "
                         "that --layer old has no equivalent of.")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--no-neg-eigval", action="store_true",
                    help="pin beta to (0,1] so the gate is measured in isolation")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "kernel", "naive_recurrent", "naive_chunk"],
                    help="passed straight to the layer; auto = kernel if Triton "
                         "can launch, else fla's naive_recurrent_kda")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--out", default="wordproblem.jsonl")
    a = ap.parse_args()

    if a.summary:
        summarise([json.loads(l) for l in open(a.summary)])
        raise SystemExit(0)

    dev = pick_device(a.device)
    backend = a.backend
    cfg = dict(steps=a.steps, train_len=a.train_len, batch=a.batch,
               eval_batch=a.eval_batch, ckpt_dir=a.ckpt_dir, lr=a.lr,
               lr_schedule=a.lr_schedule, restrict_gens=a.restrict_gens,
               cycle_curriculum_lens=[int(x) for x in a.cycle_curriculum_lens.split(",")],
               gate_init_style=a.gate_init_style, beta_init_style=a.beta_init_style,
               a5_theory_init=a.a5_theory_init,
               a5_theory_heads=a.a5_theory_heads,
               a5_theory_saturation=a.a5_theory_saturation,
               layer=a.layer,
               old_eps_floor=a.old_eps_floor, old_qk_scale=a.old_qk_scale,
               old_gauge=a.old_gauge, old_log_space=a.old_log_space, weight_decay=a.weight_decay,
               old_fproj=a.old_fproj, old_qkv_silu=a.old_qkv_silu,
               old_fproj_rank=a.old_fproj_rank,
               old_beta_nobias=a.old_beta_nobias,
               old_per_head_norm=a.old_per_head_norm,
               freeze_a_log=a.freeze_a_log,
               heads=a.heads, head_dim=a.head_dim, d_model=a.d_model, readout=a.readout,
               maxout_agg=a.maxout_agg, readout_hidden=a.readout_hidden,
               drop_silu=a.drop_silu,
               maxout_codewords=a.maxout_codewords,
               optimizer=a.optimizer, muon_lr=a.muon_lr,
               muon_momentum=a.muon_momentum, muon_ns_steps=a.muon_ns_steps,
               muon_scope=a.muon_scope,
               allow_neg_eigval=not a.no_neg_eigval, log_every=a.log_every,
               len_dist=a.len_dist,
               pareto_alpha=a.pareto_alpha, max_train_len=a.max_train_len,
               curriculum_threshold=a.curriculum_threshold,
               curriculum_check_every=a.curriculum_check_every,
               curriculum_passes=a.curriculum_passes,
               curriculum_eval_batch=a.curriculum_eval_batch,
               eval_len=a.eval_len,
               curriculum_lens=[int(x) for x in a.curriculum_lens.split(",")],
               eval_curve=a.eval_curve, eval_continuous=a.eval_continuous,
               eval_curve_lens=[int(x) for x in a.eval_curve_lens.split(",")],
               dp_num_householder=a.dp_num_householder,
               dp_no_conv_silu=a.dp_no_conv_silu)
    if cfg["len_dist"] == "performance_curriculum":
        if not 0 < cfg["curriculum_threshold"] <= 1:
            ap.error("--curriculum-threshold must be in (0, 1]")
        if cfg["curriculum_check_every"] < 1 or cfg["curriculum_passes"] < 1:
            ap.error("curriculum check interval and passes must be positive")
        if cfg["curriculum_eval_batch"] < 1:
            ap.error("--curriculum-eval-batch must be positive")
        if any(a >= b for a, b in zip(cfg["curriculum_lens"],
                                      cfg["curriculum_lens"][1:])):
            ap.error("performance curriculum lengths must be strictly increasing")
    tasks = ["s2", "z5", "s3", "s4"] if a.all else [a.task]
    n = len(tasks) * len(a.gates) * a.seeds
    print(f"device {dev} | backend {backend} | torch {torch.__version__} | "
          f"beta in (0,{2 if cfg['allow_neg_eigval'] else 1}] | {n} runs -> {a.out}",
          flush=True)
    if not HAS_TRITON:
        print("  (no launchable Triton: the layer falls back to fla's own "
              "reference\n   in fla.ops.kda.naive -- naive_recurrent_kda unless "
              "--backend naive_chunk)", flush=True)

    rows, done = [], 0
    with open(a.out, "a") as fh:
        for task in tasks:
            V = len(perm_group(task)[0])
            E = a.eval_len or 4 * a.train_len
            print(f"\n=== {task}  ({V} elements, chance {1/V:.3f})  "
                  f"train@{a.train_len} test@{E} ===", flush=True)
            for g in a.gates:
                for sd in range(a.seed_start, a.seed_start + a.seeds):
                    r = run(task, g, cfg, sd, dev, backend)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    rows.append(r)
                    done += 1
                    acc = " ".join(f"L{kk}={vv:.3f}" for kk, vv in r["acc"].items())
                    print(f"  [{done}/{n}] {g:<18} s{sd}  {acc}  "
                          f"a<0 {r['frac_alpha_neg']:.2f}  flip {r['flip_rate']:.2f}"
                          f"  [{r['secs']}s]", flush=True)
    summarise(rows)
