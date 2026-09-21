"""
Group word problems for signed channel-wise gates in a KDA-style layer.
Tests the predictions of Table 2 on a laptop.  torch + numpy only.

One layer, one token per group element, train short / test long.  Length
extrapolation is the discriminator: a model that memorises in-distribution
still fails at 4x the training length, so the recurrence has to do the work.

Four arms, differing ONLY in the gate parameterisation:

  kda            alpha = sigmoid(.)      in [0,1]     beta = sigmoid(.)   in [0,1]
  beta2          alpha = sigmoid(.)      in [0,1]     beta = 2*sigmoid(.) in [0,2]
  signed-static  alpha = s * |2sig(.)-1|, s FIXED     beta = 2*sigmoid(.) in [0,2]
  signed         alpha = 2*sigmoid(.)-1  in [-1,1]    beta = 2*sigmoid(.) in [0,2]

The load-bearing comparison is signed-static vs signed on s3.  The two have
IDENTICAL spectra; only the static-flip-channel theorem separates them, and no
eigenvalue argument does.  Everything else is a control.

Quick start
-----------
  python group_wordproblem.py --task z5                 # ~15 min, 4 arms
  python group_wordproblem.py --task s3 --seeds 3       # the decisive one
  python group_wordproblem.py --summary results.jsonl   # render the table

Results are appended to the .jsonl as each run finishes, so an interrupted
sweep loses at most one run.
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = ["kda", "beta2", "signed-static", "signed"]
TASKS = ["parity", "z5", "s3", "a4", "s4", "a5", "s5"]


# --------------------------------------------------------------------------- #
# groups
# --------------------------------------------------------------------------- #
def _sign(p):
    s, p = 1, list(p)
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            s = -s
    return s


def perm_group(name):
    """(elements, multiplication table, identity index)."""
    if name == "parity":
        els = [(0, 1), (1, 0)]                                     # S_2
    elif name == "z5":
        els = [tuple((i + k) % 5 for i in range(5)) for k in range(5)]    # C_5
    elif name == "s3":
        els = sorted(itertools.permutations(range(3)))
    elif name == "a4":
        els = [p for p in itertools.permutations(range(4)) if _sign(p) == 1]
    elif name == "s4":
        els = sorted(itertools.permutations(range(4)))
    elif name == "a5":
        els = [p for p in itertools.permutations(range(5)) if _sign(p) == 1]
    elif name == "s5":
        els = sorted(itertools.permutations(range(5)))
    else:
        raise ValueError(name)
    idx = {e: i for i, e in enumerate(els)}
    tab = np.zeros((len(els), len(els)), dtype=np.int64)
    for i, a in enumerate(els):                                    # (a.b)(x)=a(b(x))
        for j, b in enumerate(els):
            tab[i, j] = idx[tuple(a[b[x]] for x in range(len(a)))]
    return els, tab, idx[tuple(range(len(els[0])))]


def make_batch(table, ident, batch, length, rng, device):
    """x: (B,T) group elements; y: (B,T) running product x_t . ... . x_1."""
    x = rng.integers(0, table.shape[0], size=(batch, length))
    y = np.empty_like(x)
    acc = np.full(batch, ident, dtype=np.int64)
    for t in range(length):
        acc = table[x[:, t], acc]
        y[:, t] = acc
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# --------------------------------------------------------------------------- #
# layer
# --------------------------------------------------------------------------- #
class GatedDeltaLayer(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, arm,
                 gate_init="spread", tie=False, tau=0.1):
        super().__init__()
        self.h, self.n, self.arm, self.tie, self.tau = n_heads, head_dim, arm, tie, tau
        inner = n_heads * head_dim
        self.q = nn.Linear(d_model, inner, bias=False)
        self.k = nn.Linear(d_model, inner, bias=False)
        self.v = nn.Linear(d_model, inner, bias=False)
        self.a = nn.Linear(d_model, inner)                         # decay logits
        self.b = nn.Linear(d_model, n_heads)                       # rate logits
        self.o = nn.Linear(inner, d_model, bias=False)
        with torch.no_grad():
            self.a.weight.mul_(0.1)
            if arm in ("kda", "beta2"):
                self.a.bias.fill_(2.0)                             # alpha ~ 0.88
            elif gate_init == "spread":
                self.a.bias.uniform_(-1.5, 1.5)                    # mixed signs
            else:
                self.a.bias.uniform_(1.0, 2.0)                     # all alpha > 0
            self.b.bias.fill_(0.5)
        if arm == "signed-static":                                 # frozen P = {0}
            s = torch.ones(n_heads, head_dim)
            s[:, 0] = -1.0
            self.register_buffer("sign", s)

    def gates(self, x):
        B, T, _ = x.shape
        a = self.a(x).view(B, T, self.h, self.n)
        b = self.b(x).view(B, T, self.h)
        if self.arm == "kda":
            return torch.sigmoid(a), torch.sigmoid(b)
        if self.arm == "beta2":
            return torch.sigmoid(a), 2 * torch.sigmoid(b)
        beta = 2 * torch.sigmoid(b)
        if self.arm == "signed-static":
            mag = torch.abs(2 * torch.sigmoid(a) - 1)
            if self.tie:
                mag = mag.mean(-1, keepdim=True).expand_as(mag)
            return self.sign * mag, beta
        if self.arm == "signed":
            alpha = 2 * torch.sigmoid(a) - 1
            if self.tie:
                # tanh soft-sign keeps a gradient path across alpha = 0.
                # torch.sign() has zero gradient and would freeze the sign
                # pattern at initialisation.
                alpha = torch.tanh(alpha / self.tau) * \
                        alpha.abs().mean(-1, keepdim=True)
            return alpha, beta
        raise ValueError(self.arm)

    def forward(self, x, collect=False):
        B, T, _ = x.shape
        sh = (B, T, self.h, self.n)
        q = F.normalize(self.q(x).view(sh), dim=-1)
        k = F.normalize(self.k(x).view(sh), dim=-1)
        v = self.v(x).view(sh)
        alpha, beta = self.gates(x)
        H = x.new_zeros(B, self.h, self.n, self.n)
        outs = []
        for t in range(T):
            at, kt, vt, bt = alpha[:, t], k[:, t], v[:, t], beta[:, t]
            S = at.unsqueeze(-1) * H                               # Diag(alpha) H
            kS = torch.einsum("bhn,bhnd->bhd", kt, S)              # k^T S
            w = bt.unsqueeze(-1) * (kS - vt)
            H = S - kt.unsqueeze(-1) * w.unsqueeze(-2)             # delta update
            outs.append(torch.einsum("bhn,bhnd->bhd", q[:, t], H))
        o = torch.stack(outs, 1).reshape(B, T, self.h * self.n)
        out = self.o(F.normalize(o, dim=-1))
        return (out, alpha.detach(), k.detach()) if collect else out


class Model(nn.Module):
    def __init__(self, vocab, d_model=48, n_heads=4, head_dim=12,
                 arm="signed", gate_init="spread", tie=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.layer = GatedDeltaLayer(d_model, n_heads, head_dim, arm, gate_init, tie)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                                 nn.Linear(4 * d_model, vocab))

    def forward(self, x, collect=False):
        e = self.emb(x)
        if collect:
            h, alpha, k = self.layer(e, collect=True)
            return self.mlp(self.norm(e + h)), alpha, k
        return self.mlp(self.norm(e + self.layer(e)))


# --------------------------------------------------------------------------- #
# train / eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, table, ident, lengths, device, seed, batch=256):
    """Accuracy on the final quarter of each sequence: the part needing state."""
    model.eval()
    out = {}
    for L in lengths:
        x, y = make_batch(table, ident, batch, L, np.random.default_rng(seed), device)
        p = model(x).argmax(-1)
        out[L] = (p[:, -(L // 4):] == y[:, -(L // 4):]).float().mean().item()
    model.train()
    return out


@torch.no_grad()
def diagnostics(model, table, ident, device, seed, batch=128, L=48):
    """Fraction of alpha<0, |P| per token, and phi = 2 arccos(rho)."""
    x, _ = make_batch(table, ident, batch, L, np.random.default_rng(seed), device)
    _, alpha, k = model(x, collect=True)
    m = (alpha < 0).float()
    rho2 = (m * k.pow(2)).sum(-1).clamp(0, 1)
    phi = 2 * torch.arccos(rho2.sqrt().clamp(-1, 1))
    return dict(frac_alpha_neg=round(m.mean().item(), 4),
                q_mean=round(m.sum(-1).mean().item(), 3),
                phi_mean=round(phi.mean().item(), 4),
                phi_std=round(phi.std().item(), 4))


def run(task, arm, cfg, seed, device):
    torch.manual_seed(seed)
    els, table, ident = perm_group(task)
    V = len(els)
    model = Model(V, cfg["d_model"], cfg["heads"], cfg["head_dim"], arm,
                  cfg["gate_init"], cfg["tie"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-12)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.1)
    rng = np.random.default_rng(seed + 1234)
    T, t0 = cfg["train_len"], time.time()
    for it in range(cfg["steps"]):
        x, y = make_batch(table, ident, cfg["batch"], T, rng, device)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if cfg["log_every"] and (it + 1) % cfg["log_every"] == 0:
            acc = evaluate(model, table, ident, [T], device, 999, 128)[T]
            print(f"      {it+1:5d}  loss {loss.item():.4f}  acc@{T} {acc:.3f}",
                  flush=True)
    lens = [T, 2 * T, 4 * T]
    return dict(task=task, arm=arm, seed=seed, secs=round(time.time() - t0, 1),
                chance=round(1.0 / V, 4),
                acc={str(L): round(v, 4) for L, v in
                     evaluate(model, table, ident, lens, device, 999).items()},
                **diagnostics(model, table, ident, device, 999))


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def summarise(path):
    rows = [json.loads(l) for l in open(path)]
    best = {}
    for r in rows:                                    # best of seeds
        L4 = str(max(int(x) for x in r["acc"]))
        v = r["acc"][L4]
        key = (r["task"], r["arm"])
        if key not in best or v > best[key][0]:
            best[key] = (v, r)
    tasks = sorted({t for t, _ in best}, key=TASKS.index)
    w = 20
    print(f"\n{'arm':16s}" + "".join(t.center(w) for t in tasks))
    print(f"{'':16s}" + "".join("acc@4x (frac a<0)".center(w) for _ in tasks))
    print("-" * (16 + w * len(tasks)))
    for arm in ARMS:
        line = f"{arm:16s}"
        for t in tasks:
            if (t, arm) in best:
                v, r = best[(t, arm)]
                c = r["chance"]
                mid = c + 0.5 * (1.0 - c)          # halfway chance -> perfect
                mark = "OK" if v > 0.90 else ("~" if v > mid else "xx")
                line += f"{mark} {v:.3f} ({r['frac_alpha_neg']:.2f})".center(w)
            else:
                line += "-".center(w)
        print(line)
    print("\nOK >0.90    ~ partial (above halfway to perfect)    xx at/near chance")
    print("acc@4x : accuracy on the last quarter at 4x the training length")
    print("frac a<0 : channels with a negative decay gate at convergence")
    print("           (if ~0 for a signed arm, the range extension went unused)")


# --------------------------------------------------------------------------- #
def pick_device(p):
    if p != "auto":
        return p
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", metavar="FILE", help="render a table from a .jsonl")
    ap.add_argument("--task", default="s3", choices=TASKS)
    ap.add_argument("--arm", default=None, choices=ARMS)
    ap.add_argument("--all", action="store_true", help="parity,z5,s3,s4 x 4 arms")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--train-len", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--gate-init", default="spread", choices=["spread", "positive"])
    ap.add_argument("--tie", action="store_true", help="tie |alpha| within a head")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()

    if a.summary:
        summarise(a.summary); sys.exit(0)

    dev = pick_device(a.device)
    cfg = dict(steps=a.steps, train_len=a.train_len, batch=a.batch, lr=a.lr,
               heads=a.heads, head_dim=a.head_dim, d_model=a.d_model,
               gate_init=a.gate_init, tie=a.tie, log_every=a.log_every)
    tasks = ["parity", "z5", "s3", "s4"] if a.all else [a.task]
    arms = ARMS if (a.all or a.arm is None) else [a.arm]
    n = len(tasks) * len(arms) * a.seeds
    print(f"device {dev} | torch {torch.__version__} | {n} runs -> {a.out}", flush=True)

    done = 0
    with open(a.out, "a") as fh:
        for task in tasks:
            V = len(perm_group(task)[0])
            print(f"\n=== {task}  ({V} elements, chance {1/V:.3f})  "
                  f"train@{a.train_len} test@{4*a.train_len} ===", flush=True)
            for arm in arms:
                for sd in range(a.seeds):
                    r = run(task, arm, cfg, sd, dev)
                    fh.write(json.dumps(r) + "\n"); fh.flush()
                    done += 1
                    acc = "  ".join(f"L{k}={v:.3f}" for k, v in r["acc"].items())
                    print(f"  [{done}/{n}] {arm:14s} s{sd}  {acc}   "
                          f"a<0 {r['frac_alpha_neg']:.2f}  q {r['q_mean']:.1f}"
                          f"  [{r['secs']}s]", flush=True)
    print(f"\ndone. summarise with:\n  python {os.path.basename(__file__)} "
          f"--summary {a.out}", flush=True)