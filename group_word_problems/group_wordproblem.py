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


def sign_vector(els):
    """+1/-1 per group element; the S_n -> Z/2 homomorphism."""
    return np.array([_sign(e) for e in els], dtype=np.int64)


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
def evaluate(model, table, ident, lengths, device, seed, batch=256, sgn=None):
    """Accuracy on the final quarter of each sequence: the part needing state.

    If `sgn` is given (the S_n -> Z/2 homomorphism), also report accuracy on the
    sign alone.  On S_3 the only proper quotient is Z/2, so a model stuck at
    element-accuracy 1/3 with sign-accuracy ~1 has learned the sign and nothing
    more -- which distinguishes 'partial progress' from 'a hard plateau'.
    """
    model.eval()
    out, sout = {}, {}
    for L in lengths:
        x, y = make_batch(table, ident, batch, L, np.random.default_rng(seed), device)
        p = model(x).argmax(-1)
        tail = slice(-(L // 4), None)
        out[L] = (p[:, tail] == y[:, tail]).float().mean().item()
        if sgn is not None:
            g = torch.from_numpy(sgn).to(p.device)
            sout[L] = (g[p[:, tail]] == g[y[:, tail]]).float().mean().item()
    model.train()
    return (out, sout) if sgn is not None else out


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


def eval_grid(train_len, max_len):
    """Lengths to trace the extrapolation curve on, always including train_len."""
    g = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256]
    g = [L for L in g if 4 <= L <= max_len]
    return sorted(set(g + [train_len, max_len]))


def run(task, arm, cfg, seed, device, ckpt_dir=None):
    torch.manual_seed(seed)
    els, table, ident = perm_group(task)
    V = len(els)
    model = Model(V, cfg["d_model"], cfg["heads"], cfg["head_dim"], arm,
                  cfg["gate_init"], cfg["tie"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
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
    sgn = sign_vector(els)
    acc, sacc = evaluate(model, table, ident, lens, device, 999, sgn=sgn)
    curve = evaluate(model, table, ident,
                     eval_grid(T, cfg["max_eval_len"]), device, 999, batch=512)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(dict(state=model.state_dict(), cfg=cfg, task=task, arm=arm,
                        seed=seed, acc={str(L): round(v, 4) for L, v in acc.items()}),
                   os.path.join(ckpt_dir, f"{task}_{arm}_s{seed}.pt"))
    return dict(task=task, arm=arm, seed=seed, secs=round(time.time() - t0, 1),
                chance=round(1.0 / V, 4), train_len=T,
                acc={str(L): round(v, 4) for L, v in acc.items()},
                sign_acc={str(L): round(v, 4) for L, v in sacc.items()},
                curve={str(L): round(v, 4) for L, v in curve.items()},
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
                sg = r.get("sign_acc", {}).get(L4)
                tag = f"/{sg:.2f}" if sg is not None else ""
                line += f"{mark} {v:.3f}{tag}".center(w)
            else:
                line += "-".center(w)
        print(line)
    print("\nOK >0.90    ~ partial (above halfway to perfect)    xx at/near chance")
    print("acc@4x / sgn : element accuracy, then accuracy on the sign homomorphism")
    print("  on S_3 the only proper quotient is Z/2, so acc~0.33 with sgn~1.00")
    print("  means the model learned the sign and nothing more")
    print("frac a<0 : channels with a negative decay gate at convergence")
    print("           (if ~0 for a signed arm, the range extension went unused)")


def recurve(ckpt_dir, out, device, max_len):
    """Reload saved models and re-evaluate on a dense length grid (no retraining)."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*.pt"))):
        ck = torch.load(f, map_location=device, weights_only=False)
        c, task, arm = ck["cfg"], ck["task"], ck["arm"]
        els, table, ident = perm_group(task)
        m = Model(len(els), c["d_model"], c["heads"], c["head_dim"], arm,
                  c["gate_init"], c["tie"]).to(device)
        m.load_state_dict(ck["state"])
        grid = eval_grid(c["train_len"], max_len)
        cur = evaluate(m, table, ident, grid, device, 999, batch=512)
        rows.append(dict(task=task, arm=arm, seed=ck["seed"],
                         chance=round(1 / len(els), 4), train_len=c["train_len"],
                         acc=ck["acc"], frac_alpha_neg=ck.get("frac_alpha_neg", 0.0),
                         curve={str(L): round(v, 4) for L, v in cur.items()}))
        print(f"  recurved {os.path.basename(f)}", flush=True)
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} dense curves -> {out}")


def plot(path, out=None, scaled=True, dpi=200):
    """Length-extrapolation curves, one panel per task, one line per arm."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in open(path)]
    for r in rows:                       # fall back to the 3-point acc field so
        r.setdefault("curve", r["acc"])  # runs made before --plot existed still work
    if not rows:
        sys.exit("empty results file")
    tasks = sorted({r["task"] for r in rows}, key=TASKS.index)

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "font.size": 11,
    })
    title = {"parity": "Parity", "z5": r"Group $\mathbb{Z}/5$", "s3": r"Group $S_3$",
             "a4": r"Group $A_4$", "s4": r"Group $S_4$", "a5": r"Group $A_5$",
             "s5": r"Group $S_5$"}
    colour = {"kda": "#3B0F5C", "beta2": "#B03A2E",
              "signed-static": "#C9A227", "signed": "#4A7FA5"}
    label = {"kda": "KDA (shipped)", "beta2": r"$+\,\beta\in[0,2]$",
             "signed-static": r"$+\,$signed, static $\mathcal{P}$",
             "signed": r"$+\,$signed (ours)"}

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.1 * len(tasks), 2.7),
                             squeeze=False)
    for ax, task in zip(axes[0], tasks):
        sub = [r for r in rows if r["task"] == task]
        chance, tl = sub[0]["chance"], sub[0].get("train_len", 24)
        for arm in ARMS:
            rs = [r for r in sub if r["arm"] == arm]
            if not rs:
                continue
            lens = sorted(int(k) for k in rs[0]["curve"])
            if len(lens) < 4:
                ax.set_xscale("linear")          # sparse: linear axis reads better
            # best of seeds at each length, matching the table protocol
            ys = [max(r["curve"][str(L)] for r in rs) for L in lens]
            if scaled:
                ys = [max(0.0, (y - chance) / (1 - chance)) for y in ys]
            ax.plot(lens, ys, lw=2.0, color=colour[arm], label=label[arm],
                    solid_capstyle="round")
        ax.axvline(tl, color="k", ls="-.", lw=1.4, zorder=3)
        ax.annotate("train", xy=(tl, 1.02), xytext=(2, 0), textcoords="offset points",
                    fontsize=7.5, color="0.35", ha="left", va="top")
        ax.grid(True, ls="--", lw=0.7, color="0.85", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64, 128, 256])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_ylim(-0.04, 1.08)
        lo, hi = min(lens), max(lens)
        if lo >= tl:                       # data starts at the training length:
            lo = tl * 0.82                 # pad so the marker is not on the spine
        ax.set_xlim(lo, hi * 1.03)
        ax.set_title(title.get(task, task))
        ax.set_xlabel("Sequence length")
    axes[0][0].set_ylabel("Scaled accuracy" if scaled else "Accuracy")
    if len(tasks) == 1:
        axes[0][0].legend(frameon=False, fontsize=8.5, loc="center left")
    else:
        axes[0][-1].legend(frameon=False, fontsize=8.5, loc="center left",
                           bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out = out or os.path.splitext(path)[0] + "_extrapolation.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=dpi)
    print(f"wrote {out} and {out.replace('.pdf', '.png')}")
    if scaled:
        print("scaled accuracy: 0 = chance, 1 = perfect; dash-dot = training length")


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
    ap.add_argument("--plot", metavar="FILE", help="length-extrapolation plot from a .jsonl")
    ap.add_argument("--plot-out", default=None)
    ap.add_argument("--raw-acc", action="store_true",
                    help="plot raw accuracy instead of scaled (0=chance)")
    ap.add_argument("--max-eval-len", type=int, default=0,
                    help="longest evaluation length (default 8x train-len)")
    ap.add_argument("--task", default="s3", choices=TASKS)
    ap.add_argument("--arm", default=None, choices=ARMS)
    ap.add_argument("--all", action="store_true", help="parity,z5,s3,s4 x 4 arms")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--train-len", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=12)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--gate-init", default="spread", choices=["spread", "positive"])
    ap.add_argument("--tie", action="store_true", help="tie |alpha| within a head")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--save-ckpt", metavar="DIR", default=None,
                    help="save each trained model so curves can be redrawn later")
    ap.add_argument("--recurve", metavar="DIR", default=None,
                    help="reload checkpoints from DIR and write a dense curve")
    a = ap.parse_args()

    if a.summary:
        summarise(a.summary); sys.exit(0)
    if a.plot:
        plot(a.plot, a.plot_out, scaled=not a.raw_acc); sys.exit(0)
    if a.recurve:
        recurve(a.recurve, a.out, pick_device(a.device),
                a.max_eval_len or 8 * a.train_len); sys.exit(0)

    dev = pick_device(a.device)
    cfg = dict(steps=a.steps, train_len=a.train_len, batch=a.batch, lr=a.lr,
               heads=a.heads, head_dim=a.head_dim, d_model=a.d_model,
               gate_init=a.gate_init, tie=a.tie, log_every=a.log_every,
               max_eval_len=a.max_eval_len or 8 * a.train_len)
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
                    r = run(task, arm, cfg, sd, dev, a.save_ckpt)
                    fh.write(json.dumps(r) + "\n"); fh.flush()
                    done += 1
                    acc = "  ".join(f"L{k}={v:.3f}" for k, v in r["acc"].items())
                    s4 = r["sign_acc"][str(4 * a.train_len)]
                    print(f"  [{done}/{n}] {arm:14s} s{sd}  {acc}   "
                          f"sgn@4x {s4:.3f}  a<0 {r['frac_alpha_neg']:.2f}"
                          f"  q {r['q_mean']:.1f}  [{r['secs']}s]", flush=True)
    print(f"\ndone. summarise with:\n  python {os.path.basename(__file__)} "
          f"--summary {a.out}", flush=True)
