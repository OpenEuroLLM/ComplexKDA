"""Fit the ladder's losses as a scaling law, per arm, and against the paper's.

    lm_scaling/scaling_fit.py                 # the fits and the comparison
    lm_scaling/scaling_fit.py --loo           # and leave-one-out stability

THE FORMS, both from arXiv:2608.28308 (Ajroldi et al.), whose ladder, corpus
(high-quality Nemotron-CC) and tokenizer (GPT-NeoX-20B) this study adopts:

    Chinchilla   L(N, D) = E + A N^-alpha + B D^-beta
    Skaling      L(N, D) = E + (A N^-alpha + B D^-beta)^k        (Videau et al.)

The paper finds the separable Chinchilla form fits its middle and extrapolates
badly (held-out RMSE at 1.7B of 0.0174 against Skaling's 0.0056), which is worth
knowing before quoting a Chinchilla exponent from thirty.

WHAT THIRTY CELLS CAN AND CANNOT IDENTIFY. Each arm has 6 rungs x 5 budgets: N
spans 36x and D spans 8.3x. That is enough to pin `alpha` and nothing else. `E`
is unidentifiable over so short a lever -- it trades off against A and B, and a
too-high E buys itself steeper exponents -- which is why the per-arm fits here
put E between 1.08 and 1.26 where the paper, fitting to 1.7B and 300BT, puts it
at 0.9638. The paper's own Chinchilla fit carries +-100% on A and +-250% on B
for the same reason, with fifty times the data.

That spread is the argument for the two treatments of E this study reports --
every arm free to choose its own, and E pinned to the paper's -- and for
judging them OUT OF SAMPLE, which is what `scaling_holdout.py` does.

WHAT IS ROBUST is the comparison BETWEEN arms. Fitting one shared shape with a
per-arm offset reproduces the paired mean differences exactly, under both
functional forms, because that is all the data can support: the arms differ by a
constant, not by an exponent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

# The MEGATRON ladder is the scaling record: the torchtitan one it replaced
# sat 0.049 nats away on identical configuration, which is larger than most
# of what this fit measures.
LADDER_TSV = _HERE / "harvest" / "megatron_ladder.tsv"
TEX_DIR = _HERE / "tex"

# Paper-facing names, matching the figures.
RUNGS = ("47M", "124M", "302M", "588M", "983M", "1.7B")
LABEL = {"attn": "Transformer++", "attn-qknorm": "Transformer++ (QK-norm)",
         "gdn": "Gated DeltaNet",
         "kda-sig-lowrank": "KDA", "ckda-shipped-lowrank": "CKDA",
         "kda-sig-hybrid-lowrank": "KDA + attn 3:1",
         "ckda-shipped-hybrid-lowrank": "CKDA + attn 3:1"}

#: Which arm to hold against the paper's own law, best first.
#:
#: attn-qknorm, NOT attn, when we have it. The paper's runs use QK-norm, so its
#: law was fitted to a normed transformer; comparing our un-normed arm against
#: it charges the difference to the law. Measured at 47M/6BT in Megatron,
#: QK-norm is worth 0.0384, which is an order of magnitude larger than the
#: offset this comparison is trying to resolve.
REFERENCE_ARMS = ("attn-qknorm", "attn")

# FITTED TO THE REFERENCE STUDY'S OWN PUBLISHED CELLS, not transcribed from its
# table -- lm_scaling/reference/fit_reference.py, using their procedure (A and B
# in log space, Huber delta 1e-3, their 2,000-start grid) on the 51 best-HP cells
# of reference/oellm_loss_post_annealing.csv.
#
# The table's rounded values do not survive evaluation. At 50M/6BT their
# measurement is 2.9272; this fit says 2.9278 and the transcribed Skaling row
# says 3.0810, because a 0.07 difference in E and a 0.017 in k compound. Across
# all 51 cells the transcribed rows sit +0.1054 (Skaling) and +0.9830
# (Chinchilla) above the data, so anything compared against them inherits that.
#
# The exponents were right all along: refitting recovers beta 0.3494 against
# their published 0.3496, and every other Skaling term lands inside their own
# standard errors. `se` is still THEIRS, from the table -- our refit does not
# re-derive the intervals.
#
# Chinchilla is kept for comparison and should be read as poorly determined:
# rmse 0.0194 against Skaling's 0.0047 on the same cells, and the authors' own
# standard errors on A and B are +-100% and +-250%.
PAPER = {
    "chinchilla": dict(E=1.3659, A=109.255, alpha=0.2603, B=69.3002, beta=0.2167,
                       rmse=0.01935,
                       se=dict(E=0.25, A=2.1e2, alpha=0.05, B=1.3e3, beta=0.081)),
    "skaling": dict(E=0.9638, A=19637.5, alpha=0.4833, B=4540.49, beta=0.3494,
                    k=0.3932, rmse=0.00473,
                    se=dict(E=0.087, A=1.4e4, alpha=0.03, B=2.9e3, beta=0.022, k=0.033)),
}


#: Arms already reported as missing from the matched table, so the note is
#: printed once rather than once per cell.
_WARNED: set[str] = set()


def load(path: Path):
    import geometries

    table = geometries.table()
    rows = []
    for line in path.read_text().splitlines()[1:]:
        f = line.split("\t")
        # The `schedule` column arrived with the medium rungs. Rows written
        # before it are single-stage, which is what the lower ladder ran.
        arm, size, bt, gbs, lr, steps, tokens, loss = f[:8]
        sched = f[8] if len(f) > 8 else "single"
        e = table[size]
        if arm in e["archs"]:
            n = float(e["archs"][arm]["N"])
        else:
            # An arm the matched table does not model takes the rung's target.
            # Right for attn-qknorm, whose qk_norm adds 2 x head_dim per layer
            # (1,536 at 47M, 0.003%) and which is otherwise attention. Said out
            # loud because a MISTYPED arm lands here too, and would otherwise
            # be fitted at the wrong N without a word.
            n = float(e["target_N"])
            if arm not in _WARNED:
                _WARNED.add(arm)
                # EACH RUNG'S OWN target, not this one's -- the message is
                # printed once, on whichever row came first, and naming only
                # that rung read as if every row were fitted at its N.
                print(f"note: {arm!r} is not in the matched table; fitting "
                      f"every rung at its own target N (e.g. {n:,.0f} "
                      f"at {size})", file=sys.stderr)
        rows.append(dict(arm=arm, size=size, bt=int(bt), N=n, D=float(tokens),
                         L=float(loss), chain=sched == "chain"))
    return rows


def chinchilla(N, D, E, A, alpha, B, beta, **_):
    return E + A * np.asarray(N, float) ** -alpha + B * np.asarray(D, float) ** -beta


def skaling(N, D, E, A, alpha, B, beta, k, **_):
    return E + (A * np.asarray(N, float) ** -alpha
                + B * np.asarray(D, float) ** -beta) ** k


def fit_separable(rows):
    """Chinchilla per arm, by grid over (alpha, beta) with a linear solve inside.

    With the exponents fixed the model is linear in E, A and B, so this finds the
    global optimum of the grid rather than whatever a nonlinear solver reaches
    from a guessed start -- which matters when the parameters are as correlated
    as these.
    """
    N = np.array([r["N"] for r in rows])
    D = np.array([r["D"] for r in rows])
    L = np.array([r["L"] for r in rows])
    ch = np.array([float(r.get("chain", False)) for r in rows])
    cols = [np.ones_like(L)] if not ch.any() else [np.ones_like(L), ch]
    best = None
    for a in np.arange(0.05, 1.505, 0.005):
        X1 = N ** -a
        for b in np.arange(0.05, 1.505, 0.005):
            X = np.column_stack(cols + [X1, D ** -b])
            coef, *_ = np.linalg.lstsq(X, L, rcond=None)
            resid = L - X @ coef
            sse = float(resid @ resid)
            if best is None or sse < best["sse"]:
                best = dict(alpha=float(a), beta=float(b), E=float(coef[0]),
                            delta=float(coef[1]) if ch.any() else 0.0,
                            A=float(coef[-2]), B=float(coef[-1]), sse=sse,
                            rmse=float(np.sqrt(sse / len(L))))
    return best


def _sci(x, sig=4):
    """A number a LaTeX table can show: 5.249e+05 is not typeset, it is leaked."""
    if x == 0:
        return "0"
    import math
    e = int(math.floor(math.log10(abs(x))))
    if -2 <= e <= 3:
        out = f"{x:.{max(0, sig - 1 - e)}f}"
        # Strip trailing zeros ONLY after a decimal point: "4630".rstrip("0") is
        # 463, which is how a paper table loses a significant digit silently.
        return out.rstrip("0").rstrip(".") if "." in out else out
    return rf"${x / 10 ** e:.{sig - 1}f}{{\times}}10^{{{e}}}$"


def fit_grid(N, D, L, alphas=None, betas=None, chain=None):
    """Chinchilla by grid, with the whole grid solved at once.

    The columns are SCALED to O(1) before the normal equations: N^-alpha is
    ~1e-4 and D^-beta ~1e-7 against an intercept of 1, which conditions X'X at
    ~1e14 and silently moves the argmin -- an unscaled version of this put beta
    at 0.48 where the reference lstsq fit says 0.645.
    """
    alphas = np.arange(0.05, 1.501, 0.005) if alphas is None else alphas
    betas = np.arange(0.05, 1.501, 0.005) if betas is None else betas
    na, nb, n = len(alphas), len(betas), len(N)
    # A FOURTH COLUMN when the table mixes schedules: 1 on a cooldown branched
    # off a shared stable trunk, 0 on a run that warmed, held and decayed by
    # itself. The chained cells are all at the LONG budgets, so any systematic
    # between the two schedules goes straight into the D direction and steepens
    # beta -- the one exponent the extra rungs were bought to pin. It is an
    # indicator, already O(1), so it needs none of the scaling the power
    # columns do.
    #
    # WHETHER THERE IS A SYSTEMATIC AT ALL IS NOT SETTLED HERE. On torchtitan
    # the chained form measured 0.004-0.016 nats BELOW single-stage, which was
    # very likely that framework's in-order reader rather than the schedule.
    # This fit recovers a small offset of the OPPOSITE sign, and cannot
    # separate it from the rung split it is collinear with. The direct
    # measurement on this backend is `ladder_chaincheck` in
    # megatron_ext/submit_ladder.py, which has not been run -- so the column is
    # carried to keep the systematic out of beta, and is not read as a result.
    ch = None if chain is None or not np.any(chain) else np.asarray(chain, float)
    ncol = 3 + (ch is not None)
    u = N[None, :] ** -alphas[:, None]
    v = D[None, :] ** -betas[:, None]
    su, sv = u.mean(1, keepdims=True), v.mean(1, keepdims=True)
    u, v = u / su, v / sv
    X = np.empty((na, nb, n, ncol))
    X[..., 0] = 1.0
    X[..., 1] = u[:, None, :]
    X[..., 2] = v[None, :, :]
    if ch is not None:
        X[..., 3] = ch
    XtX = np.einsum("abni,abnj->abij", X, X)
    XtL = np.einsum("abni,n->abi", X, L)
    coef = np.linalg.solve(XtX, XtL[..., None])[..., 0]
    pred = np.einsum("abni,abi->abn", X, coef)
    sse = ((pred - L) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    return (dict(alpha=float(alphas[i]), beta=float(betas[j]), E=float(coef[i, j, 0]),
                 A=float(coef[i, j, 1] / su[i, 0]), B=float(coef[i, j, 2] / sv[j, 0]),
                 delta=float(coef[i, j, 3]) if ch is not None else 0.0,
                 sse=float(sse[i, j]), rmse=float(np.sqrt(sse[i, j] / n))),
            pred[i, j])


def bootstrap(N, D, L, fitted, reps=400, seed=0, chain=None):
    """Residual bootstrap: 95% intervals with the design held fixed.

    Resampling CELLS is wrong for a single arm: a draw that loses a whole rung
    leaves E, A and B unidentifiable and sends the intervals to the grid
    bounds. The grid was chosen; the randomness lives in the residuals.
    `bootstrap_arms` below resamples cells instead, which it can afford because
    it fits every arm at once and a draw there loses individual cells rather
    than a rung.
    """
    rng = np.random.default_rng(seed)
    out = {"alpha": [], "beta": [], "E": [], "A": [], "B": [], "delta": []}
    resid = L - fitted
    for _ in range(reps):
        f, _ = fit_grid(N, D, fitted + resid[rng.integers(0, len(L), len(L))],
                        chain=chain)
        for k in out:
            out[k].append(f[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in out.items()}


def emit_latex(rows, arms, out_path, reps=400, cells_only=False):
    """Three tables: per-cell losses, per-arm fits with intervals, joint fits.

    `cells_only` emits the FIRST table alone and fits nothing. That table is a
    transcription of the harvest -- no exponents, no bootstrap -- so it costs
    milliseconds where the other two cost twenty minutes of resampling, and it
    is the one worth refreshing every time a cell lands.
    """
    EOL = " \\\\"
    cells = sorted({(r["size"], r["bt"]) for r in rows},
                   key=lambda c: (RUNGS.index(c[0]), c[1]))
    # Which cells came off a shared stable trunk, so the table can say so
    # rather than presenting two schedules as one measurement.
    chained = {(r["size"], r["bt"]) for r in rows if r["chain"]}
    by = {a: [r for r in rows if r["arm"] == a] for a in arms}
    fits, cis = {}, {}
    for a in ([] if cells_only else arms):
        N = np.array([r["N"] for r in by[a]])
        D = np.array([r["D"] for r in by[a]])
        L = np.array([r["L"] for r in by[a]])
        C = np.array([r["chain"] for r in by[a]])
        fits[a], fitted = fit_grid(N, D, L, chain=C)
        cis[a] = bootstrap(N, D, L, fitted, reps=reps, chain=C)

    out = []
    out.append("% Generated by lm_scaling/scaling_fit.py --latex -- regenerate, do not edit.")
    out.append(f"%   lm_scaling/scaling_fit.py --latex --reps {reps}")
    out.append(f"% {len(rows)} annealed endpoints, {len(arms)} arms, "
               f"N {min(r['N'] for r in rows):,.0f}-{max(r['N'] for r in rows):,.0f}, "
               f"D {min(r['D'] for r in rows) / 1e9:.0f}"
               f"-{max(r['D'] for r in rows) / 1e9:.0f}BT.")
    out.append("%")
    out.append(r"\begin{table}[t]")
    out.append(r"  \centering")
    out.append(r"  \small")
    out.append(r"  \caption{Validation loss at every cell of the ladder: annealed "
               r"endpoints on our held-out Nemotron-CC split. Best per row in bold. "
               r"These compare the arms with each other and with no published table. "
               r"$\dagger$ marks a cooldown branched off a shared stable trunk rather "
               r"than a run that warmed, held and decayed on its own. Every fit below "
               r"carries a free offset for that distinction, because the chained cells "
               r"are exactly the long-budget ones and an unmodelled difference between "
               r"the two schedules would be absorbed into $\beta$; the offset is small "
               r"and is not separately identified, and no measurement on this backend "
               r"establishes that the schedules differ at all.}")
    out.append(r"  \label{tab:ladder-cells}")
    out.append(r"  \begin{tabular}{l" + "r" * len(arms) + "}")
    out.append(r"    \toprule")
    out.append("    Cell & " + " & ".join(LABEL.get(a, a) for a in arms) + EOL)
    out.append(r"    \midrule")
    for size, bt in cells:
        vals = {a: next((r["L"] for r in by[a] if r["size"] == size and r["bt"] == bt),
                        None) for a in arms}
        present = [v for v in vals.values() if v is not None]
        best = min(present) if present else None
        cs = []
        for a in arms:
            v = vals[a]
            if v is None:
                cs.append("--")
            elif v == best:
                cs.append(rf"\textbf{{{v:.4f}}}")
            else:
                cs.append(f"{v:.4f}")
        tag = r"$^{\dagger}$" if (size, bt) in chained else ""
        out.append(f"    {size}/{bt:d}BT{tag} & " + " & ".join(cs) + EOL)
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    if cells_only:
        out_path.write_text("\n".join(out) + "\n")
        return fits, cis
    out.append("")
    out.append("%")
    out.append(r"\begin{table}[t]")
    out.append(r"  \centering")
    out.append(r"  \small")
    out.append(r"  \setlength{\tabcolsep}{4pt}")
    dlev = max(r["D"] for r in rows) / min(r["D"] for r in rows)
    ncell = ", ".join(f"{len(by[a])}" for a in arms)
    out.append(r"  \caption{Chinchilla fit $L(N,D)=E+AN^{-\alpha}+BD^{-\beta}$, one fit "
               rf"per arm on its own cells ({ncell} respectively, in column order -- the "
               r"arms do not share a grid, because only the dense baseline and the signed "
               r"arms were run at the upper rungs). Brackets are 95\% intervals from "
               rf"{reps} residual bootstrap resamples over a data lever of "
               rf"${dlev:.1f}\times$. $\alpha$ is the exponent this ladder pins; $\beta$ "
               r"and $E$ still trade against $A$ and $B$, and where an arm's cells are "
               r"few or lopsided the interval says so.}")
    out.append(r"  \label{tab:scaling-fit-per-arm}")
    out.append(r"  \begin{tabular}{lcccrrr}")
    out.append(r"    \toprule")
    out.append(r"    Model & $\alpha$ & $\beta$ & $E$ & $A$ & $B$ & RMSE" + EOL)
    out.append(r"    \midrule")
    for a in arms:
        f, c = fits[a], cis[a]
        out.append(f"    {LABEL.get(a, a)} & "
                   f"{f['alpha']:.3f} [{c['alpha'][0]:.2f}, {c['alpha'][1]:.2f}] & "
                   f"{f['beta']:.3f} [{c['beta'][0]:.2f}, {c['beta'][1]:.2f}] & "
                   f"{f['E']:.3f} [{c['E'][0]:.2f}, {c['E'][1]:.2f}] "
                   f"& {_sci(f['A'])} & {_sci(f['B'])} & {f['rmse']:.4f}{EOL}")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out.append("")
    out.append("%")
    out.append(r"\begin{table}[t]")
    out.append(r"  \centering")
    out.append(r"  \small")
    out.append(r"  \caption{One shape shared across arms with a per-arm offset, which is "
               rf"what {len(rows)} cells support. Reference rows are the paper this ladder "
               r"follows (arXiv:2608.28308, Table 3), fitted to 1.7B and 300BT on the same "
               r"corpus and tokenizer. $\delta_{\mathrm{chain}}$ is carried but NOT "
               r"identified: every upper-rung cell is chained and every lower-rung cell is "
               r"single-stage, so the column is collinear with the rung split and trades "
               r"directly against $\alpha$. Dropping it moves $\alpha$ by 0.010 and "
               r"$\beta$ by 0.010, both inside their intervals.}")
    out.append(r"  \label{tab:scaling-fit-joint}")
    out.append(r"  \begin{tabular}{lrrrrr}")
    out.append(r"    \toprule")
    out.append(r"    Fit & $\alpha$ & $\beta$ & $k$ & $A$ & $B$" + EOL)
    out.append(r"    \midrule")
    joint = {}
    for form, label in (("chinchilla", "ours, Chinchilla"), ("skaling", "ours, Skaling")):
        shape, offs, rmse, delta = fit_joint(rows, arms, form)
        joint[form] = (shape, offs, rmse, delta)
        k = f"{shape['k']:.3f}" if "k" in shape else "--"
        out.append(f"    {label} & {shape['alpha']:.3f} & {shape['beta']:.3f} & {k} & "
                   f"{_sci(shape['A'])} & {_sci(shape['B'])}{EOL}")
    for form, label in (("chinchilla", "paper, Chinchilla"), ("skaling", "paper, Skaling")):
        p = PAPER[form]
        k = f"{p['k']:.3f}" if "k" in p else "--"
        out.append(f"    {label} & {p['alpha']:.3f} & {p['beta']:.3f} & {k} & "
                   f"{_sci(p['A'])} & {_sci(p['B'])}{EOL}")
    out.append(r"    \midrule")
    out.append(r"    \multicolumn{6}{l}{\emph{per-arm offset $E$, shared Skaling shape}}"
               + EOL)
    shape, offs, rmse, delta = joint["skaling"]
    base = min(offs.values())
    for a in arms:
        out.append(rf"    {LABEL.get(a, a)} & \multicolumn{{5}}{{l}}{{$E={offs[a]:.4f}$ "
                   rf"\quad ({offs[a] - base:+.4f} vs best)}}{EOL}")
    if delta:
        out.append(r"    \midrule")
        out.append(rf"    \multicolumn{{6}}{{l}}{{$\delta_{{\mathrm{{chain}}}}={delta:+.4f}$ -- one free "
                   rf"offset for a cooldown branched off a shared stable trunk}}{EOL}")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out) + "\n"
    out_path.write_text(text)

    # Each table ALSO on its own, so a paper can \input one without the other
    # two. Same generated content, split on the environment boundaries rather
    # than re-derived -- two copies of a fitted number that can disagree is
    # exactly the failure this project keeps finding.
    head = text.split(r"\begin{table}")[0]
    blocks = [r"\begin{table}" + b.split(r"\end{table}")[0] + "\\end{table}\n"
              for b in text.split(r"\begin{table}")[1:]]
    for name, block in zip(("cells", "params", "joint"), blocks):
        (out_path.parent / f"{out_path.stem}_{name}.tex").write_text(head + block)
    print(f"  split into {len(blocks)} standalone tables: "
          + ", ".join(f"{out_path.stem}_{n}.tex" for n in ("cells", "params", "joint")[:len(blocks)]))
    return fits, cis


def fit_joint(rows, arms, form):
    """One shared shape, one offset per arm. Huber, many starts, as the paper."""
    from scipy.optimize import least_squares

    N = np.array([r["N"] for r in rows])
    D = np.array([r["D"] for r in rows])
    L = np.array([r["L"] for r in rows])
    idx = np.array([arms.index(r["arm"]) for r in rows])
    # One extra free parameter for the schedule, not one per arm: the offset is
    # a property of the recipe, and every arm at a rung was chained the same way.
    ch = np.array([float(r["chain"]) for r in rows])
    nE = len(arms)
    nD = int(ch.any())

    def predict(theta):
        off = theta[idx] + (ch * theta[nE] if nD else 0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if form == "chinchilla":
                A, alpha, B, beta = theta[nE + nD:]
                return off + A * N ** -alpha + B * D ** -beta
            A, alpha, B, beta, k = theta[nE + nD:]
            return off + (A * N ** -alpha + B * D ** -beta) ** k

    rng = np.random.default_rng(0)
    best = None
    for _ in range(400):
        shape = ([10 ** rng.uniform(1, 5), rng.uniform(0.1, 0.8),
                  10 ** rng.uniform(2, 6), rng.uniform(0.1, 0.8)]
                 if form == "chinchilla" else
                 [10 ** rng.uniform(2, 6), rng.uniform(0.2, 0.9),
                  10 ** rng.uniform(2, 6), rng.uniform(0.1, 0.9),
                  rng.uniform(0.2, 0.9)])
        t0 = np.concatenate([np.full(nE, rng.uniform(0.8, 2.2)),
                             np.full(nD, -0.01), shape])
        try:
            r = least_squares(lambda th: predict(th) - L, t0, loss="huber",
                              f_scale=0.01, max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    pred = predict(best.x)
    names = (["A", "alpha", "B", "beta"] if form == "chinchilla"
             else ["A", "alpha", "B", "beta", "k"])
    return (dict(zip(names, best.x[nE + nD:])),
            {arm: float(best.x[i]) for i, arm in enumerate(arms)},
            float(np.sqrt(((pred - L) ** 2).mean())),
            float(best.x[nE]) if nD else 0.0)


def fit_shared_E(rows, arms, form, per_arm="A", E_fixed=None, starts=600,
                 x0=None, seed=0):
    """ONE E for every arm; whichever other parameters you name vary per arm.

    `E` is the entropy of the data -- what no model of any architecture can
    remove -- so letting it vary per arm asks the fit to attribute a property
    of the CORPUS to the mixer. Here it is shared, and the architecture shows
    up where it belongs, in the reducible term.

    `per_arm` names the parameters that get one value per arm; the rest are
    shared. It accepts a string of letters or a sequence of names:

        "A"      capacity differs                A_arm N^-alpha + B D^-beta
        "B"      data efficiency differs         A N^-alpha + B_arm D^-beta
        "AB"     both prefactors
        "all"    every parameter but E -- each arm gets its own exponents too,
                 which asks whether the arms SCALE differently rather than
                 merely sitting at different prefactors

    `E_fixed` pins E rather than fitting it, and the reference's own E is the
    value to pin: same corpus, same held-out split, fitted over a ladder
    reaching 1.7B and 300BT. E is the worst-determined parameter here and
    trades directly against A and B, so pinning it leaves the exponents to the
    data rather than to slack in the asymptote.
    """
    from scipy.optimize import least_squares

    names = ["A", "alpha", "B", "beta"] + (["k"] if form == "skaling" else [])
    # MATCH ON THE NAME, NOT ITS FIRST LETTER. `n[0].upper() in per_arm` reads
    # naturally and is wrong: "A" also selects "alpha" and "B" also selects
    # "beta", so per_arm="AB" silently freed every exponent as well and the
    # A/B decomposition it reported was not the one the docstring describes.
    # It showed up as a 3x spread in A where the fixed-exponent fit gives 5%.
    LETTER = {"A": "A", "B": "B", "K": "k"}
    if per_arm == "all":
        varies = set(names)
    elif isinstance(per_arm, str):
        unknown = [c for c in per_arm if c.upper() not in LETTER]
        if unknown:
            raise ValueError(f"per_arm {per_arm!r}: {unknown} name no "
                             f"prefactor; pass a sequence such as "
                             f"('A', 'alpha') to free an exponent")
        varies = {LETTER[c.upper()] for c in per_arm} & set(names)
    else:
        varies = set(per_arm)
        if not varies <= set(names):
            raise ValueError(f"per_arm {sorted(varies - set(names))} not in "
                             f"{names}")

    N = np.array([r["N"] for r in rows])
    D = np.array([r["D"] for r in rows])
    L = np.array([r["L"] for r in rows])
    idx = np.array([arms.index(r["arm"]) for r in rows])
    ch = np.array([float(r["chain"]) for r in rows])
    nE, nD = (0 if E_fixed is not None else 1), int(ch.any())

    # theta = [E?] [dchain?] then each name's block, shared or per-arm
    at, width = {}, {}
    pos = nE + nD
    for n in names:
        width[n] = len(arms) if n in varies else 1
        at[n] = pos
        pos += width[n]
    zeros = np.zeros_like(idx)

    def take(theta, n):
        block = theta[at[n]:at[n] + width[n]]
        return block[idx if width[n] > 1 else zeros]

    def predict(theta):
        base = E_fixed if E_fixed is not None else theta[0]
        E = base + (ch * theta[nE] if nD else 0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            red = (take(theta, "A") * N ** -take(theta, "alpha")
                   + take(theta, "B") * D ** -take(theta, "beta"))
            return E + (red if form == "chinchilla" else red ** take(theta, "k"))

    def guess(rng, n):
        if n == "A":
            return 10 ** rng.uniform(1, 5, width[n])
        if n == "B":
            return 10 ** rng.uniform(2, 6, width[n])
        if n == "k":
            return rng.uniform(0.2, 0.9, width[n])
        return rng.uniform(0.1, 0.8, width[n])

    rng = np.random.default_rng(seed)
    best = None
    # A WARM START, for the bootstrap. Refitting 200 resamples from 600 random
    # starts each is 120,000 optimisations; from the full-data optimum it is
    # 200 plus a handful of restarts, and a resample's optimum is near the
    # one it was resampled from. Cold fits pass x0=None and are unaffected.
    if x0 is not None:
        try:
            best = least_squares(lambda th: predict(th) - L, np.asarray(x0),
                                 loss="huber", f_scale=0.01, max_nfev=8000)
        except Exception:
            best = None
    for _ in range(starts):
        t0 = np.concatenate([np.full(nE, rng.uniform(0.8, 2.2)),
                             np.full(nD, -0.01)]
                            + [guess(rng, n) for n in names])
        try:
            r = least_squares(lambda th: predict(th) - L, t0, loss="huber",
                              f_scale=0.01, max_nfev=8000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    t = best.x
    # Per-arm RMSE, so a joint fit can say which arm it fits badly. The
    # pooled number cannot: it is one value repeated down the table.
    resid = predict(t) - L
    by_arm = {a: float(np.sqrt((resid[idx == i] ** 2).mean()))
              for i, a in enumerate(arms) if (idx == i).any()}
    out = dict(E=float(E_fixed) if E_fixed is not None else float(t[0]),
               E_fixed=E_fixed is not None, theta=t.copy(), rmse_by_arm=by_arm,
               rmse=float(np.sqrt(((predict(t) - L) ** 2).mean())),
               dchain=float(t[nE]) if nD else 0.0,
               n_params=len(t), per_arm=sorted(varies))
    for n in names:
        block = t[at[n]:at[n] + width[n]]
        out[n] = ({a: float(block[i]) for i, a in enumerate(arms)}
                  if width[n] > 1 else float(block[0]))
    return out


def fit_isolated(rows, arms, form, starts=600):
    """Each arm fitted entirely on its own: its own E, exponents and offset.

    The honest baseline for "does sharing E help?", and the weakest of the
    three: an arm contributes ~25 cells spanning 36x in N and 8x in D, and E
    is not identifiable over that lever. It trades against A and B, and a
    too-large E buys itself steeper exponents, so the isolated E values scatter
    far more than the corpus they were all trained on can justify. Reported
    because that scatter IS the argument for the other two fits.
    """
    out = {}
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        out[a] = fit_shared_E(sub, [a], form, per_arm="all", starts=starts)
    return out


def _resample(rows, arms, rng):
    """Resample rows with replacement, WITHIN each arm.

    Resampling the pooled rows would let an arm lose most of its cells, or all
    of them, and an arm with no data has no parameters to report. Stratifying
    keeps every arm at its own count, so each bootstrap replicate has the same
    design as the data.
    """
    out = []
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        idx = rng.integers(0, len(sub), len(sub))
        out.extend(sub[i] for i in idx)
    return out


def bootstrap_arms(rows, arms, form, per_arm="all", E_fixed=None, reps=200,
                   starts=6, seed=0, isolated=False, point_starts=600):
    """Percentile intervals for every fitted parameter of a multi-arm fit.

    NOT `bootstrap` above, and the difference is the design. That one holds a
    small fixed grid and resamples the RESIDUALS, because a draw that misses
    a rung leaves E, A and B unidentifiable. Here each arm brings thirty cells
    over six rungs and five budgets, so a draw can
    lose individual cells without losing a rung, and resampling cells captures
    what the residual scheme cannot: that a cell's loss is one noisy draw and
    the ladder could have come out differently.

    There is no analytic covariance worth quoting here: the loss is Huber, not
    Gaussian, the model is multi-modal, and the residuals are not independent
    across budgets sharing a trunk. Resampling asks the question directly --
    how much would this number move on another draw of the same ladder?

    Returns `{name: (lo, hi)}` for shared parameters and
    `{name: {arm: (lo, hi)}}` for per-arm ones, at the 2.5th and 97.5th
    percentiles of `reps` replicates.
    """
    # The point estimate is a COLD fit -- it is the answer being reported, and
    # every replicate is warm-started from it, so it is the one place the full
    # multi-start search has to run.
    point = (fit_isolated(rows, arms, form, starts=point_starts) if isolated
             else fit_shared_E(rows, arms, form, per_arm=per_arm,
                               E_fixed=E_fixed, starts=point_starts))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        rs = _resample(rows, arms, rng)
        try:
            if isolated:
                f = {}
                for a in arms:
                    sub = [r for r in rs if r["arm"] == a]
                    f[a] = fit_shared_E(sub, [a], form, per_arm="all",
                                        starts=starts, x0=point[a]["theta"])
            else:
                f = fit_shared_E(rs, arms, form, per_arm=per_arm,
                                 E_fixed=E_fixed, starts=starts,
                                 x0=point["theta"])
        except Exception:
            continue
        draws.append(f)

    def ci(vals):
        v = np.sort(np.asarray(vals, float))
        if not len(v):
            return (float("nan"), float("nan"))
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

    out = {}
    if isolated:
        keys = [k for k in point[arms[0]] if isinstance(point[arms[0]][k], float)]
        for k in keys:
            out[k] = {a: ci([d[a][k] for d in draws if a in d]) for a in arms}
        return point, out, len(draws)
    for k, v in point.items():
        if isinstance(v, dict):
            out[k] = {a: ci([d[k][a] for d in draws]) for a in arms}
        elif isinstance(v, float):
            out[k] = ci([d[k] for d in draws])
    return point, out, len(draws)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=LADDER_TSV)
    ap.add_argument("--latex", nargs="?", const=str(TEX_DIR / "scaling_fit.tex"),
                    default=None, metavar="PATH",
                    help="write the tables as LaTeX and exit")
    ap.add_argument("--cells-only", action="store_true",
                    help="write ONLY the per-cell loss table, fitting nothing. "
                         "Seconds rather than twenty minutes, and the right "
                         "thing to run when a new cell lands")
    ap.add_argument("--reps", type=int, default=400,
                    help="residual bootstrap replicates behind the intervals")
    ap.add_argument("--loo", action="store_true",
                    help="leave-one-out ranges, which is where the honesty is")
    a = ap.parse_args(argv)

    rows = load(a.data)
    arms = [x for x in LABEL if any(r["arm"] == x for r in rows)] or \
        sorted({r["arm"] for r in rows})
    # The harvest is a record of what RAN; this is a record of what the study
    # compares. Abandoned arms (the plateau round, stopped after a cell or two)
    # are in the table on purpose and would otherwise join the joint fit with
    # one point each and no way to identify an offset.
    dropped = sorted({r["arm"] for r in rows} - set(arms))
    if dropped:
        rows = [r for r in rows if r["arm"] in arms]
        print(f"ignoring {len(dropped)} arm(s) outside the comparison: "
              + ", ".join(dropped))
    if a.latex:
        fits, cis = emit_latex(rows, arms, Path(a.latex), reps=a.reps,
                               cells_only=a.cells_only)
        print(f"wrote {a.latex}")
        if a.cells_only:
            return 0
        for arm in arms:
            f, c = fits[arm], cis[arm]
            print(f"  {arm:30s} alpha {f['alpha']:.3f} [{c['alpha'][0]:.2f},{c['alpha'][1]:.2f}]"
                  f"  beta {f['beta']:.3f} [{c['beta'][0]:.2f},{c['beta'][1]:.2f}]"
                  f"  E {f['E']:.3f}  rmse {f['rmse']:.4f}")
        return 0
    print(f"{len(rows)} cells, {len(arms)} arms, N {min(r['N'] for r in rows):,.0f}"
          f"-{max(r['N'] for r in rows):,.0f} ({max(r['N'] for r in rows) / min(r['N'] for r in rows):.1f}x), "
          f"D {min(r['D'] for r in rows) / 1e9:.0f}-{max(r['D'] for r in rows) / 1e9:.0f}BT "
          f"({max(r['D'] for r in rows) / min(r['D'] for r in rows):.1f}x)\n")

    ref = next((a for a in REFERENCE_ARMS if any(r["arm"] == a for r in rows)),
               None)
    print(f"=== does the paper's own law predict our {ref} arm? ===")
    attn = sorted([r for r in rows if r["arm"] == ref], key=lambda r: (r["N"], r["D"]))
    for name, fn in (("chinchilla", chinchilla), ("skaling", skaling)):
        p = {k: v for k, v in PAPER[name].items() if k != "se"}
        d = np.array([r["L"] - fn(r["N"], r["D"], **p) for r in attn])
        print(f"  {name:11s} mean offset {d.mean():+.4f}  rmse {np.sqrt((d ** 2).mean()):.4f}"
              f"  spread across cells {d.max() - d.min():.4f}")
    print("  A CONSTANT offset with a small spread means our runs reproduce the")
    print("  law's SHAPE and sit on a different intercept.")
    if ref == "attn-qknorm":
        print("  The remaining intercept is beta2 0.95 against their 0.99 and no")
        print("  biases against their biases: 0.0240 at 47M/6BT, measured.")
    else:
        print("  attn-qknorm was not in this table, so the comparison is against")
        print("  the UN-NORMED arm and carries QK-norm's 0.0384 as well.")
    print()

    print(f"=== per-arm Chinchilla fits ({len(rows)} cells over {len(arms)} arms) ===")
    print(f"{'arm':30s} {'alpha':>7s} {'beta':>7s} {'E':>8s} {'A':>11s} {'B':>11s}"
          f" {'dchain':>8s} {'rmse':>8s}")
    for arm in arms:
        f = fit_separable([r for r in rows if r["arm"] == arm])
        print(f"{arm:30s} {f['alpha']:7.3f} {f['beta']:7.3f} {f['E']:8.4f} "
              f"{f['A']:11.4g} {f['B']:11.4g} {f['delta']:+8.4f} {f['rmse']:8.5f}")
    if a.loo:
        print("\n=== leave-one-out ranges (what the cells actually pin) ===")
        for arm in arms:
            sub = [r for r in rows if r["arm"] == arm]
            fs = [fit_separable([r for j, r in enumerate(sub) if j != i])
                  for i in range(len(sub))]
            print(f"{arm:30s} alpha {min(f['alpha'] for f in fs):.3f}-"
                  f"{max(f['alpha'] for f in fs):.3f}  beta "
                  f"{min(f['beta'] for f in fs):.3f}-{max(f['beta'] for f in fs):.3f}"
                  f"  E {min(f['E'] for f in fs):.3f}-{max(f['E'] for f in fs):.3f}")

    for form in ("chinchilla", "skaling"):
        shape, offsets, rmse, delta = fit_joint(rows, arms, form)
        p = PAPER[form]
        print(f"\n=== joint {form} fit: one shape, one offset per arm ({len(rows)} cells) ===")
        print("  ours : " + "  ".join(f"{k} {v:.4g}" for k, v in shape.items())
              + f"   rmse {rmse:.5f}")
        print("  paper: " + "  ".join(
            f"{k} {p[k]:.4g}" for k in shape if k in p)
            + f"   (E {p['E']:.3g} +- {p['se']['E']:.3g})")
        base = min(offsets.values())
        for arm in arms:
            print(f"    E[{arm:30s}] {offsets[arm]:+.4f}   "
                  f"{offsets[arm] - base:+.4f} vs best")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
