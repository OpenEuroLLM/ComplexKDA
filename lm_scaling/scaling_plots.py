"""Chinchilla-style figures: loss against compute, and the compute-optimal front.

    lm_scaling/scaling_plots.py --out lm_scaling/tex

WHAT IS PLOTTED, and what is deliberately not. Every point is an ANNEALED
ENDPOINT -- a run that warmed up, held and decayed to its own budget. The
intermediate validations of a WSD run are NOT endpoints: they sit at constant
learning rate, above where the same run lands after its cooldown, and plotting
them as a training curve is the classic way to draw a scaling plot that is
optimistic by a tenth of a nat. Chinchilla's own Approach 1 carries the same
caveat for cosine schedules.

So the "curves" here are what this ladder actually measures: for each (arm,
rung), five annealed endpoints at 6, 12, 20, 30 and 50BT, joined to show the
direction of more data at fixed capacity. The lower envelope across rungs is the
empirical compute-optimal front, and the fitted law's front is drawn over it.

THE FIT is the Skaling form, L = E + (A N^-alpha + B D^-beta)^k, with one shape
shared across arms and a per-arm offset -- which is what 180 cells support.

HOW MUCH E IS WORTH. E is not identifiable over this ladder's lever, so the two
fits that bracket it are reported: every arm free to choose its own, and E
pinned to the reference's value. In sample the choice barely moves the
residuals, which is the point -- what separates the two is PREDICTION, not
description. So the reported fits hold the ladder's largest cell out and score
the error there: `scaling_holdout.py`, and `tex/scaling_holdout.tex` for the
summary over both treatments and both functional forms.

The separable Chinchilla form is drawn here for comparison where it helps, but
the paper this ladder follows finds it extrapolates badly and so do we. Theirs
is the cleaner statement of it, since they hold out a whole rung: held out at
1.7B, Chinchilla misses by 0.0174 where Skaling misses by 0.0056.
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

import plot_style  # noqa: E402  (needs the sys.path bootstrap above)

#: The Megatron ladder, 180 annealed endpoints. NOT ladder_full.tsv: that is
#: the torchtitan campaign this study superseded, and defaulting to it means a
#: regeneration silently redraws every figure from outdated numbers.
LADDER = _HERE / "harvest" / "megatron_ladder.tsv"

#: Per-step validation trajectories -- torchtitan only, and there is no
#: Megatron equivalent to switch to. The Megatron cells run eval_interval 10^6
#: and validate ONCE, at their annealed endpoint, so a run contributes a point
#: and not a curve. The training-curves figure is therefore drawn only when
#: --curves is passed explicitly, so that it cannot quietly mix a torchtitan
#: trajectory into a Megatron figure set.
CURVES = None
# Per-step validation curves, if you have them: `harvest_runs.py --curves`
# writes them from a run directory. Not committed -- they are ~70 KB a
# campaign and only the ENDPOINTS are what the fit uses.
CURVES_HINT = "harvest/<campaign>_val_curves.tsv"
# Paper-facing names and a stable colour per arm.
LABEL = {"attn": "Transformer++", "attn-qknorm": "Transformer++ (QK-norm)",
         "gdn": "Gated DeltaNet",
         "kda-sig-lowrank": "KDA", "ckda-shipped-lowrank": "CKDA",
         "kda-sig-hybrid-lowrank": "KDA + attn 3:1",
         "ckda-shipped-hybrid-lowrank": "CKDA + attn 3:1"}
# attn-qknorm was not in this map: the script was written for the torchtitan
# ladder, whose arms included gdn and no normed attention. It is the baseline
# the advantage figures now subtract, so it needs both a colour and a label.
COLOR = {"attn": "#999999", "attn-qknorm": "#444444", "gdn": "#8c6d31",
         "kda-sig-lowrank": "#1f77b4", "ckda-shipped-lowrank": "#d62728",
         "kda-sig-hybrid-lowrank": "#2ca02c", "ckda-shipped-hybrid-lowrank": "#9467bd"}
#: Arms the COMPUTE figure leaves out -- it alone, not the rest of the set.
#: `attn` is the un-normed transformer, and `attn-qknorm` is the baseline this
#: study compares against: the reference law was fitted to a normed one, and
#: QK-norm is worth 0.0190 to attention, which is larger than the gaps the
#: figure is about. Two grey lines a fifth of a nat apart, only one of which is
#: the comparison, is a reader's first question and a distraction from the
#: answer. Every other figure still draws both.
COMPUTE_SKIP = ("attn",)
RUNGS = ("47M", "124M", "302M", "588M", "983M", "1.7B")
NOISE = 0.0020  # two bit-identical configs have landed this far apart
MARK = {"47M": "o", "124M": "s", "302M": "^", "588M": "D", "983M": "v", "1.7B": "P"}


def load(path: Path):
    import geometries

    t = geometries.table()
    rows = []
    for line in path.read_text().splitlines()[1:]:
        f = line.split("\t")
        arm, size, bt, gbs, lr, steps, tokens, loss = f[:8]
        # Rows written before the medium rungs carry no `schedule`; the lower
        # ladder ran single-stage, which is what that absence means.
        chain = (f[8] if len(f) > 8 else "single") == "chain"
        e = t[size]
        n = float(e["archs"][arm]["N"]) if arm in e["archs"] else float(e["target_N"])
        rows.append(dict(arm=arm, size=size, bt=int(bt), N=n, D=float(tokens),
                         L=float(loss), C=6.0 * n * float(tokens), chain=chain))
    return rows


def load_curves(path: Path, table):
    """Every run's validation trajectory, as (compute, loss) along training.

    The step-1 validation is dropped: it is the untrained model at loss ~10.85
    and would own the y-axis by itself.
    """
    runs = {}
    if not path.exists():
        return runs
    for line in path.read_text().splitlines()[1:]:
        arm, size, bt, gbs, step, tokens, loss = line.split("\t")
        if float(loss) > 5.0:
            continue
        e = table[size]
        n = float(e["archs"][arm]["N"]) if arm in e["archs"] else float(e["target_N"])
        runs.setdefault((arm, size, int(bt)), []).append(
            (6.0 * n * float(tokens), float(loss)))
    for k in runs:
        runs[k].sort()
    return runs


def isolated_fits(rows, arms, data, cache, starts=600, refit=False):
    """Per-arm Skaling fits -- each arm its own E, exponents and k -- cached.

    NOT `fit_joint` below, and the difference is what a front can then say.
    The joint fit gives one shape and a per-arm offset, so `E` drops out of
    the argmin over N: every arm gets the SAME N*(C) and fronts that are
    translations of one another, parallel and unable to cross. Fitting each
    arm alone lets the exponents differ, so its front can bend differently
    and two arms can converge or cross with compute.

    THE COST OF THAT FREEDOM, which the figure inherits: an arm alone brings
    30 cells over a 36x lever in N, and `E` is not identifiable over that. It
    trades against A and B, a too-high E buys steeper exponents, and the
    per-arm E values scatter further than one corpus can justify. A crossing
    between two fronts here is therefore not evidence of a crossing -- read it
    against the bootstrap intervals in tex/scaling_holdout_isolated.tex before
    it becomes a claim.

    CACHED because this is six multi-start fits, minutes, against the ~1 min
    the whole figure set otherwise costs; the key carries the data file, its
    mtime, the arms and the start budget, so a changed ladder refits itself
    and only a matching one is reused. `--refit-iso` forces it.
    """
    import hashlib
    import json

    # KEYED ON CONTENT, NOT MTIME. An mtime key can never match in a clone:
    # git stamps every checked-out file with the checkout time, so a committed
    # cache would be discarded on the first run by everyone but its author,
    # and the figure would cost six 600-start fits instead of reading a file.
    # The digest of the ladder is what the fits actually depend on.
    digest = hashlib.sha256(Path(data).read_bytes()).hexdigest()[:16]
    key = dict(data=Path(data).name, sha256=digest,
               n=len(rows), arms=list(arms), starts=int(starts), form="skaling")
    if cache and Path(cache).exists() and not refit:
        try:
            blob = json.loads(Path(cache).read_text())
            if blob.get("key") == key:
                print(f"isolated fits: reused {cache}")
                return blob["fits"]
        except Exception as e:      # a truncated or hand-edited cache
            print(f"isolated fits: ignoring {cache} ({e})")
    import scaling_fit as SF

    print(f"isolated fits: fitting {len(arms)} arms at {starts} starts ...",
          flush=True)
    fits = SF.fit_isolated(rows, list(arms), "skaling", starts=starts)
    out = {a: {k: v for k, v in f.items() if isinstance(v, (int, float, bool))}
           for a, f in fits.items()}
    if cache:
        Path(cache).write_text(json.dumps(dict(key=key, fits=out), indent=1))
        print(f"isolated fits: wrote {cache}")
    return out


def fit_joint(rows, arms):
    """Skaling with one shared shape and one offset per arm."""
    from scipy.optimize import least_squares

    N = np.array([r["N"] for r in rows])
    D = np.array([r["D"] for r in rows])
    L = np.array([r["L"] for r in rows])
    idx = np.array([arms.index(r["arm"]) for r in rows])
    # One free offset for a cooldown branched off a shared stable trunk. On this
    # ladder it is NOT identified -- every upper-rung cell is chained and every
    # lower-rung cell is single-stage, so the column is collinear with the rung
    # split and trades directly against alpha. It is here so the fit does not
    # silently attribute the schedule to the architecture; it is not a
    # measurement of the schedule. See scaling_fit.py for the sensitivity.
    ch = np.array([float(r["chain"]) for r in rows])
    nE, nD = len(arms), int(ch.any())

    def predict(th):
        with np.errstate(over="ignore", invalid="ignore"):
            A, al, B, be, k = th[nE + nD:]
            off = th[idx] + (ch * th[nE] if nD else 0.0)
            return off + (A * N ** -al + B * D ** -be) ** k

    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        t0 = np.concatenate([np.full(nE, rng.uniform(1.0, 2.2)), np.full(nD, -0.01),
                             [10 ** rng.uniform(2, 6), rng.uniform(0.2, 0.9),
                              10 ** rng.uniform(2, 7), rng.uniform(0.2, 0.9),
                              rng.uniform(0.2, 0.9)]])
        try:
            r = least_squares(lambda th: predict(th) - L, t0, loss="huber",
                              f_scale=0.01, max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    A, al, B, be, k = best.x[nE + nD:]
    return (dict(A=A, alpha=al, B=B, beta=be, k=k),
            {a: best.x[i] for i, a in enumerate(arms)})


#: Serif axes, to sit in a LaTeX document without the figure text reading as
#: a different typeface from the body. Only the text: lines, colours and the
#: mathtext engine are left alone.
def _style(plt, size=11.5):
    """The package-wide style, plus the size hierarchy these figures need.

    `plot_style.apply` sets one base size and lets everything inherit it. That
    is right for a single-panel figure and wrong for a six-panel row: the row
    is wider than any column it lands in, so it is scaled down on inclusion and
    every point of type goes with it. The ticks in particular carry five budget
    labels under a narrow panel and collide before the axis labels do, so they
    stay below the base size rather than at it.
    """
    fam = plot_style.apply(plt, size=size)
    plt.rcParams.update({
        "axes.labelsize": size,
        "axes.titlesize": size,
        "xtick.labelsize": size - 2.5,
        "ytick.labelsize": size - 2.5,
        "legend.fontsize": size - 1.5,
    })
    return fam


#: The baseline the advantage figures subtract. QK-norm, not plain attention:
#: the reference ladder this study replicates uses it, it is worth ~0.027 nats
#: on its own, and a gap measured against the un-normed arm silently counts
#: that as part of the architecture's advantage.
ADV_BASE = "attn-qknorm"


def advantage(rows, sizes, bts, base=ADV_BASE):
    """Each arm's gap to the dense baseline, in nats and in equivalent data.

    Nats are what we measure; nobody has an intuition for them. The second
    reading is the one a reader can price: how much MORE data the baseline
    would need, at this budget, to reach the loss the arm already has. From
    the baseline's own local return on data at that cell,

        dL/dlnD  (a centred secant over its three budgets at this rung)

    a gap of `d` nats is worth `exp(d / slope)` times the tokens.

    LOCAL, and on purpose. The tempting version fits the baseline's own
    L = e + b D^-beta at the rung and inverts it, which extrapolates past
    20BT and is worthless there: three points fix `e` with no interval at
    all, and at 47M the fitted e = 3.056 sits 0.014 above the hybrid's
    measured 3.070, so the multiplier reads 15.5x. The secant cannot say
    anything about budgets it did not measure, which is the correct amount
    for this grid to say.
    """
    L = {(r["arm"], r["size"], r["bt"]): r["L"] for r in rows}
    D = {(r["arm"], r["size"], r["bt"]): r["D"] for r in rows}
    out = {}
    for size in sizes:
        have = [b for b in bts if (base, size, b) in L]
        if len(have) < 2:
            continue
        lnD = np.log([D[(base, size, b)] for b in have])
        slope = np.gradient(np.array([L[(base, size, b)] for b in have]), lnD)
        for arm in {r["arm"] for r in rows} - {base}:
            for j, b in enumerate(have):
                if (arm, size, b) not in L:
                    continue
                d = L[(arm, size, b)] - L[(base, size, b)]
                # Third reading: the gap as a FRACTION of the baseline loss.
                #
                # Read it with care. L carries the irreducible floor E ~ 1.19,
                # which no architecture can touch, and L falls 2.95 -> 2.14
                # across this grid. So d/L grows with N partly because the
                # denominator shrinks, not because the architecture pulls
                # further ahead: 47M/50B -0.84% against 1.7B/50B -1.89% is a
                # 2.2x growth where the nats grow 1.7x. Dividing by the
                # REDUCIBLE part, d/(L - E), is the principled version and
                # gives 2.9x -- but it inherits E's uncertainty, and at
                # Ajroldi's E = 0.964 rather than our 1.185 the same cell moves
                # from -4.23% to -3.42%. The equivalent-data multiplier beside
                # it is the relative measure that needs no E at all.
                out[(arm, size, b)] = (d, float(np.exp(d / slope[j])),
                                       100.0 * d / L[(base, size, b)])
    return out


def frontier(shape, E, cs):
    """min over N of L(N, C/(6N)) -- the compute-optimal loss at each C."""
    out_L, out_N = [], []
    for c in cs:
        n = np.geomspace(1e7, 1e11, 600)
        d = c / (6.0 * n)
        with np.errstate(over="ignore", invalid="ignore"):
            l = E + (shape["A"] * n ** -shape["alpha"]
                     + shape["B"] * d ** -shape["beta"]) ** shape["k"]
        i = int(np.nanargmin(l))
        out_L.append(l[i])
        out_N.append(n[i])
    return np.array(out_L), np.array(out_N)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Once, here, so EVERY figure gets it -- not just the advantage pair. The
    # helper sets it too, harmlessly, for when it is called on its own.
    _style(plt)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=LADDER)
    ap.add_argument("--out", type=Path, default=_HERE / "tex")
    ap.add_argument("--curves", type=Path, default=CURVES,
                    help="per-step validation trajectories for the training-"
                         "curves figure. Off by default: only the torchtitan "
                         "campaign has them, and mixing its trajectories into "
                         "a Megatron figure set is a provenance error that "
                         "nothing in the filenames would reveal.")
    ap.add_argument("--iso-cache", type=Path,
                    default=_HERE / "tex" / "isolated_fits.json",
                    help="where the per-arm fits behind the compute-optimal "
                         "fronts are cached")
    ap.add_argument("--iso-starts", type=int, default=600,
                    help="multi-starts per arm for those fits")
    ap.add_argument("--refit-iso", action="store_true",
                    help="refit them even if the cache matches")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    rows = load(a.data)
    arms = [x for x in LABEL if any(r["arm"] == x for r in rows)]
    # The harvest records every run; these figures show the comparison. Arms
    # abandoned after a cell or two (the plateau round) have no curve to draw.
    rows = [r for r in rows if r["arm"] in arms]
    shape, E = fit_joint(rows, arms)
    print("joint Skaling fit: " + "  ".join(f"{k} {v:.4g}" for k, v in shape.items()))
    for arm in arms:
        print(f"   E[{arm:30s}] {E[arm]:+.4f}")

    # --- Figure 1: loss against compute, with the fitted front -------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    cs = np.geomspace(min(r["C"] for r in rows) * 0.6,
                      max(r["C"] for r in rows) * 2.5, 120)
    shown = [x for x in arms if x not in COMPUTE_SKIP] or list(arms)
    # THE FRONT FOLLOWS A DRAWN ARM. `best_arm` picked over every fitted arm
    # would label the figure with a curve whose points are not on it.
    best_arm = min(shown, key=lambda x: E[x])
    # ONE FRONT PER ARM, from that arm's OWN fit rather than from the shared
    # shape: under the joint fit E is additive and drops out of the argmin
    # over N, so those fronts are translations of one another and cannot
    # cross whatever the arms do. These can.
    iso = isolated_fits(rows, shown, a.data, a.iso_cache, a.iso_starts,
                        a.refit_iso)
    for arm in shown:
        f = iso[arm]
        print(f"   front[{arm:30s}] E {f['E']:.4f}  alpha {f['alpha']:.3f}  "
              f"beta {f['beta']:.3f}  k {f['k']:.3f}  rmse {f['rmse']:.5f}")
        ax.plot(cs, frontier(f, f["E"], cs)[0], color=COLOR[arm], lw=1.1,
                ls="--", alpha=0.75, zorder=2)
    # Figures 4 and 5 draw the best arm's front and its N*; keep that one.
    fl, fn = frontier(shape, E[best_arm], cs)
    for arm in shown:
        first = True
        for size in RUNGS:
            pts = sorted([r for r in rows if r["arm"] == arm and r["size"] == size],
                         key=lambda r: r["C"])
            if not pts:
                continue
            # EVERY MARKER FILLED. Chained and single-stage cells were drawn
            # apart (hollow for a cooldown branched off a shared trunk) while
            # that distinction was still load-bearing; it is not. The ladder
            # reports one annealed endpoint per cell either way, the two
            # schedules land on the same curve, and a second visual channel
            # here reads as a second result. `chain` still rides on the rows
            # and still separates the fits -- it just no longer marks the plot.
            ax.plot([p["C"] for p in pts], [p["L"] for p in pts],
                    color=COLOR[arm], marker=MARK[size], ms=4.5, lw=1.0, alpha=0.85,
                    label=LABEL[arm] if first else None)
            first = False
    ax.set_xscale("log")
    ax.set_xlabel("Training Compute  C = 6ND  [FLOPs]")
    ax.set_ylabel(r"Validation Loss $\downarrow$")
    # NO TITLE; the caption lives in the document.
    ax.grid(alpha=0.25, which="both")
    # The dashed style gets ONE neutral entry. Five coloured "front, <arm>"
    # rows would double the legend to say the same thing the colours already
    # say, and the reader has to map colour to arm only once.
    _h = ax.get_legend_handles_labels()[0]
    _h.append(plt.Line2D([], [], color="0.35", ls="--", lw=1.1,
                         label="compute-optimal front (fit)"))
    ax.legend(handles=_h, fontsize=7, loc="upper right", ncol=2)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"scaling_loss_vs_compute.{ext}", dpi=160)
    plt.close(fig)

    # --- Figure 2: loss against N at each budget (the offset structure) ----
    # Only budgets where some arm has a REAL sweep in N. At 50BT the ladder
    # holds two rungs of attention and nothing else, and a line through two
    # points across a 12x span in N is not a measurement of curvature -- it is
    # a segment that looks like one.
    budgets = [b for b in sorted({r["bt"] for r in rows})
               if max(len({r["size"] for r in rows if r["bt"] == b and r["arm"] == x})
                      for x in arms) >= 3]
    # TWO ROWS, same reason as the advantage figures: one row of five panels
    # is 20in wide, has to be shrunk to fit a page, and takes its type down
    # with it. A 2 x ceil(n/2) grid keeps the panels near square.
    ncol = -(-len(budgets) // 2)
    fig, axg = plt.subplots(2, ncol, figsize=(3.1 * ncol, 3.0 * 2),
                            sharex=True, sharey=True, squeeze=False)
    axes = [axg[i // ncol][i % ncol] for i in range(len(budgets))]
    # Any cell the budgets do not fill is blank, not an empty framed box.
    for j in range(len(budgets), 2 * ncol):
        axg[j // ncol][j % ncol].axis("off")
    for i, (ax, bt) in enumerate(zip(axes, budgets)):
        for arm in arms:
            pts = sorted([r for r in rows if r["arm"] == arm and r["bt"] == bt],
                         key=lambda r: r["N"])
            if len(pts) < 2:
                continue
            ax.plot([p["N"] for p in pts], [p["L"] for p in pts],
                    color=COLOR[arm], lw=1.1, label=LABEL[arm])
            for q in pts:
                ax.plot(q["N"], q["L"], marker="o", ms=4, color=COLOR[arm])
        ax.set_xscale("log")
        # A panel is "bottom" if nothing is drawn beneath it -- which includes
        # the last panel of the top row when the grid is ragged. sharex hides
        # its ticks, so they are turned back on: an axis label over a bare
        # frame is worse than no label.
        if i + ncol >= len(budgets):
            ax.set_xlabel("$N$")
            ax.tick_params(labelbottom=True)
        # The budget goes inside the panel, top left, clear of the curves --
        # which fall left to right, so that corner is the empty one.
        ax.text(0.03, 0.06, f"$D$ = {bt}B", transform=ax.transAxes,
                ha="left", va="bottom")
        ax.grid(alpha=0.25, which="both")
    fig.supylabel(r"Validation Loss $\downarrow$")
    # NO TITLE; the caption lives in the document.
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"scaling_loss_vs_N.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2b: loss against D at each rung (the transpose) -----------
    # The same 180 endpoints read along the other axis. Figure 2 asks how far
    # a rung gets you at a fixed budget; this asks how far a budget gets you
    # at a fixed rung, which is the axis the advantage figures are drawn on
    # and the one beta is fitted from. Same filter as Figure 2 for the same
    # reason: three points before a line through them is called a curve.
    sizes_d = [s for s in RUNGS
               if max(len({r["bt"] for r in rows
                           if r["size"] == s and r["arm"] == x})
                      for x in arms) >= 3]
    ncol = -(-len(sizes_d) // 2)
    fig, axg = plt.subplots(2, ncol, figsize=(3.1 * ncol, 3.0 * 2),
                            sharex=True, sharey=True, squeeze=False)
    axes = [axg[i // ncol][i % ncol] for i in range(len(sizes_d))]
    for j in range(len(sizes_d), 2 * ncol):
        axg[j // ncol][j % ncol].axis("off")
    for i, (ax, size) in enumerate(zip(axes, sizes_d)):
        for arm in arms:
            pts = sorted([r for r in rows if r["arm"] == arm and r["size"] == size],
                         key=lambda r: r["D"])
            if len(pts) < 2:
                continue
            ax.plot([p["D"] for p in pts], [p["L"] for p in pts],
                    color=COLOR[arm], lw=1.1, label=LABEL[arm])
            for q in pts:
                ax.plot(q["D"], q["L"], marker="o", ms=4, color=COLOR[arm])
        ax.set_xscale("log")
        # Ticks AT the measured budgets and labelled with the NOMINAL one, as
        # in the advantage figures: the realised D is the budget rounded to a
        # whole number of steps, and "6.00624B" under the 6B cell is noise
        # dressed as precision. Decade ticks would put "10^10" under a panel
        # whose five points are 6, 12, 20, 30 and 50B.
        ds = sorted({(r["D"], r["bt"]) for r in rows if r["size"] == size})
        ax.set_xticks([d for d, _ in ds], [f"{b}B" for _, b in ds])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        # Same ragged-grid rule as Figure 2: a panel with nothing beneath it
        # gets its ticks back, because an axis label over a bare frame is
        # worse than no label.
        if i + ncol >= len(sizes_d):
            ax.set_xlabel("$D$")
            ax.tick_params(labelbottom=True)
        # Loss falls left to right here too, so the bottom left corner is the
        # empty one.
        ax.text(0.03, 0.06, f"$N$ = {size}", transform=ax.transAxes,
                ha="left", va="bottom")
        ax.grid(alpha=0.25, which="both")
    fig.supylabel(r"Validation Loss $\downarrow$")
    # NO TITLE; the caption lives in the document.
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"scaling_loss_vs_D.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: what the front implies for N_opt(C) ---------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(cs, fn, color="#1f77b4", lw=1.6)
    lo, hi = np.log(cs[0]), np.log(cs[-1])
    slope = float(np.polyfit(np.log(cs), np.log(fn), 1)[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compute C [FLOPs]")
    ax.set_ylabel("compute-optimal N")
    ax.set_title(f"Compute-optimal model size: N* $\\propto$ C$^{{{slope:.2f}}}$")
    ax.grid(alpha=0.25, which="both")
    for r in rows:
        ax.plot(r["C"], r["N"], ".", color=COLOR[r["arm"]], ms=3, alpha=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"scaling_n_opt.{ext}", dpi=160)
    plt.close(fig)
    # --- Figure 4: the Chinchilla Approach-1 view --------------------------
    #
    # One line per RUN, in compute. Runs of the same (arm, rung) share their
    # constant-LR trajectory and peel off where each one begins its cooldown, so
    # the fan of endpoints below each bundle is the anneal -- and is exactly why
    # a point ON a curve is not an endpoint.
    import geometries as _geo
    curves = load_curves(a.curves, _geo.table()) if a.curves else {}
    if not a.curves:
        print("  training curves: SKIPPED. The Megatron cells validate once, "
              "at their endpoint, so they have no trajectory to draw. Pass "
              f"--curves {CURVES_HINT} for per-step curves, "
              "which are a different campaign.")
    # The harvest holds every run; the figure holds the comparison.
    curves = {k: v for k, v in curves.items() if k[0] in arms}
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
        # One colour per rung, dark to light with size. Only rungs that have a
        # validation trajectory get a legend entry -- `ladder_val_curves.tsv`
        # covers the lower ladder, and the medium rungs' curves have not been
        # harvested, so their endpoints appear without a line behind them.
        RUNG_C = {"47M": "#4c72b0", "124M": "#dd8452", "302M": "#55a868",
                  "588M": "#c44e52", "983M": "#8172b3", "1.7B": "#937860"}
        for ax, mode in zip(axes, ("rung", "arm")):
            for (arm, size, bt), pts in curves.items():
                c = RUNG_C[size] if mode == "rung" else COLOR[arm]
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=c, lw=0.8, alpha=0.30, zorder=1)
            for r in rows:
                c = RUNG_C[r["size"]] if mode == "rung" else COLOR[r["arm"]]
                ax.plot(r["C"], r["L"], "o", color=c, ms=4.0, zorder=3,
                        markeredgecolor="white", markeredgewidth=0.4)
            # The empirical front: the best ANNEALED endpoint at or below each C.
            pts = sorted(rows, key=lambda r: r["C"])
            fx, fy, best = [], [], 1e9
            for r in pts:
                best = min(best, r["L"])
                fx.append(r["C"])
                fy.append(best)
            ax.plot(fx, fy, color="black", lw=1.6, zorder=4,
                    label="empirical compute-optimal front")
            ax.plot(cs, fl, color="black", lw=1.1, ls="--", alpha=0.7, zorder=4,
                    label="fitted front (Skaling)")
            ax.set_xscale("log")
            ax.set_xlabel("training compute  C = 6ND  [FLOPs]")
            ax.grid(alpha=0.25, which="both")
            ax.set_title("coloured by rung" if mode == "rung" else "coloured by arm")
        axes[0].set_ylabel("validation loss")
        drawn = {r["size"] for r in rows}
        handles = [plt.Line2D([], [], color=RUNG_C[s2], lw=2, label=s2)
                   for s2 in RUNG_C if s2 in drawn]
        axes[0].legend(handles=handles + [
            plt.Line2D([], [], color="black", lw=1.6, label="empirical front"),
            plt.Line2D([], [], color="black", lw=1.1, ls="--", label="fitted front")],
            fontsize=7, loc="upper right")
        axes[1].legend(handles=[plt.Line2D([], [], color=COLOR[a2], lw=2, label=LABEL[a2])
                                for a2 in arms], fontsize=7, loc="upper right")
        fig.suptitle("Validation loss along training, in compute -- lines are runs at "
                     "constant LR, dots are annealed endpoints", y=1.01, fontsize=10)
        for ext in ("pdf", "png"):
            fig.savefig(a.out / f"scaling_training_curves.{ext}", dpi=160,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"training curves: {len(curves)} runs, "
              f"{sum(len(v) for v in curves.values())} validation points")

    # --- Figure 5: the gap to attention, in nats and in equivalent data ----
    #
    # Figures 1-4 plot the loss itself, and at this scale the loss is the
    # wrong variable: the axis spans 0.58 nats and the comparison that decides
    # the paper is 0.002-0.05 of it, so every arm draws the same line. This one
    # subtracts the baseline and plots what is left.
    sizes = [s for s in RUNGS if any(r["size"] == s for r in rows)]
    bts = sorted({r["bt"] for r in rows})
    adv = advantage(rows, sizes, bts)
    # THE FIGURE DROPS UN-NORMED ATTENTION. With QK-norm as the baseline,
    # plain Transformer++ plots ABOVE zero -- correct, and worth ~0.027 nats,
    # but a line above the baseline in a figure about arms that beat it reads
    # as a contradiction at a glance. The number stays in the printed report,
    # where it quantifies what QK-norm is worth without being a line on a
    # chart that means the opposite of every other line.
    others = [x for x in arms if x != ADV_BASE]
    plotted = [x for x in others if x != "attn"]
    # TWICE: once with the equivalent-data row and once without. The multiplier
    # is a derived quantity -- how much more data Transformer++ would need to
    # reach the same loss -- and it is the more arresting of the two, which is
    # exactly why a reader should be able to see the measured nats on their
    # own. Same panels, same colours, same noise band, so the two read as the
    # same figure with one row removed rather than as two different claims.
    # "rel" is implemented and deliberately not emitted: d/L divides by a
    # loss that carries the irreducible floor, so its growth with N is partly
    # the denominator shrinking. Nats is the measured quantity and the
    # equivalent-data multiplier is the relative reading that needs no E.
    for mode in ("both", "nats", "data"):
        for orient in ("h", "v", "2row"):
            _advantage_figure(a, rows, arms, plotted, sizes, adv, mode,
                              orient)
    _report_advantage(rows, others, sizes, adv)

    lo, hi = min(r["bt"] for r in rows), max(r["bt"] for r in rows)
    print(f"\nN* exponent from the fitted front: {slope:.3f}"
          f"   (Chinchilla reports ~0.5; ours is only as good as beta, which the"
          f" {lo}-{hi}BT lever pins)")
    # Five: four plus the nats-only advantage variant, six with --curves.
    print(f"wrote {11 + bool(a.curves)} figures to {a.out}/")
    return 0


def _advantage_figure(a, rows, arms, others, sizes, adv, mode, orient="h"):
    """The gap to the baseline, as `mode` asks for it.

    "both"  nats on top, equivalent data below -- the two readings together
    "nats"  the measured quantity on its own
    "data"  the derived multiplier on its own

    Three figures rather than one, because the multiplier is the more
    arresting number and a reader should be able to see the nats without it,
    and because either row on its own wants the full height of a panel.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    _style(plt)
    # A notch below the shared defaults: these panels carry five tick labels
    # each and a legend spanning the figure, and the law figures they sit
    # beside are single-panel and can afford more.
    matplotlib.rcParams.update({
        "font.size": 9.5, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    })

    # Which rows this variant draws, top to bottom.
    panes = {"both": ("nats", "data"), "nats": ("nats",), "data": ("data",),
             "rel": ("rel",)}[mode]
    # VERTICAL stacks the rungs instead of the readings, for a one-column
    # layout. Only single-reading modes: stacking six rungs times two readings
    # is twelve panels, which is a figure nobody reads.
    vert = orient == "v"
    grid = orient == "2row"
    ncol = -(-len(sizes) // 2)
    if grid:
        # A 2x3 grid is nearer square than a 1x6 row, so it is scaled down less
        # on inclusion and needs less type to survive the trip.
        # BIGGER than the one-row variants, not smaller. The 2x3 panels are
        # 4.05 x 2.4in against the row's 2.7 x 2.6, and it is nearer square
        # overall, so it is scaled down least of the three on inclusion -- and
        # since it is scaled down least, it can carry the largest type. These
        # are the previous values raised 30%, which is the point at which the
        # 6B/12B/20B/30B/50B ticks still clear each other under a 4.05in panel.
        matplotlib.rcParams.update({
            "font.size": 13.5, "axes.labelsize": 14.5,
            "xtick.labelsize": 12, "ytick.labelsize": 12,
            "legend.fontsize": 12.5,
        })
    if (vert or grid) and len(panes) != 1:
        return
    if grid:
        # Six rungs as 2x3. One row is 16in wide and must be shrunk to fit a
        # page; one column is 10in tall. This is the shape that survives both.
        # Wider than tall: 2.9 x 2.7 panels came out nearly square, and the
        # curves are flat enough that height is spent on empty space.
        fig, axes = plt.subplots(2, ncol, figsize=(4.05 * ncol, 2.4 * 2),
                                 sharex=True, sharey=True, squeeze=False)
    elif vert:
        fig, axes = plt.subplots(len(sizes), 1, figsize=(4.4, 1.7 * len(sizes)),
                                 sharex=True, sharey=True, squeeze=False)
    else:
        fig, axes = plt.subplots(len(panes), len(sizes),
                                 figsize=(2.7 * len(sizes), 2.6 * len(panes)),
                                 sharex="col", sharey="row", squeeze=False)
    for col, size in enumerate(sizes):
        if grid:
            by = {panes[0]: axes[col // ncol][col % ncol]}
        elif vert:
            by = {panes[0]: axes[col][0]}
        else:
            by = {kind: axes[i][col] for i, kind in enumerate(panes)}
        if "nats" in by:
            # The noise floor: two bit-identical configs have landed 0.0020
            # apart, so nothing inside this band is a result.
            by["nats"].axhspan(-NOISE, NOISE, color="0.85", zorder=0)
            by["nats"].axhline(0.0, color=COLOR[ADV_BASE], lw=1.2)
        if "data" in by:
            by["data"].axhline(1.0, color=COLOR[ADV_BASE], lw=1.2)
        if "rel" in by:
            by["rel"].axhspan(-100 * NOISE / 2.5, 100 * NOISE / 2.5,
                              color="0.85", zorder=0)
            by["rel"].axhline(0.0, color=COLOR[ADV_BASE], lw=1.2)
        for arm in others:
            cells = [(r["D"], adv[(arm, size, r["bt"])])
                     for r in rows if r["arm"] == arm and r["size"] == size
                     and (arm, size, r["bt"]) in adv]
            if not cells:
                continue
            cells.sort()
            d = [c[0] for c in cells]
            IDX = {"nats": 0, "data": 1, "rel": 2}
            for kind in panes:
                by[kind].plot(d, [c[1][IDX[kind]] for c in cells],
                              color=COLOR[arm], marker="o", ms=4, lw=1.2,
                              label=LABEL[arm] if kind == panes[0] else None)
        for ax in by.values():
            ax.set_xscale("log")
            ax.grid(alpha=0.25, which="both")
        # LOG on the multiplier, because it is a ratio: x1.4 against x1.0 has
        # to read as the same distance as x1.0 against x0.71, and on a linear
        # axis 47M's x2.2 flattens every upper rung into the bottom eighth of
        # the panel -- exactly where the pure-vs-hybrid split lives. Ticks are
        # written out because log minor ticks default to unlabelled and a
        # reader cannot price an unlabelled gridline.
        if "data" in by:
            by["data"].set_yscale("log")
            ticks = [0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.4]
            by["data"].set_yticks(ticks, [f"{t:g}" for t in ticks])
            by["data"].yaxis.set_minor_formatter(
                matplotlib.ticker.NullFormatter())
        # Ticks AT the measured budgets, in the unit the campaign is planned
        # in. The default decade ticks put "6 x 10^9" under a point whose whole
        # meaning is that it is the 6B cell. The NOMINAL budget, not D/1e9:
        # the realised count is the budget rounded to a whole number of steps,
        # and "6.00624B" under a cell called 6B is noise dressed as precision.
        last = by[panes[-1]]
        ds = sorted({(r["D"], r["bt"]) for r in rows if r["size"] == size})
        last.set_xticks([d for d, _ in ds], [f"{b}B" for _, b in ds])
        last.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if vert or grid:
            # The rung goes INSIDE the panel. Stacked titles would spend as
            # much height on labels as on data.
            # TOP LEFT. Bottom right collided with the 302M curves, which are
            # the ones that reach furthest down; the top left corner is empty
            # in every panel because every arm sits below the baseline.
            # Below the zero line and its noise band, which sit near the top
            # of every panel: at 0.94 the label ran into them.
            last.text(0.03, 0.84, f"$N$ = {size}", transform=last.transAxes,
                      ha="left", va="top")
            bottom = (col >= len(sizes) - ncol) if grid \
                else (col == len(sizes) - 1)
            if bottom:
                last.set_xlabel("$D$")
        else:
            last.set_xlabel("$D$")
            axes[0][col].set_title(f"N = {size}")
    # The arrow points the way BETTER is, and that is not the same direction
    # in both panels: a more negative loss delta is better, but a larger
    # equivalent-data multiplier is -- it is how much more data the baseline
    # would need to catch up. Marking both with the same arrow would be wrong.
    YL = {"nats": r"$\Delta$ val loss (nats) $\downarrow$",
          "data": r"equivalent data ($\times$ tokens) $\uparrow$",
          "rel": r"$\Delta$ val loss (\% of baseline) $\downarrow$"}
    if vert or grid:
        fig.supylabel(YL[panes[0]])
    else:
        for i, kind in enumerate(panes):
            axes[i][0].set_ylabel(YL[kind])
    # Stacked, "lower left" lands on the 47M curves; the panels are wide and
    # short, so the legend goes outside, above the first one.
    if vert or grid:
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, ncol=2 if vert else len(lab), loc="lower center",
                   bbox_to_anchor=(0.5, 0.985), frameon=False)
    else:
        # ABOVE THE ROW. At this type size the legend no longer fits inside a
        # panel: in the first attempt it spilled out of the 47M axes and over
        # the 124M ones. Figure-level, centred, one row of entries.
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, ncol=len(lab), loc="lower center",
                   bbox_to_anchor=(0.5, 0.99), frameon=False)
    # NO TITLE. The caption lives in the document, where it can be edited
    # without regenerating the figure.
    stem = {"both": "scaling_advantage", "nats": "scaling_advantage_nats",
            "data": "scaling_advantage_data",
            "rel": "scaling_advantage_relative"}[mode] \
        + {"h": "", "v": "_vertical", "2row": "_2row"}[orient]
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"{stem}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _report_advantage(rows, others, sizes, adv):
    print("\ngap to attention  (nats, equivalent-data multiplier):")
    for arm in others:
        for size in sizes:
            cs2 = [(r["bt"], adv[(arm, size, r["bt"])]) for r in rows
                   if r["arm"] == arm and r["size"] == size
                   and (arm, size, r["bt"]) in adv]
            if not cs2:
                continue
            cs2.sort()
            print(f"  {LABEL[arm]:24s} {size:>5s}  " + "  ".join(
                f"{b:>3d}BT {v[0]:+.4f} x{v[1]:.2f}" for b, v in cs2))


if __name__ == "__main__":
    raise SystemExit(main())
