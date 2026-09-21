#!/usr/bin/env python3
"""The held-out cell, drawn: where the law lands on a point it never saw.

    python lm_scaling/holdout_plot.py                  # all three panels
    python lm_scaling/holdout_plot.py --fit pinned     # panel (a) for another fit

READS THE JSON `scaling_holdout.py` writes and FITS NOTHING. The fits behind
these tables cost an hour, and a figure that refits to draw them would be a
second estimate of the same quantity -- a different seed, or tomorrow's data
file, and the figure and the table disagree in the third decimal with no way
to tell which is in the paper. One fit, one dump, one set of numbers.

THE THREE PANELS, and the question each answers:

 (a) RESIDUALS AT THE TOP RUNG. Is the held-out miss unusual? The in-fit cells
     of the same rung are the yardstick: if the open marker at the held-out
     budget sits in the same band as the filled ones beside it, the fit is not
     doing anything different when it extrapolates.

 (b) THE ERROR PER ARM, every fit. Signed, so a form that lands the level but
     leans one way shows it, with the bootstrap interval of the fitted surface
     and the ladder's own +-0.0020 nondeterminism as a band. An error inside
     that band is not a miss anybody can measure.

 (c) HELD-OUT RMSE AGAINST THE CONTROL, per fit and per functional form. The
     filled marker is the fit that never saw the cell, the open one the same
     fit with the cell included, and the segment between them is the price of
     extrapolating as opposed to the cell simply sitting off the surface.

COLOURS AND FONTS come from `plot_style` and `scaling_plots.COLOR`, which the
rest of the document's figures use. An arm keeps its colour here, so a reader
who has learned the palette in the loss curves does not learn it again.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import plot_style                # noqa: E402
import scaling_fit as SF         # noqa: E402
import scaling_plots as SP       # noqa: E402

TEX = _HERE / "tex"
#: A marker per fit, so panel (b) can carry both fits in one arm's colour.
#: Shape says WHICH FIT, colour says which arm -- never both on one channel.
#: Extra keys are harmless: a JSON from an older run that carried more fits
#: still plots, it just falls back to the default marker for the ones dropped.
FITMARK = {"isolated": "o", "pinned": "^"}
FITLABEL = {"isolated": "isolated $E$", "pinned": "pinned $E$"}
#: Panel (c) compares FORMS, not arms, so it needs ink of its own rather than
#: borrowing an arm's colour and implying a relation that is not there.
FORMC = {"skaling": "#222222", "chinchilla": "#e08214"}


def load(path: Path):
    d = json.loads(path.read_text())
    d["byfit"] = {f["kind"]: f for f in d["fits"]}
    return d


def held_of(fit, arm, cell=None):
    for h in fit["held"]:
        if h["arm"] == arm and (cell is None or h["cell"] == cell):
            return h
    return None


def rmse(vals):
    v = np.asarray(list(vals), float)
    return float(np.sqrt((v ** 2).mean())) if len(v) else float("nan")


def panel_residuals(ax, d, kind, legend="full"):
    """(a) Residual against budget at the held-out rung."""
    fit = d["byfit"][kind]
    rung = d["holdout"][0].split("/")[0]
    rows, pred = d["rows"], fit["pred"]
    ax.axhspan(-SP.NOISE, SP.NOISE, color="0.88", zorder=0, lw=0)
    ax.axhline(0.0, color="0.35", lw=0.9, zorder=1)
    for arm in d["arms"]:
        idx = sorted([i for i, r in enumerate(rows)
                      if r["arm"] == arm and r["size"] == rung],
                     key=lambda i: rows[i]["bt"])
        if not idx:
            continue
        x = [rows[i]["bt"] for i in idx]
        y = [pred[i] - rows[i]["L"] for i in idx]
        c = SP.COLOR[arm]
        ax.plot(x, y, color=c, lw=0.8, alpha=0.6, zorder=2,
                label=SF.LABEL.get(arm, arm))
        keep = [j for j, i in enumerate(idx) if not rows[i]["held"]]
        ax.plot([x[j] for j in keep], [y[j] for j in keep], ls="none",
                marker="o", ms=3.8, color=c, mec="white", mew=0.4, zorder=3)
        # THE HELD-OUT CELL, hollow and larger, with the bootstrap interval of
        # the fitted surface. Filled markers are cells the fit was given; the
        # distinction has to be visible without the caption.
        for j, i in enumerate(idx):
            if not rows[i]["held"]:
                continue
            h = held_of(fit, arm, f"{rows[i]['size']}/{rows[i]['bt']}")
            if h and not any(math.isnan(v) for v in h["ci"]):
                ax.plot([x[j]] * 2, h["ci"], color=c, lw=1.0, alpha=0.9, zorder=3)
            ax.plot([x[j]], [y[j]], ls="none", marker="o", ms=6.5, mfc="white",
                    mec=c, mew=1.4, zorder=4)
    ax.set_xscale("log")
    ax.set_xticks(sorted({r["bt"] for r in rows if r["size"] == rung}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("tokens (BT)")
    ax.set_ylabel("predicted $-$ measured (nats)")
    ax.set_title(f"(a) residuals at {rung}, {FITLABEL.get(kind, kind)} fit",
                 fontsize="medium")
    # A KEY FOR THE MARKERS, not just the colours. Hollow means the fit never
    # saw the cell, and that is the whole subject of the panel -- leaving it
    # to the caption puts the one thing a reader must know in the one place a
    # reader skips. `legend="marks"` keeps ONLY that key, for the combined
    # figure, where panel (b) already names every arm down its own axis and a
    # second copy of the palette costs four rows of headroom to say nothing.
    if legend:
        h = ([] if legend == "marks" else ax.get_legend_handles_labels()[0])
        h += [plt.Line2D([], [], ls="none", marker="o", ms=3.8, color="0.35",
                         label="in the fit"),
              plt.Line2D([], [], ls="none", marker="o", ms=6.0, mfc="white",
                         mec="0.35", mew=1.3, label="held out")]
        # RESERVE the strip rather than hunt for a gap: eight entries over six
        # lines and a whisker leave no clear corner, and every `loc` tried put
        # the key on top of an arm. Headroom above the data is empty by
        # construction, so the legend goes there and the panel keeps its shape.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (0.15 if legend == "marks" else 0.46) * (hi - lo))
        ax.legend(handles=h, fontsize=6.2, frameon=False, loc="upper center",
                  ncol=2 if legend == "marks" else 4, handlelength=1.3,
                  columnspacing=1.1, labelspacing=0.3, borderaxespad=0.25)
    return ax


def panel_errors(ax, d, kinds, legend=True):
    """(b) Signed held-out error per arm, one marker per fit."""
    cells = d["holdout"]
    keys = [(a, c) for a in d["arms"] for c in cells]
    multi = len(cells) > 1
    ax.axvspan(-SP.NOISE, SP.NOISE, color="0.88", zorder=0, lw=0)
    ax.axvline(0.0, color="0.35", lw=0.9, zorder=1)
    for row, (arm, cell) in enumerate(keys):
        for m, kind in enumerate(kinds):
            h = held_of(d["byfit"][kind], arm, cell)
            if h is None:
                continue
            off = (m - (len(kinds) - 1) / 2) * 0.155
            c = SP.COLOR[arm]
            if not any(math.isnan(v) for v in h["ci"]):
                ax.plot(h["ci"], [row + off] * 2, color=c, lw=0.9, alpha=0.75,
                        zorder=2)
            ax.plot([h["err"]], [row + off], ls="none", marker=FITMARK[kind],
                    ms=4.2, color=c, mec="white", mew=0.35, zorder=3)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{SF.LABEL.get(a, a)}" + (f" {c}" if multi else "")
                        for a, c in keys], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_ylim(len(keys) - 0.45, -1.75 if legend else -0.55)
    ax.set_xlabel("predicted $-$ measured (nats)")
    ax.set_title(f"(b) held-out error at {', '.join(cells)}", fontsize="medium")
    if legend:
        ax.legend(handles=[plt.Line2D([], [], ls="none", marker=FITMARK[k],
                                      color="0.35", ms=4.2,
                                      label=FITLABEL.get(k, k)) for k in kinds],
                  fontsize=6.4, frameon=False, loc="upper left", ncol=2,
                  handlelength=1.0, columnspacing=1.0, borderaxespad=0.2)
    return ax


def panel_rmse(ax, ds, kinds, legend=True):
    """(c) Held-out RMSE and its control, per fit, for each functional form."""
    for row, kind in enumerate(kinds):
        for m, d in enumerate(ds):
            fit = d["byfit"].get(kind)
            if fit is None:
                continue
            off = (m - (len(ds) - 1) / 2) * 0.20
            c = FORMC.get(d["form"], "0.3")
            held = rmse(h["err"] for h in fit["held"])
            ax.plot([held], [row + off], marker="o", ms=5.0, color=c,
                    mec="white", mew=0.4, zorder=3)
            if fit.get("control"):
                ctrl = rmse(x["err"] for x in fit["control"])
                ax.plot([ctrl, held], [row + off] * 2, color=c, lw=1.0,
                        alpha=0.7, zorder=2)
                ax.plot([ctrl], [row + off], marker="o", ms=5.0, mfc="white",
                        mec=c, mew=1.2, zorder=3)
    ax.axvline(SP.NOISE, color="0.45", lw=0.9, ls=":", zorder=1)
    ax.text(SP.NOISE, -0.72, " run-to-run floor", fontsize=6.4, color="0.35",
            ha="left", va="center")
    ax.set_xscale("log")
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([FITLABEL.get(k, k) for k in kinds], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_ylim(len(kinds) - 0.45, -0.95)
    ax.set_xlabel("RMSE at the held-out cell (nats)")
    ax.set_title("(c) held out vs. control", fontsize="medium")
    if legend:
        h = [plt.Line2D([], [], ls="none", marker="o", ms=5,
                        color=FORMC.get(d["form"], "0.3"),
                        label=d["form"].capitalize()) for d in ds]
        h += [plt.Line2D([], [], ls="none", marker="o", ms=5, mfc="white",
                         mec="0.35", label="control (cell in the fit)")]
        ax.legend(handles=h, fontsize=6.6, frameon=False, loc="best",
                  handlelength=1.0)
    return ax


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=TEX / "scaling_holdout.json",
                    help="the dump scaling_holdout.py writes")
    ap.add_argument("--other", type=Path, default=TEX / "scaling_holdout_chinchilla.json",
                    help="a second form for panel (c); skipped if absent")
    # NO HARDCODED DEFAULT. This named "shared" until that fit was dropped, at
    # which point every invocation died on "--fit 'shared' is not in the dump".
    # Unset means the first fit the dump actually carries.
    ap.add_argument("--fit", default=None,
                    help="which fit panel (a) draws; default is the first in the dump")
    ap.add_argument("--out", type=Path, default=TEX)
    ap.add_argument("--stem", default="holdout")
    a = ap.parse_args(argv)

    if not a.json.exists():
        raise SystemExit(f"{a.json} not found -- run lm_scaling/scaling_holdout.py "
                         f"first; it writes the dump beside its tables.")
    d = load(a.json)
    ds = [d]
    if a.other and a.other.exists():
        ds.append(load(a.other))
    else:
        print(f"note: {a.other} not found; panel (c) shows {d['form']} only")
    kinds = [k for k in FITMARK if k in d["byfit"]]
    if not kinds:
        raise SystemExit(f"{a.json} carries no fit this script knows: "
                         f"{sorted(d['byfit'])}")
    if a.fit is None:
        a.fit = kinds[0]
    if a.fit not in d["byfit"]:
        raise SystemExit(f"--fit {a.fit!r} is not in the dump; it has {kinds}")
    print(f"{a.json.name}: {d['n_fit']} of {d['n_all']} cells, holdout "
          f"{', '.join(d['holdout'])}, fits {', '.join(kinds)}")

    plot_style.apply(plt, size=9.0)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.35))
    panel_residuals(axes[0], d, a.fit, legend="marks")
    panel_errors(axes[1], d, kinds)
    panel_rmse(axes[2], ds, kinds)
    for ext in ("pdf", "png"):
        fig.savefig(a.out / f"{a.stem}.{ext}", dpi=160)
    plt.close(fig)
    print(f"-> {a.out / (a.stem + '.pdf')}")

    # EACH PANEL ON ITS OWN as well, because a paper takes one of these into a
    # column and the other two into an appendix, and a cropped screenshot of
    # a three-panel figure is how a wrong axis label reaches print.
    for name, draw, size in (
            ("residuals", lambda ax: panel_residuals(ax, d, a.fit), (4.3, 3.3)),
            ("error", lambda ax: panel_errors(ax, d, kinds), (4.6, 3.3)),
            ("rmse", lambda ax: panel_rmse(ax, ds, kinds), (4.3, 2.9))):
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        for ext in ("pdf", "png"):
            fig.savefig(a.out / f"{a.stem}_{name}.{ext}", dpi=160)
        plt.close(fig)
        print(f"-> {a.out / f'{a.stem}_{name}.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
