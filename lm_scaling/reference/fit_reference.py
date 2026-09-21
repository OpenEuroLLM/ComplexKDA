"""Fit the reference study's OWN data with the reference study's OWN procedure.

    lm_scaling/reference/fit_reference.py

WHY THIS EXISTS. `scaling_fit.PAPER` held constants transcribed from the
paper's table, and neither row reproduces the measurements the authors publish
in `oellm_loss_post_annealing.csv`:

    PAPER[chinchilla] minus their data:  mean +0.9830  sd 0.2747
    PAPER[skaling]    minus their data:  mean +0.1054  sd 0.0221

A reference you cannot evaluate is not a reference. Rather than guess at the
convention the table used, this refits their published cells directly, with the
procedure their own scripts use (github.com/OpenEuroLLM/dense_english_scaling_laws,
scripts/bootstrap_estimation_chinchilla.py):

  * the same functional form, E + A N^-alpha + B D^-beta;
  * A and B optimised as log A and log B, so both stay positive;
  * Huber with delta 1e-3, which is what makes the fit robust to the handful
    of cells where the HP sweep did not find the optimum;
  * their initial grid -- E in [-1, 1], log A and log B in [0, 20], alpha and
    beta in [0, 2] -- 2,000 starts, because this objective has local minima
    and a single start lands in whichever one the guess was nearest.

ONE CELL PER (N, D): the CSV is an HP sweep, so the cell's value is the MINIMUM
over learning rate, batch size, beta2 and seed. Fitting the sweep itself would
fit the hyperparameter grid, not the scaling law.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
CSV = _HERE / "oellm_loss_post_annealing.csv"
HUBER_DELTA = 1e-3


def best_cells(path=CSV):
    """Their grid, one loss per (N, D): the best the HP sweep achieved."""
    best = {}
    for r in csv.DictReader(open(path)):
        k = (float(r["N"]), float(r["D"]))
        L = float(r["loss"])
        if k not in best or L < best[k]:
            best[k] = L
    N = np.array([k[0] for k in best]); D = np.array([k[1] for k in best])
    return N, D, np.array(list(best.values()))


def fit(N, D, L, form="chinchilla"):
    from scipy.optimize import least_squares

    lnN, lnD = np.log(N), np.log(D)

    def predict(th):
        E, logA, al, logB, be = th[:5]
        with np.errstate(over="ignore", invalid="ignore"):
            core = np.exp(logA - al * lnN) + np.exp(logB - be * lnD)
            return E + (core ** th[5] if form == "skaling" else core)

    best = None
    for E0 in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for lA in (0.0, 5.0, 10.0, 20.0):
            for lB in (0.0, 5.0, 10.0, 20.0):
                for a0 in (0.0, 0.5, 1.0, 1.5, 2.0):
                    for b0 in (0.0, 0.5, 1.0, 1.5, 2.0):
                        t0 = [E0, lA, a0, lB, b0] + ([0.5] if form == "skaling" else [])
                        try:
                            r = least_squares(lambda th: predict(th) - L, t0,
                                              loss="huber", f_scale=HUBER_DELTA,
                                              max_nfev=2000)
                        except Exception:
                            continue
                        if best is None or r.cost < best.cost:
                            best = r
    E, logA, al, logB, be = best.x[:5]
    out = dict(E=float(E), A=float(np.exp(logA)), alpha=float(al),
               B=float(np.exp(logB)), beta=float(be))
    if form == "skaling":
        out["k"] = float(best.x[5])
    resid = predict(best.x) - L
    out["rmse"] = float(np.sqrt((resid ** 2).mean()))
    return out


def main() -> int:
    N, D, L = best_cells()
    print(f"{len(L)} cells from their sweep: N {N.min():,.0f}-{N.max():,.0f} "
          f"({N.max()/N.min():.0f}x), D {D.min()/1e9:.0f}-{D.max()/1e9:.0f}BT "
          f"({D.max()/D.min():.0f}x)\n")
    for form in ("chinchilla", "skaling"):
        f = fit(N, D, L, form)
        k = f"  k {f['k']:.4f}" if "k" in f else ""
        print(f"{form:<11s} E {f['E']:.4f}  A {f['A']:.6g}  alpha {f['alpha']:.4f}"
              f"  B {f['B']:.6g}  beta {f['beta']:.4f}{k}   rmse {f['rmse']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
