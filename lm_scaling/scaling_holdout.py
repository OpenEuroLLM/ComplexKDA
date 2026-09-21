#!/usr/bin/env python3
"""The fits of scaling_fit_variants.py with the ladder's largest cell HELD OUT.

    python lm_scaling/scaling_holdout.py                  # 1.7B/50BT held out
    python lm_scaling/scaling_holdout.py --holdout 1.7B   # the whole top rung
    python lm_scaling/scaling_holdout.py --reps 0         # no intervals, minutes

WHY. Every number scaling_fit_variants.py reports is IN-SAMPLE: the law is fitted to
all 180 endpoints and its RMSE is measured on the same 180. That says how well
the form DESCRIBES the ladder and nothing about whether it PREDICTS a cell it
has not seen, which is the only thing a scaling law is for. The study this
ladder follows (arXiv:2608.28308) makes exactly this split, and it is where
their separable fit falls over: held out at 1.7B, Chinchilla misses by 0.0174
where Skaling misses by 0.0056.

WHAT IS HELD OUT. By default the single most expensive cell -- the top rung at
the longest budget, 1.7B/50BT -- which is one row per arm, six of 180. It is
the corner the fit must extrapolate to in BOTH N and D at once, and it is the
cell a reader asks about. `--holdout` takes `SIZE/BT` for a cell or a bare
`SIZE` for a whole rung, repeatable.

(The top rung of THIS ladder is 1.7B, by target parameter count with embeddings
counted. The 1.3B model is the separate fwedu replication; it is not a cell
here, and 1.7B/50BT is the largest point these fits have.)

WHAT IS REPORTED, per fit and per arm: predicted loss at the held-out cell, the
measured one, and the signed difference. Three summaries under them -- RMSE
over arms; the MEAN SIGNED error, because a law that extrapolates with the
right shape onto a wrong offset shows up there and not in the RMSE; and, as the
control, the residual the same fit leaves at that same cell when the cell is IN
the fit. The distance between those last two is the price of extrapolating,
separated from the part of the miss that is just the cell sitting off the
surface.

WHAT AN ERROR IS WORTH. Two bit-identical configurations of this ladder have
measured 0.0020 nats apart, so an error of that size is the measurement and not
the law. The bootstrap interval printed beside each error is the spread of the
FITTED SURFACE over resampled ladders; it does not include that cell noise, so
a miss inside the interval is a miss the fit itself cannot resolve.

WRITES NOTHING that scaling_fit.py or scaling_fit_variants.py own --
tex/scaling_holdout*.tex only.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import scaling_fit as SF      # noqa: E402  the forms, the loader, the fitters
import scaling_fit_variants as S3  # noqa: E402  the LaTeX table writer and arm order

LADDER = _HERE / "harvest" / "megatron_ladder.tsv"

#: The fits of scaling_fit_variants.py, by name; `main` maps each to the keyword
#: arguments SF.fit_shared_E takes, except `isolated`, which routes to
#: SF.fit_isolated and fits every arm alone. These must stay in step with that
#: script's `specs`: the point here is the SAME fits on 174 cells, not 180, so
#: a fit that differs in any other way makes the comparison say nothing.
#:
#: TWO, NOT FIVE. The paper reports the two treatments of E that bracket the
#: question -- every arm fitted alone, and E pinned to the reference's value --
#: and an earlier version of this code also carried `shared`, `shared_beta` and
#: `pinned_beta`. Those are genuine fits and they are gone on purpose: five
#: variants of one comparison in a release invites the reader to hunt for the
#: one that says what they want, and the three dropped ones were never in the
#: paper. `scaling_fit_variants.py` carries the same two.
SPECS = ("isolated", "pinned")


def _repo_relative(path) -> str:
    """A path as the repository sees it.

    `--data` defaults to an absolute path under this file, and this dump
    ships; recording it verbatim stamps the developer's home directory into
    a released artefact.
    """
    p = Path(path).resolve()
    root = Path(__file__).resolve().parent.parent
    return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)


def cell_key(row) -> str:
    return f"{row['size']}/{row['bt']}"


def largest_cell(rows) -> str:
    """The top rung of the ladder at its longest budget.

    By SF.RUNGS order, not by string or by fitted N: the rungs are named
    ("47M" ... "1.7B") and sorting those as text puts 124M above 47M and 1.7B
    below both.
    """
    rung = max({r["size"] for r in rows}, key=SF.RUNGS.index)
    bt = max(r["bt"] for r in rows if r["size"] == rung)
    return f"{rung}/{bt}"


def split(rows, holdout):
    """(kept, held out) by `SIZE/BT` cell or bare `SIZE` rung.

    Raises on a spec that matches nothing. A typo'd rung that silently held out
    zero cells would report an "extrapolation error" measured on cells the fit
    had just been given, which is the one failure this script must not have.
    """
    sizes = {r["size"] for r in rows}
    cells = {cell_key(r) for r in rows}
    keep_out = set()
    for spec in holdout:
        if spec in cells:
            keep_out.add(spec)
        elif spec in sizes:
            keep_out |= {cell_key(r) for r in rows if r["size"] == spec}
        else:
            raise SystemExit(f"--holdout {spec!r} matches no cell or rung; "
                             f"rungs are {sorted(sizes, key=SF.RUNGS.index)}")
    kept = [r for r in rows if cell_key(r) not in keep_out]
    test = [r for r in rows if cell_key(r) in keep_out]
    return kept, test, sorted(keep_out, key=lambda c: (SF.RUNGS.index(c.split("/")[0]),
                                                       int(c.split("/")[1])))


def value(fit, name, arm):
    """One parameter out of a fit, whether it is shared or per arm."""
    v = fit[name]
    return v[arm] if isinstance(v, dict) else v


def predict(fit, form, arm, N, D, chain):
    """The law evaluated from a returned fit dict.

    SF.fit_shared_E builds its prediction inside a closure over a packed theta
    and returns only the unpacked parameters, so predicting a cell the fit
    never saw means rebuilding the form here. `check_predictor` below asserts
    this reproduces the fit's own in-sample RMSE, because a silent mismatch in
    the chained offset or in the k exponent would land entirely in the
    extrapolation error and read as a finding.
    """
    E = value(fit, "E", arm) + (fit["dchain"] if chain else 0.0)
    red = (value(fit, "A", arm) * N ** -value(fit, "alpha", arm)
           + value(fit, "B", arm) * D ** -value(fit, "beta", arm))
    return E + (red if form == "chinchilla" else red ** value(fit, "k", arm))


def rows_of(fit, form, rows, arm=None):
    """Predictions for `rows`, taking `arm` from each row unless pinned."""
    return np.array([predict(fit, form, arm or r["arm"], r["N"], r["D"], r["chain"])
                     for r in rows])


def check_predictor(point, form, rows, arms, isolated, tol=5e-9):
    """Does `predict` reproduce what the fitter said its RMSE was?"""
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        fit = point[a] if isolated else point
        resid = rows_of(fit, form, sub, arm=a) - np.array([r["L"] for r in sub])
        got = float(np.sqrt((resid ** 2).mean()))
        want = fit["rmse"] if isolated else fit["rmse_by_arm"][a]
        if not math.isclose(got, want, abs_tol=tol):
            raise SystemExit(
                f"predictor disagrees with the fitter on {a}: rebuilt RMSE "
                f"{got:.8f} against the fit's {want:.8f}. scaling_fit.py's "
                f"parameterisation has moved; fix `predict` before trusting "
                f"any number below.")


def cold_fit(rows, arms, form, isolated, kw, starts, seed=0):
    if isolated:
        return SF.fit_isolated(rows, arms, form, starts=starts)
    return SF.fit_shared_E(rows, arms, form, starts=starts, seed=seed, **kw)


def converged_fit(rows, arms, form, isolated, kw, starts):
    """A cold fit, refit from a second seed, keeping the better optimum.

    Same guard as SF.bootstrap_arms and for the same reason: the multi-start
    search on the E-pinned forms has landed in different local minima on
    different seeds, and here every replicate warm-starts from this fit AND
    every extrapolated number is read off it.
    """
    point = cold_fit(rows, arms, form, isolated, kw, starts, seed=0)
    if isolated:
        return point
    alt = cold_fit(rows, arms, form, isolated, kw, starts, seed=1)
    if not math.isclose(alt["rmse"], point["rmse"], rel_tol=1e-6):
        better = min((alt, point), key=lambda f: f["rmse"])
        print(f"    WARNING: multi-start not converged at starts={starts}: "
              f"seeds gave rmse {point['rmse']:.5f} and {alt['rmse']:.5f}. "
              f"Reporting the better ({better['rmse']:.5f}); RAISE --starts "
              f"until they agree.", file=sys.stderr, flush=True)
        point = better
    return point


def pct(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not len(v):
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def bootstrap(rows, test, arms, form, isolated, kw, point, reps, starts, seed=0):
    """Intervals on the parameters AND on the held-out prediction, one loop.

    SF.bootstrap_arms discards its replicates once it has the parameter
    percentiles, so an interval on the extrapolated LOSS cannot be recovered
    from it -- and the loss is what this script reports. Same resampling
    scheme as there (with replacement, stratified within each arm, warm-started
    from the point fit); it just keeps the predictions on the way past.
    """
    rng = np.random.default_rng(seed)
    draws, preds = [], []
    for i in range(reps):
        rs = SF._resample(rows, arms, rng)
        try:
            if isolated:
                f = {a: SF.fit_shared_E([r for r in rs if r["arm"] == a], [a],
                                        form, per_arm="all", starts=starts,
                                        x0=point[a]["theta"]) for a in arms}
            else:
                f = SF.fit_shared_E(rs, arms, form, starts=starts,
                                    x0=point["theta"], **kw)
        except Exception:
            continue
        draws.append(f)
        preds.append([predict(f[r["arm"]] if isolated else f, form, r["arm"],
                              r["N"], r["D"], r["chain"]) for r in test])
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{reps} replicates", flush=True)
    ci = {}
    if isolated and draws:
        keys = [k for k, v in point[arms[0]].items() if isinstance(v, float)]
        ci = {k: {a: pct([d[a][k] for d in draws if a in d]) for a in arms}
              for k in keys}
    elif draws:
        for k, v in point.items():
            if isinstance(v, dict):
                ci[k] = {a: pct([d[k][a] for d in draws]) for a in arms}
            elif isinstance(v, float):
                ci[k] = pct([d[k] for d in draws])
    P = np.array(preds) if preds else np.zeros((0, len(test)))
    pred_ci = [pct(P[:, j]) if len(P) else (float("nan"),) * 2
               for j in range(len(test))]
    return ci, pred_ci, len(draws)


def jsonable(o):
    """Drop the packed theta, and turn numpy scalars into JSON numbers."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items() if k != "theta"}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def dump_json(path, results, rows, test, arms, form, a, cells):
    """Every fitted number, once, so a figure never refits to draw a table.

    The tables here cost an hour of fitting, and until this existed the only
    record of a fit was the LaTeX it had already been formatted into. A plot
    would then have had to refit -- and two fits of the same model, from
    different seeds or a different day's data file, disagree in the third
    decimal and put two numbers for one quantity into one paper. The figure
    reads this file instead.

    `pred` is over ALL rows, held-out ones included, so a residual plot can
    show the held-out miss against the in-sample scatter at the same rung.
    """
    import json

    every = rows + test
    held_keys = {(r["arm"], cell_key(r)) for r in test}
    out = dict(
        form=form, data=_repo_relative(a.data), arms=list(arms), holdout=list(cells),
        n_fit=len(rows), n_all=len(every), starts=a.starts,
        rows=[dict(arm=r["arm"], size=r["size"], bt=r["bt"], N=r["N"],
                   D=r["D"], L=r["L"], chain=bool(r["chain"]),
                   held=(r["arm"], cell_key(r)) in held_keys) for r in every],
        fits=[])
    for res in results:
        point = res["point"]
        pred = [predict(point[r["arm"]] if res["isolated"] else point, form,
                        r["arm"], r["N"], r["D"], r["chain"]) for r in every]
        out["fits"].append(dict(
            kind=res["kind"], isolated=res["isolated"],
            n_params=res["n_params"], rmse=res["rmse"], aic=res["aic"],
            reps=res["reps"], params=jsonable(point), ci=jsonable(res["ci"]),
            pred=jsonable(pred), held=jsonable(res["held"]),
            control=jsonable(res["control"])))
    path.write_text(json.dumps(out, indent=1))
    return path


def run_spec(kind, kw, isolated, rows, test, arms, form, a):
    """One fit on the reduced ladder, plus its control fit on the full one."""
    print(f"  fitting {kind} on {len(rows)} cells ...", flush=True)
    point = converged_fit(rows, arms, form, isolated, kw, a.starts)
    check_predictor(point, form, rows, arms, isolated)
    ci, pred_ci, got = ({}, [(float("nan"),) * 2] * len(test), 0)
    if a.reps:
        ci, pred_ci, got = bootstrap(rows, test, arms, form, isolated, kw,
                                     point, a.reps, a.boot_starts)
    held = []
    for r, (lo, hi) in zip(test, pred_ci):
        fit = point[r["arm"]] if isolated else point
        p = predict(fit, form, r["arm"], r["N"], r["D"], r["chain"])
        # AN INTERVAL ON THE ERROR, not on the loss. The bootstrap moves the
        # predicted surface; the measurement it is compared against is one
        # fixed number, so subtracting it shifts the interval without widening
        # it -- and an interval quoted next to a signed error has to be in the
        # same units as that error or it reads as a contradiction.
        held.append(dict(arm=r["arm"], cell=cell_key(r), L=r["L"], pred=p,
                         err=p - r["L"], pred_ci=(lo, hi),
                         ci=(lo - r["L"], hi - r["L"])))
    control = None
    if a.control:
        print(f"  control: same fit on all {len(rows) + len(test)} cells ...",
              flush=True)
        full = converged_fit(rows + test, arms, form, isolated, kw, a.starts)
        check_predictor(full, form, rows + test, arms, isolated)
        control = []
        for r in test:
            fit = full[r["arm"]] if isolated else full
            p = predict(fit, form, r["arm"], r["N"], r["D"], r["chain"])
            control.append(dict(arm=r["arm"], cell=cell_key(r), pred=p,
                                err=p - r["L"]))
    n = len(rows)
    npar = (sum(p["n_params"] for p in point.values()) if isolated
            else point["n_params"])
    rss = ((sum((p["rmse"] ** 2) * sum(1 for r in rows if r["arm"] == arm)
                for arm, p in point.items()) if isolated
            else (point["rmse"] ** 2) * n))
    return dict(kind=kind, point=point, ci=ci, reps=got, held=held,
                control=control, n_params=npar, isolated=isolated,
                rmse=math.sqrt(rss / n), aic=n * math.log(rss / n) + 2 * npar)


def summarise(errs):
    e = np.array(errs, float)
    return dict(rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                worst=float(e[np.argmax(np.abs(e))]))


def report(res, arms, test, form, ref):
    """Console report for one fit: every arm, then the three summaries."""
    print(f"\n=== {res['kind']} ({res['n_params']} params, in-sample rmse "
          f"{res['rmse']:.5f}) ===")
    # THE CELL IN THE LABEL as soon as more than one is held out -- a whole
    # rung is five rows per arm, and five rows headed by the same arm name are
    # unreadable and easy to misread as replicates of one measurement.
    multi = len({h["cell"] for h in res["held"]}) > 1
    w = 30 + (10 if multi else 0)
    print(f"  {'arm':<{w}s} {'measured':>9s} {'predicted':>10s} {'error':>9s} "
          f"{'error [2.5, 97.5]':>19s} {'control':>9s}")
    ctrl = {(c["arm"], c["cell"]): c["err"] for c in (res["control"] or [])}
    for h in res["held"]:
        lo, hi = h["ci"]
        iv = "--" if math.isnan(lo) else f"[{lo:+.4f}, {hi:+.4f}]"
        c = ctrl.get((h["arm"], h["cell"]))
        label = f"{h['arm']} {h['cell']}" if multi else h["arm"]
        print(f"  {label:<{w}s} {h['L']:9.4f} {h['pred']:10.4f} "
              f"{h['err']:+9.4f} {iv:>19s} "
              f"{('%+.4f' % c) if c is not None else '--':>9s}")
    s = summarise([h["err"] for h in res["held"]])
    line = (f"  held out: rmse {s['rmse']:.4f}   mean signed {s['bias']:+.4f}"
            f"   worst {s['worst']:+.4f}")
    if ctrl:
        cs = summarise(list(ctrl.values()))
        line += (f"\n  control (cell in the fit): rmse {cs['rmse']:.4f}   "
                 f"mean signed {cs['bias']:+.4f}")
    print(line)
    # THE ORDERING, not just the level. The paper's claim is a gap between
    # arms at a cell, and a law can miss every arm by +0.01 and still get every
    # gap right -- or land the levels and invert the ranking. Both are worth
    # seeing, and only this table shows the second.
    if ref and any(h["arm"] == ref for h in res["held"]):
        for cell in sorted({h["cell"] for h in res["held"]}):
            at = {h["arm"]: h for h in res["held"] if h["cell"] == cell}
            if ref not in at:
                continue
            print(f"  advantage vs {ref} at {cell} (negative = better):")
            for arm in arms:
                if arm == ref or arm not in at:
                    continue
                dm = at[arm]["L"] - at[ref]["L"]
                dp = at[arm]["pred"] - at[ref]["pred"]
                print(f"    {arm:<30s} measured {dm:+.4f}   predicted "
                      f"{dp:+.4f}   error {dp - dm:+.4f}")


def error_table(results, arms, test, form, cells, n_fit, n_all, suffix=""):
    """The held-out error as LaTeX: a column per fit, a row per arm."""
    kinds = [r["kind"] for r in results]
    meas = {(h["arm"], h["cell"]): h["L"] for h in results[0]["held"]}
    rows_out = sorted({(h["arm"], h["cell"]) for h in results[0]["held"]},
                      key=lambda t: (arms.index(t[0]), t[1]))
    reps = max(r["reps"] for r in results)
    has_control = all(r["control"] for r in results)
    out = [
        "% Generated by lm_scaling/scaling_holdout.py -- regenerate, do not edit.",
        f"%   held out {', '.join(cells)}: fitted on {n_fit} of {n_all} cells.",
        r"\begin{table}[t]", r"  \centering", r"  \small",
        "  \\caption{Held-out error at "
        + ", ".join(c + "BT" for c in cells)
        + f", the largest cell{'s' if len(cells) > 1 else ''} of the ladder. "
        f"Each column is the fit of that name refitted on the remaining "
        f"{n_fit} of {n_all} endpoints and then evaluated at the held-out "
        "cell; entries are predicted minus measured, in nats"
        + (f", with 2.5--97.5 percentiles of {reps} bootstrap replicates of "
           "the fitted surface" if reps else "")
        + ". "
        + ("The \\emph{control} rows repeat the two summaries for the same "
           "fits with the held-out cell included, so the distance between "
           "them is the price of extrapolating rather than of the cell "
           "sitting off the surface. " if has_control else "")
        + "Two bit-identical runs of this ladder have measured $0.0020$ "
        "apart, which is the floor any of these numbers can mean.}",
        # SUFFIXED, like the variant tables at the bottom of `main`. Without
        # it every form writes `tab:scaling-holdout`, so inputting two of them
        # is a multiply-defined label and the four parameter tables -- which all
        # \ref this one -- resolve to whichever error table came last.
        r"  \label{tab:scaling-holdout" + suffix.replace("_", "-") + "}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{lr" + "r" * len(kinds) + "}", r"    \toprule",
        "    Arm & Measured & " + " & ".join(k.replace("_", r"\_") for k in kinds)
        + r" \\", r"    \midrule",
    ]
    for arm, cell in rows_out:
        cells_tex = []
        for r in results:
            h = next(x for x in r["held"] if x["arm"] == arm and x["cell"] == cell)
            lo, hi = h["ci"]
            cells_tex.append(S3.sub(h["err"], None if math.isnan(lo) else (lo, hi),
                                    "{:+.4f}"))
        label = SF.LABEL.get(arm, arm)
        if len(cells) > 1:
            label += f" ({cell})"
        out.append(f"    {label} & ${meas[(arm, cell)]:.4f}$ & "
                   + " & ".join(cells_tex) + r" \\")
    out.append(r"    \midrule")
    for name, key in (("RMSE over held-out rows", "rmse"),
                      ("mean signed", "bias")):
        vals = [summarise([h["err"] for h in r["held"]])[key] for r in results]
        fmt = "{:.4f}" if key == "rmse" else "{:+.4f}"
        out.append(f"    {name} & & "
                   + " & ".join(f"${fmt.format(v)}$" for v in vals) + r" \\")
    if has_control:
        for name, key in (("RMSE, control", "rmse"),
                          ("mean signed, control", "bias")):
            vals = [summarise([c["err"] for c in r["control"]])[key]
                    for r in results]
            fmt = "{:.4f}" if key == "rmse" else "{:+.4f}"
            out.append(f"    {name} & & "
                       + " & ".join(f"${fmt.format(v)}$" for v in vals) + r" \\")
    out.append(r"    \midrule")
    out.append("    in-sample RMSE & & "
               + " & ".join(f"${r['rmse']:.5f}$" for r in results) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=LADDER)
    ap.add_argument("--form", default="skaling", choices=["skaling", "chinchilla"])
    ap.add_argument("--holdout", action="append", metavar="SIZE/BT",
                    help="cell (`1.7B/50`) or whole rung (`1.7B`) to hold out; "
                         "repeatable. Default: the ladder's largest cell")
    ap.add_argument("--specs", nargs="+", default=list(SPECS), choices=SPECS,
                    help="which of scaling_fit_variants.py's fits to run")
    ap.add_argument("--reps", type=int, default=200,
                    help="bootstrap replicates; 0 skips the intervals")
    ap.add_argument("--starts", type=int, default=600,
                    help="multi-starts for each point fit")
    ap.add_argument("--boot-starts", type=int, default=6,
                    help="extra random starts per replicate, on top of the "
                         "warm start from the point fit")
    ap.add_argument("--no-control", dest="control", action="store_false",
                    help="skip the same fits WITH the held-out cell, which are "
                         "what says whether a miss is extrapolation or noise")
    ap.add_argument("--out", type=Path, default=_HERE / "tex")
    ap.add_argument("--no-tables", dest="tables", action="store_false",
                    help="write the JSON dump but none of the .tex tables. The "
                         "Chinchilla form is run for the comparison in "
                         "holdout_plot.py's third panel, not for tables the "
                         "paper does not report")
    a = ap.parse_args(argv)

    rows = SF.load(a.data)
    arms = [x for x in S3.ORDER if any(r["arm"] == x for r in rows)]
    dropped = sorted({r["arm"] for r in rows} - set(arms))
    if dropped:
        print(f"  not reported, dropped: {', '.join(dropped)}", flush=True)
    rows = [r for r in rows if r["arm"] in arms]
    n_all = len(rows)
    fit_rows, test, cells = split(rows, a.holdout or [largest_cell(rows)])
    if not test:
        raise SystemExit("nothing held out")
    Eref = SF.PAPER[a.form]["E"]
    kws = dict(isolated=(True, {}),
               pinned=(False, dict(per_arm="all", E_fixed=Eref)))
    ref = next((x for x in SF.REFERENCE_ARMS if x in arms), None)

    print(f"{n_all} cells, {len(arms)} arms, form {a.form}, E_ref {Eref}")
    print(f"held out: {', '.join(cells)} -- {len(test)} rows, "
          f"fitting on {len(fit_rows)}")
    for c in cells:
        r = next(r for r in test if cell_key(r) == c)
        print(f"  {c:<10s} N {r['N']:,.0f}  D {r['D'] / 1e9:.0f}BT  "
              f"{'chained' if r['chain'] else 'single-stage'} cooldown")
    # THE CHAINED OFFSET has to survive the split. Every top-rung cell is a
    # cooldown branched off a shared trunk, so holding out the whole rung could
    # leave dchain carried by cells at other rungs only -- or, if those go too,
    # by nothing at all, in which case the fit sets it from noise and the
    # extrapolation inherits that.
    n_chain = sum(1 for r in fit_rows if r["chain"])
    if any(r["chain"] for r in test):
        print(f"  the held-out cells are chained; {n_chain} chained cells "
              f"remain in the fit to pin dchain")
        if not n_chain:
            print("  WARNING: no chained cell left in the fit -- dchain is "
                  "unidentified and the prediction below carries whatever the "
                  "optimiser put there", file=sys.stderr)

    results = []
    for kind in a.specs:
        isolated, kw = kws[kind]
        results.append(run_spec(kind, kw, isolated, fit_rows, test, arms,
                                a.form, a))
        report(results[-1], arms, test, a.form, ref)

    suffix = "" if a.form == "skaling" else f"_{a.form}"
    a.out.mkdir(parents=True, exist_ok=True)
    if a.tables:
        path = a.out / f"scaling_holdout{suffix}.tex"
        path.write_text(error_table(results, arms, test, a.form, cells,
                                    len(fit_rows), n_all, suffix))
        print(f"\n-> {path}")
    print(f"-> {dump_json(a.out / f'scaling_holdout{suffix}.json', results, fit_rows, test, arms, a.form, a, cells)}")
    # The parameter tables too, in scaling_fit_variants.py's own layout, so the
    # exponents fitted WITHOUT the top cell can be read against the published
    # ones column for column. Its writer, not a second copy of it.
    # WHAT EACH FIT IS, spelled out rather than cross-referenced. These
    # captions used to open "As Table~\ref{tab:scaling-<kind>}", pointing at
    # the in-sample tables of scaling_fit_variants.py -- which this release no
    # longer ships, because the paper reports the held-out fits only. A caption
    # that references a table the reader does not have is worse than a longer
    # caption, and \ref to a missing label builds as "??".
    WHAT = {
        "isolated": "Each architecture fitted in isolation: its own irreducible "
                    "loss $E$, its own exponents, its own chained-cell offset",
        "pinned": "Every architecture sharing one $E$, pinned to the value of "
                  "\\citet{ajroldi2026scaling}, with its own exponents and "
                  "chained-cell offset",
    }
    dash = suffix.replace("_", "-")   # the label form of `suffix`
    for r in results:
        what = WHAT.get(r["kind"], f"The {r['kind'].replace('_', ' ')} fit")
        cap = (f"{what}, refitted with {', '.join(cells)} held out "
               f"({len(fit_rows)} of {n_all} endpoints). RMSE is in-sample, "
               f"over that arm's remaining cells; the held-out error is in "
               f"Table~\\ref{{tab:scaling-holdout{dash}}}.")
        if r["reps"]:
            cap += (f" Bootstrap intervals are the 2.5th--97.5th percentiles "
                    f"of {r['reps']} replicates, resampled within each arm.")
        txt = S3.table(r["kind"], r["point"], r["ci"], r["reps"], len(fit_rows),
                       arms, a.form, cap,
                       f"tab:scaling-holdout-{r['kind'].replace('_', '-')}{dash}")
        if not a.tables:
            continue
        p = a.out / f"scaling_holdout_{r['kind']}{suffix}.tex"
        p.write_text(txt)
        print(f"-> {p}")

    print(f"\n{'fit':<12s} {'params':>6s} {'in-sample':>10s} {'held rmse':>10s} "
          f"{'bias':>9s} {'control':>9s} {'AIC':>10s} {'reps':>5s}")
    for r in results:
        s = summarise([h["err"] for h in r["held"]])
        c = (summarise([x["err"] for x in r["control"]])["rmse"]
             if r["control"] else float("nan"))
        print(f"{r['kind']:<12s} {r['n_params']:>6d} {r['rmse']:>10.5f} "
              f"{s['rmse']:>10.4f} {s['bias']:>+9.4f} {c:>9.4f} "
              f"{r['aic']:>10.1f} {r['reps']:>5d}")
    # THE ONLY PUBLISHED NUMBER OF THE SAME KIND. Ajroldi et al. hold their
    # 1.7B rung out of a ladder reaching 300BT and report 0.0056 for Skaling
    # against 0.0174 for Chinchilla. Ours is a smaller extrapolation -- one
    # cell of a corner, not a whole rung, and over a shorter lever -- so it
    # should be the easier problem, and a number far above theirs means the
    # form is being asked to do something it is not doing.
    print("  for scale: arXiv:2608.28308 hold out their whole 1.7B rung and "
          "report held-out RMSE 0.0056 (Skaling) and 0.0174 (Chinchilla).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
