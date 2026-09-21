"""The Megatron ladder beside the reference's own published cells.

    lm_scaling/megatron_ext/compare_table.py            # on the cluster
    lm_scaling/megatron_ext/compare_table.py --tsv FILE # from a harvest

Prints one row per cell where all six arms have finished, with the reference's
number in the same row so the attention baseline can be checked rather than
assumed.

THE REFERENCE COLUMN is the BEST hyperparameter configuration that study ran
at that cell, from its own released table (reference/oellm_loss_post_annealing.csv,
arXiv:2608.28308). Two things follow, and both cut against reading the gap as
a defect in our runs:

  * Their number is a MINIMUM over a sweep of 6-28 configurations per cell.
    Ours is a single run at the b*/eta* their fitted law predicts. A minimum
    over a sweep is optimistically biased against a single draw at the
    predicted optimum, and by construction they cannot be equal.

  * Their best beta2 is 0.99 at 47M and 124M but 0.95 from 302M up -- which
    is ours. So the configurations CONVERGE as the rung grows: at the lower
    rungs we differ in beta2 and biases, at the upper rungs in biases alone.
    A gap that shrinks with N is that convergence, not our arm getting better.

Their N differs from ours by less than 0.2% at every rung (47,557,632 against
our 47,637,888, and so on), which is what makes the rows comparable at all.
"""

from __future__ import annotations

import argparse
import collections
import csv
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REFERENCE = _HERE.parents[0] / "reference" / "oellm_loss_post_annealing.csv"
#: DERIVED FROM THE LADDER DEFINITIONS, not typed out. This list was written
#: by hand and then ladder_302M_upper was added to submit_ladder without being
#: added here, so 302M/30BT and 302M/50BT read as 0/6 in the table for hours
#: after they finished -- cells we had already paid for, invisible.
#:
#: ladder_chaincheck is DELIBERATELY excluded. It reruns 47M/12BT and 20BT as
#: a trunk-plus-cooldown to measure what that construction is worth against a
#: standalone run, and the chained form lands ~0.009 nats LOWER. Folding it in
#: would silently replace two single-stage numbers with chained ones and quietly
#: change the very quantity the experiment exists to measure.


def _groups():
    # Same import dance as `--partial` below: this file is run as a script
    # from the repo root, where sys.path[0] is its own directory, and is also
    # imported as part of the package by the tests.
    try:
        import submit_ladder as SL
    except ImportError:
        from lm_scaling.megatron_ext import submit_ladder as SL
    return tuple(f"megatron_{k}" for k in (*SL.CELLS, *SL.CHAINS)
                 if k != "ladder_chaincheck")


GROUPS = _groups()
ARMS = ["attn", "attn-qknorm", "kda-sig-lowrank", "ckda-shipped-lowrank",
        "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"]
#: Column headings. Spelled out rather than abbreviated: this table is read
#: away from the code that produced it, and "CKDA+at" does not say which arm
#: is signed, which is hybrid, or at what ratio.
SHORT = {"attn": "attn",
         "attn-qknorm": "attn+qknorm",
         "kda-sig-lowrank": "KDA",
         "ckda-shipped-lowrank": "CKDA",
         "kda-sig-hybrid-lowrank": "KDA+attn3:1",
         "ckda-shipped-hybrid-lowrank": "CKDA+attn3:1"}
WIDTH = 14
#: Their N for each of our rungs. Written out rather than matched by proximity,
#: so a rung we have no reference for stays blank instead of borrowing a
#: neighbour's number.
THEIR_N = {"47M": 47557632, "124M": 124455168, "302M": 301880320,
           "588M": 588544000, "983M": 983138304, "1.7B": 1713504256}
RUNG = {r: i for i, r in enumerate(THEIR_N)}


def reference_cells():
    """{(rung, budget_bt): (loss, lr, gbsz, beta2, n_configs)} at their best HP."""
    by = collections.defaultdict(list)
    with open(REFERENCE) as fh:
        for r in csv.DictReader(fh):
            by[(int(r["N"]), round(float(r["D"]) / 1e9))].append(r)
    n_to_rung = {v: k for k, v in THEIR_N.items()}
    out = {}
    for (n, d), runs in by.items():
        if n not in n_to_rung:
            continue
        best = min(runs, key=lambda r: float(r["loss"]))
        out[(n_to_rung[n], d)] = (float(best["loss"]), best["lr"],
                                  best["gbsz"], best["beta2"], len(runs))
    return out


#: The harvested ladder, committed: 180 rows, one per finished run. This is
#: what the paper's table was built from.
_REPO_HARVEST = _HERE.parent / "harvest" / "megatron_ladder.tsv"


def ours(tsv=None):
    if tsv:
        text = Path(tsv).read_text()
    else:
        text = subprocess.run(
            [sys.executable, str(_HERE / "harvest_megatron.py"), *GROUPS],
            capture_output=True, text=True).stdout
    cells = collections.defaultdict(dict)
    for r in csv.DictReader(text.splitlines(), delimiter="\t"):
        cells[(r["size"], int(r["budget_bt"]))][r["arm"]] = float(r["loss"])
    return cells


#: Paper-facing arm names for the LaTeX table. The terminal table abbreviates
#: to fit 80 columns; a paper has room to say which arm is signed and which is
#: hybrid, so it does.
TEX_LABEL = {"attn": "Transformer++",
             "attn-qknorm": "Transformer++ (QK-norm)",
             "kda-sig-lowrank": "KDA",
             "ckda-shipped-lowrank": "CKDA",
             "kda-sig-hybrid-lowrank": "KDA + attn 3:1",
             "ckda-shipped-hybrid-lowrank": "CKDA + attn 3:1"}


def latex(cells, ref, path):
    """Every measured validation loss, one row per cell.

    The raw numbers behind every fit and figure, so a reader can check any of
    them without rerunning the harvester. Best per row in bold; the reference
    column is the best of the published hyperparameter sweep at that cell, and
    is blank where they report none.
    """
    rows = sorted(cells, key=lambda k: (RUNG.get(k[0], 99), k[1]))
    out = [
        "% Generated by lm_scaling/megatron_ext/compare_table.py --latex.",
        f"%   {sum(len(v) for v in cells.values())} annealed endpoints, "
        f"{len(rows)} cells, {len(ARMS)} arms.",
        r"\begin{table}[t]", r"  \centering", r"  \small",
        "  \caption{Validation loss at every cell of the Megatron ladder: "
        "annealed endpoints on the held-out Nemotron-CC split, one run per "
        "cell at the predicted $b^\\star$ and $\\eta^\\star$. Best per row in "
        "bold. The final column is the best cell of the hyperparameter sweep "
        "of \\citet{ajroldi2026scaling} at the same $(N, D)$, blank where they "
        "report none; it is a reference for the attention baseline, not a "
        "competitor to the other arms.}",
        r"  \label{tab:scaling-data}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{l" + "r" * (len(ARMS) + 1) + "}",
        r"    \toprule",
        "    Cell & " + " & ".join(TEX_LABEL[a] for a in ARMS)
        + r" & \citet{ajroldi2026scaling} \\",
        r"    \midrule",
    ]
    prev = None
    for key in rows:
        rung, bt = key
        if prev is not None and rung != prev:
            out.append(r"    \midrule")
        prev = rung
        got = cells[key]
        best = min(got.values()) if got else None
        cs = []
        for a in ARMS:
            v = got.get(a)
            if v is None:
                cs.append("--")
            elif v == best:
                cs.append(f"\\textbf{{{v:.4f}}}")
            else:
                cs.append(f"{v:.4f}")
        r = ref.get(key)
        cs.append(f"{r[0]:.4f}" if r else "--")
        out.append(f"    {rung}/{bt}BT & " + " & ".join(cs) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}%", r"  }",
            r"\end{table}", ""]
    path.write_text("\n".join(out))
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # DEFAULTS TO THE COMMITTED HARVEST. It used to default to running
    # `harvest_megatron.py`, which only works on the cluster and returns an
    # empty stdout anywhere else -- so `--latex` off the cluster silently wrote
    # a table with no cells over the real one. Measured, on this file.
    ap.add_argument("--tsv", default=str(_REPO_HARVEST),
                    help="a harvest file; the committed one by default. Pass "
                         "an empty string to run the harvester on the cluster "
                         "instead")
    ap.add_argument("--latex", nargs="?", metavar="PATH",
                    const=str(_HERE.parents[0] / "tex" / "scaling_data.tex"),
                    help="also write the raw-number table as LaTeX")
    ap.add_argument("--partial", action="store_true",
                    help="show cells that are not yet complete")
    a = ap.parse_args(argv)

    ref, cells = reference_cells(), ours(a.tsv)
    if a.latex:
        # Before --partial pads the grid: the raw-number table reports what was
        # MEASURED, so a cell with no run belongs in the terminal view that
        # tracks progress, not in the paper's data table.
        out = Path(a.latex)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = latex(cells, ref, out)
        print(f"  wrote {n} cells to {out}")
    if a.partial:
        # Every cell the campaign PLANS, so a row with nothing in it yet still
        # appears. Without this the table silently omits exactly the cells that
        # need attention -- a rung whose chains were never submitted looks
        # identical to a rung that does not exist.
        try:
            sys.path.insert(0, str(_HERE))
            import submit_ladder as SL

            for _name, cs in SL.CELLS.items():
                for rung, tok, _g, _l in cs:
                    cells.setdefault((rung, round(tok / 1e9)), {})
            for _n, chains in SL.CHAINS.items():
                for rung, _g, _l, budgets in chains:
                    for b in budgets:
                        cells.setdefault((rung, round(b / 1e9)), {})
        except Exception as e:
            print(f"note: could not read the planned grid ({e})", file=sys.stderr)
    keys = [k for k in sorted(cells, key=lambda c: (RUNG.get(c[0], 9), c[1]))
            if a.partial or len(cells[k]) >= len(ARMS)]
    if not keys:
        print("no complete cells yet")
        return 0

    head = f"{'cell':<12}" + "".join(f"{SHORT[x]:>{WIDTH}}" for x in ARMS)
    print(head + f"{'have':>6}{'Ajroldi ref':>{WIDTH}}{'attn+qk - ref':>{WIDTH}}")
    print("-" * (18 + WIDTH * (len(ARMS) + 2)))
    for k in keys:
        v = cells[k]
        row = f"{k[0] + '/' + str(k[1]) + 'BT':<12}"
        row += "".join(f"{v[x]:>{WIDTH}.4f}" if x in v else f"{'--':>{WIDTH}}"
                       for x in ARMS)
        row += f"{sum(1 for x in ARMS if x in v)}/{len(ARMS)}".rjust(6)
        r = ref.get(k)
        if r and "attn-qknorm" in v:
            row += f"{r[0]:>{WIDTH}.4f}{v['attn-qknorm'] - r[0]:>+{WIDTH}.4f}"
        elif r:
            row += f"{r[0]:>{WIDTH}.4f}{'--':>{WIDTH}}"
        else:
            row += f"{'--':>{WIDTH}}{'--':>{WIDTH}}"
        print(row)

    print("\ndelta vs attn+qknorm  (negative = the arm beats qk-normed attention)")
    print(f"{'cell':<12}" + "".join(f"{SHORT[x]:>{WIDTH}}" for x in ARMS)
          + f"{'vs Ajroldi':>{WIDTH}}")
    print("-" * (12 + WIDTH * (len(ARMS) + 1)))
    for k in keys:
        v = cells[k]
        if "attn-qknorm" not in v:
            continue
        b = v["attn-qknorm"]
        row = f"{k[0] + '/' + str(k[1]) + 'BT':<12}"
        row += "".join(f"{v[x] - b:>+{WIDTH}.4f}" if x in v else f"{'--':>{WIDTH}}"
                       for x in ARMS)
        r = ref.get(k)
        row += f"{b - r[0]:>+{WIDTH}.4f}" if r else f"{'--':>{WIDTH}}"
        print(row)

    print("\nreference = best of their HP sweep at that cell; ours = one run at "
          "the predicted b*/eta*")
    print(f"{'cell':<12}{'their best HP':<34}{'beta2 matches ours?':<22}")
    for k in keys:
        r = ref.get(k)
        if not r:
            continue
        same = "yes" if r[3] == "0.95" else f"no (theirs {r[3]})"
        print(f"{k[0] + '/' + str(k[1]) + 'BT':<12}"
              f"lr {r[1]}, gbsz {r[2]}, beta2 {r[3]}, n={r[4]:<3}"
              f"      {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
