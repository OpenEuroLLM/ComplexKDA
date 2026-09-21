"""Pair the ladder's downstream numbers, and ask whether they agree with the loss.

    lm_scaling/ladder_downstream.py                      # the paired statistics
    lm_scaling/ladder_downstream.py --per-task           # and every task

WHY THIS EXISTS. At 1.3B the downstream average favours the signed arm while the
validation loss favours the baseline, and one run per arm cannot tell that from
noise. The ladder has EIGHTEEN paired cells: three rungs, three budgets, two
pairs. The mean of eighteen paired differences has a standard error near 0.15
points where a single cell has 0.56, which is the difference between a
measurement and an anecdote.

WHAT IT CAN AND CANNOT SAY. These models are trained on Nemotron with the NeoX
tokenizer, so the numbers compare our arms with each other and with no published
table. Below 302M most of the suite sits at chance -- HellaSwag near 27,
ARC-c near 23, OpenBookQA near 25 -- so the accuracy average there is mostly
BoolQ and PIQA majority-class behaviour. WikiText and LAMBADA PERPLEXITY are the
readout that means something at every rung, and they carry a second advantage:
they are a different corpus from the training loss, so agreement between them is
evidence and disagreement is a finding.

THE SIGN CONVENTION, once: every difference is SIGNED MINUS BASELINE, and lower
is better for perplexity and loss while higher is better for accuracy. So a
negative perplexity difference and a positive accuracy difference both favour the
signed arm, and the `agree` column says whether loss and downstream point the
same way in that cell.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_table import ACC_GDN2, row  # noqa: E402

# COMMITTED: the 180 cells behind the scaling table, so the analysis runs
# without the cluster. Point --results at a rerun to compare.
RESULTS = Path(__file__).resolve().parent / "harvest" / "ladder_downstream"
VAL_LOSS = Path(__file__).resolve().parent / "harvest" / "megatron_ladder.tsv"
# The Megatron ladder's full grid. The torchtitan ladder this replaced stopped
# at 302M/20BT, and leaving those bounds here would silently analyse 18 of the
# 180 cells that exist -- which is how a pairing figure quoted in a paper
# caption ends up computed from a sixth of the data.
SIZES = ("47M", "124M", "302M", "588M", "983M", "1.7B")
BUDGETS = (6, 12, 20, 30, 50)
PAIRS = (("kda-sig-lowrank", "ckda-shipped-lowrank", "pure"),
         ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank", "hybrid"))
# Documents per task, for the binomial standard error of an accuracy difference.
N_DOCS = {"lambada_openai": 5153, "piqa": 1838, "hellaswag": 10042,
          "winogrande": 1267, "arc_easy": 2376, "arc_challenge": 1172,
          "openbookqa": 500, "social_iqa": 1954, "boolq": 3270}


def load_results(results: Path, sizes=SIZES) -> dict:
    """{(arm, size, budget): harness result}, whatever step each cell ended on."""
    out = {}
    # `_step<N>` is OPTIONAL. The sbatch sweeps name a result after the step the
    # cell ended on; the autoexp sweep names it after the cell alone, because
    # there the step is resolved inside the job and the config never sees it.
    # Globbing only the first form silently found nothing for the second.
    for f in sorted(results.glob("*.json")):
        stem = f.stem.rsplit("_step", 1)[0]
        for size in sizes:
            marker = f"_{size}_wsd"
            if marker in stem:
                arm, rest = stem.split(marker)
                out[(arm, size, int(rest.rstrip("B")))] = json.loads(f.read_text())
                break
    return out


def load_val_loss(path: Path) -> dict:
    """`{(arm, size, budget): loss}` from either ladder's harvest.

    Two formats, because there are two ladders. `ladder_val_loss.tsv` is five
    bare columns (`arm size budget step loss`); `megatron_ladder.tsv` is
    `harvest_megatron.py`'s output with a HEADER, which is what tells the two
    apart -- and a chained cell there appears once per budget exactly as a
    single-stage one does, so nothing else has to know which it was.
    """
    if not path.exists():
        return {}
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        return {}
    out = {}
    head = lines[0].split("\t")
    if head[:3] == ["arm", "size", "budget_bt"]:
        cols = {name: i for i, name in enumerate(head)}
        for line in lines[1:]:
            f = line.split("\t")
            out[(f[cols["arm"]], f[cols["size"]], int(f[cols["budget_bt"]]))] = \
                float(f[cols["loss"]])
        return out
    for line in lines:
        arm, size, bt, _step, loss = line.split()
        out[(arm, size, int(bt))] = float(loss)
    return out


#: Column order and headings, from `compare_table` so the downstream table and
#: `scaling_data.tex` name the same arms the same way and in the same order. A
#: second copy here is how one table calls an arm "CKDA" and the other
#: "cKDA", and a reader has to work out whether they are the same thing.
def tex_arms() -> tuple[list, dict]:
    try:
        from megatron_ext import compare_table as CT
    except ImportError:  # run as a script from lm_scaling/
        sys.path.insert(0, str(_HERE / "megatron_ext"))
        import compare_table as CT
    return list(CT.ARMS), dict(CT.TEX_LABEL)


#: What a column can hold. `better` picks which value gets the bold: accuracy is
#: better high, perplexity better low, and hard-coding `min` would silently
#: award the bold to the worst arm on the two perplexity tables.
METRICS = {
    "avg9": (lambda r: 100 * r["avg"], "{:.2f}", max,
             "Downstream accuracy, averaged over the nine tasks of Gated "
             "DeltaNet-2's Table 2 (per cent)",
             # The caveat belongs on THIS table and only this one.
             "Below 302M most of the suite is at chance -- HellaSwag near 27, "
             "ARC-c near 23, OpenBookQA near 25 -- so the average at the lower "
             "rungs is largely majority-class behaviour on BoolQ and PIQA. The "
             "perplexity tables carry signal at every rung."),
    "wiki": (lambda r: r["wiki.ppl"], "{:.2f}", min,
             "WikiText word-level perplexity",
             "Measured on a different corpus from the training loss, so "
             "agreement between the two is evidence rather than restatement."),
    "lmb": (lambda r: r["lmb.ppl"], "{:.2f}", min, "LAMBADA perplexity",
            "Measured on a different corpus from the training loss, so "
            "agreement between the two is evidence rather than restatement."),
}


def latex(res: dict, sizes, budgets, metric: str, path: Path) -> int:
    """One row per cell, one column per arm -- the shape of `scaling_data.tex`.

    THE CAPTION CARRIES THE CAVEAT, because this table will be read away from
    the code and away from the analysis. These models are trained on Nemotron
    with the NeoX tokenizer, so the numbers compare the arms with each other and
    with no published table; and below 302M most of the suite sits at chance, so
    the accuracy average there is largely BoolQ and PIQA majority-class
    behaviour. A reader who takes a 47M accuracy column at face value is being
    misled by the table, not by the data.
    """
    arms, label = tex_arms()
    pick, fmt, better, what, caveat = METRICS[metric]
    cells = [(s, b) for s in sizes for b in budgets
             if any((a, s, b) in res for a in arms)]
    out = [
        "% Generated by lm_scaling/ladder_downstream.py --latex.",
        f"%   {sum(1 for _ in res)} evaluated cells, {len(cells)} rows, "
        f"{len(arms)} arms, metric {metric}.",
        r"\begin{table}[t]", r"  \centering", r"  \small",
        "  \\caption{" + what + " at every cell of the Megatron ladder, one "
        "run per cell at the predicted $b^\\star$ and $\\eta^\\star$. "
        + ("Highest" if better is max else "Lowest") + " per row in bold. "
        "Trained on Nemotron-CC with the NeoX tokenizer, so these compare the "
        "arms with each other and with no published table. " + caveat + "}",
        f"  \\label{{tab:downstream-{metric}}}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{l" + "r" * len(arms) + "}",
        r"    \toprule",
        "    Cell & " + " & ".join(label[a] for a in arms) + r" \\",
        r"    \midrule",
    ]
    prev = None
    for size, bt in cells:
        if prev is not None and size != prev:
            out.append(r"    \midrule")
        prev = size
        got = {a: pick(row(res[(a, size, bt)], ACC_GDN2))
               for a in arms if (a, size, bt) in res}
        best = better(got.values()) if got else None
        cs = []
        for a in arms:
            if a not in got:
                cs.append("--")
            elif got[a] == best:
                cs.append("\\textbf{" + fmt.format(got[a]) + "}")
            else:
                cs.append(fmt.format(got[a]))
        out.append(f"    {size}/{bt}BT & " + " & ".join(cs) + r" \\")
    out += [r"    \bottomrule", r"  \end{tabular}%", r"  }",
            r"\end{table}", ""]
    path.write_text("\n".join(out))
    return len(cells)


def avg_se(a: dict, b: dict) -> float:
    """Standard error of the difference of two 9-task averages, in points.

    Binomial per task and independent across tasks -- the arms are scored on the
    same items, so a paired estimate would be tighter and this is conservative.
    """
    var = 0.0
    for _, task, metric in ACC_GDN2:
        m = metric.split(",")[0]
        va, vb = a[task][f"{m},none"], b[task][f"{m},none"]
        n = N_DOCS[task]
        var += va * (1 - va) / n + vb * (1 - vb) / n
    return 100 * math.sqrt(var) / len(ACC_GDN2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--val-loss", type=Path, default=VAL_LOSS)
    ap.add_argument("--per-task", action="store_true")
    # The rungs and budgets to TABULATE. Defaults are the torchtitan ladder's
    # nine cells; the Megatron ladder adds 30 and 50BT at the lower rungs, and a
    # budget missing from this list is silently absent from the table rather
    # than reported -- which is why it is an argument and not a constant.
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--budgets", default=",".join(str(b) for b in BUDGETS))
    ap.add_argument("--latex", metavar="PATH", default=None,
                    help="write a LaTeX table of --latex-metric to PATH, in the "
                         "shape of tex/scaling_data.tex, and print nothing else")
    ap.add_argument("--latex-metric", default="avg9", choices=sorted(METRICS),
                    help="avg9 (accuracy average), wiki or lmb (perplexity)")
    a = ap.parse_args(argv)
    sizes = tuple(s for s in a.sizes.split(",") if s)
    budgets = tuple(int(b) for b in a.budgets.split(",") if b)

    res = load_results(a.results, sizes)
    if a.latex:
        if not res:
            raise SystemExit(f"no results under {a.results}")
        out = Path(a.latex)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = latex(res, sizes, budgets, a.latex_metric, out)
        print(f"wrote {out}: {n} cells, metric {a.latex_metric}")
        return 0
    val = load_val_loss(a.val_loss)
    if not res:
        raise SystemExit(f"no results under {a.results}")
    have = sorted({(s, b) for _, s, b in res})
    print(f"{len(res)} cells evaluated, {len(have)} (rung, budget) combinations\n")

    arms = ["kda-sig-lowrank", "ckda-shipped-lowrank",
            "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"]
    print("=== wiki ppl / lmb ppl / avg9 per cell ===")
    print(f"{'cell':12s}" + "".join(f"{a_[:14]:>22s}" for a_ in arms))
    for size in sizes:
        for bt in budgets:
            cells = []
            for arm in arms:
                r = res.get((arm, size, bt))
                cells.append("                     -" if r is None else
                             (lambda x: f"{x['wiki.ppl']:7.2f}{x['lmb.ppl']:7.2f}"
                                        f"{100 * x['avg']:8.2f}")(row(r, ACC_GDN2)))
            print(f"{size + '/' + str(bt) + 'BT':12s}" + "".join(cells))

    print("\n=== signed minus baseline, per cell "
          "(ppl: negative favours signed; avg9: positive favours signed) ===")
    print(f"{'pair':7s} {'cell':11s} {'d wiki':>8s} {'d lmb':>8s} {'d avg9':>8s} "
          f"{'se':>5s} {'d loss':>8s}  agree?")
    collected = {tag: {"wiki": [], "lmb": [], "avg": [], "loss": [], "agree": 0,
                       "n": 0} for _, _, tag in PAIRS}
    for base, signed, tag in PAIRS:
        for size in sizes:
            for bt in budgets:
                rb, rs = res.get((base, size, bt)), res.get((signed, size, bt))
                if rb is None or rs is None:
                    continue
                b, s = row(rb, ACC_GDN2), row(rs, ACC_GDN2)
                dw = s["wiki.ppl"] - b["wiki.ppl"]
                dl = s["lmb.ppl"] - b["lmb.ppl"]
                da = 100 * (s["avg"] - b["avg"])
                dv = val.get((signed, size, bt), float("nan")) - \
                    val.get((base, size, bt), float("nan"))
                c = collected[tag]
                c["wiki"].append(dw)
                c["lmb"].append(dl)
                c["avg"].append(da)
                c["loss"].append(dv)
                c["n"] += 1
                # Do loss and downstream point the same way? Loss down is better,
                # accuracy up is better, so agreement is opposite signs.
                agree = "yes" if (dv < 0) == (da > 0) else "no"
                c["agree"] += agree == "yes"
                print(f"{tag:7s} {size + '/' + str(bt) + 'BT':11s} {dw:+8.3f} "
                      f"{dl:+8.3f} {da:+8.2f} {avg_se(rb, rs):5.2f} {dv:+8.4f}  {agree}")

    print("\n=== paired means (n cells per pair) ===")
    for _, _, tag in PAIRS:
        c = collected[tag]
        if c["n"] < 2:
            continue
        for key, label, unit in (("wiki", "wiki ppl", ""), ("lmb", "lmb ppl", ""),
                                 ("avg", "avg9", " pts"), ("loss", "val loss", "")):
            v = [x for x in c[key] if not math.isnan(x)]
            if len(v) < 2:
                continue
            m, sd = st.mean(v), st.stdev(v)
            se = sd / len(v) ** 0.5
            print(f"  {tag:7s} {label:9s} mean {m:+.4f}{unit}  sd {sd:.4f}  "
                  f"se {se:.4f}  t {m / se:+.2f}  (n={len(v)})")
        print(f"  {tag:7s} loss and downstream agree in {c['agree']} of {c['n']} cells")

    allavg = [d for c in collected.values() for d in c["avg"]]
    if len(allavg) > 1:
        m, sd = st.mean(allavg), st.stdev(allavg)
        print(f"\n  ALL {len(allavg)} cells, avg9: mean {m:+.3f} pts  sd {sd:.3f}  "
              f"se {sd / len(allavg) ** 0.5:.3f}  t {m / (sd / len(allavg) ** 0.5):+.2f}")
        print("  Positive favours the signed arm. Compare with the 1.3B result "
              "(no BOS): -0.04 (pure) and +0.45 (hybrid).")

    if a.per_task:
        print("\n=== per task, mean signed minus baseline over cells (points) ===")
        for base, signed, tag in PAIRS:
            print(f"  {tag}:")
            for col, task, metric in ACC_GDN2:
                m = metric.split(",")[0]
                ds = [100 * (res[(signed, s, b)][task][f"{m},none"]
                             - res[(base, s, b)][task][f"{m},none"])
                      for s in sizes for b in budgets
                      if (signed, s, b) in res and (base, s, b) in res]
                if len(ds) > 1:
                    print(f"    {col:8s} {st.mean(ds):+7.2f}  "
                          f"(sd {st.stdev(ds):5.2f}, n={len(ds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
