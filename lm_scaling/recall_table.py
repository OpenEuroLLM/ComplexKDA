"""Render the fwedu arms' recall-intensive results -- Based's SQuAD/SWDE/FDA.

    lm_scaling/recall_table.py --step 190976
    lm_scaling/recall_table.py --step 190976 --pairs
    lm_scaling/recall_table.py --step 190976 --format markdown
    lm_scaling/recall_table.py --step 190976 --format latex \
        > lm_scaling/tex/fwedu_1p3B_recall.tex

THE TABLE is the three recall-intensive tasks of Arora et al.
(arXiv:2402.18668), which lm-eval ships as `squad_completion`, `swde` and
`fda`. Each gives the model a document and an answer prefix and scores whether
the gold value is CONTAINED in what it generates. Produced by
`eval_recall.sbatch`.

WHY THIS IS ITS OWN TABLE and not three more columns of the GDN one. Those
tasks are multiple-choice likelihood comparisons over world knowledge; these
are open-ended generation over a document that is in the prompt. The published
averages this campaign is compared against are defined over the common-sense
tasks alone, so adding a column to them would silently redefine the number
every published row is quoted at. And the separation is the point of the
measurement: the common-sense suite is where a fixed-size recurrent state is
expected to be fine, and in-context recall is where it is expected not to be.

ALL FOUR ARMS, in one block each. Unlike the RULER table there are no
transcribed published rows here -- none of the papers this campaign is
compared against reports this suite at 1.3B/100BT FineWeb-Edu -- so every row
is ours and the comparison is OURS-versus-OURS. If a published row is ever
transcribed, drop it in `published_recall_1p3B.json` in the shape
`niah_table.py` uses and it will print underneath.

WHICH TOKENISATION. These rows are scored with `--context-prefix none`, pinned
by `eval_recall.sbatch`: a bare context, which is lm-eval's own convention and
matches a corpus written with `add_special_tokens=False`. The FIRST pass of
this table (job 1917940) predates that flag and was measured with an implicit
`<s>` in front of every document, because this harness tokenised with specials
on and the fwedu tokenizer copy sets `add_bos_token: True`. Those results are
measured but not published, and are NOT this table -- a run's convention is
recorded in the `.meta.json` sidecar beside its results.

THE COMPARISON IS PAIRED, and here that is not a seed argument: these are fixed
validation sets, so all four arms answer the same 2,984 / 1,111 / 1,102
questions. The binomial se printed by `--pairs` is therefore CONSERVATIVE for
the difference, the same way the ladder's was (observed cell-to-cell sd 0.43
against a binomial se of 0.53, because both arms scored identical items).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eval_table import DISPLAY  # noqa: E402

RESULTS = Path("/e/scratch/e-sta-openeurollm/poeppel1/eval_recall/results")
PUBLISHED = _HERE / "published_recall_1p3B.json"   # absent today; see docstring

ARMS = ["kda-sig-lowrank", "ckda-shipped-lowrank",
        "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"]

# (column, harness task, documents). The counts are the validation splits
# `stage_recall.py` prints and verifies offline; they are here because a result
# file records the SCORE and not what it was scored over, and the standard
# error of a proportion needs the denominator. A run made with `--limit` has a
# different n -- which is why limited runs are tagged and refused below rather
# than rescaled.
GRID = [("SQuAD", "squad_completion", 2984),
        ("SWDE", "swde", 1111),
        ("FDA", "fda", 1102)]

# lm-eval reports this task's metric under this key; the filter name is part of
# it. `contains` is `np.mean` over per-document 0/1, so it arrives as a
# FRACTION and the table is in percent.
METRIC = "contains,none"

# Filename markers that mean "this file is a diagnostic, not a measurement",
# read the same way niah_table.py reads them. `_smoke` and `_limit` are here
# for a reason the RULER table learned the hard way: a `--limit`ed run is a
# real model measured at a smaller n, and merged into this table it would
# silently coarsen a number whose se is quoted from the full split.
DIAGNOSTIC_TAGS = ("_smoke", "_limit", "_diag", "_ablate", "_window", "_samples")


def read_arm(results: Path, arm: str, step: int) -> dict:
    """One arm's tasks, merged over every pass. Missing files are missing
    cells, not an error: a partial run should still render.

    Globbed rather than named so a later pass ADDS tasks instead of replacing a
    file -- the same reason eval_ruler.sbatch's TAG exists. The unsuffixed
    `{arm}_step{step}.json` is included; anything carrying a diagnostic marker
    is not.
    """
    out = {}
    for path in sorted(results.glob(f"{arm}_step{step}*.json")):
        if any(mark in path.name for mark in DIAGNOSTIC_TAGS):
            continue
        # SIDECARS ARE NOT RESULTS. eval_downstream writes `<out>.samples.json`
        # (`--log-samples`: a LIST of documents per task) and `<out>.meta.json`
        # (what the run was -- context prefix, tokenizer, limit) beside the
        # results file, and both match the glob above. Merging either puts
        # something that is not a metrics dict where scores belong; `.meta.json`
        # does not even fail quietly, because `{"tasks": [...]}` reaches
        # `dict.update` as a list and raises.
        #
        # Tested by SUFFIX COUNT rather than by name, so the next sidecar is
        # excluded before it is written: an arm name carries hyphens and a step
        # is digits, so a real results file has exactly one suffix.
        if len(path.suffixes) > 1:
            continue
        for task, vals in json.loads(path.read_text()).items():
            out.setdefault(task, {}).update(vals)
    return out


def cell(res: dict, task: str):
    """One cell as a PERCENTAGE, or None when the task was not run.

    A value above 1.0 means the harness stopped returning a fraction, and every
    number in the table would then be 100x too big -- so it stops rather than
    renders, the same guard niah_table.py carries.
    """
    vals = res.get(task)
    if not vals:
        return None
    raw = vals.get(METRIC, vals.get("contains"))
    if raw is None:
        return None
    raw = float(raw)
    if raw > 1.0:
        raise SystemExit(
            f"{task} came back as {raw}, above 1.0. This renderer assumes "
            f"lm-eval's `contains`, which is a mean over 0/1 and so a "
            f"fraction; if the harness now returns percent, every number here "
            f"is 100x too big.")
    return 100.0 * raw


def row_avg(res: dict):
    """The average over the three tasks -- or None if any is missing.

    None rather than an average over two: a row whose average is computed over
    a different set of tasks than the row above it is not comparable with it,
    and nothing in the printed table would say so.
    """
    vals = [cell(res, task) for _, task, _ in GRID]
    return sum(vals) / len(vals) if all(v is not None for v in vals) else None


def collect(results: Path, arms, step: int) -> dict:
    return {arm: read_arm(results, arm, step) for arm in arms}


def _fmt(v):
    return "    --" if v is None else f"{v:6.1f}"


def _rows(res, arms):
    rows = [(f"{DISPLAY.get(arm, arm)} (ours)", res[arm],
             "hybrid" if "hybrid" in arm else "recurrent") for arm in arms]
    # Recurrent first, so each block's pair sits together.
    rows.sort(key=lambda r: r[2] != "recurrent")
    return rows


BLOCKS = {"recurrent": "Recurrent models",
          "hybrid": "Attention or hybrid models (3:1 full NoPE attention)"}


def render_text(rows, pub, step):
    cols = [g for g, _, _ in GRID] + ["Avg."]
    head = f"{'':28s}" + "".join(f"{c:>6s}" for c in cols)
    print(f"Recall-intensive tasks, fwedu_1p3B at step {step:,} "
          f"(contains-match %, scored at the 4,096-token context)")
    # The denominators go in a sentence rather than a header row: at the six
    # characters a cell is wide, "n=2984" is exactly six and the counts print
    # as "n=2984n=1111".
    print("documents: " + ", ".join(f"{g} {n:,}" for g, _, n in GRID))
    print(head)
    print("-" * len(head))
    kind = None
    for label, res, row_kind in rows:
        if row_kind != kind:
            if kind is not None:
                print("-" * len(head))
            print(f"  [{BLOCKS[row_kind]}]")
            kind = row_kind
        line = f"{label:28s}"
        for _, task, _ in GRID:
            line += _fmt(cell(res, task))
        print(line + _fmt(row_avg(res)))
    if pub:
        print("-" * len(head))
        for name, r in pub["rows"].items():
            line = f"{name:28s}" + "".join(
                _fmt(r.get(t)) for _, t, _ in GRID)
            print(line + _fmt(r.get("avg")))
        print(f"\npublished rows: {pub['source']}")


def render_markdown(rows, pub, step):
    cols = [g for g, _, _ in GRID] + ["Avg."]
    print(f"### Recall-intensive tasks, fwedu_1p3B at step {step:,}\n")
    print("| Model | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    kind = None
    for label, res, row_kind in rows:
        if row_kind != kind:
            print(f"| *{BLOCKS[row_kind]}* |" + " |" * len(cols))
            kind = row_kind
        cells = [_fmt(cell(res, task)).strip() for _, task, _ in GRID]
        print(f"| {label} | " + " | ".join(cells + [_fmt(row_avg(res)).strip()])
              + " |")
    if pub:
        for name, r in pub["rows"].items():
            cells = [_fmt(r.get(t)).strip() for _, t, _ in GRID]
            print(f"| {name} | " + " | ".join(cells + [_fmt(r.get('avg')).strip()])
                  + " |")
    print()
    print("Contains-match accuracy (%), higher is better; "
          + ", ".join(f"{g} n={n:,}" for g, _, n in GRID)
          + ". Every arm answers the same documents, so the differences are "
            "paired. Source: Arora et al., arXiv:2402.18668, as implemented in "
            "lm-eval-harness (`squad_completion`, `swde`, `fda`).")


def _tex(v):
    return "---" if v is None else f"{v:.1f}"


def render_latex(rows, pub, step):
    """The paper's table. Requires `booktabs`."""
    cols = [g for g, _, _ in GRID] + ["Avg."]
    print("% generated by lm_scaling/recall_table.py --format latex "
          f"--step {step}; DO NOT EDIT -- regenerate instead")
    print("%   requires \\usepackage{booktabs}")
    print("%   ALL ROWS ARE OURS: no paper this campaign compares against "
          "reports this")
    print("%   suite at 1.3B/100BT FineWeb-Edu, so this table is "
          "ours-versus-ours.")
    print("\\begin{table}[t]")
    print("  \\centering")
    print("  \\small")
    print(f"  \\begin{{tabular}}{{l{'c' * len(cols)}}}")
    print("    \\toprule")
    print("    Model & " + " & ".join(cols) + " \\\\")
    print("    \\midrule")
    kind = None
    for label, res, row_kind in rows:
        if row_kind != kind:
            if kind is not None:
                print("    \\midrule")
            print(f"    \\multicolumn{{{1 + len(cols)}}}{{l}}"
                  f"{{\\textit{{{BLOCKS[row_kind]}}}}} \\\\")
            kind = row_kind
        cells = [_tex(cell(res, task)) for _, task, _ in GRID]
        print(f"    {label} & " + " & ".join(cells + [_tex(row_avg(res))])
              + " \\\\")
    if pub:
        print("    \\midrule")
        for name, r in pub["rows"].items():
            cells = [_tex(r.get(t)) for _, t, _ in GRID]
            print(f"    {name} & " + " & ".join(cells + [_tex(r.get('avg'))])
                  + " \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print(f"  \\caption{{Recall-intensive tasks at 1.3B parameters / 100B "
          f"FineWeb-Edu tokens: contains-match accuracy (\\%) on the suite of "
          f"Arora et al. (SQuAD-completion, SWDE, FDA), scored at the models' "
          f"own 4{{,}}096-token context over "
          # `{,}` rather than a bare comma, so the thousands separator does not
          # read as the list separator beside it.
          + ", ".join(f"{n:,}".replace(",", "{,}") + f" ({g})"
                      for g, _, n in GRID) + " documents. "
          f"These tasks put the answer in the prompt, so they measure "
          f"in-context recall rather than knowledge. All rows are ours.}}")
    print("  \\label{tab:fwedu-1p3b-recall}")
    print("\\end{table}")


RENDERERS = {"text": render_text, "markdown": render_markdown,
             "latex": render_latex}

PAIRS = (("kda-sig-lowrank", "ckda-shipped-lowrank", "recurrent"),
         ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank", "hybrid"))


def se_points(pa: float, pb: float, n: int) -> float:
    """Standard error of the difference of two proportions, in POINTS.

    Computed as if the two samples were independent, which they are not --
    every arm answers the same documents -- so this OVERSTATES the error on the
    difference. That direction is deliberate and it is the direction the ladder
    measured: cell-to-cell sd 0.43 against a binomial se of 0.53. A z printed
    from it is a floor, not an estimate.
    """
    a, b = pa / 100.0, pb / 100.0
    return 100.0 * math.sqrt(a * (1 - a) / n + b * (1 - b) / n)


def render_pairing(res, step):
    """Signed minus baseline, per task, within each stack."""
    print(f"\nPaired difference, signed minus bounded gate (step {step:,})")
    print("Fixed validation sets: both arms answer the same documents, so the")
    print("binomial se below is CONSERVATIVE for the difference.")
    for lo, hi, kind in PAIRS:
        if lo not in res or hi not in res:
            continue
        print(f"\n  {kind} -- {hi} minus {lo}")
        print(f"    {'task':10s} {'baseline':>9s} {'signed':>8s} {'delta':>8s} "
              f"{'se':>6s} {'z':>6s}")
        deltas, ses = [], []
        for group, task, n in GRID:
            a, b = cell(res[lo], task), cell(res[hi], task)
            if a is None or b is None:
                print(f"    {group:10s} {'--':>9s} {'--':>8s} {'--':>8s} "
                      f"{'--':>6s} {'--':>6s}")
                continue
            se = se_points(a, b, n)
            deltas.append(b - a)
            ses.append(se)
            print(f"    {group:10s} {a:9.2f} {b:8.2f} {b - a:+8.2f} "
                  f"{se:6.2f} {(b - a) / se if se else 0:+6.2f}")
        if len(deltas) == len(GRID):
            # The average is over the three tasks, so its se is the quadratic
            # mean of theirs over three -- not their sum.
            avg_se = math.sqrt(sum(s * s for s in ses)) / len(ses)
            avg = sum(deltas) / len(deltas)
            print(f"    {'avg3':10s} {row_avg(res[lo]):9.2f} "
                  f"{row_avg(res[hi]):8.2f} {avg:+8.2f} {avg_se:6.2f} "
                  f"{avg / avg_se if avg_se else 0:+6.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--format", choices=sorted(RENDERERS), default="text")
    ap.add_argument("--no-published", action="store_true")
    ap.add_argument("--pairs", action="store_true",
                    help="append signed-minus-bounded per task, with the "
                         "binomial se of the difference")
    a = ap.parse_args(argv)

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    res = collect(a.results, arms, a.step)
    pub = (json.loads(PUBLISHED.read_text())
           if PUBLISHED.exists() and not a.no_published else None)
    RENDERERS[a.format](_rows(res, arms), pub, a.step)
    if a.pairs:
        render_pairing(res, a.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
