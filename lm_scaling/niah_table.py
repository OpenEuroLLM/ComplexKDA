"""Render the fwedu arms' RULER needle results as Gated DeltaNet-2's Table 3.

    lm_scaling/niah_table.py --step 190976
    lm_scaling/niah_table.py --step 190976 --format markdown
    lm_scaling/niah_table.py --step 190976 --format latex > lm_scaling/tex/fwedu_1p3B_niah.tex

THE TABLE is S-NIAH-1/2/3 and MK-NIAH-1 from RULER, per context length, for
1.3B models trained on 100B FineWeb-Edu tokens at 4K sequences. Ours are
produced by `eval_ruler.sbatch`; the published rows are transcribed in
`published_gdn2_niah_1p3B.json` and printed underneath so the comparison is on
one screen.

THE GRID IS NOT RECTANGULAR. The paper reports S-NIAH-1 and -2 at 1K/2K/4K/8K
and S-NIAH-3 and MK-NIAH-1 at 1K/2K/4K. A cell outside that grid is left BLANK
rather than filled: our own run can produce an 8K MK-NIAH-1 number, and putting
it in a column no published row has is how a table stops being a comparison.

TWO SENTINELS TO KNOW, both of which look like results if they are not handled:

  -1.0   `common_utils.aggregate_metrics` returns this for a length with no
         samples -- i.e. a cell that was never run. It is not a score of -100%.
  0.0    a real score. A recurrent model at 8K genuinely does hit it, so a
         zero cell must survive to the table while a -1 must not.

THE PUBLISHED ROWS ARE RECURRENT ONLY -- see the note in the transcription.
Ours default to the two recurrent arms for that reason. `--arms` will add the
hybrids, and then they print as their own block behind their own caveat: those
rows answer a different question (full NoPE attention every fourth layer, not
2K sliding-window) and are evaluated only at or below their 4,096-token
training context, so they are OURS-versus-OURS and not a row of that table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from eval_table import DISPLAY  # noqa: E402

# COMMITTED, so the RULER table regenerates without the cluster.
# `eval_ruler.sbatch` writes fresh ones; point --results at those to compare.
RESULTS = Path(__file__).resolve().parent / "harvest" / "ruler"
PUBLISHED = _HERE / "published_gdn2_niah_1p3B.json"

# The recurrent arms, in the order the pairing reads. The hybrids are not in
# the DEFAULT on purpose -- the published block they would sit in uses 2K
# sliding-window attention where ours use full NoPE attention every fourth
# layer -- but they can be asked for, and then they print in their own block
# under their own heading rather than in the published one's column order.
ARMS = ["kda-sig-lowrank", "ckda-shipped-lowrank"]
HYBRID_ARMS = ["kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"]

# Why the hybrids get a rule and a sentence instead of two more rows.
HYBRID_NOTE = ("ours, hybrid -- 3:1 full NoPE attention. NOT comparable to the "
               "rows below: the published hybrids use 2K sliding-window "
               "attention. Their 8K column is EXTRAPOLATION, past the 4,096 "
               "tokens their attention layers trained at -- NoPE, so no "
               "position table is exceeded, but a sliding-window hybrid "
               "extends by construction where this had to be measured.")

# (column group, harness task, the lengths the paper reports it at).
GRID = [("S-NIAH-1", "niah_single_1", [1024, 2048, 4096, 8192]),
        ("S-NIAH-2", "niah_single_2", [1024, 2048, 4096, 8192]),
        ("S-NIAH-3", "niah_single_3", [1024, 2048, 4096]),
        ("MK-NIAH-1", "niah_multikey_1", [1024, 2048, 4096])]

# published_gdn2_niah_1p3B.json's key per column group.
PUB_KEY = {"S-NIAH-1": "s-niah-1", "S-NIAH-2": "s-niah-2",
           "S-NIAH-3": "s-niah-3", "MK-NIAH-1": "mk-niah-1"}

NOT_RUN = -1.0

# Filename markers that mean "this file is a diagnostic, not a measurement".
# eval_ruler.sbatch's TAG puts them there; read_arm refuses to merge them.
# Any TAG containing one of these marks a file as diagnostic. `_samples` and
# `_window` are here for the same reason as `_ablate`: a windowed arm is not
# the model that trained, and a --limit'd sample run is a real model at n=200,
# which merged over an n=500 cell would silently coarsen a published number.
DIAGNOSTIC_TAGS = ("_ablate", "_window", "_samples", "_diag")


def read_arm(results: Path, arm: str, step: int) -> dict:
    """EVERY pass of one arm, merged. Missing files are missing cells, not an
    error: a smoke run produces one pass and should still render.

    Globbed rather than named, so a later pass ADDS cells instead of replacing
    a file. The hybrids' 8K column was run separately, months of GPU-time after
    their 1K/2K/4K -- writing it into `_long.json` would have destroyed the
    three lengths already there.

    THE SENTINEL MUST NOT WIN A MERGE. `common_utils.process_results` always
    returns `{"4096": -1.0, <the length it ran>: score}`, so an 8K-only pass
    carries a -1.0 for 4096. Merged naively over a pass that really did run
    4096, that sentinel overwrites a real score with "never run" and the cell
    silently empties.
    """
    out = {}
    for path in sorted(results.glob(f"{arm}_step{step}_*.json")):
        # DIAGNOSTIC RUNS ARE NOT RESULTS. `--ablate-attn` writes a real
        # results file for a model that does not exist -- a hybrid with its
        # attention silenced -- and the glob above would merge it into the
        # table as readily as a measurement. Diagnostics are tagged and skipped
        # here as well as written to their own directory: one of those two is
        # enough right up until someone points --results at the wrong place.
        if any(mark in path.name for mark in DIAGNOSTIC_TAGS):
            continue
        # SIDECARS ARE NOT RESULTS. eval_downstream writes `<out>.meta.json`
        # (what the run was -- context prefix, tokenizer, limit) and, under
        # `--log-samples`, `<out>.samples.json`, both beside the results file
        # and both matching the glob above. `.meta.json` does not fail
        # quietly: its `{"tasks": [...]}` reaches the loop below and raises
        # `'str' object has no attribute 'items'` from somewhere that looks
        # like a corrupt result.
        #
        # Tested by SUFFIX COUNT rather than by name, so the next sidecar is
        # excluded before it is written: an arm name carries hyphens and a
        # step is digits, so a real results file has exactly one suffix.
        if len(path.suffixes) > 1:
            continue
        for task, vals in json.loads(path.read_text()).items():
            dst = out.setdefault(task, {})
            for key, value in vals.items():
                if (key in dst and isinstance(value, (int, float))
                        and float(value) == NOT_RUN):
                    continue
                dst[key] = value
    return out


def cell(vals: dict, length: int):
    """One cell as a PERCENTAGE, or None when it was not run.

    lm-eval reports the metric under "<length>,none" -- the filter name is part
    of the key -- and `string_match_all` returns a FRACTION. The published table
    is in percent, so this multiplies; a value that is already above 1 means the
    harness changed under us and the table would be wrong by 100x, so it stops
    rather than renders.
    """
    if vals is None:
        return None
    raw = vals.get(f"{length},none", vals.get(str(length)))
    if raw is None:
        return None
    raw = float(raw)
    if raw == NOT_RUN:
        return None
    if raw > 1.0:
        raise SystemExit(
            f"a RULER cell came back as {raw}, above 1.0. This renderer assumes "
            f"lm-eval's `string_match_all`, which returns a fraction; if the "
            f"harness now returns percent, every number here is 100x too big.")
    return 100.0 * raw


def collect(results: Path, arms, step: int) -> dict:
    return {arm: read_arm(results, arm, step) for arm in arms}


def _fmt(v):
    # Six wide, because "100.0" is five and every recurrent row's 1K cells are
    # 100.0 -- at five the columns run together and the table reads "100.0100.0".
    return "    --" if v is None else f"{v:6.1f}"


def render_text(rows, pub, step):
    groups = [(g, lens) for g, _, lens in GRID]
    head1 = f"{'':28s}" + "".join(
        f"{g:^{6 * len(lens)}s}" for g, lens in groups)
    head2 = f"{'':28s}" + "".join(
        "".join(f"{(str(n // 1024) + 'K'):>6s}" for n in lens) for _, lens in groups)
    print(f"RULER needle retrieval, fwedu_1p3B at step {step:,} "
          f"(accuracy %, 500 samples a cell)")
    print(head1)
    print(head2)
    print("-" * len(head2))
    kind = None
    for label, res, row_kind in rows:
        if row_kind != kind:
            if kind is not None:
                print("-" * len(head2))
            if row_kind == "hybrid":
                print(f"  [{HYBRID_NOTE}]")
            kind = row_kind
        line = f"{label:28s}"
        for _, task, lens in GRID:
            for n in lens:
                line += _fmt(cell(res.get(task), n))
        print(line)
    if pub:
        print("-" * len(head2))
        for name, r in pub["rows"].items():
            line = f"{name:28s}"
            for group, _, lens in GRID:
                for n in lens:
                    line += _fmt(r.get(PUB_KEY[group], {}).get(str(n)))
            print(line)
        print(f"\npublished rows: {pub['source']}")
        print(f"  {pub['note']}")


def render_markdown(rows, pub, step):
    cols = [(g, n) for g, _, lens in GRID for n in lens]
    print(f"### RULER needles, fwedu_1p3B at step {step:,}\n")
    print("| Model | " + " | ".join(f"{g} {n // 1024}K" for g, n in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    kind = None
    for label, res, row_kind in rows:
        if row_kind != kind and row_kind == "hybrid":
            print(f"| *{HYBRID_NOTE}* |" + " |" * len(cols))
        kind = row_kind
        cells = [_fmt(cell(res.get(task), n)).strip()
                 for _, task, lens in GRID for n in lens]
        print(f"| {label} | " + " | ".join(cells) + " |")
    if pub:
        for name, r in pub["rows"].items():
            cells = [_fmt(r.get(PUB_KEY[g], {}).get(str(n))).strip() for g, n in cols]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print(f"\nPublished rows: {pub['source']}. {pub['note']}")


def _tex(v):
    """One cell for LaTeX. An unrun cell is an em dash, never a number."""
    return "---" if v is None else f"{v:.1f}"


def context_prefix_of(results: Path, arms, step: int) -> str:
    """Which `--context-prefix` these cells were measured with.

    WHY THIS IS IN THE HEADER. Both conventions were measured, and a rendered
    table is otherwise indistinguishable between them. Only the no-BOS one is
    published, so the header is what tells a reader which they are holding. A leading `<s>` moved single-
    needle cells here by up to 18 points, so a file that does not say which it
    is cannot be compared with anything.

    A run predating `--context-prefix` wrote no sidecar, and every such run
    tokenised with specials ON; "bos" is therefore the right reading of a
    missing file rather than "unknown".
    """
    seen = set()
    for arm in arms:
        for meta in sorted(results.glob(f"{arm}_step{step}_*.meta.json")):
            seen.add(json.loads(meta.read_text()).get("context_prefix", "bos"))
        if not list(results.glob(f"{arm}_step{step}_*.meta.json")):
            seen.add("bos")
    return "+".join(sorted(seen)) if seen else "unknown"


def render_latex(rows, pub, step, prefix="unknown", results=None):
    """The paper's table.

    FOURTEEN numeric columns in four groups, so the column spec is grouped and
    every group gets its own `\\cmidrule` -- one rule spanning all of them reads
    as a single 14-wide header and loses which length belongs to which task.
    `\\tabcolsep` is tightened and the body set `\\small` because at the default
    the table is wider than \\textwidth and silently overflows the margin.

    Requires `booktabs`. Cells that were not run print `---`, never 0.0.
    """
    cols = [(g, n) for g, _, lens in GRID for n in lens]
    ncol = len(cols)
    print("% generated by lm_scaling/niah_table.py --format latex "
          f"--step {step}; DO NOT EDIT -- regenerate instead")
    print(f"%   published rows transcribed in lm_scaling/{PUBLISHED.name}")
    # RELATIVE TO THE REPOSITORY. `results` defaults to an absolute path under
    # this file, so printing it verbatim stamps the developer's home directory
    # into a file that ships.
    where = ""
    if results:
        r = Path(results).resolve()
        root = Path(__file__).resolve().parent.parent
        where = f", from {r.relative_to(root) if r.is_relative_to(root) else r}"
    print(f"%   OUR cells: --context-prefix {prefix}{where}")
    print("%   requires \\usepackage{booktabs}")
    print("\\begin{table}[t]")
    print("  \\centering")
    print("  \\small")
    print("  \\setlength{\\tabcolsep}{3.5pt}")
    # One group of `c`s per task, so the visual grouping matches the header.
    spec = "l" + "".join(" " + "c" * len(lens) for _, _, lens in GRID)
    print(f"  \\begin{{tabular}}{{{spec}}}")
    print("    \\toprule")
    spans = " & ".join(f"\\multicolumn{{{len(lens)}}}{{c}}{{{g}}}"
                       for g, _, lens in GRID)
    print(f"    Model & {spans} \\\\")
    # A cmidrule per group: 2-5, 6-9, 10-12, 13-15 for the paper's grid.
    at, rules = 2, []
    for _, _, lens in GRID:
        rules.append(f"\\cmidrule(lr){{{at}-{at + len(lens) - 1}}}")
        at += len(lens)
    print("    " + " ".join(rules))
    print("     & " + " & ".join(f"{n // 1024}K" for _, n in cols) + " \\\\")
    print("    \\midrule")
    kind = None
    for label, res, row_kind in rows:
        if row_kind != kind and row_kind == "hybrid":
            print("    \\midrule")
            print(f"    \\multicolumn{{{1 + ncol}}}{{l}}{{\\footnotesize\\itshape "
                  f"{HYBRID_NOTE}}} \\\\")
        kind = row_kind
        cells = [_tex(cell(res.get(task), n))
                 for _, task, lens in GRID for n in lens]
        print(f"    {label} & " + " & ".join(cells) + " \\\\")
    if pub:
        print("    \\midrule")
        print(f"    \\multicolumn{{{1 + ncol}}}{{l}}{{\\footnotesize\\itshape "
              f"published, recurrent}} \\\\")
        for name, r in pub["rows"].items():
            cells = [_tex(r.get(PUB_KEY[g], {}).get(str(n))) for g, n in cols]
            print(f"    {name} & " + " & ".join(cells) + " \\\\")
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print(f"  \\caption{{RULER needle retrieval at 1.3B parameters / 100B "
          f"FineWeb-Edu tokens, 4K training sequences; accuracy (\\%) over 500 "
          f"samples per cell. Published rows are the recurrent block of "
          f"{pub['source'] if pub else 'n/a'}. "
          f"Our hybrid rows are evaluated only at or below 4,096 tokens, the "
          f"context their attention layers trained at.}}")
    print("  \\label{tab:fwedu-1p3b-niah}")
    print("\\end{table}")


RENDERERS = {"text": render_text, "markdown": render_markdown, "latex": render_latex}

# The pairing: each bounded-gate arm and the signed arm it is matched with.
PAIRS = (("kda-sig-lowrank", "ckda-shipped-lowrank", "recurrent"),
         ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank", "hybrid"))


def render_pairing(res, step):
    """Signed minus bounded, per cell.

    THE COMPARISON IS PAIRED, and that is a property of RULER rather than a
    choice: `prepare_niah` fixes RANDOM_SEED = 42, so both arms are asked the
    same 500 questions in the same order at every length. The difference is
    therefore over matched items, and the per-cell binomial error -- 2.2 points
    at p = 0.5, n = 500 -- OVERSTATES the error on the difference. The ladder
    saw exactly this: cell-to-cell sd 0.43 against a binomial se of 0.53,
    because both arms scored identical items.

    What it does NOT license is reading every positive cell as a win. Fourteen
    cells invite one to be large by chance; the question this row answers is
    whether the signs agree across cells, not whether any single cell is big.
    """
    print(f"\nPaired difference, signed minus bounded gate (step {step:,})")
    print("Same 500 samples per cell for both arms (RULER seed 42), so this is")
    print("a matched comparison; read the AGREEMENT ACROSS CELLS, not one cell.")
    for lo, hi, kind in PAIRS:
        if lo not in res or hi not in res:
            continue
        diffs, line = [], f"  {kind:10s}"
        for _, task, lens in GRID:
            for n in lens:
                a, b = cell(res[lo].get(task), n), cell(res[hi].get(task), n)
                if a is None or b is None:
                    line += "    --"
                    continue
                diffs.append(b - a)
                line += f"{b - a:+6.1f}"
        print(line)
        if diffs:
            pos = sum(1 for d in diffs if d > 0)
            mean = sum(diffs) / len(diffs)
            print(f"    {'':8s}mean {mean:+.2f} points over {len(diffs)} cells; "
                  f"signed ahead in {pos}/{len(diffs)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--format", choices=sorted(RENDERERS), default="text")
    ap.add_argument("--no-published", action="store_true",
                    help="ours alone, without the transcribed rows")
    ap.add_argument("--pairs", action="store_true",
                    help="append signed-minus-bounded per cell. The samples are "
                         "identical across arms (RULER seed 42), so the "
                         "difference is paired and the binomial se is "
                         "conservative for it")
    a = ap.parse_args(argv)

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    res = collect(a.results, arms, a.step)
    rows = [(f"{DISPLAY.get(arm, arm)} (ours)", res[arm],
             "hybrid" if "hybrid" in arm else "recurrent") for arm in arms]
    # Recurrent first, so the hybrid block and its caveat sit between our
    # comparable rows and the published ones rather than above both.
    rows.sort(key=lambda r: r[2] != "recurrent")
    pub = None if a.no_published else json.loads(PUBLISHED.read_text())
    if a.format == "latex":
        RENDERERS[a.format](rows, pub, a.step,
                            context_prefix_of(a.results, arms, a.step), a.results)
    else:
        RENDERERS[a.format](rows, pub, a.step)
    if a.pairs:
        render_pairing(res, a.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
