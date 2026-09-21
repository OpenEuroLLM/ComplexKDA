"""Render the fwedu arms' downstream results the way the table they answer is.

    lm_scaling/eval_table.py --step 190976
    lm_scaling/eval_table.py --results DIR --step 190976 --markdown
    lm_scaling/eval_table.py --step 190976 --convention gdn2 --format paper \
        > lm_scaling/tex/fwedu_1p3B.tex     # the paper's file: both tables

Gated DeltaNet's Table 3 (arXiv:2412.06464) is the 1.3B/100BT FineWeb-Edu table
later papers copy rows from. It reports, per model: wikitext and LAMBADA
perplexity, then accuracy on LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c,
SIQA and BoolQ, and an average of the accuracies.

WHICH METRIC, because the harness reports several per task and the table names
one. HellaSwag and ARC-c are reported length-normalized (`acc_norm`) and the
rest plain (`acc`) -- that is the convention of the published table and of the
rows it is compared with. Mixing them silently moves HellaSwag by 2-4 points,
which is larger than every difference this campaign is measuring.

THE AVERAGE is over the eight accuracies, perplexities excluded. It is the
column readers compare, so it is computed here rather than by hand, and an arm
missing a task gets no average at all rather than one over seven.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The published rows this campaign is measured against, transcribed from the
# paper rather than remembered. Printed under our own so the comparison is on
# one screen -- and so nobody has to re-fetch a PDF to read a result.
PUBLISHED = Path(__file__).resolve().parent / "published_1p3B.json"  # the default convention's

# The campaign's four arms, in the order the pairing reads: each linear arm
# beside the signed arm it is paired with.
ARMS = ["kda-sig-lowrank", "ckda-shipped-lowrank",
        "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"]

# (column, task, harness metric). Perplexities first, then the accuracies the
# average runs over.
PPL = [("wiki.ppl", "wikitext", "word_perplexity,none"),
       ("lmb.ppl", "lambada_openai", "perplexity,none")]

# Gated DeltaNet's Table 3: eight accuracies, ARC-c and HellaSwag normalized.
ACC = [("lmb.acc", "lambada_openai", "acc,none"),
       ("piqa", "piqa", "acc,none"),
       ("hella", "hellaswag", "acc_norm,none"),
       ("wino", "winogrande", "acc,none"),
       ("arc-e", "arc_easy", "acc,none"),
       ("arc-c", "arc_challenge", "acc_norm,none"),
       ("siqa", "social_iqa", "acc,none"),
       ("boolq", "boolq", "acc,none")]

# Gated DeltaNet-2's Table 2: NINE accuracies -- OpenBookQA joins -- and ARC-c
# is plain `acc` there, not normalized. The difference is not cosmetic: the same
# Gated DeltaNet model reads 38.39 in the older table and 35.15 in the newer
# one, which is larger than the gap between most of the rows being compared.
ACC_GDN2 = [("lmb.acc", "lambada_openai", "acc,none"),
            ("piqa", "piqa", "acc,none"),
            ("hella", "hellaswag", "acc_norm,none"),
            ("wino", "winogrande", "acc,none"),
            ("arc-e", "arc_easy", "acc,none"),
            ("arc-c", "arc_challenge", "acc,none"),
            ("obqa", "openbookqa", "acc,none"),
            ("siqa", "social_iqa", "acc,none"),
            ("boolq", "boolq", "acc,none")]

_HERE = Path(__file__).resolve().parent

# The per-arm scores, COMMITTED, so the tables in the paper can be regenerated
# without the cluster. `eval_fwedu.sbatch` writes fresh ones under
# $EVAL_ROOT/results; point --results there to compare a rerun against what we
# published. DECLARED HERE and not at the top of the file: `_HERE` is defined
# on the line above, and the original position was forty lines before it.
RESULTS = _HERE / "harvest" / "downstream"
RECALL_RESULTS = _HERE / "harvest" / "recall"
CONVENTIONS = {
    "gdn": (ACC, _HERE / "published_1p3B.json"),
    "gdn2": (ACC_GDN2, _HERE / "published_gdn2_1p3B.json"),
}

# Based's recall-intensive suite, appended to either convention with
# `--with-recall`. Produced by eval_recall.sbatch, read from its own directory,
# and rendered by recall_table.py on its own -- this is the joined view.
RECALL = [("squad", "squad_completion", "contains,none"),
          ("swde", "swde", "contains,none"),
          ("fda", "fda", "contains,none")]

# WHY THESE COLUMNS DO NOT JOIN THE AVERAGE. `Avg.` is the number every
# published row in this table is quoted at, and both papers define it over
# their common-sense accuracies alone. Adding three columns to it would move
# our rows and not theirs, which is a rigged comparison rather than a richer
# one. The recall suite therefore carries `Rec.`, an average over its own three
# tasks, and the published rows are blank in all four -- no paper here reports
# them.
RECALL_AVG = "rec.avg"


def read_recall(results: Path, arm: str, step: int) -> dict | None:
    f = results / f"{arm}_step{step}.json"
    return json.loads(f.read_text()) if f.exists() else None


def context_prefix_of(results: Path, arm: str, step: int) -> str:
    """How this run tokenised the start of a context.

    `eval_downstream.py` records it in a `.meta.json` sidecar. A results file
    with NO sidecar was written before `--context-prefix` existed, and that
    harness tokenised with specials on: the fwedu tokenizer copy sets
    `add_bos_token: True`, so those runs carry an implicit `<s>`. Missing
    therefore means "bos" -- it is not unknown, it is the old default.
    """
    meta = results / f"{arm}_step{step}.meta.json"
    if not meta.exists():
        return "bos"
    return json.loads(meta.read_text()).get("context_prefix", "bos")


def check_one_convention(downstream: Path, recall: Path, arms, step: int) -> str:
    """Refuse to put two tokenisations in one table.

    THE FAILURE THIS PREVENTS is a table that renders perfectly and is not a
    comparison: the common-sense columns measured with a leading `<s>` and the
    recall columns without one, in the same row, under one model name. Nothing
    in the output would say so, and the sidecar that could have said so is the
    file this reads.

    Costs nothing when they agree and stops the run when they do not, because
    the fix is a re-run and not a footnote.
    """
    seen = {}
    for arm in arms:
        for tag, d in (("downstream", downstream), ("recall", recall)):
            if (d / f"{arm}_step{step}.json").exists():
                seen.setdefault(context_prefix_of(d, arm, step), []).append(
                    f"{tag}:{arm}")
    if len(seen) > 1:
        lines = "\n".join(f"    --context-prefix {p}: {', '.join(w)}"
                          for p, w in sorted(seen.items()))
        raise SystemExit(
            f"refusing to join tables measured with different context "
            f"prefixes:\n{lines}\n"
            f"  A leading <s> changes what was scored, so these columns are "
            f"not one measurement of one model. Re-run the lagging side "
            f"(eval_fwedu.sbatch / eval_recall.sbatch both take "
            f"CONTEXT_PREFIX) and join then. `--force-mixed-prefix` renders it "
            f"anyway, for looking at -- never for publishing.")
    return next(iter(seen), "none")


def read(results: Path, arm: str, step: int) -> dict | None:
    f = results / f"{arm}_step{step}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def row(res: dict, acc=None, rec: dict | None = None) -> dict:
    acc = acc if acc is not None else ACC
    out = {}
    for col, task, metric in PPL + acc:
        vals = res.get(task) or {}
        out[col] = vals.get(metric)
    accs = [out[c] for c, _, _ in acc]
    out["avg"] = sum(accs) / len(accs) if all(a is not None for a in accs) else None
    if rec is not None:
        for col, task, metric in RECALL:
            out[col] = (rec.get(task) or {}).get(metric)
        got = [out[c] for c, _, _ in RECALL]
        out[RECALL_AVG] = (sum(got) / len(got)
                           if all(g is not None for g in got) else None)
    return out


# Paper-facing row labels. `ckda-shipped-lowrank` is a filename; a table needs a
# name a reader can use, and what identifies an arm is its gate and its stack,
# not the init tag that distinguishes it from an arm nobody published.
DISPLAY = {
    "kda-sig-lowrank": "KDA, bounded gate",
    "ckda-shipped-lowrank": "CKDA",
    "kda-sig-hybrid-lowrank": "KDA + attn 3:1",
    "ckda-shipped-hybrid-lowrank": "CKDA + attn 3:1",
}

# The published tables' own column names, so a generated table can sit beside a
# transcribed one without a reader having to map the headings.
HEADERS = {"wiki.ppl": "Wiki.", "lmb.ppl": "LMB.", "lmb.acc": "LMB.",
           "piqa": "PIQA", "hella": "Hella.", "wino": "Wino.", "arc-e": "ARC-e",
           "arc-c": "ARC-c", "obqa": "OBQA", "siqa": "SIQA", "boolq": "BoolQ",
           "avg": "Avg.",
           "squad": "SQuAD", "swde": "SWDE", "fda": "FDA", "rec.avg": "Rec."}

# Our own held-out validation loss at the final step, which is NOT in any table
# here: it is measured on our FineWeb-Edu split and compares our arms with each
# other and with nobody else. Kept beside the downstream numbers because the two
# disagree about the pairs, and a reader who sees only one is missing the result.
VAL_LOSS = {"kda-sig-lowrank": 2.0287, "ckda-shipped-lowrank": 2.0316,
            "kda-sig-hybrid-lowrank": 2.0181, "ckda-shipped-hybrid-lowrank": 2.0195}

PAIRS = (("kda-sig-lowrank", "ckda-shipped-lowrank", "recurrent"),
         ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank", "hybrid"))

# The replication check: the SAME baseline-vs-complex pairing, scored the same
# way, over the SIXTY paired cells of the Megatron scaling ladder (six rungs,
# 47M to 1.7B, at 6-50BT). Recomputed when the ladder moved to Megatron: the
# torchtitan ladder it replaced read -0.452 over 18 cells, and quoting that
# here would attribute a superseded framework's number to this one.
#
# It belongs in the paper file because it is the reason the caption there does
# not claim a downstream advantage. This campaign is n = 1 per arm; measured
# eighteen times the same difference goes the OTHER way, and a single
# favourable draw published without that context becomes a claim it cannot
# support. `test_eval_table.py` RECOMPUTES these from `harvest/ladder_downstream/`
# rather than pinning the literals -- an earlier version pinned them, the ladder
# was re-scored, and the mean here sat at -0.027 against a measured -0.020 until
# the recomputation caught it.
LADDER_AVG9 = {"mean": -0.020, "se": 0.084, "cells": 60, "negative": 31}


def load_published(path: Path):
    """The published rows IN OUR UNITS: accuracies as fractions, ppl as printed.

    The papers print percentages and the harness reports fractions. Two unit
    systems in one table is how a bolded "best" lands on the wrong row, so the
    conversion happens once, here.
    """
    pub = json.loads(path.read_text())
    rows = {name: {k: (v if (k.endswith("ppl") or k == "group") else v / 100)
                   for k, v in r.items()}
            for name, r in pub["rows"].items()}
    return pub, rows


def best(values, col: str):
    """The winning value in a column: lowest perplexity, highest accuracy."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    return min(vals) if col.endswith("ppl") else max(vals)


def fmt(col: str, v: float | None, width: int = 8) -> str:
    # Every cell exactly as wide as its header, so the columns line up under it.
    if v is None:
        return f"{'-':>{width}s}"
    return f"{v:{width}.2f}" if col.endswith("ppl") else f"{100 * v:{width}.1f}"


def cell(col: str, v: float | None, winner=None, mark: str = "**{}**") -> str:
    """A typeset cell, two decimals as both papers print them.

    Bolding compares the PRINTED value, not the stored one. Ours arrive from the
    harness at full precision and the published rows were transcribed at two
    decimals, so 0.727965 and 0.7280 both print 72.80 while one is strictly
    larger -- bolding that one and not the other puts two identical numbers in a
    column with a bold on only one of them, which reads as a bug in the table
    rather than as the seven-thousandth of a point it is.
    """
    if v is None:
        return "--"

    def shown(x):
        return f"{x:.2f}" if col.endswith("ppl") else f"{100 * x:.2f}"
    s = shown(v)
    return mark.format(s) if winner is not None and s == shown(winner) else s


def family(arm: str) -> str:
    """Which block a row belongs in. Both papers separate the hybrids."""
    return "hybrid" if "hybrid" in arm else "recurrent"


def blocks(rows: dict, pub_rows: dict, show_published: bool):
    """(title, [(label, values)]) in the order a table should print them.

    Published rows first inside each family, ours after, so a reader compares
    downward within a family rather than across the page.
    """
    out = []
    for fam, title in (("recurrent", "Recurrent models"),
                       ("hybrid", "Attention or hybrid models")):
        entries = []
        if show_published:
            entries += [(name, r) for name, r in pub_rows.items()
                        if r.get("group") == fam]
        entries += [(f"{DISPLAY.get(a, a)} (ours)", r) for a, r in rows.items()
                    if family(a) == fam and r is not None]
        if entries:
            out.append((title, entries))
    return out


def winners(entries, cols):
    """The bolded value per column, over the rows given.

    WHICH ROWS is the whole question, and it is `--bold`. Pooling both blocks
    makes one winner per column across the table, and since a hybrid beats a
    pure recurrent model on nearly everything, nine of twelve bolds land in the
    hybrid block and the recurrent block reads as though nothing won there --
    including our own baseline row, which takes none. Bolding WITHIN each block
    is what a blocked table means: the recurrent rows are compared with each
    other, which is the comparison the block exists to make.
    """
    return {c: best([r[c] for _, r in entries], c) for c in cols}


def deltas(rows: dict):
    """Signed minus baseline, per pair: the experiment, not the rows."""
    out = []
    for base, signed, tag in PAIRS:
        b, s = rows.get(base), rows.get(signed)
        if not (b and s) or b["avg"] is None or s["avg"] is None:
            continue
        out.append((tag,
                    100 * (s["avg"] - b["avg"]),
                    s["wiki.ppl"] - b["wiki.ppl"],
                    s["lmb.ppl"] - b["lmb.ppl"],
                    VAL_LOSS.get(signed, float("nan")) - VAL_LOSS.get(base, float("nan"))))
    return out


def render_text(rows, pub, pub_rows, cols, acc, step, convention, show_published,
                bold="block"):
    # `bold` is accepted and ignored: this one is for reading in a terminal and
    # marks nothing.
    arc = {t: m for _, t, m in acc}["arc_challenge"].split(",")[0]
    head = f"{'arm':30s}" + "".join(f"{c:>8s}" for c in cols)
    sep = "-" * len(head)
    print(f"fwedu_1p3B downstream at step {step}   "
          f"[{convention} convention: {len(acc)} accuracies, ARC-c {arc}]")
    print(head)
    print(sep)
    for arm, r in rows.items():
        if r is None:
            print(f"{arm:30s}" + "  not evaluated")
            continue
        print(f"{arm:30s}" + "".join(fmt(c, r[c]) for c in cols))
    print(sep)
    if show_published:
        print(f"published, {pub['setup'].split(',', 1)[1].strip()} "
              f"({pub['source'].split(',')[-2].strip()} "
              f"{pub['source'].split(',')[-1].strip()}):")
        for name, r in pub_rows.items():
            print(f"{name:30s}" + "".join(fmt(c, r[c]) for c in cols))
        print(sep)
        print("  A published row is the FINAL step of a full budget. An "
              "intermediate checkpoint of a")
        print("  cosine run is not a smaller run: its learning rate has not "
              "annealed, and most of the")
        print("  perplexity gain arrives in the last fifth.")
        print(sep)
    for tag, davg, dwiki, dlmb, dval in deltas(rows):
        print(f"  {tag:9s} signed - baseline: avg acc {davg:+.2f} pts | "
              f"wiki ppl {dwiki:+.2f} | lmb ppl {dlmb:+.2f} | val loss {dval:+.4f}")


def render_markdown(rows, pub, pub_rows, cols, acc, step, convention,
                    show_published, bold="block"):
    print(f"### fwedu_1p3B at step {step:,} ({convention} convention: "
          f"{len(acc)} accuracies, "
          f"ARC-c { {t: m for _, t, m in acc}['arc_challenge'].split(',')[0]})")
    print()
    grouped = blocks(rows, pub_rows, show_published)
    pooled = winners([e for _, entries in grouped for e in entries], cols)
    print("| Model | " + " | ".join(HEADERS.get(c, c) for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for title, entries in grouped:
        win = winners(entries, cols) if bold == "block" else pooled
        print(f"| *{title}* |" + " |" * len(cols))
        for label, r in entries:
            print(f"| {label} | "
                  + " | ".join(cell(c, r[c], win[c]) for c in cols) + " |")
    print()
    scope = "in the column within its block" if bold == "block" else \
            "in the column, across both blocks"
    print(f"Perplexities lower is better, accuracies (%) higher; **bold** is the "
          f"best value {scope}. Published rows: {pub['source']}.")
    print()
    for tag, davg, dwiki, dlmb, dval in deltas(rows):
        print(f"- {tag}: signed − baseline = **{davg:+.2f}** points of average "
              f"accuracy, wiki ppl {dwiki:+.2f}, LAMBADA ppl {dlmb:+.2f}, "
              f"held-out loss {dval:+.4f} (our split; lower is better).")


def render_latex(rows, pub, pub_rows, cols, acc, step, convention, show_published,
                 bold="block", with_header=True, prefix="unknown"):
    n_acc = len(acc)
    spec = "l" + "r" * (len(cols))
    ppl_end, acc_end = 3, 3 + n_acc
    # The recall suite, when present, is its own span: it is neither a
    # perplexity nor one of the accuracies the `Avg.` runs over, and putting it
    # under the accuracy rule would say it was.
    n_rec = sum(1 for c in cols if c in {c2 for c2, _, _ in RECALL}) \
        + (1 if RECALL_AVG in cols else 0)
    grouped = blocks(rows, pub_rows, show_published)
    pooled = winners([e for _, entries in grouped for e in entries], cols)
    if with_header:
        print("% Generated by lm_scaling/eval_table.py -- regenerate rather than edit:")
        print(f"%   lm_scaling/eval_table.py --step {step} --convention {convention} "
              f"--format latex" + (" --with-recall" if n_rec else ""))
        print(f"% Ours: fwedu_1p3B, {len(rows)} arms at step {step}, scored at the "
              f"model's 4,096-token context.")
        # WHICH TOKENISATION. A leading <s> moves these numbers -- on the RULER
        # cells by up to 18 points -- so a table that does not say which
        # convention it is cannot be compared with one that does.
        print(f"% Context prefix: {prefix}.")
        print(f"% Published: {pub['source']}.")
        print(f"% {pub.get('metrics', '')}")
    print(r"\begin{table*}[t]")
    print(r"  \centering")
    print(r"  \footnotesize")
    print(r"  \setlength{\tabcolsep}{3.5pt}")
    caption = (f"Language modelling and zero-shot common-sense reasoning at 1.3B "
               f"parameters on 100B tokens of FineWeb-Edu. Our rows are trained "
               f"with the published recipe (AdamW, peak LR $4\\times10^{{-4}}$, "
               f"cosine to $0.1\\times$ peak after a 1B-token warm-up, 0.5M-token "
               f"batches, 4K sequences) and scored with lm-eval-harness at the "
               f"model's 4{{,}}096-token context. Perplexities lower is better, "
               f"accuracies higher; the average is over the "
               f"{n_acc} accuracies, following the reference table. "
               + ("Best per column within each block in bold."
                  if bold == "block" else
                  "Best per column across both blocks in bold.")
               + (" \\textbf{Rec.} averages the recall-intensive tasks of Arora "
                  "et al. (SQuAD-completion, SWDE, FDA), which put the answer "
                  "in the prompt; they are \\emph{not} part of Avg., which "
                  "follows the reference table, and no published row here "
                  "reports them." if n_rec else "")
               + f" Published numbers from {pub['source']}.")
    print(f"  \\caption{{{caption}}}")
    print(f"  \\label{{tab:fwedu-1p3b-{convention}}}")
    print(f"  \\begin{{tabular}}{{{spec}}}")
    print(r"    \toprule")
    spans = (f"    & \\multicolumn{{2}}{{c}}{{ppl $\\downarrow$}} & "
             f"\\multicolumn{{{n_acc}}}{{c}}{{accuracy (\\%) $\\uparrow$}} & ")
    rules = (f"    \\cmidrule(lr){{2-{ppl_end}}} \\cmidrule(lr){{4-{acc_end}}} "
             f"\\cmidrule(lr){{{acc_end + 1}-{acc_end + 1}}}")
    if n_rec:
        spans += (f"\\multicolumn{{{n_rec}}}{{c}}"
                  f"{{recall (\\%) $\\uparrow$}} ")
        rules += f" \\cmidrule(lr){{{acc_end + 2}-{acc_end + 1 + n_rec}}}"
    print(spans + r"\\")
    print(rules)
    print("    Model & " + " & ".join(HEADERS.get(c, c) for c in cols) + r" \\")
    for title, entries in grouped:
        win = winners(entries, cols) if bold == "block" else pooled
        print(r"    \midrule")
        print(f"    \\multicolumn{{{len(cols) + 1}}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for label, r in entries:
            # r"\textbf{{{}}}" survives .format as \textbf{<value>}; doubling the
            # braces any other way drops the value and bolds an empty cell.
            cells = " & ".join(cell(c, r[c], win[c], r"\textbf{{{}}}")
                               for c in cols)
            print(f"    {label} & {cells} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table*}")
    if with_header:
        print("%")
        print("% The pairing, which is the experiment (signed minus baseline):")
        for tag, davg, dwiki, dlmb, dval in deltas(rows):
            print(f"%   {tag:9s} avg acc {davg:+.2f} pts | wiki ppl {dwiki:+.2f} | "
                  f"lmb ppl {dlmb:+.2f} | held-out loss {dval:+.4f} (our split)")


def render_paper(rows, pub, pub_rows, cols, acc, step, convention, show_published,
                 bold="block", prefix="unknown"):
    """The drop-in file: the comparison table, then the pairing that is the point.

    The comparison table is what a reader looks at and the pairing table is what
    this campaign actually measured -- one arm against the arm it varies, at the
    same seed and data order, where the published rows are other people's runs
    under a recipe we copied but did not share a machine with.

    They are in one file because publishing the first without the second is how
    a favourable draw becomes a claim. It cannot be one: the two metrics
    disagree in sign, and the same pairing over sixty ladder cells reads
    -0.020 +- 0.084, which is indistinguishable from zero.
    """
    print("% " + "=" * 74)
    print("%  The 1.3B / 100BT FineWeb-Edu experiments: two tables, drop-in.")
    print("%")
    print("%  Generated by lm_scaling/eval_table.py -- REGENERATE, do not edit:")
    print(f"%    lm_scaling/eval_table.py --step {step} --convention {convention} \\")
    print("%        --format paper > lm_scaling/tex/fwedu_1p3B.tex")
    print("%")
    print(f"%  Ours: fwedu_1p3B, {len(rows)} arms at step {step}, scored with "
          "lm-eval-harness")
    print(f"%  Context prefix: {prefix} -- see eval_downstream.py "
          "--context-prefix.")
    print("%  at the model's own 4,096-token context (a 2,048 window inflates "
          "WikiText")
    print("%  perplexity by a consistent 5%).")
    print(f"%  Published rows: {pub['source']},")
    print(f"%  transcribed in lm_scaling/{CONVENTIONS[convention][1].name} and "
          "checked against")
    print("%  that paper's own printed average by tests/lm/test_eval_table.py.")
    print("%")
    print("%  NEEDS: booktabs. No new bibliography keys -- the published rows "
          "are")
    print("%  labelled as the source table labels them; add \\citep{} if the "
          "venue")
    print("%  wants per-row citations.")
    print("%")
    print("%  WHAT MAY NOT BE CLAIMED FROM TABLE 1: a downstream advantage. The "
          "signed")
    print(f"%  arm leads the {len(acc)}-task average here, at n = 1. The SAME "
          f"pairing over")
    print(f"%  the ladder's {LADDER_AVG9['cells']} paired cells TRAILS by "
          f"{abs(LADDER_AVG9['mean']):.3f} points "
          f"(se {LADDER_AVG9['se']:.3f}), negative in")
    print(f"%  {LADDER_AVG9['negative']} of the {LADDER_AVG9['cells']}, and "
          f"held-out loss goes the other way in both pairs")
    print("%  here as well. See config/experiments/ladder_results.md. Table 2 "
          "is in")
    print("%  this file so that the one is not read without the other.")
    print("% " + "=" * 74)
    render_latex(rows, pub, pub_rows, cols, acc, step, convention,
                 show_published, bold=bold, with_header=False)
    print()
    render_pairing(rows, acc, convention)


def render_pairing(rows, acc, convention):
    """Table 2: signed minus baseline, on both metrics, with their disagreement.

    Held-out loss is OUR split, so it is comparable between our arms and with
    nobody else -- which is exactly why it is the more trustworthy half of this
    table: same tokenizer, same data order, same seed, same step.
    """
    rowsets = deltas(rows)
    if not rowsets:
        return
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \small")
    caption = (
        f"The experiment, at 1.3B/100BT on FineWeb-Edu: CKDA minus the "
        f"bounded-gate KDA baseline it varies, at the same seed, data order and "
        f"step. Held-out loss is our own FineWeb-Edu split (nats, lower is "
        f"better); the accuracy average is the {len(acc)} tasks of "
        f"Table~\\ref{{tab:fwedu-1p3b-{convention}}}. \\textbf{{The two metrics "
        f"disagree in sign}}, and neither difference is resolvable at one seed: "
        f"the measured seed-noise floor of this pipeline is $\\sim$0.002 nats, "
        f"and the same pairing across the scaling ladder's "
        f"{LADDER_AVG9['cells']} paired cells trails by "
        f"{abs(LADDER_AVG9['mean']):.2f} points of average accuracy "
        f"(se {LADDER_AVG9['se']:.2f}), negative in {LADDER_AVG9['negative']} of "
        f"{LADDER_AVG9['cells']}. The supported claim is that the signed gate "
        f"matches the baseline, not that it beats it.")
    print(f"  \\caption{{{caption}}}")
    print(r"  \label{tab:fwedu-1p3b-pairing}")
    print(r"  \begin{tabular}{lrrrrrr}")
    print(r"    \toprule")
    print(r"    & \multicolumn{3}{c}{held-out loss $\downarrow$} "
          r"& \multicolumn{2}{c}{ppl $\downarrow$} & accuracy $\uparrow$ \\")
    print(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}")
    print(r"    Stack & baseline & signed & $\Delta$ & $\Delta$ Wiki. "
          r"& $\Delta$ LMB. & $\Delta$ Avg. (pts) \\")
    print(r"    \midrule")
    names = {"recurrent": "Pure (KDA)", "hybrid": "Hybrid (KDA + attn 3:1)"}
    # By tag, not by position: `deltas` drops a pair whose arms are not both
    # evaluated, and zipping against PAIRS would then label the surviving row
    # with the other pair's losses.
    arms = {tag: (base, signed) for base, signed, tag in PAIRS}
    for tag, davg, dwiki, dlmb, dval in rowsets:
        base, signed = arms[tag]
        # Signed quantities in math mode, so a difference prints a minus sign
        # and not a hyphen -- they are a glyph apart and this table is nothing
        # but signs.
        print(f"    {names.get(tag, tag)} & {VAL_LOSS[base]:.4f} & "
              f"{VAL_LOSS[signed]:.4f} & ${dval:+.4f}$ & ${dwiki:+.2f}$ & "
              f"${dlmb:+.2f}$ & ${davg:+.2f}$ \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


RENDERERS = {"text": render_text, "markdown": render_markdown,
             "latex": render_latex, "paper": render_paper}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--format", choices=sorted(RENDERERS), default="text",
                    help="`text` to read, `markdown` for a results record, "
                         "`latex` for the comparison table alone, `paper` for "
                         "the drop-in file: that table plus the pairing, which "
                         "is what this campaign varied and what the comparison "
                         "table must not be published without.")
    ap.add_argument("--markdown", action="store_true",
                    help="deprecated alias for --format markdown")
    ap.add_argument("--no-published", action="store_true",
                    help="ours alone, without the published rows")
    ap.add_argument("--bold", choices=("block", "global"), default="block",
                    help="which rows a bolded cell wins against. `block` is "
                         "within Recurrent or within Hybrid, which is what a "
                         "blocked table means; `global` pools them, and since "
                         "a hybrid beats a pure recurrent model on nearly "
                         "every column that leaves the recurrent block with "
                         "almost no bold at all.")
    ap.add_argument("--convention", choices=sorted(CONVENTIONS), default="gdn",
                    help="which paper's rules: `gdn` is Gated DeltaNet's Table 3 "
                         "(eight accuracies, ARC-c normalized); `gdn2` is Gated "
                         "DeltaNet-2's Table 2 (nine, with OpenBookQA, ARC-c "
                         "plain). Each prints its own published rows.")
    ap.add_argument("--with-recall", action="store_true",
                    help="append Based's recall-intensive suite (SQuAD-completion, "
                         "SWDE, FDA) from --recall-results, with its own `Rec.` "
                         "average. The published rows are blank there -- no paper "
                         "in either convention reports this suite -- and the "
                         "table's own `Avg.` is left over the common-sense "
                         "accuracies, because that is the number every published "
                         "row is quoted at.")
    ap.add_argument("--recall-results", type=Path, default=RECALL_RESULTS)
    ap.add_argument("--force-mixed-prefix", action="store_true",
                    help="join columns measured with DIFFERENT context prefixes. "
                         "For looking at, never for publishing: a leading <s> "
                         "changes what was scored, so the row is then not one "
                         "measurement of one model.")
    a = ap.parse_args(argv)

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    acc, published = CONVENTIONS[a.convention]
    cols = [c for c, _, _ in PPL + acc] + ["avg"]
    if a.with_recall:
        cols += [c for c, _, _ in RECALL] + [RECALL_AVG]
        if not a.force_mixed_prefix:
            check_one_convention(a.results, a.recall_results, arms, a.step)
    rows = {}
    for arm in arms:
        res = read(a.results, arm, a.step)
        rec = read_recall(a.recall_results, arm, a.step) if a.with_recall else None
        rows[arm] = row(res, acc, rec) if res else None

    pub, pub_rows = ({}, {})
    show = published.exists() and not a.no_published
    if show:
        pub, pub_rows = load_published(published)

    # EVERY ROW CARRIES EVERY COLUMN, so a renderer can index rather than probe.
    # The published rows have no recall columns at all, and an arm evaluated on
    # one side and not the other has some of them; both must print `--` instead
    # of raising on a key that was never going to be there.
    for r in list(rows.values()) + list(pub_rows.values()):
        if r is not None:
            for c in cols:
                r.setdefault(c, None)

    fmt_name = "markdown" if a.markdown else a.format
    kw = {"bold": a.bold}
    # Only the two rendered-for-print formats carry a header to put it in.
    if fmt_name in ("latex", "paper"):
        kw["prefix"] = context_prefix_of(a.results, arms[0], a.step)
    RENDERERS[fmt_name](rows, pub, pub_rows, cols, acc, a.step, a.convention,
                        show, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
