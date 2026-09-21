"""The scaling ladder's architectures, as the table a paper prints.

    lm_scaling/arch_table.py                    > lm_scaling/tex/arch_table.tex
    lm_scaling/arch_table.py --format markdown     # to read
    lm_scaling/arch_table.py --sizes 47M 124M 302M --campaigns ladder
    lm_scaling/arch_table.py --fixed-points        # rungs only, no 1.3B column

WHY GENERATED. Every number here already exists in `ladder_matched.json`, which
`param_match.py` wrote and which `titan_ext.ladder_flavors` builds the models
from, or in the campaign plans. A transcribed table is a second copy that can
disagree with the models actually trained, and the disagreement is invisible:
1344 and 1408 look equally plausible in a column. The warmup of the FineWeb-Edu
column is 1,908 steps and not the 1,907 one computes by hand from "1B tokens",
which is the whole argument in one number.

THE SPLIT IS COMPUTED, NOT ASSUMED. A field is printed once, spanning every
column, only if every column agrees; otherwise it becomes a row. That matters
because the columns are not all rungs: `replication_1p3B.json` is the 1.3B
FineWeb-Edu replication, and it follows the published Gated DeltaNet recipe
rather than the ladder's. Adding it moves head dimension (128 against 64),
vocabulary (32,000 against 50,304) and tying (untied against tied) out of the
constant block by itself. A table that assumed those were constant would print
the ladder's values under a column where all three are false.

PARAMETER MATCHING is why the intermediate size differs per arm and the
parameter counts do not. Each arm's mixer has its own budget -- a gated delta
rule carries projections a transformer does not -- so `param_match` solves for
the `d_ffn` that brings every arm close to its column's target, and the table
prints both so a reader can check the match rather than take it on trust. The
caption states the WORST deviation over the cells shown, computed: 0.81%, which
is close but is not the "fraction of a percent" one is tempted to write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

LADDER = _HERE / "ladder_matched.json"

#: Recipes for the fixed-point campaigns, resolved once from the hydra
#: planner and committed. See `fixed_recipe`.
RECIPES = _HERE / "campaign_recipes.json"

# The arms the ladder campaign actually runs, in the order a reader compares
# them: the attention baseline, the published linear baseline, then our pair
# and their hybrids. `ladder_matched.json` carries 81 arms because it is also
# the ablation table; a paper table is not a list of everything measured.
ARMS = [
    ("attn", "Transformer++"),
    ("attn-qknorm", "Transformer++ (QK-norm)"),
    ("kda-sig-lowrank", "KDA (bounded gate)"),
    ("ckda-shipped-lowrank", "CKDA"),
    ("kda-sig-hybrid-lowrank", "KDA + attn 3:1"),
    ("ckda-shipped-hybrid-lowrank", "CKDA + attn 3:1"),
]

#: Arms that are another arm's geometry with a non-width change. `attn-qknorm`
#: is `attn` plus Megatron's `--qk-layernorm`, which adds normalisation and no
#: width, so `submit_ladder.ARMS` builds it from `attn`'s matched MLP and the
#: matched table has no separate entry for it. The rows come out identical on
#: purpose: that IS the statement, and printing "--" for the arm that every
#: comparison in the paper is measured against would be worse.
#:
#: LADDER RUNGS ONLY. These aliases say what the Megatron ladder built, so they
#: do not reach the fixed-point columns: the FineWeb-Edu campaign defines an
#: `attn` geometry as its matching reference but ran no QK-norm arm, and
#: inheriting one there would invent a configuration.
GEOM_ALIAS = {"attn-qknorm": "attn"}

# `gdn` USED TO BE THE SECOND ROW. It was run on the torchtitan ladder, which
# this release supersedes and does not ship, and it was never run on the
# Megatron one -- so its geometry had no results anywhere in the repository to
# go with it, while `attn-qknorm`, the baseline all 180 cells are compared
# against, had no row at all. `param_match.py` still carries the arm, so
# `--arms gdn ...` brings the column back.

SIZES = ["47M", "124M", "302M", "588M", "983M", "1.7B"]


def load(path: Path = LADDER) -> dict:
    return json.loads(path.read_text())


def recipe(campaigns) -> dict:
    """{size: {"lr": .., "gbs": ..}} from the campaigns that run each rung.

    Read from the planner rather than from the yaml, because a rung's peak
    learning rate is set by a sweep group and not by a literal. Optional: the
    geometry table alone needs no hydra, and this import does.
    """
    import plan as P

    out: dict[str, dict[str, set]] = {}
    for c in campaigns:
        for j in P.plans(c):
            a = j.config.aux
            d = out.setdefault(str(a["size"]), {"lr": set(), "gbs": set(),
                                                "warmup": set(), "decay": set()})
            d["lr"].add(float(a["lr"]))
            d["gbs"].add(int(a["gbs"]))
            # PER FIELD, not per job, because the ladder STAGES its upper rungs:
            # the stable run carries the warmup and cooldown_steps 0, and the
            # cooldown that branches off it carries warmup 0 and the cooldown.
            # Filtering whole jobs on either one reports the other as absent --
            # "0 / 2000" warmups if cooldown stages are kept, "no cooldown" at
            # 588M and above if they are dropped. Both were wrong here first.
            if int(a["warmup_steps"]) > 0:
                d["warmup"].add(int(a["warmup_steps"]))
            if int(a["cooldown_steps"]) > 0:
                d["decay"].add(float(a["decay_fraction"]))
    return out


def fixed_recipe(campaign: str) -> dict:
    """The recipe of a fixed-point campaign, from `campaign_recipes.json`.

    A COMMITTED FACT, not a re-derivation. `recipe()` reads these out of the
    hydra planner, and the planner needs `hydra_staged_sweep` and the campaign's
    config layer -- neither of which is part of this release. Regenerating this
    table on a clean clone therefore used to fail at the FineWeb-Edu column with
    `ModuleNotFoundError`, which made `reproduce_tables.sh` unrunnable for
    anyone but us.

    So the resolved numbers ship, the way every other expensively-derived fact
    in this repository does, and `--write-recipes` regenerates them where the
    planner IS importable. `--campaigns` still routes back through `recipe()`.
    """
    data = json.loads(RECIPES.read_text())
    if campaign not in data:
        raise SystemExit(
            f"{RECIPES.name} has no recipe for {campaign!r}; it has "
            f"{sorted(data)}. Regenerate it with --write-recipes {campaign} "
            f"in an environment where the hydra planner imports.")
    return data[campaign]


def ladder_recipe() -> dict:
    """{size: {"lr": .., "gbs": .., "warmup": .., "decay": ..}} for the rungs.

    From `megatron_ext/submit_ladder.py`, which IS the ladder: its `CELLS` are
    the single-stage (rung, budget, batch, lr) rows and its `CHAINS` the
    trunk-plus-cooldown ones, and both are what was submitted. Warmup and the
    cooldown fraction come out of the root config the submitter copies into the
    autoexp checkout. Read rather than transcribed, for the reason in the module
    docstring -- this table's job is to report what ran, and a literal here is
    how it comes to disagree with the runs.

    The torchtitan campaigns this used to read are gone: the ladder moved to
    Megatron, and their configs are not part of the release. `--campaigns`
    still routes back to `recipe()`, which is how the FineWeb-Edu column gets
    its numbers -- that campaign's plan does live here.
    """
    import yaml

    from megatron_ext import submit_ladder as S

    cfg = yaml.safe_load(
        (_HERE / "megatron_ext" / "config" / "ckda_ladder_jupiter.yaml").read_text())
    warmup = int(cfg["backend"]["megatron"]["lr_warmup_iters"])
    decay = float(cfg["aux"]["decay_fraction"])

    out: dict[str, dict[str, set]] = {}

    def add(rung, gbs, lr):
        d = out.setdefault(rung, {"lr": set(), "gbs": set(),
                                  "warmup": set(), "decay": set()})
        d["lr"].add(float(lr))
        d["gbs"].add(int(gbs))
        # Every cell warms up and every cell cools down, whether it does so in
        # one job or in a trunk plus a branch. The staging is an economy, not a
        # schedule -- so unlike `recipe()`, which reads it off per-job fields
        # and has to filter the zeros out, there is nothing to filter here.
        d["warmup"].add(warmup)
        d["decay"].add(decay)

    for cells in S.CELLS.values():
        for rung, _tokens, gbs, lr in cells:
            add(rung, gbs, lr)
    for chains in S.CHAINS.values():
        for rung, gbs, lr, _budgets in chains:
            add(rung, gbs, lr)
    return out


def worst_match(tab, sizes, arms):
    """The largest |N - target| / target over the cells shown, as a percent.

    Computed rather than described: "parameter-matched" is the table's central
    claim and the tolerance is the evidence for it. It is 0.81% at 47M, which
    is not "a fraction of a percent" however much one would like to write that.
    """
    worst, where = 0.0, None
    for s in sizes:
        tgt = tab[s]["target_N"]
        for arm, label in arms:
            a = tab[s]["archs"].get(GEOM_ALIAS.get(arm, arm))
            if not a:
                continue
            d = 100 * (a["N"] - tgt) / tgt
            if abs(d) > abs(worst):
                worst, where = d, f"{label} at {s}"
    return worst, where


def _fmt_n(n: int) -> str:
    return f"{n / 1e6:.2f}"


def _thin(x: int) -> str:
    """50304 -> 50{,}304, the spelling the reference table uses."""
    s = f"{x:,}"
    return s.replace(",", "{,}")


def _set_row(rec, sizes, key, fmt):
    """One cell per rung, joining the distinct values a rung was run at.

    The ladder sweeps batch size on purpose -- b*(N, D) steps 32 -> 64 -> 128
    with the budget, so a rung run at several budgets was run at several
    batches, and 47M was run at all three. Printing the first would be a quiet
    lie; printing all of them is the fact.

    Peak learning rate happens to be single-valued per rung under the Megatron
    ladder. It was not under the torchtitan one this replaced, where 588M
    carried two, and the joining stays because that is a property of which
    cells ran and not of the code.
    """
    cells = []
    for s in sizes:
        vals = sorted(rec.get(s, {}).get(key, ()))
        cells.append(" / ".join(fmt(v) for v in vals) if vals else "--")
    return cells


def columns(tab, sizes, fixed, vocab, seq, rec):
    """One dict per column: the geometry, the recipe and the corpus.

    A column is a rung OR a fixed point. `replication_1p3B.json` is not a rung
    -- it has its own vocabulary, its own head dimension, untied embeddings and
    a cosine schedule -- which is exactly why it is worth a column and exactly
    why the constant block has to be computed rather than assumed.
    """
    cols = []
    for s in sizes:
        e = tab[s]
        cols.append({"name": s, "geo": e, "archs": e["archs"], "rung": True,
                     "vocab": vocab, "seq": seq, "corpus": "Nemotron-CC",
                     "sched": "WSD", "rec": rec.get(s, {})})
    for path, label, campaign, corpus, sched in fixed:
        e = json.loads(Path(path).read_text())
        # The recipe from the campaign's own plan, like the rungs': a warmup of
        # 1,907 steps is "1B tokens" divided by a batch size, and writing it as
        # a literal here is how it comes to disagree with the runs.
        r = fixed_recipe(campaign).get(str(e.get("tag", "")), {}) if campaign else {}
        cols.append({"name": label, "geo": e, "archs": e["archs"], "rung": False,
                     "vocab": e.get("vocab_size", vocab),
                     "seq": e.get("seq_len", seq), "corpus": corpus,
                     "sched": sched, "rec": r})
    return cols


def _resolve(c, arm):
    """An arm's geometry in one column, through GEOM_ALIAS on rungs only."""
    e = c["archs"].get(arm)
    if e is None and c.get("rung"):
        e = c["archs"].get(GEOM_ALIAS.get(arm, arm))
    return e


def _arch(c, arm, field, fmt):
    """One column's value for an arm's geometry, or "--"."""
    e = _resolve(c, arm)
    return fmt(e[field]) if e else "--"


def _cell(c, field):
    """One column's value for a scalar field, or None if it has none."""
    g = c["geo"]
    if field == "d_model":
        return str(g["d_model"])
    if field == "n_layers":
        return str(g["n_layers"])
    if field == "n_heads":
        return str(g["n_heads"])
    if field == "head_dim":
        return str(g["head_dim"])
    if field == "vocab":
        return _thin(c["vocab"])
    if field == "seq":
        return _thin(c["seq"])
    if field == "tie":
        return {True: "yes", False: "no"}[bool(g["tie_word_embeddings"])]
    if field == "corpus":
        return c["corpus"]
    if field == "sched":
        return c["sched"]
    if field == "act":
        return "SwiGLU"
    if field in ("lr", "gbs", "warmup", "decay"):
        vals = sorted(c["rec"].get(field, ()))
        if not vals:
            return None
        f = {"lr": lambda v: f"{v:g}", "gbs": lambda v: str(int(v)),
             "warmup": lambda v: str(int(v)),
             "decay": lambda v: (f"{100 * v:.0f}\\%" if v else "none")}[field]
        return " / ".join(f(v) for v in vals)
    raise KeyError(field)


# (field, label). Order is the order a reader wants them, not the order they
# happen to be constant in -- the split below decides that from the data.
FIELDS = [("d_model", "Hidden size"), ("n_layers", "Num.\\ hidden layers"),
          ("n_heads", "Num.\\ attention heads"), ("head_dim", "Head dimension"),
          ("act", "Hidden activation"), ("seq", "Max.\\ position embeddings"),
          ("vocab", "Vocabulary size"), ("tie", "Tied embeddings"),
          ("corpus", "Training corpus"), ("sched", "LR schedule"),
          ("lr", "Peak learning rate"),
          ("gbs", "Global batch size (sequences)"),
          ("warmup", "Warmup steps"), ("decay", "Cooldown fraction of budget")]


def split(cols):
    """(varies, constant): a field is constant only if every column agrees.

    Computed, never assumed. Adding the 1.3B FineWeb-Edu point moves head
    dimension, vocabulary and tying out of the constant block, and a table that
    assumed them would print the ladder's values under a column where they are
    false.
    """
    varies, constant = [], []
    for field, label in FIELDS:
        vals = [_cell(c, field) for c in cols]
        if any(v is None for v in vals):
            varies.append((label, [v or "--" for v in vals]))
        elif len(set(vals)) == 1:
            constant.append((label, vals[0]))
        else:
            varies.append((label, vals))
    return varies, constant


def rows(cols, arms):
    """(label, [value per column]) for the per-arm blocks."""
    out = []
    out.append(("Intermediate size", None))
    for arm, label in arms:
        out.append((f"\\quad {label}",
                    [_arch(c, arm, "d_ffn", str)
                     for c in cols]))
    out.append(("Total parameters (millions)", None))
    for arm, label in arms:
        out.append((f"\\quad {label}",
                    [_arch(c, arm, "N", _fmt_n)
                     for c in cols]))
    return out


def render_latex(cols, arms, worst, source: Path):
    n = len(cols)
    varies, constant = split(cols)
    print(f"% Generated by lm_scaling/arch_table.py from {source.name} -- "
          f"regenerate, do not edit:")
    print(r"%   lm_scaling/arch_table.py > lm_scaling/tex/arch_table.tex")
    print("% NEEDS: booktabs, graphicx (for \\resizebox).")
    print(r"\begin{table}[h!]")
    print(r"\centering")
    print(r"\resizebox{\linewidth}{!}{%")
    print(f"\\begin{{tabular}}{{l{'c' * n}}}")
    print(r"\toprule")
    print(r"\textbf{Config} & "
          + " & ".join(f"\\textbf{{{c['name']}}}" for c in cols) + r" \\")
    print(r"\midrule")
    for label, vals in varies:
        print(f"{label} & " + " & ".join(vals) + r" \\")
    for label, vals in rows(cols, arms):
        if vals is None:
            print(r"\addlinespace")
            print(f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{{label}}}}} \\\\")
            continue
        print(f"{label} & " + " & ".join(vals) + r" \\")
    if constant:
        print(r"\midrule")
        print(f"\\multicolumn{{{n + 1}}}{{l}}{{\\textit{{Constant across "
              f"every column}}}} \\\\")
        for label, v in constant:
            print(f"{label} & \\multicolumn{{{n}}}{{c}}{{{v}}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    w, where = worst
    print(f"\\caption{{Architectures of the scaling ladder, with the "
          f"1.3B/100BT FineWeb-Edu replication beside it. Every arm is "
          f"parameter-matched to its column's target: the intermediate size "
          f"(the latent dimension of the SwiGLU MLP) absorbs the difference "
          f"between each mixer's own parameter budget, so the MLP width differs "
          f"between arms and the totals do not, to within ${abs(w):.2f}\\%$ "
          f"(largest deviation: {where}). The ladder trains on Nemotron-CC "
          f"under AdamW with $(\\beta_1,\\beta_2)=(0.9,0.95)$ and weight "
          f"decay $0.1$; where a rung shows several values it was run at all of "
          f"them. The FineWeb-Edu column follows the published Gated DeltaNet "
          f"recipe instead, which is why its vocabulary, head dimension, tying "
          f"and schedule differ.}}")
    print(r"\label{tab:ladder-configs}")
    print(r"\end{table}")


def render_markdown(cols, arms, worst, source: Path):
    names = [c["name"] for c in cols]
    varies, constant = split(cols)
    print(f"### Architectures (from `{source.name}` and the fixed points)")
    print()
    print("| Config | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")

    def clean(s):
        return (s.replace("\\quad ", "· ").replace("\\ ", " ")
                 .replace("\\%", "%").replace("{,}", ","))

    for label, vals in varies:
        print(f"| {clean(label)} | " + " | ".join(clean(v) for v in vals) + " |")
    for label, vals in rows(cols, arms):
        if vals is None:
            print(f"| **{clean(label)}** |" + " |" * len(names))
            continue
        print(f"| {clean(label)} | " + " | ".join(vals) + " |")
    for label, v in constant:
        val = clean(v)
        print(f"| {clean(label)} *(const)* | " + " | ".join([val] * len(names)) + " |")
    w, where = worst
    print()
    print(f"Parameter-matched to within **{abs(w):.2f}%** (largest deviation: {where}).")


RENDERERS = {"latex": render_latex, "markdown": render_markdown}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path, default=LADDER)
    ap.add_argument("--sizes", nargs="+", default=SIZES)
    ap.add_argument("--arms", nargs="+",
                    help="arm keys; default is the six the ladder runs")
    # EMPTY by default, which means `ladder_recipe()` -- straight out of the
    # Megatron submitter. It used to default to the torchtitan campaigns, whose
    # configs are not in the release; naming campaigns here still routes the
    # recipe back through the hydra planner, which is how the FineWeb-Edu
    # column gets its own.
    ap.add_argument("--campaigns", nargs="*", default=[],
                    help="plans to read the per-rung recipe from; empty (the "
                         "default) reads it from megatron_ext/submit_ladder.py "
                         "instead, and needs no hydra config layer")
    ap.add_argument("--fixed-points", nargs="*",
                    default=[str(_HERE / "replication_1p3B.json")],
                    help="extra columns that are NOT rungs of the ladder: their "
                         "vocabulary, head dim, tying and schedule differ, which "
                         "is why they are columns and not rows")
    ap.add_argument("--write-recipes", nargs="*", metavar="CAMPAIGN",
                    help="regenerate campaign_recipes.json from the hydra "
                         "planner and exit; needs hydra_staged_sweep and the "
                         "campaign's config layer, neither of which ships here")
    ap.add_argument("--vocab", type=int, default=50304)      # nemotron_neox
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--format", choices=sorted(RENDERERS), default="latex")
    a = ap.parse_args(argv)

    if a.write_recipes is not None:
        camps = a.write_recipes or ["fwedu_1p3B"]
        out = {c: {k: {f: sorted(v) for f, v in d.items()}
                   for k, d in recipe([c]).items()} for c in camps}
        RECIPES.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"wrote {RECIPES} for {', '.join(camps)}")
        return 0

    tab = load(a.table)
    missing = [s for s in a.sizes if s not in tab]
    if missing:
        raise SystemExit(f"{a.table.name} has no rung {missing}; "
                         f"it has {sorted(tab)}")
    arms = ([(k, dict(ARMS).get(k, k)) for k in a.arms] if a.arms else ARMS)
    # The rungs' recipe comes from the Megatron submitter by default; naming
    # campaigns routes it back through the hydra planner instead.
    rec = recipe(a.campaigns) if a.campaigns else ladder_recipe()
    fixed = [(f, "1.3B (FineWeb-Edu)", "fwedu_1p3B", "FineWeb-Edu", "cosine")
             for f in a.fixed_points]
    for f, *_ in fixed:
        if not Path(f).exists():
            raise SystemExit(f"no such fixed point: {f}")
    cols = columns(tab, a.sizes, fixed, a.vocab, a.seq_len, rec)
    # The tolerance over everything shown, fixed points included: the caption
    # quotes it, so it has to be computed over the same cells the reader sees.
    worst, where = 0.0, None
    for c in cols:
        tgt = c["geo"]["target_N"]
        for arm, label in arms:
            e = _resolve(c, arm)
            if not e:
                continue
            d = 100 * (e["N"] - tgt) / tgt
            if abs(d) > abs(worst):
                worst, where = d, f"{label} at {c['name']}"
    RENDERERS[a.format](cols, arms, (worst, where), a.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
