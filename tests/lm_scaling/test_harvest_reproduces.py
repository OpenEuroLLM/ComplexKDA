"""The committed results still produce the committed tables.

`lm_scaling/harvest/` holds the expensive half of the paper: 180 harvested
ladder cells, 180 downstream evaluations, the four 1.3B arms, the RULER runs and
the gate spectrum. Everything in `lm_scaling/tex/` is derived from those by a
script in this repository, and the point of committing the data is that a reader
can re-derive it rather than take the tables on trust.

That only holds while the derivation still runs. These tests are the cheap end
of it -- the parts that need no fitting and no bootstrap, so they can sit in the
normal suite. `lm_scaling/reproduce_tables.sh --check` is the whole of it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

LM = Path(__file__).resolve().parents[2] / "lm_scaling"


def run(*args):
    p = subprocess.run([sys.executable, *args], capture_output=True, text=True,
                       cwd=LM.parent)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout


@pytest.mark.parametrize("name", ["megatron_ladder.tsv", "ladder_downstream",
                                  "downstream", "ruler", "gate_spectrum"])
def test_the_expensive_results_are_present(name):
    """A missing harvest turns every table below into a silent empty one."""
    assert (LM / "harvest" / name).exists()


def test_ladder_downstream_still_prints_its_committed_table():
    """180 evaluation JSONs in, one table out, byte for byte.

    `harvest/megatron_downstream_table.txt` is that table as it was when the
    numbers went into the paper. It is derived, not measured -- the measurement
    is the 180 JSONs beside it -- so it is worth keeping only as the assertion
    that the derivation has not moved, which is what this is.
    """
    got = run(str(LM / "ladder_downstream.py"))
    want = (LM / "harvest" / "megatron_downstream_table.txt").read_text()
    assert got == want


def test_arch_table_reproduces():
    """Geometry and recipe, from `ladder_matched.json` and the submitter."""
    got = run(str(LM / "arch_table.py"))
    assert got == (LM / "tex" / "arch_table.tex").read_text()


@pytest.mark.parametrize("convention", ["gdn", "gdn2"])
def test_1p3B_downstream_tables_reproduce(convention):
    got = run(str(LM / "eval_table.py"), "--step", "190976",
              "--convention", convention, "--format", "latex")
    assert got == (LM / "tex" / f"fwedu_1p3B_table_{convention}.tex").read_text()


def test_ruler_table_reproduces():
    got = run(str(LM / "niah_table.py"), "--format", "latex", "--step", "190976")
    assert got == (LM / "tex" / "fwedu_1p3B_niah.tex").read_text()


def test_ladder_avg9_is_what_the_ladder_measures():
    """The one number in a paper caption that a reader cannot check.

    `eval_table.LADDER_AVG9` is the 1.3B caption's reason for saying "matches"
    rather than "beats": the same pairing over the ladder's 60 paired cells,
    pooled. It is a constant in a file about a different experiment, so nothing
    about rendering the 1.3B table would notice it going stale -- and it did.
    When the ladder moved to Megatron the mean was updated to -0.027 while the
    committed evaluations say -0.020, and the test that was supposed to pin it
    compared it against a literal in the results document, which carried the
    same wrong value. Both were consistent and both were wrong.

    So recompute it here, from the 180 evaluation JSONs, the way
    `ladder_downstream.py` does.
    """
    import math
    import statistics as st
    import sys

    if str(LM) not in sys.path:
        sys.path.insert(0, str(LM))
    import ladder_downstream as L
    from eval_table import ACC_GDN2, LADDER_AVG9, row

    res = L.load_results(L.RESULTS, L.SIZES)
    deltas = []
    for base, signed, _tag in L.PAIRS:
        for size in L.SIZES:
            for budget in L.BUDGETS:
                a = res.get((base, size, budget))
                b = res.get((signed, size, budget))
                if a is None or b is None:
                    continue
                # `row` is what turns a harness dump into the nine scored tasks
                # and their average; the raw JSON has no "avg" of its own, and
                # which nine it is (ACC_GDN2) is the whole reporting convention.
                d = 100 * (row(b, ACC_GDN2)["avg"] - row(a, ACC_GDN2)["avg"])
                if not math.isnan(d):
                    deltas.append(d)

    assert len(deltas) == LADDER_AVG9["cells"]
    assert sum(1 for d in deltas if d < 0) == LADDER_AVG9["negative"]
    sd = st.stdev(deltas)
    assert round(st.mean(deltas), 3) == LADDER_AVG9["mean"]
    assert round(sd / len(deltas) ** 0.5, 3) == LADDER_AVG9["se"]


def test_the_architecture_table_describes_arms_that_have_results():
    """A column for an architecture nobody trained reads as a result.

    `arch_table.py` renders one row per arm, with its matched MLP width and
    parameter count at every rung. Those numbers exist for any arm
    `param_match.py` can describe, whether or not it was ever run -- so the
    table shipped a fully specified `gdn` row for an arm trained only on the
    superseded torchtitan ladder, which this release does not contain, while
    `attn-qknorm`, the baseline every cell in `harvest/megatron_ladder.tsv` is
    compared against, had no row at all.

    The arms of the table and the arms of the campaign are the same six.
    """
    import csv
    import sys

    if str(LM) not in sys.path:
        sys.path.insert(0, str(LM))
    import arch_table
    from megatron_ext import submit_ladder

    with open(LM / "harvest" / "megatron_ladder.tsv") as f:
        measured = {r["arm"] for r in csv.DictReader(f, delimiter="\t")}

    shown = {k for k, _label in arch_table.ARMS}
    assert shown == measured
    # and the ladder submits exactly those, so all three agree
    assert set(submit_ladder.ARMS) == measured


def test_every_alias_stands_in_for_a_real_geometry():
    """`GEOM_ALIAS` must name arms the submitter actually builds that way."""
    import sys

    if str(LM) not in sys.path:
        sys.path.insert(0, str(LM))
    import arch_table
    from megatron_ext import submit_ladder

    for arm, geom in arch_table.GEOM_ALIAS.items():
        assert submit_ladder.ARMS[arm]["geom"] == geom


def test_the_results_document_quotes_the_harvest():
    """`ladder_results.md` is hand-written around a table of 180 losses.

    Unlike everything in `tex/`, it is not generated -- it is prose with a
    table in the middle, and the table was pasted in. That is the arrangement
    where a number goes stale without anything noticing, so every cell of it is
    read back against `harvest/megatron_ladder.tsv` here.
    """
    import collections
    import csv
    import re

    cells = collections.defaultdict(dict)
    with open(LM / "harvest" / "megatron_ladder.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cells[f'{r["size"]}/{r["budget_bt"]}BT'][r["arm"]] = float(r["loss"])

    # The column order of the document's table, which its own header states.
    order = ["attn", "attn-qknorm", "kda-sig-lowrank", "ckda-shipped-lowrank",
             "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"]

    doc = (LM / "config" / "experiments" / "ladder_results.md").read_text()
    checked = 0
    for line in doc.splitlines():
        m = re.match(r"\|\s*([0-9.]+[MB]/\d+BT)\s*\|(.+)\|\s*$", line)
        if not m:
            continue
        cell = m.group(1)
        values = [x.strip() for x in m.group(2).split("|") if x.strip()]
        if cell not in cells or len(values) != len(order):
            continue
        for arm, got in zip(order, values):
            assert float(got) == cells[cell][arm], f"{cell} {arm}"
            checked += 1

    # All 30 cells x 6 arms, so a table that silently loses rows fails too.
    assert checked == 180


def test_the_lockfile_and_the_readme_name_the_same_commits():
    """Two of the pinned commits are ours, and three files quote them.

    `megatron_stack.lock` holds the full SHAs; `lm_scaling/README.md` quotes
    them short, beside the branch a reader has to push. A short SHA copied by
    hand is the kind of thing that survives a rebase of the branch it names and
    then points at nothing, so it is checked rather than trusted.

    The external checkout itself is gitignored and absent for a reader, so
    there is nothing here to compare against a tree -- this is the lockfile
    against the prose about it.
    """
    import re

    lock = (LM / "megatron_stack.lock").read_text()
    readme = (LM / "README.md").read_text()

    pins = {}
    for line in lock.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if parts[0] in ("repo", "submodule"):
            pins[parts[1]] = {"sha": parts[3],
                              "branch": parts[4] if len(parts) > 4 else ""}

    ours = {"oellm-autoexp", "submodules/Megatron-LM"}
    assert ours <= set(pins)

    for name in sorted(ours):
        sha, branch = pins[name]["sha"], pins[name]["branch"]
        assert len(sha) == 40 and re.fullmatch(r"[0-9a-f]{40}", sha), name
        # The branch to push is named in the lockfile and in the README.
        assert branch == "feat/complex-kda", f"{name} pins branch {branch!r}"
        assert branch in readme, f"{branch} not named in README"
        # And the README's short form is a prefix of the real thing.
        assert sha[:8] in readme, f"{name}: README does not quote {sha[:8]}"


def test_the_results_document_states_the_1p3B_pairing_it_ships():
    """The doc's headline 1.3B figures against the committed scores.

    `test_eval_table.py` pins the LADDER figure this way already. The 1.3B
    pairing was not pinned, and it drifted: the document carried +0.52 (pure)
    and +0.86 (hybrid) from an evaluation that prefixed every context with
    `<s>`, while `harvest/downstream/` held the re-run without it, where the
    pure pair is -0.04. Nothing failed -- the tables regenerated correctly from
    the new data and the prose kept the old numbers beside them.
    """
    import json
    import re
    import sys

    if str(LM) not in sys.path:
        sys.path.insert(0, str(LM))
    from eval_table import ACC_GDN2, PAIRS, RESULTS, row

    def avg9(arm):
        f = RESULTS / f"{arm}_step190976.json"
        return 100 * row(json.loads(f.read_text()), ACC_GDN2)["avg"]

    measured = {tag: avg9(signed) - avg9(base) for base, signed, tag in PAIRS}
    doc = (LM / "config" / "experiments" / "fwedu_1p3B_results.md").read_text()

    # The bullets under the nine-task table, one per pair.
    for tag, key in (("recurrent", "recurrent"), ("hybrid", "hybrid")):
        m = re.search(rf"^- {key}: signed . baseline = \*\*([+-−][0-9.]+)\*\* points",
                      doc, re.M)
        assert m, f"the document no longer states the {key} pairing"
        stated = float(m.group(1).replace("−", "-"))
        assert abs(stated - measured[tag]) < 5e-3, (
            f"{key}: document says {stated:+.2f}, harvest says {measured[tag]:+.2f}")


def test_the_1p3B_scores_are_the_no_bos_convention():
    """What the paper reports, and the sidecar that records it.

    Only ONE convention ships: scored without a BOS token, which is how this
    corpus was tokenised. An earlier pass prefixed every context with `<s>`
    because the fwedu tokenizer copy sets `add_bos_token: True` and the harness
    honours it; those readings were measured and are not published, so a file
    here must be able to say which it is rather than be assumed.
    `eval_downstream.py` writes a `.meta.json` beside each result; a missing
    sidecar means the old default.
    """
    import sys

    if str(LM) not in sys.path:
        sys.path.insert(0, str(LM))
    from eval_table import PAIRS, RESULTS, context_prefix_of

    arms = {a for pair in PAIRS for a in pair[:2]}
    for arm in sorted(arms):
        assert context_prefix_of(RESULTS, arm, 190976) == "none", arm
    # And the superseded convention is NOT shipped: a reader who finds both
    # has to work out which produced the tables, which is the confusion the
    # sidecar exists to prevent.
    assert not (LM / "harvest" / "downstream_bos").exists()
    assert not (LM / "harvest" / "ruler_bos").exists()


#: The scripts `reproduce_tables.sh` runs, relative to `lm_scaling/`.
REGENERATORS = [
    "scaling_holdout.py", "holdout_plot.py", "scaling_plots.py",
    "megatron_ext/compare_table.py", "ladder_downstream.py", "eval_table.py",
    "niah_table.py", "arch_table.py", "gate_spectrum_plot.py",
]

#: The submission stack. Public, but installed into the login venv, and needed
#: to PLAN a campaign rather than to read one back.
ORCHESTRATION = ("hydra_staged_sweep", "slurm_gen", "monitor", "plan")

_BLOCK = """
import sys
class _Blocker:
    def find_module(self, name, path=None): return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {block!r}:
            raise ImportError(
                "the orchestration layer is not available when regenerating "
                "tables; this import is the bug " + name)
sys.meta_path.insert(0, _Blocker())
import runpy
# argv[0] IS the program name; argparse parses argv[1:]. Passing ["--help"]
# alone leaves argparse with no arguments at all, which starts a real 600-start
# fit instead of printing usage.
sys.argv = [{script!r}] + {argv!r}
runpy.run_path({script!r}, run_name="__main__")
"""


def _without_orchestration(script, argv):
    """Run a regenerator in a process where the submission stack cannot import.

    The point of the blocker rather than just a bare environment: this suite is
    run in BOTH environments, and under the login venv -- which has the whole
    stack -- an import of it is invisible. That is exactly how the FineWeb-Edu
    column came to resolve its recipe through the hydra planner and make
    `reproduce_tables.sh` unrunnable for anyone who had only installed the
    analysis dependencies. Blocking the import reproduces a clean checkout in
    either environment.
    """
    code = _BLOCK.format(block=set(ORCHESTRATION), argv=argv, script=str(script))
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=LM.parent)


@pytest.mark.parametrize("name", REGENERATORS)
def test_regenerating_a_table_needs_no_submission_stack(name):
    """`reproduce_tables.sh` says it reads only what is in the repository.

    It cannot also need the campaign planner and the config layer that feeds it:
    those are how the runs were LAUNCHED, and a reader re-deriving the tables has
    neither. Anything a regenerator needs from them has to ship as data -- which
    is what `campaign_recipes.json` is.
    """
    p = _without_orchestration(LM / name, ["--help"])
    assert "this import is the bug" not in p.stderr, \
        f"{name} imports the submission stack:\n{p.stderr[-1500:]}"


def test_the_architecture_table_reproduces_without_the_planner():
    """The regression this test exists for, end to end rather than on --help.

    `--help` returns before `main` does any work, so it would not have caught
    the planner import: that one is inside `recipe()`, reached only when a
    column actually resolves its recipe.
    """
    p = _without_orchestration(LM / "arch_table.py", [])
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout == (LM / "tex" / "arch_table.tex").read_text(), \
        "arch_table.tex is not what arch_table.py emits on a clean checkout"
