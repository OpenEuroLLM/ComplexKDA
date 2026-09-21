"""The architecture table reports the recipe the ladder actually ran.

`lm_scaling/tex/arch_table.tex` states a peak learning rate, a batch size, a
warmup and a cooldown fraction per rung. Those four rows used to be read from
the torchtitan campaign plans; the ladder moved to Megatron, and the rows moved
with it -- to `megatron_ext/submit_ladder.py` for the cells and to the root
config for the schedule. The first version of that move went out with the old
numbers still in the file (588M at "0.0005 / 0.001", batches up to 512), which
is exactly the failure the generator exists to prevent, so it is checked here.
"""

import csv
import collections
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

LM = Path(__file__).resolve().parents[2] / "lm_scaling"


@pytest.fixture(scope="module")
def arch_table():
    import sys
    if str(LM) not in sys.path:
        sys.path.insert(0, str(LM))
    return pytest.importorskip("arch_table")


def test_recipe_matches_the_runs_it_describes(arch_table):
    """Every (rung, lr) and (rung, batch) the table states was submitted.

    Against `harvest/megatron_ladder.tsv`, the harvested result of the campaign
    -- 180 cells, one row per run. Set equality in both directions: a value in
    the table that no run used is a claim about an experiment nobody did, and a
    value a run used that the table omits is a rung reported as narrower than
    it was swept.
    """
    rec = arch_table.ladder_recipe()

    ran = collections.defaultdict(lambda: {"lr": set(), "gbs": set()})
    with open(LM / "harvest" / "megatron_ladder.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ran[row["size"]]["lr"].add(float(row["lr"]))
            ran[row["size"]]["gbs"].add(int(row["gbs"]))

    assert set(rec) >= set(ran)
    for size, got in sorted(ran.items()):
        assert rec[size]["lr"] == got["lr"], size
        assert rec[size]["gbs"] == got["gbs"], size


def test_chained_cooldown_matches_the_single_stage_one():
    """One schedule shape, described in two places.

    A single-stage cell decays over `aux.decay_fraction` of its budget, from
    the root config. A chain's cooldowns are built in Python and carry
    `COOLDOWN_FRAC`. They have to be the same number -- otherwise the chained
    rungs run a different WSD than the ones they are fitted jointly with, and
    nothing in the output would say so.
    """
    import sys
    if str(LM) not in sys.path:
        sys.path.insert(0, str(LM))
    from megatron_ext import submit_ladder

    cfg = yaml.safe_load(
        (LM / "megatron_ext" / "config" / "ckda_ladder_jupiter.yaml").read_text())
    assert submit_ladder.COOLDOWN_FRAC == float(cfg["aux"]["decay_fraction"])


def test_table_on_disk_is_what_the_generator_emits(arch_table, capsys):
    """`tex/arch_table.tex` is generated, so it should equal a regeneration."""
    arch_table.main([])
    assert capsys.readouterr().out == (LM / "tex" / "arch_table.tex").read_text()
