"""`replication_1p3B.json` is GENERATED, and generated files drift.

It is written by `replication_1p3B.py` and committed, because building five
1.3B models to count their parameters takes minutes and no config should pay
that at load. The cost of committing a derived file is that nothing notices
when the source moves and the artefact does not.

It has happened. An earlier round of this study found eight arms whose
committed kwargs did not match `ARCHS`: four carried another arm's
`spread_frac` and were therefore identical to each other, three were missing
the `beta_floor_frac` they were named for, and one had lost the single knob it
existed to test.

That is the same failure as a kwarg being pinned across an axis the study
varies: a dropped kwarg does not raise, it makes two arms equal, and the sweep
measures nothing while looking healthy.

The kwargs are checked and not the geometry: `d_ffn` and `N` come from
counting parameters in a built model, which is what takes the minutes. The
gate and init knobs cost no parameters, so a kwarg can drift without moving a
single width -- which is exactly why it went unseen.
"""

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))

pytest.importorskip("fla", reason="param_match imports the fla layers")

import param_match as pm  # noqa: E402
import replication_1p3B as R  # noqa: E402

TABLE = json.loads((_ROOT / "lm_scaling" / "replication_1p3B.json").read_text())
# Chosen by the generator when it writes the entry, not by ARCHS.
_DERIVED = {"arch", "mixer", "d_ffn", "N", "delta_pct", "hybrid"}


@pytest.mark.parametrize("arch", sorted(TABLE["archs"]))
def test_every_committed_kwarg_matches_archs(arch):
    want = {k: v for k, v in pm.ARCHS[arch]["kwargs"].items() if k not in _DERIVED}
    got = TABLE["archs"][arch].get("kwargs", {})
    assert got == want, (
        f"{arch} in replication_1p3B.json disagrees with ARCHS. "
        "Regenerate: python lm_scaling/replication_1p3B.py")
    assert TABLE["archs"][arch].get("hybrid") == pm.ARCHS[arch].get("hybrid"), arch
    assert TABLE["archs"][arch]["mixer"] == pm.ARCHS[arch]["mixer"], arch


def test_the_table_covers_every_arm_the_generator_would_write():
    """An arm added to ARMS and never regenerated cannot be run: the resolvers
    RAISE on a missing entry, which is loud -- but only once someone asks for
    it. An arm in the table and not in ARMS is worse, because regenerating
    would silently DROP it."""
    assert set(TABLE["archs"]) == {"attn", *R.ARMS}, (
        "replication_1p3B.json and replication_1p3B.ARMS disagree; "
        "regenerate, and check nothing was dropped")


def test_no_two_arms_are_the_same_model():
    """The property the drift destroyed. Two arms carrying the same kwargs and
    the same stack are one model under two names, and the axis between them
    measures nothing."""
    seen = {}
    for arm, entry in TABLE["archs"].items():
        if arm == "attn":
            continue
        key = (json.dumps(entry["kwargs"], sort_keys=True),
               json.dumps(entry.get("hybrid"), sort_keys=True))
        assert key not in seen, (
            f"{arm} and {seen[key]} are the same model under two names")
        seen[key] = arm


@pytest.mark.parametrize("baseline,signed", [
    ("kda-sig-lowrank", "ckda-shipped-lowrank"),
    ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"),
])
def test_a_pair_differs_only_in_the_gate(baseline, signed):
    """THE EXPERIMENT. Each pair is meant to isolate the decay gate's sign, so
    the two arms must differ in `gate` and `allow_neg_eigval` and in nothing
    else -- and must be matched to the parameter, or the comparison carries a
    width difference as well."""
    a, b = TABLE["archs"][baseline], TABLE["archs"][signed]
    differing = {k for k in set(a["kwargs"]) | set(b["kwargs"])
                 if a["kwargs"].get(k) != b["kwargs"].get(k)}
    assert differing == {"gate", "allow_neg_eigval"}, (
        f"{baseline} vs {signed} differ in {sorted(differing)}; the pair is "
        "supposed to vary the gate's sign and nothing else")
    assert a["kwargs"]["allow_neg_eigval"] is False
    assert b["kwargs"]["allow_neg_eigval"] is True
    assert (a["d_ffn"], a["N"]) == (b["d_ffn"], b["N"]), (
        "the pair is the comparison and must be parameter-matched exactly")
    assert a.get("hybrid") == b.get("hybrid"), "same stack, or it is a different axis"
