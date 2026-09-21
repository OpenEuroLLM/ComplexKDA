"""The config reads the tables; it does not carry copies of them.

Every failure of the first campaign was one number living in more than one
place. These resolvers exist so a YAML can say `${matched:1.3B,kda-sig-lowrank,d_ffn}`
instead of repeating 5440, and the tests below pin the two properties that
make that worth doing: the values match the tables, and a missing entry
RAISES instead of defaulting.

The second is the important half. A resolver that quietly returns a default
for an unmeasured cell reintroduces exactly the failure it replaced -- runs
have OOM'd five minutes in on a ceiling that looked plausible and was measured
at the wrong context.
"""

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))

# The citations need the composition primitives that hydra_staged_sweep
# registers, so the whole module skips where that library is absent. It is a
# real dependency of the config layer -- see lm_scaling/make_login_venv.sh.
pytest.importorskip("hydra_staged_sweep",
                    reason="config layer needs hydra_staged_sweep")
pytest.importorskip("compoconf", reason="config layer needs compoconf")

import resolvers  # noqa: E402

TABLE = json.loads((_ROOT / "lm_scaling" / "replication_1p3B.json").read_text())
REGIME = "gh200_seq4096_titan_g4"
ARMS = ("kda-sig-lowrank", "ckda-shipped-lowrank",
        "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank")


@pytest.fixture(autouse=True)
def _install():
    resolvers.install()


@pytest.mark.parametrize("arch", sorted(TABLE["archs"]))
def test_matched_agrees_with_the_table(arch):
    entry = TABLE["archs"][arch]
    assert resolvers.matched("1.3B", arch, "d_ffn") == entry["d_ffn"]
    assert resolvers.matched("1.3B", arch, "N") == entry["N"]


def test_rung_gives_the_geometry():
    assert resolvers.rung("1.3B", "head_dim") == TABLE["head_dim"]
    assert resolvers.rung("1.3B", "d_model") == TABLE["d_model"]
    assert resolvers.rung("1.3B", "n_layers") == TABLE["n_layers"]


def test_a_missing_arch_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no arch"):
        resolvers.matched("1.3B", "not-an-arm", "d_ffn")
    with pytest.raises(KeyError, match="no rung"):
        resolvers.matched("9000M", "kda-sig-lowrank", "d_ffn")


def test_arch_kwarg_gives_the_canonical_definition():
    """The baseline is the beta in [0,1] arm. An earlier campaign built its
    baseline with allow_neg_eigval=True from a separate hardcoded block, so it
    was really the -neg arm and the comparison it fed was not the one its name
    claimed."""
    assert resolvers.arch_kwarg("1.3B", "kda-sig-lowrank", "allow_neg_eigval") is False
    assert resolvers.arch_kwarg("1.3B", "ckda-shipped-lowrank", "allow_neg_eigval") is True
    assert resolvers.arch_kwarg("1.3B", "kda-sig-lowrank", "gate") == "sigmoid"
    assert resolvers.arch_kwarg("1.3B", "ckda-shipped-lowrank", "gate") == "signed_sigmoid2"


@pytest.mark.parametrize("arch", ARMS)
def test_peak_mbs_is_the_measured_value(arch):
    assert resolvers.peak_mbs(REGIME, "1.3B", arch) == 4


def test_an_unmeasured_ceiling_raises():
    """Guessing is what the probe exists to avoid: the ceilings halve for the
    linear arms between 2048 and 4096 and do not move for attention, so there
    is no safe interpolation."""
    with pytest.raises(KeyError, match="no measured micro-batch ceiling"):
        resolvers.peak_mbs(REGIME, "1.3B", "not-an-arm")


def test_the_stack_and_the_world_are_part_of_the_key():
    """A ceiling belongs to the STACK and the WORLD SIZE, not to the
    architecture alone. Different model construction, different FSDP wrapping
    and different activation lifetimes move it, and FSDP's per-GPU fixed cost
    falls as the world grows. A section nobody has measured raises rather than
    borrowing the neighbouring one."""
    with pytest.raises(KeyError, match="gh200_seq4096_titan_g64"):
        resolvers.peak_mbs("gh200_seq4096_titan_g64", "1.3B", "kda-sig-lowrank")
    with pytest.raises(KeyError, match="a100_seq4096_titan_g4"):
        resolvers.peak_mbs("a100_seq4096_titan_g4", "1.3B", "kda-sig-lowrank")


def test_every_titan_ceiling_is_a_power_of_two():
    """A global batch that is a power of two, over a power-of-two world, leaves
    a power of two per GPU -- and only a power-of-two micro-batch divides it
    with integral gradient accumulation. A ceiling of 5 or 21 cannot be used as
    one even where it is true."""
    import yaml

    spec = yaml.safe_load((_ROOT / "lm_scaling" / "peak_mbs.yaml").read_text())
    for name, section in spec.items():
        if "_titan" not in name:
            continue
        for key, mbs in section.items():
            assert mbs >= 1 and not (mbs & (mbs - 1)), f"{name}/{key} = {mbs}"


def test_the_flavor_names_the_nosilu_variant():
    """`drop_silu` selects a SEPARATE flavor rather than mutating one, so a
    running job's restart cannot pick up a changed definition."""
    assert resolvers.flavor("1.3B", "kda-sig-lowrank", False) == "kda-sig-lowrank-1.3B"
    assert resolvers.flavor("1.3B", "kda-sig-lowrank", True) == "kda-sig-lowrank-1.3B-nosilu"


def test_the_micro_batch_is_composed_in_yaml_not_in_a_resolver():
    """The cap, the divisibility and the fallback are three steps, and they
    belong in the config where they can be read and overridden -- not inside
    one `${micro_batch:...}` that hides all three.

    Here the campaign's own numbers: 128 sequences over 32 GPUs is 4 each, and
    4 is also the measured ceiling, so accumulation stays 1. The composition
    still has to notice when the two disagree -- at 8 GPUs the per-GPU share is
    16 and the ceiling caps it at 4.
    """
    def resolve(gpus):
        cfg = OmegaConf.create({
            "model": "kda-sig-lowrank", "size": "1.3B", "gpus": gpus,
            "aux": {
                "gbs": 128,
                "peak": f"${{peak_mbs:{REGIME},${{size}},${{model}}}}",
                "want": "${oc.divi:${aux.gbs},${gpus}}",
                "mbs": "${oc.eval:'max(d for d in range(1, ${aux.want} + 1) "
                       "if ${aux.want} % d == 0 and d <= ${aux.peak})'}",
                "grad_accum": "${oc.divi:${aux.want},${aux.mbs}}",
            },
        })
        return OmegaConf.to_container(cfg, resolve=True)["aux"]

    got = resolve(32)
    assert (got["peak"], got["want"], got["mbs"], got["grad_accum"]) == (4, 4, 4, 1)

    got = resolve(8)
    assert got["want"] == 16
    assert got["mbs"] == 4, "the measured ceiling must cap the micro-batch"
    assert got["grad_accum"] == 4
    assert 128 % (got["mbs"] * 8) == 0, "gradient accumulation needs an integer"


def test_resolvers_are_citations_not_computations():
    """A resolver that takes five arguments and returns a decision has moved
    the experiment out of the config and into Python, which is the situation
    this layer exists to leave. Citations read one table and stop."""
    import inspect

    for name, fn in resolvers._RESOLVERS.items():
        n = len(inspect.signature(fn).parameters)
        assert n <= 3, (
            f"{name} takes {n} arguments; that is composition, and it belongs "
            "in the YAML using oc.divi/oc.cdivi/oc.eval")


def test_budget_tag_keeps_small_budgets_distinct():
    """`{D:.0f}B` collapses 0.05B and 0.1B onto "0B", which would make two runs
    share one config file and one checkpoint name."""
    assert resolvers.budget_tag(0.05e9) != resolvers.budget_tag(0.1e9)
    assert resolvers.budget_tag(100e9) == "100B"
    assert resolvers.budget_tag(6e9) == "6B"


def test_the_whole_chain_resolves_through_omegaconf():
    """What the campaign config does: geometry, micro-batch, steps and name --
    every one composed in the config out of citations and arithmetic
    primitives, none of it delegated to a resolver that decides."""
    cfg = OmegaConf.create({
        "model": "ckda-shipped-lowrank", "size": "1.3B", "gpus": 32,
        "aux": {
            "tokens": 100_000_000_000,
            "seq_len": 4096,
            "n_params": "${matched:${size},${model},N}",
            "d_ffn": "${matched:${size},${model},d_ffn}",
            "head_dim": "${rung:${size},head_dim}",
            "gbs": 128,
            "peak": f"${{peak_mbs:{REGIME},${{size}},${{model}}}}",
            "want": "${oc.divi:${aux.gbs},${gpus}}",
            "mbs": "${oc.eval:'max(d for d in range(1, ${aux.want} + 1) "
                   "if ${aux.want} % d == 0 and d <= ${aux.peak})'}",
            "steps": "${oc.cdivi:${aux.tokens},${oc.muli:${aux.gbs},${aux.seq_len}}}",
            "tag": "cosine${budget_tag:${aux.tokens}}",
        },
    })
    got = OmegaConf.to_container(cfg, resolve=True)["aux"]
    assert got["n_params"] == TABLE["archs"]["ckda-shipped-lowrank"]["N"]
    assert got["d_ffn"] == TABLE["archs"]["ckda-shipped-lowrank"]["d_ffn"]
    assert got["head_dim"] == 128
    assert got["mbs"] == 4
    assert got["tag"] == "cosine100B"
    # 100BT before the 256-step grid rounding the campaign applies on top.
    assert got["steps"] == 190_735
