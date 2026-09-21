"""SiLU on q/k/v as a campaign setting, not a change to any arm's definition.

A running job re-reads its flavor from the geometry tables at every restart,
and SiLU has no parameters, so flipping `drop_silu` on an existing arm would
load its checkpoint into a different model without a word. The switch is a
second flavor per arm instead, `<arch>-<rung>-nosilu`, which a campaign selects
through `aux.drop_silu`. Runs that started with SiLU keep naming the flavor
they started with.
"""

import sys
from dataclasses import make_dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lm_scaling"))
sys.path.insert(0, str(ROOT))

import geometries  # noqa: E402
import resolvers  # noqa: E402

FIELDS = ["dim", "n_layers", "n_heads", "n_kv_heads", "head_dim", "hidden_dim",
          "vocab_size", "max_seq_len", "norm_eps", "rope_theta", "qk_norm",
          "depth_init", "mixer"]


def _flavors():
    ext = pytest.importorskip("lm_scaling.titan_ext")
    names = list(dict.fromkeys(FIELDS + list(ext.FLAModelArgs.__dataclass_fields__)))
    return ext.ladder_flavors(make_dataclass("A", [(f, object, None) for f in names]))


def test_every_arm_of_our_layer_gets_a_nosilu_flavor_that_differs_only_there():
    fla = _flavors()
    n = 0
    for rung, r in geometries.table().items():
        for arch, a in r["archs"].items():
            key, variant = f"{arch}-{rung}", f"{arch}-{rung}{geometries.NOSILU}"
            if a.get("mixer") not in geometries.DROPS_SILU:
                assert variant not in fla, f"{variant}: its layer cannot drop SiLU"
                continue
            assert variant in fla, f"{variant} is not registered"
            base, ns = vars(fla[key]), vars(fla[variant])
            # The base flavor is the table's definition, untouched.
            assert base["drop_silu"] == (a.get("kwargs") or {}).get("drop_silu"), key
            assert ns["drop_silu"] is True
            assert ({k: v for k, v in ns.items() if k != "drop_silu"}
                    == {k: v for k, v in base.items() if k != "drop_silu"}), key
            n += 1
    assert n > 0


@pytest.mark.parametrize("arch,drop,want", [
    ("kda-sig-lowrank", True, "kda-sig-lowrank-1.3B-nosilu"),
    # A STRING, because an override reaches the resolver as one and "false"
    # is truthy in Python.
    ("ckda-shipped-lowrank", "true", "ckda-shipped-lowrank-1.3B-nosilu"),
    ("kda-sig-hybrid-lowrank", True, "kda-sig-hybrid-lowrank-1.3B-nosilu"),
    ("ckda-shipped-hybrid-lowrank", True, "ckda-shipped-hybrid-lowrank-1.3B-nosilu"),
    # Upstream layers keep their SiLU, and the name says so.
    ("attn", True, "attn-1.3B"),
    ("kda-sig-lowrank", False, "kda-sig-lowrank-1.3B"),
    ("kda-sig-lowrank", "false", "kda-sig-lowrank-1.3B"),
])
def test_the_flavor_a_run_names(arch, drop, want):
    assert resolvers.flavor("1.3B", arch, drop) == want


def test_an_unknown_arm_is_refused_rather_than_named():
    with pytest.raises(KeyError):
        resolvers.flavor("1.3B", "no-such-arm", True)


def test_every_flavor_the_resolver_can_name_is_registered():
    fla = _flavors()
    for rung, r in geometries.table().items():
        for arch, a in r["archs"].items():
            if a.get("mixer") in geometries.DROPS_SILU:
                assert resolvers.flavor(rung, arch, True) in fla
