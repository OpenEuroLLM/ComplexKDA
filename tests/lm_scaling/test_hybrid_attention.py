"""What a hybrid's ATTENTION layers are, as opposed to how many there are.

The hybrids follow current practice (Qwen3-Next, Kimi): the full-attention
layers carry an output gate and no positional encoding. The dense `attn` arm
does NOT -- it is the classical baseline, deliberately.

Both settings default off, so an arm gets them only by asking. That matters
because the two are not symmetric in how they fail: the gate costs parameters
and would be caught by the count guard, while NoPE is parameter-free and a
dropped `attn_use_rope` would build a RoPE model whose parameter count is
perfect.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))
sys.path.insert(0, str(_ROOT))

pytest.importorskip("fla", reason="the attention layer lives in fla")

import param_match as pm  # noqa: E402


def test_only_the_hybrids_ask_for_a_gated_nope_attention():
    """The dense arm stays classical, which is the choice this encodes."""
    for arm in ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"):
        spec = pm.ARCHS[arm]["hybrid"]
        assert spec["attn_output_gate"] is True
        assert spec["attn_use_rope"] is False
    for arm in ("attn", "kda-sig-lowrank", "ckda-shipped-lowrank"):
        assert "hybrid" not in pm.ARCHS[arm]


def test_the_gate_costs_what_the_match_pays_for():
    """One hidden x (n_heads * head_dim) matrix per attention layer. If the
    count and the layer ever disagree, the hybrids stop being
    parameter-matched and the ladder compares models of different sizes."""
    from fla.layers.attn import Attention

    plain = Attention(hidden_size=256, num_heads=4, output_gate=False)
    gated = Attention(hidden_size=256, num_heads=4, output_gate=True)
    measured = (sum(p.numel() for p in gated.parameters())
                - sum(p.numel() for p in plain.parameters()))
    assert measured == 256 * 4 * 64, "the layer's cost"
    # and param_match charges exactly that
    assert measured == 256 * 4 * 64


def test_nope_is_parameter_free_and_actually_off():
    """Parameter-free is why it needs its own test: the count guard cannot
    see a rotation that is still being applied."""
    from fla.layers.attn import Attention

    with_rope = Attention(hidden_size=256, num_heads=4, use_rope=True)
    without = Attention(hidden_size=256, num_heads=4, use_rope=False)
    assert sum(p.numel() for p in with_rope.parameters()) == \
        sum(p.numel() for p in without.parameters())
    assert with_rope.rotary is not None
    assert without.rotary is None, "the rotary module must not be built at all"


def test_a_hybrid_is_counted_with_its_gate():
    """`count_hybrid` builds the count from the PLAIN arm by difference, so
    the gate has to be added explicitly -- there is no gated model to measure
    it from."""
    args = dict(d_model=896, n_heads=14, n_layers=20, d_ffn=3264, head_dim=64,
                vocab_size=50304, seq_len=4096, tie=True)
    arm = "ckda-shipped-hybrid-lowrank"
    gated = pm.count_hybrid(arm, **args)

    original = pm.ARCHS[arm]["hybrid"]
    pm.ARCHS[arm]["hybrid"] = {"attn_every": 4}
    try:
        plain = pm.count_hybrid(arm, **args)
    finally:
        pm.ARCHS[arm]["hybrid"] = original

    n_attn = len([i for i in range(20) if (i + 1) % 4 == 0])
    assert gated - plain == n_attn * 896 * 14 * 64


def test_the_flavor_refuses_a_hybrid_setting_the_args_class_cannot_hold():
    """A dropped setting builds a different model under the same name, and
    for NoPE the parameter count does not notice at all."""
    import titan_ext

    fields = titan_ext._FLAExtras.__dataclass_fields__
    for name in ("attn_output_gate", "attn_use_rope", "attn_every"):
        assert name in fields, name
    assert fields["attn_output_gate"].default is False
    assert fields["attn_use_rope"].default is True, (
        "RoPE must be the default, or the dense arm silently becomes NoPE")


def test_the_compared_pairs_are_parameter_identical():
    """The campaign is two PAIRS, and each pair is the comparison.

    The two members differ in the gate and in whether negative eigenvalues are
    allowed; the hybrid pair differs the same way inside a hybrid stack.
    Neither difference costs a parameter, so the two members of a pair must
    land on the same `d_ffn` and the same N EXACTLY -- not within a tolerance.
    A pair that drifts apart is comparing two model sizes and calling the
    difference a gate.

    The match against `attn` is looser and cannot be otherwise: `d_ffn` moves
    in multiples of 64, so under half a step is the best any arm can do. The
    gated attention makes the hybrids' figure worse by pushing them into the
    next quantisation bucket, and that is the price of the gate rather than a
    bug.
    """
    import json

    table = json.loads((_ROOT / "lm_scaling" / "replication_1p3B.json").read_text())
    archs = table["archs"]
    for a, b in (("kda-sig-lowrank", "ckda-shipped-lowrank"),
                 ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank")):
        assert archs[a]["d_ffn"] == archs[b]["d_ffn"], f"{a}/{b}"
        assert archs[a]["N"] == archs[b]["N"], (
            f"{a} has {archs[a]['N']:,} and {b} has {archs[b]['N']:,}; the "
            "pair is the comparison and must be matched exactly")
    # And every arm stays inside what the d_ffn grid can achieve.
    for arm, entry in archs.items():
        assert abs(entry["delta_pct"]) < 1.0, (
            f"{arm} is {entry['delta_pct']:+.3f}% off the attention baseline, "
            "which is more than one d_ffn step")


def test_a_hybrid_carries_its_pure_arms_mixer_kwargs_exactly():
    """A hybrid must differ from its pure arm in the STACK and nothing else.

    ARCHS spells the kwargs out twice -- once for `complex-kda`, once for
    `ckda-hybrid` -- so a setting added to one and forgotten in the other is
    a silent divergence. It would not be caught by the parameter count, which
    none of these knobs move, nor by the pair test, which compares the two
    hybrids against each other and would be happy for both to be wrong
    together.

    It matters because the campaign reads the pure pair and the hybrid pair as
    the same question asked twice, once without attention layers and once
    with. If the mixers differ between them, the two answers are to different
    questions.
    """
    for pure, hybrid in (("kda-sig-lowrank", "kda-sig-hybrid-lowrank"),
                         ("ckda-shipped-lowrank", "ckda-shipped-hybrid-lowrank")):
        assert pm.ARCHS[hybrid]["kwargs"] == pm.ARCHS[pure]["kwargs"], (
            f"{hybrid} does not carry {pure}'s mixer kwargs:\n"
            f"  {pure}:  {pm.ARCHS[pure]['kwargs']}\n"
            f"  {hybrid}: {pm.ARCHS[hybrid]['kwargs']}")
        assert pm.ARCHS[hybrid]["mixer"] == pm.ARCHS[pure]["mixer"]
        assert "hybrid" in pm.ARCHS[hybrid] and "hybrid" not in pm.ARCHS[pure]


def test_the_arms_state_their_gate_and_init_rather_than_inheriting_them():
    """STATED, not inherited, so the choice is visible and a change to the
    layer's defaults cannot silently move the campaign.

    All four arms start from the SHIPPED init -- every gate channel at
    alpha > 0, training to discover the negative ones -- so the signed arms
    differ from the baseline in the gate's reachable range and not in where
    they start. `output_gate` is stated for the same reason: the layer has a
    default, and an arm that relied on it would change model if the default
    did.
    """
    for arm in ("ckda-shipped-lowrank", "ckda-shipped-hybrid-lowrank"):
        kw = pm.ARCHS[arm]["kwargs"]
        assert kw["gate"] == "signed_sigmoid2"
        assert kw["allow_neg_eigval"] is True
        assert kw["gate_init_style"] == "shipped"
    for arm in ("kda-sig-lowrank", "kda-sig-hybrid-lowrank"):
        kw = pm.ARCHS[arm]["kwargs"]
        assert kw["gate"] == "sigmoid"
        assert kw["allow_neg_eigval"] is False
        assert kw["gate_init_style"] == "shipped"
    # `attn` and `gdn` are upstream layers: neither has an output gate to
    # state, which is exactly why they are the external baselines.
    for arm in pm.ARCHS:
        if arm in ("attn", "gdn"):
            continue
        assert pm.ARCHS[arm]["kwargs"]["output_gate"] == "lowrank", arm


@pytest.mark.parametrize("mixer", ["attn", "kda", "complex-kda", "gdn"])
def test_every_mixer_branch_builds_a_layer_that_accepts_its_kwargs(mixer):
    """`fla_layer_kwargs` must return kwargs the class will actually take.

    It did not. The `attn` branch passed `head_dim`, which fla's Attention
    does not accept -- it derives hidden_size // num_heads -- so every hybrid
    probe died with TypeError after allocating a node. Twenty-seven jobs
    across three arms produced no measurement, and the only symptom in
    `--collect` was "nothing fit", which reads like a memory ceiling.

    Constructing is the check, because a signature comparison would pass a
    kwarg the class accepts and then rejects on a value.
    """
    import inspect

    from titan_ext.mixer import fla_layer_kwargs

    class Args:
        dim, head_dim, norm_eps = 896, 64, 1e-6
        qk_norm = False
        attn_output_gate, attn_use_rope, rope_theta = True, False, 10000.0
        allow_neg_eigval, lower_bound, expand_v = True, -5.0, 1.0
        gate, gate_init_style, spread_frac = "signed_sigmoid2", "spread", 0.5
        output_gate = "linear"

    cls, kw = fla_layer_kwargs(mixer, Args())
    accepted = set(inspect.signature(cls.__init__).parameters)
    assert not (set(kw) - accepted), (
        f"{mixer}: {cls.__name__} does not accept "
        f"{sorted(set(kw) - accepted)}")
    layer = cls(**kw)
    assert sum(p.numel() for p in layer.parameters()) > 0


def test_the_hybrids_attention_costs_what_the_match_charges():
    """The gate is paid for out of d_ffn, so the layer and `count_hybrid`
    have to agree on its price to the parameter."""
    from titan_ext.mixer import fla_layer_kwargs

    class Args:
        dim, head_dim, norm_eps = 896, 64, 1e-6
        qk_norm, rope_theta = False, 10000.0
        attn_use_rope = False

    plain = dict(Args.__dict__)
    cls, kw = fla_layer_kwargs("attn", type("A", (Args,), {"attn_output_gate": False})())
    without = sum(p.numel() for p in cls(**kw).parameters())
    cls, kw = fla_layer_kwargs("attn", type("A", (Args,), {"attn_output_gate": True})())
    with_gate = sum(p.numel() for p in cls(**kw).parameters())
    assert with_gate - without == 896 * 14 * 64, "d_model x n_heads x head_dim"
    _ = plain
