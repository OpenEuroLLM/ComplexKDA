"""The Megatron variant builds the same model the torchtitan ladder trains.

The whole point of running the ladder on a second framework is that a
difference between the two is the FRAMEWORK and not a second implementation of
the layer. That only holds while both build the same object, and they now reach
it by different routes: torchtitan goes through `param_match.ARCHS` and
`titan_ext.mixer.fla_layer_kwargs`, Megatron through typed fields on
`TransformerConfig` read by `megatron/core/ssm/complex_kda.py`.

Two tables that must agree, in other words -- which is the arrangement this
repository keeps finding bugs in. So they are compared here rather than
inspected: same class, same parameter count, same state-dict keys and shapes,
same gate settings, same layers.

The Megatron half needs the fork checked out under `external/`, which is
gitignored. Those tests skip without it; the ones that need only our side do
not.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LM = REPO / "lm_scaling"
MEGATRON = REPO / "external" / "oellm-autoexp" / "submodules" / "Megatron-LM"

if str(LM) not in sys.path:
    sys.path.insert(0, str(LM))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _reference_layer(arm: str, d_model: int, head_dim: int, n_layers: int, role: str = "linear"):
    """The layer the TORCHTITAN ladder builds for this arm.

    Loaded by file rather than imported as a package: `titan_ext.__init__`
    registers torchtitan's model specs and imports `titan_oellm`, which is not
    installed here, so importing the package to reach one pure function fails
    on an unrelated dependency.
    """
    from param_match import ARCHS

    spec = importlib.util.spec_from_file_location("_tm", LM / "titan_ext" / "mixer.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    class _Args:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    flavor = ARCHS[arm]
    hybrid = flavor.get("hybrid") or {}
    args = _Args(
        **{**dict(drop_silu=True), **flavor["kwargs"], **hybrid},
        dim=d_model, head_dim=head_dim, norm_eps=1e-5,
        n_layers=n_layers, _layer_idx=0,
    )
    cls, kwargs = tm.fla_layer_kwargs("attn" if role == "attn" else flavor["mixer"], args)
    return cls(**kwargs)


def _megatron():
    if not MEGATRON.exists():
        pytest.skip("the Megatron fork is not checked out under external/")
    if str(MEGATRON) not in sys.path:
        sys.path.insert(0, str(MEGATRON))
    return pytest.importorskip("megatron.core.transformer.transformer_config")


def _same_module(a, b) -> list:
    """Every way two layers can differ that a parameter count would not show."""
    problems = []
    if type(a) is not type(b):
        problems.append(f"class {type(a).__name__} != {type(b).__name__}")
    na = sum(p.numel() for p in a.parameters())
    nb = sum(p.numel() for p in b.parameters())
    if na != nb:
        problems.append(f"parameters {na:,} != {nb:,}")
    ka, kb = set(a.state_dict()), set(b.state_dict())
    if ka != kb:
        problems.append(f"state_dict keys differ: {sorted(ka ^ kb)[:6]}")
    for k in ka & kb:
        if a.state_dict()[k].shape != b.state_dict()[k].shape:
            problems.append(f"{k}: {tuple(a.state_dict()[k].shape)} != "
                            f"{tuple(b.state_dict()[k].shape)}")
    return problems


# The gate settings that separate each arm from its baseline, as
# `submit_ladder.ARMS` states them.
PURE_ARMS = {
    "kda-sig-lowrank": dict(linear_gate_activation="sigmoid", linear_beta_max=1.0),
    "ckda-shipped-lowrank": dict(linear_gate_activation="signed_sigmoid2", linear_beta_max=2.0),
}


@pytest.mark.parametrize("arm", sorted(PURE_ARMS))
def test_the_variant_builds_the_reference_layer(arm):
    """Megatron's complex_kda mixer IS the layer torchtitan trains."""
    _megatron()
    from megatron.core.ssm.complex_kda import ComplexKDA
    from megatron.core.transformer.transformer_config import TransformerConfig

    d_model, head_dim, n_layers = 576, 64, 18
    config = TransformerConfig(
        num_layers=n_layers, hidden_size=d_model, num_attention_heads=d_model // head_dim,
        experimental_attention_variant="complex_kda", linear_attention_freq=n_layers + 1,
        linear_key_head_dim=head_dim, linear_drop_qkv_silu=True, layernorm_epsilon=1e-5,
        **PURE_ARMS[arm],
    )
    native = ComplexKDA(config, layer_number=1).layer
    reference = _reference_layer(arm, d_model, head_dim, n_layers)

    assert not _same_module(native, reference)
    # The three settings that ARE the experiment. A parameter count cannot see
    # any of them: signed and unsigned gates are the same shape, and dropping
    # the SiLU changes an activation rather than a tensor.
    assert native.allow_neg_eigval == reference.allow_neg_eigval
    assert native.gate == reference.gate
    # `drop_silu` is consumed at construction and not stored, so it is read
    # back where it lands -- the short convolutions' activation. Comparing
    # `layer.drop_silu` instead silently compares two AttributeErrors away to
    # nothing, which is how an earlier version of this check passed.
    for conv in ("q_conv1d", "k_conv1d", "v_conv1d"):
        assert getattr(native, conv).activation == getattr(reference, conv).activation
    assert native.q_conv1d.activation is None, "this campaign drops the q/k/v SiLU"


def test_the_hybrids_attention_is_gated_and_nope():
    """A hybrid's full-attention layers are fla's, not Megatron's.

    Substituting Megatron's costs the output gate -- ``hidden_size ** 2``
    parameters per attention layer -- so the hybrid would be a smaller model
    under the same name. It was, once: 46,819,446 parameters at the 47M rung
    against the matched table's 47,265,270.
    """
    _megatron()
    from megatron.core.ssm.complex_kda import ComplexKDAHybridAttention
    from megatron.core.transformer.transformer_config import TransformerConfig

    d_model, head_dim, n_layers = 576, 64, 18
    config = TransformerConfig(
        num_layers=n_layers, hidden_size=d_model, num_attention_heads=d_model // head_dim,
        experimental_attention_variant="complex_kda", linear_attention_freq=4,
        linear_key_head_dim=head_dim, linear_hybrid_attention="gated_nope",
        linear_gate_activation="signed_sigmoid2", linear_beta_max=2.0,
        linear_drop_qkv_silu=True, layernorm_epsilon=1e-5,
    )
    native = ComplexKDAHybridAttention(config, layer_number=4).layer
    reference = _reference_layer("ckda-shipped-hybrid-lowrank", d_model, head_dim,
                                 n_layers, role="attn")

    assert not _same_module(native, reference)
    assert native.g_proj is not None, "the hybrid's attention is output-gated"
    assert native.rotary is None, "the hybrid's attention is NoPE"


def test_tensor_parallelism_is_refused_rather_than_ignored():
    """The fla layer's projections are not parallel linears.

    Training at TP > 1 would replicate the mixer while sharding the rest of the
    layer, which is a wrong model rather than a slow one.
    """
    _megatron()
    from megatron.core.ssm.complex_kda import ComplexKDA
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=2,
        tensor_model_parallel_size=2,
        experimental_attention_variant="complex_kda", linear_attention_freq=3,
        linear_key_head_dim=64,
    )
    with pytest.raises(NotImplementedError, match="tensor_model_parallel_size"):
        ComplexKDA(config, layer_number=1)


@pytest.mark.parametrize("n_layers", [12, 18, 20, 24])
def test_the_variant_places_its_layers_where_torchtitan_does(n_layers):
    """`linear_attention_freq` and `attn_every` must mark the same layers.

    Megatron marks layer i as full attention when ``(i + 1) % freq == 0``, and
    `titan_ext.hybrid_layers` uses the same rule -- so the hybrids agree by
    construction, and the PURE arms need a freq the model is never deep enough
    to reach. `arm_overrides` derives that from the rung's depth; a literal
    would quietly grow an attention layer when a rung got deeper.
    """
    from megatron_ext.submit_ladder import ARMS, arm_overrides

    for arm, spec in ARMS.items():
        if "gate" not in spec:
            continue
        freq = next(int(o.split("=")[1]) for o in arm_overrides(arm, n_layers)
                    if "linear_attention_freq" in o)
        megatron_attn = [i for i in range(n_layers) if (i + 1) % freq == 0]
        every = spec.get("attn_every", 0)
        titan_attn = [i for i in range(n_layers) if every and (i + 1) % every == 0]
        assert megatron_attn == titan_attn, arm
        if not every:
            assert megatron_attn == [], f"{arm} is pure and must have no attention layer"


def test_every_arm_states_the_settings_the_comparison_rests_on():
    """The overrides are the experiment; a missing one is a different model."""
    from megatron_ext.submit_ladder import ARMS, arm_overrides

    for arm in ARMS:
        ov = dict(o.replace("backend.megatron.", "").split("=", 1)
                  for o in arm_overrides(arm, 18))
        # qk_layernorm is what separates the two attention arms, and it is the
        # only thing that does -- so it is stated for every arm, not just them.
        assert ov["qk_layernorm"] == ("true" if arm == "attn-qknorm" else "false")
        if arm.startswith("attn"):
            assert "experimental_attention_variant" not in ov, f"{arm} is plain attention"
            continue
        assert ov["experimental_attention_variant"] == "complex_kda"
        # Signed arms extend BOTH ranges; their baselines extend neither.
        signed = arm.startswith("ckda")
        assert ov["linear_gate_activation"] == ("signed_sigmoid2" if signed else "sigmoid")
        assert ov["linear_beta_max"] == ("2.0" if signed else "1.0")
        # Campaign-wide, and measured to matter: it costs the unsigned arms
        # 0.015-0.021 nats and the signed arms nothing.
        assert ov["linear_drop_qkv_silu"] == "true"
        assert ("linear_hybrid_attention" in ov) == ("hybrid" in arm)


def test_the_ladder_runs_without_biases_outside_the_mixer():
    """A guard the removed `--spec` module enforced at build time.

    It refused to build a spec when `add_bias_linear` or `add_qkv_bias` was on,
    because the (C)KDA layers carry their own biases and Megatron's would be a
    second set the parameter match does not account for. Nothing refuses it now
    -- the variant is Megatron's own and has no opinion -- so what is checked is
    that every cell this repository submits still turns both off.
    """
    import megatron_ext.submit_ladder as S

    table = S.geometry()
    overrides = S.cell_overrides(
        "ckda-shipped-lowrank", "124M", 6e9, 64, 0.001, table, group="test")
    flat = dict(o.replace("backend.megatron.", "").split("=", 1)
                for o in overrides if o.count("=") == 1)
    assert flat["add_qkv_bias"] == "false"
    assert flat["add_bias_linear"] == "false"
