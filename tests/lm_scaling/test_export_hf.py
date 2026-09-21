"""What an export SAYS it is, checked against the table the run was planned from.

The exporter writes a config from `geometries.table()` and renames torchtitan's
parameters to fla's. Both halves fail silently: a config that omits the hybrid's
attention spec builds a PURE linear model out of a hybrid's weights (and the
strict load then complains about tensors, not about architecture), and a rename
that swaps w1 for w3 builds a model with the right parameter count and the wrong
arithmetic. Neither shows up in a downstream number that merely looks plausible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "lm_scaling")):
    if p not in sys.path:
        sys.path.insert(0, p)

from export_hf import hf_config, rename  # noqa: E402

PURE = ("kda-sig-lowrank", "ckda-shipped-lowrank")
HYBRID = ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank")


@pytest.mark.parametrize("arm", PURE + HYBRID)
def test_the_config_is_the_arms_own_geometry(arm):
    cfg = hf_config(arm, "1.3B")
    assert cfg["hidden_size"] == 2048
    assert cfg["num_hidden_layers"] == 24
    assert cfg["head_dim"] == 128
    assert cfg["num_heads"] == 16
    assert cfg["vocab_size"] == 32000
    assert cfg["output_gate"] == "lowrank"
    assert cfg["gate_init_style"] == "shipped"
    assert cfg["gate"] == ("sigmoid" if arm.startswith("kda-sig") else "signed_sigmoid2")
    assert cfg["allow_neg_eigval"] is arm.startswith("ckda")
    # d_ffn pays for the hybrid's attention parameters, so it is NOT the same.
    assert cfg["intermediate_size"] == (5312 if "hybrid" in arm else 5440)
    # A multiple of 64. Not of 128: both values are 64 mod 128, and rounding
    # either to 5376 or 5504 moves the parameter count off the attention
    # reference the whole ladder is matched to. 64 is what the matcher rounds to
    # and what these runs train.
    assert cfg["intermediate_size"] % 64 == 0


@pytest.mark.parametrize("arm", PURE)
def test_a_pure_arm_has_no_attention_spec(arm):
    # `attn: None` is what makes every layer the mixer. A stray spec here would
    # put full attention into six layers of a pure arm.
    assert hf_config(arm, "1.3B").get("attn") is None


@pytest.mark.parametrize("arm", HYBRID)
def test_a_hybrid_declares_the_stack_it_trained(arm):
    spec = hf_config(arm, "1.3B")["attn"]
    # (i + 1) % 4 == 0 at 24 layers, which is titan_ext.hybrid_layers -- six
    # attention layers, eighteen linear, Kimi Linear's 3:1. The off-by-one
    # ([0, 4, ...] instead) is a different model under the same name.
    assert spec["layers"] == [3, 7, 11, 15, 19, 23]
    assert spec["num_heads"] == spec["num_kv_heads"] == 16
    # Gated and NoPE, as the runs trained and as fla's Attention reads them.
    assert spec["output_gate"] is True
    assert spec["use_rope"] is False
    assert spec["qkv_bias"] is False and spec["qk_norm"] is False
    assert spec["window_size"] is None


def test_the_hybrid_spec_is_what_fla_builds_from():
    """The keys the vendored block actually reads, with fla's own normalizer."""
    from fla.models.hybrid import get_hybrid_attention_spec, normalize_hybrid_attention_config

    spec = hf_config("kda-sig-hybrid-lowrank", "1.3B")["attn"]
    norm = normalize_hybrid_attention_config(spec, num_hidden_layers=24)
    # Unknown keys survive normalization -- which is what lets output_gate and
    # use_rope through, neither being a field of HybridAttentionSpec.
    assert norm["output_gate"] is True and norm["use_rope"] is False
    assert get_hybrid_attention_spec(norm, layer_idx=3) is norm
    assert get_hybrid_attention_spec(norm, layer_idx=4) is None


def test_the_feed_forward_rename_keeps_gate_and_up_apart():
    # torchtitan computes w2(silu(w1(x)) * w3(x)); fla computes
    # down_proj(swish(gate_proj(x)) * up_proj(x)). Swapping w1 and w3 gives a
    # model of the right size and the wrong output.
    assert rename("layers.5.feed_forward.w1.weight") == "model.layers.5.mlp.gate_proj.weight"
    assert rename("layers.5.feed_forward.w3.weight") == "model.layers.5.mlp.up_proj.weight"
    assert rename("layers.5.feed_forward.w2.weight") == "model.layers.5.mlp.down_proj.weight"


def test_the_mixer_and_attention_parameters_both_land_under_attn():
    assert rename("layers.0.attention.inner.q_proj.weight") == "model.layers.0.attn.q_proj.weight"
    assert rename("layers.3.attention.inner.g_proj.weight") == "model.layers.3.attn.g_proj.weight"
    assert rename("layers.0.attention_norm.weight") == "model.layers.0.attn_norm.weight"
    assert rename("layers.0.ffn_norm.weight") == "model.layers.0.mlp_norm.weight"
    assert rename("tok_embeddings.weight") == "model.embeddings.weight"
    assert rename("output.weight") == "lm_head.weight"
    # Optimizer and dataloader state are not model tensors.
    assert rename("optimizer.state.step") is None
    assert rename("dataloader.state") is None


def test_the_export_geometry_is_the_published_one_and_not_a_default():
    """Untied embeddings and a 32,000 vocabulary are the PUBLISHED model's, and
    neither is written in the table entry -- both come from `hf_config`'s
    defaults. A change to those defaults would silently export a tied head for
    a model that has none, which loads without complaint and scores wrong.
    """
    cfg = hf_config("kda-sig-lowrank", "1.3B")
    assert cfg["tie_word_embeddings"] is False
    assert cfg["vocab_size"] == 32000
    assert cfg["max_position_embeddings"] == 4096
    assert cfg["head_dim"] == 128
    assert cfg["intermediate_size"] % 64 == 0
