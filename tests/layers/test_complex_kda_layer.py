# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import math

import pytest
import torch

from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention, _beta_spread_init_params, _identity
from fla.models.complex_kda.configuration_complex_kda import ComplexKDAConfig
from fla.models.complex_kda.modeling_complex_kda import ComplexKDABlock


@pytest.mark.parametrize("logit_std", [0.2, 0.45, 1.2])
def test_beta_spread_init_matches_standard_width_and_modes(logit_std):
    weight_scale, bias = _beta_spread_init_params(logit_std)
    spread_std = weight_scale * logit_std
    q = (torch.arange(1024, dtype=torch.float64) + 0.5) / 1024
    z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)

    standard_width = torch.sigmoid(logit_std * z).std(correction=0)
    spread_width = (2.0 * torch.sigmoid(spread_std * z - bias)).std(correction=0)
    assert torch.isclose(spread_width, standard_width, atol=1e-10, rtol=0)

    log3 = math.log(3.0)
    mode_residual = log3 - bias * math.tanh(bias * log3 / spread_std**2) - 0.5 * spread_std**2
    assert abs(mode_residual) < 1e-12


def test_beta_spread_init_assigns_balanced_layer_biases():
    torch.manual_seed(0)
    layer = ComplexKimiDeltaAttention(
        hidden_size=128,
        head_dim=16,
        num_heads=8,
        use_short_conv=False,
        allow_neg_eigval=True,
        beta_init_style="spread",
    )
    bias = layer.b_proj.bias
    assert bias is not None
    assert (bias < 0).sum() == (bias > 0).sum() == 4
    assert torch.allclose(bias.abs(), bias[0].abs().expand_as(bias))


def test_beta_spread_init_requires_extended_range():
    with pytest.raises(ValueError, match="requires allow_neg_eigval=True"):
        ComplexKimiDeltaAttention(
            hidden_size=128,
            head_dim=16,
            num_heads=8,
            use_short_conv=False,
            allow_neg_eigval=False,
            beta_init_style="spread",
        )
    with pytest.raises(ValueError, match="requires `allow_neg_eigval=True`"):
        ComplexKDAConfig(allow_neg_eigval=False, beta_init_style="spread")


def test_standard_beta_init_remains_bias_free():
    layer = ComplexKimiDeltaAttention(
        hidden_size=128,
        head_dim=16,
        num_heads=8,
        use_short_conv=False,
        allow_neg_eigval=False,
    )
    assert layer.beta_init_style == "standard"
    assert layer.b_proj.bias is None


def test_model_config_propagates_drop_silu():
    config = ComplexKDAConfig(
        hidden_size=128,
        head_dim=16,
        num_heads=8,
        intermediate_size=256,
        num_hidden_layers=1,
        use_short_conv=False,
        drop_silu=True,
        fuse_norm=False,
        fuse_swiglu=False,
    )
    block = ComplexKDABlock(config, layer_idx=0)
    assert block.attn.act is _identity


def test_model_config_propagates_drop_key_silu():
    config = ComplexKDAConfig(
        hidden_size=128,
        head_dim=16,
        num_heads=8,
        intermediate_size=256,
        num_hidden_layers=1,
        use_short_conv=False,
        drop_key_silu=True,
        fuse_norm=False,
        fuse_swiglu=False,
    )
    block = ComplexKDABlock(config, layer_idx=0)
    assert block.attn.act is torch.nn.functional.silu
    assert block.attn.k_act is _identity


def test_drop_key_silu_short_conv_only_changes_key_activation():
    layer = ComplexKimiDeltaAttention(
        hidden_size=128,
        head_dim=16,
        num_heads=8,
        use_short_conv=True,
        drop_key_silu=True,
    )
    assert layer.q_conv1d.activation == "silu"
    assert layer.k_conv1d.activation is None
    assert layer.v_conv1d.activation == "silu"


def test_drop_key_silu_composes_with_conv_silu():
    config = ComplexKDAConfig(
        hidden_size=128,
        head_dim=16,
        num_heads=8,
        intermediate_size=256,
        num_hidden_layers=1,
        use_short_conv=True,
        conv_silu="qk",
        drop_key_silu=True,
        fuse_norm=False,
        fuse_swiglu=False,
    )
    block = ComplexKDABlock(config, layer_idx=0)
    assert block.attn.q_conv1d.activation == "silu"
    assert block.attn.k_conv1d.activation is None
    assert block.attn.v_conv1d.activation is None
