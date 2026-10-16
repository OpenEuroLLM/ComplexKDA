"""The fused input-projection / short-conv path of the linear-attention layers computes the
same as the module-by-module path (fuse_projections=False)."""

import pytest
import torch

from fla.layers import GatedDeltaNet, KimiDeltaAttention
from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention as ComplexKDA
from fla.utils import device


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("make", [
    lambda: KimiDeltaAttention(hidden_size=128, head_dim=32, num_heads=4, mode="chunk"),
    lambda: GatedDeltaNet(hidden_size=128, head_dim=32, num_heads=4, mode="chunk"),
    lambda: ComplexKDA(hidden_size=128, head_dim=32, num_heads=4, mode="chunk"),
    # mixed conv activations (two launches) and a full-rank output gate
    lambda: ComplexKDA(hidden_size=128, head_dim=32, num_heads=4, mode="chunk", conv_silu="qv",
                       output_gate="linear", beta_init_style="spread"),
])
def test_fused_projections_match(make, varlen):
    torch.manual_seed(0)
    layer = make().to(device)
    x = torch.randn(1 if varlen else 2, 128, 128, device=device)
    kw = {"cu_seqlens": torch.tensor([0, 50, 128], device=device, dtype=torch.int32)} if varlen else {}
    results = []
    for fused in (True, False):
        layer.fuse_projections = fused
        layer.zero_grad()
        xx = x.clone().requires_grad_()
        out = layer(xx, **kw)[0]
        out.square().mean().backward()
        results.append((out.detach(), xx.grad,
                        {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}))
    (o1, gx1, g1), (o2, gx2, g2) = results
    # one GEMM vs. several may round differently under TF32
    torch.testing.assert_close(o1, o2, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(gx1, gx2, rtol=1e-2, atol=1e-3)
    assert g1.keys() == g2.keys()
    for n in g1:
        assert (g1[n] - g2[n]).norm() <= 2e-2 * g2[n].norm() + 1e-5, n
