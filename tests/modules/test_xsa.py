# Exclusive self attention: the fused kernel, and the xsa option of the delta-rule layers.

import pytest
import torch

from fla.modules.xsa import HAVE_TRITON, exclusive_self_attention, xsa_reference
from fla.utils import device

pytestmark = pytest.mark.skipif(device != "cuda" or not HAVE_TRITON, reason="needs CUDA + Triton")


@pytest.mark.parametrize("hq,hkv,d", [(4, 4, 64), (8, 2, 64), (6, 1, 96)])
@pytest.mark.parametrize("gated", [False, True])
def test_kernel_matches_reference(hq, hkv, d, gated):
    torch.manual_seed(0)
    o = torch.randn(37, hq, d, device=device, requires_grad=True)
    v = torch.randn(37, hkv, d, device=device, requires_grad=True)
    g = torch.randn(37, hq, d, device=device, requires_grad=True) if gated else None
    dy = torch.randn(37, hq, d, device=device)
    outs = []
    for fn in (exclusive_self_attention, xsa_reference):
        for t in (o, v, g):
            if t is not None:
                t.grad = None
        fn(o, v, g).backward(dy)
        outs.append([t.grad.clone() for t in (o, v, g) if t is not None])
    for a, b in zip(*outs):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("layer", ["gdn", "kda", "complex_kda"])
def test_layer_output_is_orthogonal_to_own_values(layer, monkeypatch):
    from fla.layers import GatedDeltaNet, KimiDeltaAttention
    from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
    from fla.modules import xsa as xsa_module

    cls = {"gdn": GatedDeltaNet, "kda": KimiDeltaAttention, "complex_kda": ComplexKimiDeltaAttention}[layer]
    torch.manual_seed(0)
    m = cls(hidden_size=256, head_dim=64, num_heads=4, xsa=True).to(device).bfloat16()
    seen = {}
    orig = xsa_module.exclusive_self_attention

    def spy(o, v, gate=None, **kw):
        out = orig(o, v, gate, **kw)
        seen["out"], seen["v"] = out.detach().float(), v.detach().float()
        return out

    monkeypatch.setattr(xsa_module, "exclusive_self_attention", spy)
    x = torch.randn(2, 64, 256, device=device, dtype=torch.bfloat16, requires_grad=True)
    o, *_ = m(x)
    o.float().sum().backward()
    out, v = seen["out"], seen["v"]
    coef = (out * v).sum(-1) / v.pow(2).sum(-1).clamp(min=1e-6)
    assert coef.abs().max() < 2e-2
    assert x.grad is not None and torch.isfinite(x.grad).all()
