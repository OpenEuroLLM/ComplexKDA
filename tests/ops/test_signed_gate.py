# -*- coding: utf-8 -*-

import pytest
import torch

from fla.layers.complex_kda_layer import signed_gate
from fla.ops.kda.signed_gate import fused_signed_gate

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("activation", ["sigmoid2", "tanh"])
@pytest.mark.parametrize("with_params", [True, False])
@pytest.mark.parametrize("shape", [(2, 100, 4, 64), (1, 37, 3, 48)])
def test_matches_eager(activation, with_params, shape):
    torch.manual_seed(0)
    B, T, H, K = shape
    z = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    z[0, :3] = 0.0  # exactly zero pre-activations: sign +1, |a| gradient 0
    z[0, 3:6] = 1e-4  # tiny ones: the sign must not be decided by rounding
    dt = (0.3 * torch.randn(H * K, device="cuda")) if with_params else None
    A = (0.2 * torch.randn(H, device="cuda")) if with_params else None
    dg = torch.randn(shape, device="cuda")

    def run(fn):
        zz = z.detach().clone().requires_grad_(True)
        dd = dt.detach().clone().requires_grad_(True) if dt is not None else None
        aa = A.detach().clone().requires_grad_(True) if A is not None else None
        s, g = fn(zz, aa, dd, -5.0, activation)
        g.backward(dg)
        return s, g, zz.grad, (dd.grad if dd is not None else None), (aa.grad if aa is not None else None)

    ref = run(lambda zz, aa, dd, lb, act: signed_gate(zz, aa, dd, lb, activation=act))
    got = run(lambda zz, aa, dd, lb, act: fused_signed_gate(zz, aa, dd, lb, activation=act))
    assert got[0].dtype == torch.int8 and torch.equal(got[0], ref[0])
    torch.testing.assert_close(got[1], ref[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got[2].float(), ref[2].float(), rtol=1e-2, atol=1e-2)
    for a, b in zip(got[3:], ref[3:]):
        if b is not None:
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
