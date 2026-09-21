# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib

import pytest
import torch
import triton

from fla.ops.kda import chunk_kda
from fla.ops.kda.backends.optimized import KDAOptimizedTritonBackend
from fla.ops.kda.backends.tilelang import KDATileLangBackend
from fla.utils import IS_NVIDIA_HOPPER, assert_close, device


def _reference_gate_gradient(q, k, v, g, beta, sign, h0, cu, state_v_first, norm, do, dht):
    # retain input quantization, but evaluate the recurrence and its derivative in FP64.
    q, k, v = [x.detach() for x in (q, k, v)]
    if norm:
        q, k = [(x.float() * (x.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt()).to(x.dtype)
                for x in (q, k)]
    q, k, v = [x.double() for x in (q, k, v)]
    beta = (beta.detach().float().sigmoid() * (2 if sign is not None else 1)).double()
    gate = g.detach().double().requires_grad_()
    boundaries = [0, q.shape[1]] if cu is None else cu.tolist()
    outputs, finals = [], []
    for n, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        state = h0[n].detach().double()
        if state_v_first:
            state = state.transpose(-1, -2)
        for t in range(start, end):
            decay = gate[0, t].exp()
            if sign is not None:
                decay = decay * sign[0, t]
            state = state * decay.unsqueeze(-1)
            key = k[0, t]
            residual = v[0, t] - (key.unsqueeze(-1) * state).sum(-2)
            state = state + beta[0, t, :, None, None] * key.unsqueeze(-1) * residual.unsqueeze(-2)
            outputs.append((q[0, t].unsqueeze(-1) * state).sum(-2) * q.shape[-1] ** -0.5)
        finals.append(state.transpose(-1, -2) if state_v_first else state)
    output, final = torch.stack(outputs).unsqueeze(0), torch.stack(finals)
    return torch.autograd.grad((output, final), gate, (do.double(), dht.double()))[0]


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('signed', [False, True])
def test_reference_gate_gradient_beta_precision(dtype, signed):
    shape = (1, 1, 1, 1)
    q, k, v, do = [torch.full(shape, value, dtype=dtype) for value in (0.5, 0.75, 0.2, 1.3)]
    g, h0, dht = [torch.full(shape, value) for value in (-0.1, 0.7, 0.4)]
    beta = torch.full((1, 1, 1), 0.7, dtype=dtype)
    sign = -torch.ones(shape, dtype=torch.int8) if signed else None
    actual = _reference_gate_gradient(q, k, v, g, beta, sign, h0, None, False, False, do, dht)
    activated_beta = (beta.float().sigmoid() * (2 if signed else 1)).double().unsqueeze(-1)
    expected = ((do.double() * q.double() + dht.double()) * (1 - activated_beta * k.double().square())
                * h0.double() * g.double().exp() * (-1 if signed else 1))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('backend', ['triton', 'tilelang', 'triton_optimized', 'hybrid_optimized'])
def test_chunk_dispatch_without_kernel_launch(monkeypatch, backend):
    from fla.ops.backends import BackendRegistry
    from fla.ops.kda.backends import kda_registry

    BackendRegistry.ensure_initialized('kda')
    key = 'tilelang_baseline' if backend == 'tilelang' else backend
    selected = kda_registry._backends[key]
    marker = object()
    monkeypatch.setattr(type(selected), 'chunk_kda_verifier', lambda *args, **kwargs: (True, None))
    monkeypatch.setattr(selected, 'chunk_kda', lambda *args, **kwargs: marker)
    monkeypatch.setenv('FLA_KDA_BACKEND', backend)
    assert chunk_kda(None, None, None, None, None) is marker


@pytest.mark.skipif(not IS_NVIDIA_HOPPER, reason='optimized implementations currently require Hopper')
@pytest.mark.parametrize('backend', ['triton_optimized', 'tilelang', 'hybrid_optimized'])
@pytest.mark.parametrize('signed', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('varlen,state_v_first,norm', [(False, False, True), (True, True, True), (False, True, False)])
def test_chunk_implementations(monkeypatch, backend, signed, dtype, varlen, state_v_first, norm, gate_tolerance=None):
    if backend in ('tilelang', 'hybrid_optimized') and not KDATileLangBackend.is_available():
        pytest.skip('TileLang or CUDA compiler unavailable')
    torch.manual_seed(42)
    shape = (1, 137, 2, 60)
    q, k = [torch.randn(shape, device=device, dtype=dtype) for _ in range(2)]
    if not norm:
        q, k = [torch.nn.functional.normalize(x, dim=-1) for x in (q, k)]
    q, k = q.requires_grad_(), k.requires_grad_()
    v = torch.randn(1, 137, 2, 32, device=device, dtype=dtype).requires_grad_()
    g = (-0.02 * torch.rand(shape, device=device)).requires_grad_()
    beta = torch.randn(1, 137, 2, device=device, dtype=dtype).requires_grad_()
    sign = (torch.randint(0, 2, shape, device=device, dtype=torch.int8) * 2 - 1) if signed else None
    cu = torch.tensor([0, 33, 137], device=device, dtype=torch.long) if varlen else None
    state_shape = (32, 60) if state_v_first else (60, 32)
    h0 = torch.randn(2 if varlen else 1, 2, *state_shape, device=device).requires_grad_()
    leaves = (q, k, v, g, beta, h0)
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, sign=sign, initial_state=h0, output_final_state=True,
                  cu_seqlens=cu, state_v_first=state_v_first, use_qk_l2norm_in_kernel=norm,
                  use_beta_sigmoid_in_kernel=True, allow_neg_eigval=signed)
    monkeypatch.setenv('FLA_KDA_BACKEND', 'triton')
    ref, ref_ht = chunk_kda(**kwargs)
    do, dht = torch.randn_like(ref), torch.randn_like(ref_ht)
    expected = torch.autograd.grad((ref, ref_ht), leaves, (do, dht))
    monkeypatch.setenv('FLA_KDA_BACKEND', backend)
    actual, actual_ht = chunk_kda(**kwargs)
    # backward must retain the implementation chosen by forward.
    monkeypatch.setenv('FLA_KDA_BACKEND', 'invalid-after-forward')
    gradients = torch.autograd.grad((actual, actual_ht), leaves, (do, dht))
    reference_dg = _reference_gate_gradient(q, k, v, g, beta, sign, h0, cu, state_v_first, norm, do, dht)
    if gate_tolerance is None:
        # match the existing BF16 varlen gate-gradient contract in test_kda.py.
        gate_tolerance = 0.015 if dtype == torch.bfloat16 else 0.005
    for name, a, b in zip(('o', 'ht', 'dq', 'dk', 'dv', 'dg', 'db', 'dh0'),
                          (ref, ref_ht, *expected), (actual, actual_ht, *gradients)):
        # historical BF16 WY reductions are layout-dependent; use the mathematical reference for dg.
        if name == 'dg':
            a = reference_dg
        assert_close(name, a, b, gate_tolerance if name == 'dg' else 0.005)


@pytest.mark.skipif(not IS_NVIDIA_HOPPER, reason='optimized implementations currently require Hopper')
@pytest.mark.parametrize('bv,warps', [(32, 2), (128, 4)])
def test_chunk_wy_reduction_configs(monkeypatch, bv, warps):
    from fla.ops.kda.backends.optimized.wy import chunk_kda_bwd_kernel_wy_dqkg_fused

    tuner = chunk_kda_bwd_kernel_wy_dqkg_fused.fn
    monkeypatch.setattr(tuner, 'configs', [triton.Config({'BK': 64, 'BV': bv}, num_warps=warps, num_stages=2)])
    test_chunk_implementations(monkeypatch, 'triton_optimized', False, torch.bfloat16, True, True, True, gate_tolerance=0.005)


@pytest.mark.skipif(not IS_NVIDIA_HOPPER, reason='optimized implementations currently require Hopper')
@pytest.mark.parametrize('backend', ['triton', 'tilelang', 'triton_optimized', 'hybrid_optimized'])
def test_chunk_dispatch_route(monkeypatch, backend):
    if backend in ('tilelang', 'hybrid_optimized') and not KDATileLangBackend.is_available():
        pytest.skip('TileLang or CUDA compiler unavailable')
    from fla.ops.kda.backends import kda_registry
    torch.manual_seed(42)
    q = torch.randn(1, 65, 2, 16, device=device, requires_grad=True)
    g = -torch.rand_like(q)
    beta = torch.rand(1, 65, 2, device=device)
    expected_class = 'tilelang_baseline' if backend == 'tilelang' else backend
    selected = kda_registry._backends[expected_class]
    original = selected.chunk_kda
    calls = []

    def tracked(*args, **kwargs):
        calls.append(backend)
        return original(*args, **kwargs)

    monkeypatch.setattr(selected, 'chunk_kda', tracked)
    monkeypatch.setenv('FLA_KDA_BACKEND', backend)
    output, _ = chunk_kda(q, q, q, g, beta, use_qk_l2norm_in_kernel=True)
    output.sum().backward()
    assert calls == [backend]


def test_chunk_unknown_backend(monkeypatch):
    monkeypatch.setenv('FLA_KDA_BACKEND', 'typo')
    with pytest.raises(ValueError, match='Unknown FLA_KDA_BACKEND'):
        chunk_kda(None, None, None, None, None)


def test_chunk_verifier_cpu():
    q = torch.zeros(1, 65, 2, 16)
    accepted, reason = KDAOptimizedTritonBackend().chunk_kda_verifier(q, q, q, q, q[..., 0])
    assert not accepted and 'CUDA' in reason


@pytest.mark.skipif(not IS_NVIDIA_HOPPER, reason='optimized implementations currently require Hopper')
def test_chunk_rejection_is_strict(monkeypatch):
    monkeypatch.setenv('FLA_KDA_BACKEND', 'triton_optimized')
    q = torch.zeros(1, 65, 2, 256, device=device)
    with pytest.raises(RuntimeError, match='key dimension <= 128'):
        chunk_kda(q, q, q, q, q[..., 0])


@pytest.mark.skipif(not IS_NVIDIA_HOPPER, reason='parallel parity currently requires Hopper')
def test_parallel_parity():
    from fla.ops.kda.backends.optimized.parity import sign_cumprod_parallel
    torch.manual_seed(42)
    sign = torch.randint(0, 2, (1, 8193, 16, 128), device=device, dtype=torch.int8) * 2 - 1
    gauge = importlib.import_module('fla.ops.kda.gauge')
    assert torch.equal(gauge.kda_sign_cumprod(sign), sign_cumprod_parallel(sign, 128, 64))
