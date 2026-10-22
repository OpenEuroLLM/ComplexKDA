# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import functools
import math

import pytest
import torch
import torch.nn.functional as F
from triton.runtime.autotuner import Autotuner
from triton.runtime.errors import OutOfResources

from fla.ops.kda.naive import naive_recurrent_kda
from fla.ops.kronlstm import (
    chunk_bwd,
    chunk_fwd,
    chunk_kronlstm,
    fused_recurrent_kronlstm,
    naive_chunk_kronlstm,
    naive_recurrent_kronlstm,
)
from fla.ops.kronlstm.naive import V_NORM_FLOOR
from fla.utils import IS_NVIDIA, assert_close, device

pytestmark = pytest.mark.skipif(not IS_NVIDIA, reason='the KronLSTM kernels are validated on NVIDIA GPUs only')

TOL = {torch.float32: 5e-6, torch.bfloat16: 2e-2}
RULES = {'A': dict(decouple_erase=False), 'D': dict(target_gate=True, decouple_erase=False),
         'E': dict(target_gate=True, decouple_erase=True)}
GATES = {'none': None, 'scalar': 'gs', 'vector': 'gv'}
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16}


@pytest.fixture(scope='module', autouse=True)
def ieee_fp32_dots():
    # fp32 `tl.dot` defaults to tf32, which reads ~1e-3 against the float64 reference; the fp32 tolerances need ieee
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('TRITON_F32_DEFAULT', 'ieee')
        yield


def gen(B, T, H, K, V, seed=0, gmag=0.3):
    torch.manual_seed(seed)
    d = dict(dtype=torch.float64, device=device)
    return dict(
        q=F.normalize(torch.randn(B, T, H, K, **d), dim=-1),
        k=F.normalize(torch.randn(B, T, H, K, **d), dim=-1),
        v=F.normalize(torch.randn(B, T, H, V, **d), dim=-1),
        beta=torch.rand(B, T, H, **d) * 0.9 + 0.05,
        gs=-(torch.rand(B, T, H, **d) * 0.9 + 0.1) * gmag,
        gv=-(torch.rand(B, T, H, K, **d) * 0.9 + 0.1) * gmag,
        ct=torch.sigmoid(torch.randn(B, T, H, **d) * 5.0),
        S0=torch.randn(B, H, K, V, **d) * 0.1,
        do=torch.randn(B, T, H, V, **d),
        dS=torch.randn(B, H, K, V, **d) * 0.1,
    )


def rule_kwargs(x, rule, dt):
    # s = 2, the full range of the write strength; the layer tests run its default, s = 1
    kw = dict(RULES[rule], allow_neg_eigval=True)
    if kw.pop('target_gate', False):
        kw['target_gate'] = x['ct'].to(torch.float32 if dt == torch.bfloat16 else dt)
    return kw


def run_grads(op, x, dt, gate, rule, chunk_size=None, initial_state=True):
    """Outputs, final state and the gradients of every input of `op` under a fixed random cotangent."""
    gk = GATES[gate]
    names = ['q', 'k', 'v', 'beta'] + ([gk] if gk else []) + (['S0'] if initial_state else [])
    leaves = {n: x[n].to(dt).detach().clone().requires_grad_(True) for n in names}
    kw = rule_kwargs(x, rule, dt)
    if 'target_gate' in kw:
        kw['target_gate'] = kw['target_gate'].detach().clone().requires_grad_(True)
        leaves['ct'] = kw['target_gate']
    if chunk_size is not None:
        kw['chunk_size'] = chunk_size
    o, S = op(leaves['q'], leaves['k'], leaves['v'], leaves.get(gk) if gk else None, leaves['beta'],
              initial_state=leaves.get('S0'), output_final_state=True, **kw)
    loss = (o.double() * x['do']).sum() + (S.double() * x['dS']).sum()
    grads = torch.autograd.grad(loss, list(leaves.values()))
    return dict(zip(leaves.keys(), grads)), o, S


def assert_all_close(ref, tri, ratio, err_atol=1e-6):
    (g_ref, o_ref, S_ref), (g_tri, o_tri, S_tri) = ref, tri
    assert_close('o', o_ref, o_tri, ratio, err_atol=err_atol)
    assert_close('ht', S_ref, S_tri, ratio, err_atol=err_atol)
    for name in g_ref:
        assert_close(f'd{name}', g_ref[name], g_tri[name], ratio, err_atol=err_atol)


@pytest.mark.parametrize('rule', list(RULES))
def test_naive_recurrent_is_kda_in_kv_space(rule):
    """The state recursion is KDA's with key vec(k v^T), the target as value, beta* as beta and the gate repeated over V."""
    B, T, H, K, V = 2, 96, 3, 16, 8
    x = gen(B, T, H, K, V)
    kw = rule_kwargs(x, rule, torch.float64)
    _, S = naive_recurrent_kronlstm(x['q'], x['k'], x['v'], x['gv'], x['beta'], scale=1.0,
                                    initial_state=x['S0'], output_final_state=True, **kw)
    a = (x['k'].unsqueeze(-1) * x['v'].unsqueeze(-2)).reshape(B, T, H, K * V)
    vn = x['v'].norm(dim=-1) ** 2
    target = vn if rule == 'A' else vn * x['ct']
    if rule == 'E':
        target = target / x['beta']
    beta_star = 2.0 / vn.clamp_min(V_NORM_FLOOR) * x['beta']
    _, S_kda = naive_recurrent_kda(a, a, target.unsqueeze(-1), x['gv'].repeat_interleave(V, dim=-1), beta_star,
                                   scale=1.0, initial_state=x['S0'].reshape(B, H, K * V, 1), output_final_state=True)
    # KDA's reference accumulates in fp32
    assert_close('ht', S_kda.reshape(B, H, K, V), S, 3e-6)


@pytest.mark.parametrize('rule', list(RULES))
@pytest.mark.parametrize('gate', ['none', 'scalar'])
def test_naive_recurrent_all_equal_vector_gate_is_the_scalar_gate(gate, rule):
    x = gen(2, 64, 3, 16, 8)
    kw = rule_kwargs(x, rule, torch.float64)
    g = None if gate == 'none' else x['gs']
    g_eq = torch.zeros_like(x['gv']) if gate == 'none' else x['gs'].unsqueeze(-1).expand_as(x['gv']).contiguous()
    a = naive_recurrent_kronlstm(x['q'], x['k'], x['v'], g, x['beta'], initial_state=x['S0'], output_final_state=True, **kw)
    b = naive_recurrent_kronlstm(x['q'], x['k'], x['v'], g_eq, x['beta'], initial_state=x['S0'], output_final_state=True,
                                 **kw)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize('rule', list(RULES))
@pytest.mark.parametrize('gate', list(GATES))
def test_naive_chunk(gate, rule):
    """The chunkwise reference against the token-by-token one: output, state and every gradient, in float64."""
    x = gen(2, 128, 3, 16, 8)
    ref = run_grads(naive_recurrent_kronlstm, x, torch.float64, gate, rule)
    tri = run_grads(naive_chunk_kronlstm, x, torch.float64, gate, rule, chunk_size=32)
    assert_all_close(ref, tri, 1e-12, err_atol=0.0)


def test_naive_chunk_gradcheck():
    x = gen(1, 24, 2, 4, 4, seed=3)
    names = ['q', 'k', 'v', 'gv', 'beta', 'S0', 'ct']
    leaves = [x[n].clone().requires_grad_(True) for n in names]

    def fn(q, k, v, g, beta, S0, ct):
        return naive_chunk_kronlstm(q, k, v, g, beta, chunk_size=8, initial_state=S0, output_final_state=True,
                                    allow_neg_eigval=True, target_gate=ct, decouple_erase=True)
    assert torch.autograd.gradcheck(fn, leaves, eps=1e-6, atol=1e-6, rtol=1e-5)


# Every (H, K, V, dtype) compiles all configs of the six autotuned kernels, so the shapes are few
# and the other tests reuse them. Every variant, gate rank and dtype runs on the main shape
# (two key blocks, a partial last chunk), and so do larger gates: every exponent the kernels form is referenced
# so that it is <= 0.
# Two variants run on each edge shape: odd T below one chunk with K < 64 (the layer tests' heads),
# one key block with V = 128, V = 256, and K = 256 (four key blocks).
CHUNK_MAIN = (2, 200, 2, 128, 128)
CHUNK_EDGES = [
    ((1, 63, 4, 32, 32), 'fp32'),
    ((1, 63, 4, 32, 32), 'bf16'),
    ((2, 160, 2, 64, 128), 'bf16'),
    ((1, 128, 2, 128, 256), 'fp32'),
    ((1, 100, 2, 256, 64), 'bf16'),
]


def chunk_case(B, T, H, K, V, rule, gate, dtype, gmag=0.3):
    gid = '' if gmag == 0.3 else f'-g{gmag}'
    return pytest.param(B, T, H, K, V, rule, gate, dtype, gmag, id=f'B{B}-T{T}-H{H}-K{K}-V{V}-{rule}-{gate}-{dtype}{gid}')


CHUNK_CASES = (
    [chunk_case(*CHUNK_MAIN, rule, gate, dtype) for rule in RULES for gate in GATES for dtype in DTYPES]
    + [chunk_case(*CHUNK_MAIN, 'A', 'vector', 'fp32', gmag) for gmag in (1.0, 4.0, 8.0)]
    + [chunk_case(*shape, rule, 'vector', dtype) for shape, dtype in CHUNK_EDGES for rule in ('A', 'E')]
)


@pytest.mark.parametrize(('B', 'T', 'H', 'K', 'V', 'rule', 'gate', 'dtype', 'gmag'), CHUNK_CASES)
def test_chunk(B, T, H, K, V, rule, gate, dtype, gmag):
    x = gen(B, T, H, K, V, gmag=gmag)
    ref = run_grads(naive_chunk_kronlstm, x, torch.float64, gate, rule)
    tri = run_grads(chunk_kronlstm, x, DTYPES[dtype], gate, rule)
    # dg is a difference of nearly cancelling terms, which costs a digit in fp32 at |g| >= 4 per step
    assert_all_close(ref, tri, TOL[DTYPES[dtype]] * (4 if gmag >= 4 else 1))


def test_chunk_resume_is_bitwise():
    """Two calls chained through the final state equal one call over the whole sequence, at chunk boundaries."""
    x = gen(1, 256, 2, 128, 128)
    args = [x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')]
    with torch.no_grad():
        o_full, S_full = chunk_kronlstm(*args, output_final_state=True, decouple_erase=False)
        o1, S1 = chunk_kronlstm(*(t[:, :128] for t in args), output_final_state=True, decouple_erase=False)
        o2, S2 = chunk_kronlstm(*(t[:, 128:] for t in args), initial_state=S1, output_final_state=True,
                                decouple_erase=False)
    assert torch.equal(torch.cat([o1, o2], 1), o_full) and torch.equal(S2, S_full)


def test_chunk_backward_is_bitwise_repeatable():
    x = gen(2, 512, 2, 128, 128, seed=3)
    g_base, _, _ = run_grads(chunk_kronlstm, x, torch.bfloat16, 'vector', 'E')
    for _ in range(20):
        g_new, _, _ = run_grads(chunk_kronlstm, x, torch.bfloat16, 'vector', 'E')
        for name in g_base:
            assert torch.equal(g_new[name], g_base[name]), name


# variable-length inputs: N sequences packed along T with B = 1 and `cu_seqlens`;
# the references are dense runs of each sequence on its own.
# Sequences of any length, an empty one, K < 64 and four key blocks (K = 256), on the shapes of `test_chunk`
VARLEN_CASES = [
    pytest.param(cu, H, K, V, rule, gate, dtype, id=f'N{len(cu) - 1}-T{cu[-1]}-H{H}-K{K}-V{V}-{rule}-{gate}-{dtype}')
    for cu, H, K, V, rule, gate, dtype in [
        ([0, 15, 64, 64, 500, 513, 1000], 2, 128, 128, 'E', 'vector', 'fp32'),
        ([0, 15, 64, 64, 500, 513, 1000], 2, 128, 128, 'D', 'scalar', 'bf16'),
        ([0, 15, 64, 64, 500, 513, 1000], 2, 128, 128, 'A', 'none', 'fp32'),
        ([0, 1, 33, 160], 4, 32, 32, 'E', 'vector', 'fp32'),
        ([0, 200, 456], 2, 256, 64, 'A', 'vector', 'bf16'),
    ]
]


def gen_packed(cu, H, K, V, seed=0):
    """`gen` for `N = len(cu) - 1` sequences packed along T, with one initial state and one state cotangent per sequence."""
    x = gen(1, cu[-1], H, K, V, seed=seed)
    N = len(cu) - 1
    x['S0'], x['dS'] = x['S0'].expand(N, -1, -1, -1).clone(), x['dS'].expand(N, -1, -1, -1).clone()
    x['S0'] += torch.randn_like(x['S0']) * 0.1
    x['dS'] += torch.randn_like(x['dS']) * 0.1
    return x


def run_segments(op, x, dt, gate, rule, cu, packed, grads=True):
    """`run_grads` on packed sequences: one call with `cu_seqlens`, or one dense call per sequence, concatenated.

    An empty sequence contributes no outputs and passes its initial state through."""
    gk = GATES[gate]
    names = ['q', 'k', 'v', 'beta'] + ([gk] if gk else []) + ['S0']
    leaves = {n: x[n].to(dt).detach().clone().requires_grad_(grads) for n in names}
    kw = rule_kwargs(x, rule, dt)
    if 'target_gate' in kw:
        kw['target_gate'] = kw['target_gate'].detach().clone().requires_grad_(grads)
        leaves['ct'] = kw['target_gate']
    g = leaves.get(gk) if gk else None
    if packed:
        o, S = op(leaves['q'], leaves['k'], leaves['v'], g, leaves['beta'], initial_state=leaves['S0'],
                  output_final_state=True, cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device), **kw)
    else:
        outs, states = [], []
        for i, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
            if s == e:
                states.append(leaves['S0'][i:i + 1])
                continue
            kw_i = {**kw, **({'target_gate': kw['target_gate'][:, s:e]} if 'target_gate' in kw else {})}
            o_i, S_i = op(leaves['q'][:, s:e], leaves['k'][:, s:e], leaves['v'][:, s:e],
                          g[:, s:e] if g is not None else None, leaves['beta'][:, s:e], initial_state=leaves['S0'][i:i + 1],
                          output_final_state=True, **kw_i)
            outs.append(o_i)
            states.append(S_i)
        o, S = torch.cat(outs, 1), torch.cat(states, 0)
    if not grads:
        return {}, o, S
    loss = (o.double() * x['do']).sum() + (S.double() * x['dS']).sum()
    return dict(zip(leaves.keys(), torch.autograd.grad(loss, list(leaves.values())))), o, S


@pytest.mark.parametrize(('cu', 'H', 'K', 'V', 'rule', 'gate', 'dtype'), VARLEN_CASES)
def test_chunk_varlen(cu, H, K, V, rule, gate, dtype):
    """A packed run equals dense runs of each sequence on its own: outputs, final states and every gradient."""
    x = gen_packed(cu, H, K, V)
    ref = run_segments(chunk_kronlstm, x, DTYPES[dtype], gate, rule, cu, packed=False)
    tri = run_segments(chunk_kronlstm, x, DTYPES[dtype], gate, rule, cu, packed=True)
    assert_all_close(ref, tri, TOL[DTYPES[dtype]])


def test_rejects_unsupported_inputs():
    """Head dims outside the kernels' range are refused, and so are fp32 inputs whose state the scan cannot stage;
    `cu_seqlens` needs the sequences packed along T with B = 1 and one initial state per sequence."""
    for K, V in [(8, 32), (32, 8), (512, 32), (32, 512)]:
        x = gen(1, 16, 2, K, V)
        inputs = [x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')]
        with pytest.raises(ValueError, match='head dims'):
            chunk_kronlstm(*inputs, decouple_erase=False)
        if max(K, V) > 256:
            with pytest.raises(ValueError, match='K, V <= 256'):
                fused_recurrent_kronlstm(*inputs, decouple_erase=False)
    x = gen(1, 16, 2, 256, 256)
    with pytest.raises(ValueError, match='fp32 inputs'):
        chunk_kronlstm(*(x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')), decouple_erase=False)

    x = gen(2, 64, 2, 32, 32)
    args = [x[n].float() for n in ('q', 'k', 'v', 'gv', 'beta')]
    cu = torch.tensor([0, 32, 64], device=device)
    with pytest.raises(ValueError, match='batch size'):
        chunk_kronlstm(*args, cu_seqlens=cu, decouple_erase=False)
    with pytest.raises(ValueError, match='initial states'):
        chunk_kronlstm(*(t[:1] for t in args), initial_state=x['S0'][:1].float(), cu_seqlens=cu,
                       decouple_erase=False)
    with pytest.raises(ValueError, match='batch size'):
        fused_recurrent_kronlstm(*args, cu_seqlens=cu, decouple_erase=False)


def _autotuner(kernel):
    while not isinstance(kernel, Autotuner):
        kernel = kernel.fn
    return kernel


AUTOTUNERS = {
    'fwd_h': _autotuner(chunk_fwd.chunk_kronlstm_fwd_kernel_h),
    'fwd_intra_diag': _autotuner(chunk_fwd.chunk_kronlstm_fwd_kernel_intra_diag),
    'fwd_inter': _autotuner(chunk_fwd.chunk_kronlstm_fwd_kernel_inter),
    'bwd_dhu': _autotuner(chunk_bwd.chunk_kronlstm_bwd_kernel_dhu),
    'bwd_intra': _autotuner(chunk_bwd.chunk_kronlstm_bwd_kernel_intra),
    'bwd_local': _autotuner(chunk_bwd.chunk_kronlstm_bwd_kernel_local),
}
AUTOTUNE_CASES = [pytest.param(name, i, id=f'{name}-{i}')
                  for name, tuner in AUTOTUNERS.items() for i in range(len(tuner.configs))]


@functools.cache
def _autotune_reference():
    return run_grads(naive_chunk_kronlstm, gen(*CHUNK_MAIN), torch.float64, 'vector', 'D')


@pytest.mark.parametrize(('kernel', 'index'), AUTOTUNE_CASES)
def test_chunk_every_autotune_config(kernel, index, monkeypatch):
    """Whichever config an autotuner may select, forward and backward stay within the bf16 tolerance of the reference."""
    tuner = AUTOTUNERS[kernel]
    monkeypatch.setattr(tuner, 'configs', [tuner.configs[index]])
    try:
        tri = run_grads(chunk_kronlstm, gen(*CHUNK_MAIN), torch.bfloat16, 'vector', 'D')
    except OutOfResources:
        # the autotuner discards a config that does not fit the shape the same way
        pytest.skip('the config does not fit in shared memory at this shape')
    assert_all_close(_autotune_reference(), tri, TOL[torch.bfloat16])


REC_TOL = {torch.float32: 1e-5, torch.bfloat16: 1e-2}
# every variant, gate rank and dtype on the main shape, with and without an initial state;
# two variants on each edge shape: one token with the layer tests' heads, V = 256, and V < K
REC_MAIN = (2, 37, 2, 128, 128)
REC_EDGES = [((1, 1, 4, 32, 32), 'fp32'), ((2, 96, 2, 128, 256), 'bf16'), ((3, 5, 2, 32, 16), 'fp32')]


def rec_case(B, T, H, K, V, rule, gate, dtype, use_h0=True):
    hid = '' if use_h0 else '-no_h0'
    return pytest.param(B, T, H, K, V, rule, gate, dtype, use_h0, id=f'B{B}-T{T}-H{H}-K{K}-V{V}-{rule}-{gate}-{dtype}{hid}')


REC_CASES = (
    [rec_case(*REC_MAIN, rule, gate, dtype) for rule in RULES for gate in GATES for dtype in DTYPES]
    + [rec_case(*REC_MAIN, rule, 'vector', 'fp32', use_h0=False) for rule in ('A', 'E')]
    + [rec_case(*shape, rule, 'vector', dtype) for shape, dtype in REC_EDGES for rule in ('A', 'E')]
)


def rec_call(op, x, dt, gate, rule, initial_state=True, **kwargs):
    gk = GATES[gate]
    g = x[gk].float() if gk else None
    return op(*(x[n].to(dt) for n in ('q', 'k', 'v')), g, x['beta'].to(dt),
              initial_state=x['S0'].float() if initial_state else None, output_final_state=True, **rule_kwargs(x, rule, dt),
              **kwargs)


@pytest.mark.parametrize(('B', 'T', 'H', 'K', 'V', 'rule', 'gate', 'dtype', 'use_h0'), REC_CASES)
def test_fused_recurrent(B, T, H, K, V, rule, gate, dtype, use_h0):
    """The forward-only decode kernel against the token-by-token reference; there is no backward to test."""
    x = gen(B, T, H, K, V)
    o_ref, S_ref = rec_call(naive_recurrent_kronlstm, x, DTYPES[dtype], gate, rule, initial_state=use_h0)
    o_tri, S_tri = rec_call(fused_recurrent_kronlstm, x, DTYPES[dtype], gate, rule, initial_state=use_h0)
    assert_close('o', o_ref, o_tri, REC_TOL[DTYPES[dtype]])
    assert_close('ht', S_ref, S_tri, REC_TOL[DTYPES[dtype]])


@pytest.mark.parametrize(('cu', 'H', 'K', 'V', 'rule', 'gate', 'dtype'), [
    pytest.param(*p, id='N{}-T{}-H{}-K{}-V{}-{}-{}-{}'.format(len(p[0]) - 1, p[0][-1], *p[1:]))
    for p in [
        ([0, 5, 42, 43, 100], 2, 128, 128, 'E', 'vector', 'fp32'),
        ([0, 5, 42, 43, 100], 2, 128, 128, 'E', 'vector', 'bf16'),
        ([0, 1, 1, 2, 3], 2, 128, 128, 'D', 'scalar', 'fp32'),   # one-token sequences, one of them empty
    ]
])
def test_fused_recurrent_varlen(cu, H, K, V, rule, gate, dtype):
    """A packed decode run equals runs of each sequence on its own, in outputs and final states."""
    x = gen_packed(cu, H, K, V)
    dt = DTYPES[dtype]
    _, o_ref, S_ref = run_segments(fused_recurrent_kronlstm, x, dt, gate, rule, cu, packed=False, grads=False)
    _, o_tri, S_tri = run_segments(fused_recurrent_kronlstm, x, dt, gate, rule, cu, packed=True, grads=False)
    assert_close('o', o_ref, o_tri, REC_TOL[dt])
    assert_close('ht', S_ref, S_tri, REC_TOL[dt])


@pytest.mark.parametrize('rule', ['D', 'E'])
def test_fused_recurrent_decoding_matches_chunk(rule):
    """64 one-token steps chained through the final state land where the chunkwise kernel lands over the sequence."""
    x = gen(2, 64, 2, 128, 128, seed=5)
    o_chunk, S_chunk = rec_call(chunk_kronlstm, x, torch.float32, 'vector', rule)
    S = x['S0'].float()
    outs = []
    for t in range(64):
        xt = {n: (x[n][:, t:t + 1] if n not in ('S0', 'dS') else x[n]) for n in x}
        xt['S0'] = S
        o_t, S = rec_call(fused_recurrent_kronlstm, xt, torch.float32, 'vector', rule)
        outs.append(o_t)
    assert_close('o', o_chunk, torch.cat(outs, 1), 1e-5)
    assert_close('ht', S_chunk, S, 1e-5)


# signed forget gate f = sign * exp(g) in [-1, 1] (`sign=`), applied by the ops as a gauge on q and k;
# the oracle is the naive recurrence applying the signed decay directly
def gen_sign(x, gate, p_neg=0.3, seed=7):
    shape = x[GATES[gate]].shape
    gen_ = torch.Generator(device=device).manual_seed(seed)
    return torch.where(torch.rand(shape, generator=gen_, device=device) < p_neg, -1, 1).to(torch.int8)


SIGNED_CASES = [
    pytest.param(B, T, H, K, V, rule, gate, dtype, id=f'B{B}-T{T}-H{H}-K{K}-V{V}-{rule}-{gate}-{dtype}')
    for B, T, H, K, V, rule, gate, dtype in [
        (2, 200, 2, 64, 64, 'E', 'vector', 'fp32'),
        (2, 200, 2, 64, 64, 'A', 'vector', 'fp32'),
        (2, 200, 2, 64, 64, 'E', 'scalar', 'fp32'),
        (1, 300, 2, 128, 128, 'E', 'vector', 'bf16'),
    ]
]


@pytest.mark.parametrize(('B', 'T', 'H', 'K', 'V', 'rule', 'gate', 'dtype'), SIGNED_CASES)
def test_chunk_signed_gate(B, T, H, K, V, rule, gate, dtype):
    x = gen(B, T, H, K, V)
    sign = gen_sign(x, gate)
    ref = run_grads(functools.partial(naive_recurrent_kronlstm, sign=sign), x, torch.float64, gate, rule)
    tri = run_grads(functools.partial(chunk_kronlstm, sign=sign), x, DTYPES[dtype], gate, rule)
    assert_all_close(ref, tri, TOL[DTYPES[dtype]])


def test_chunk_all_positive_sign_is_unsigned():
    x = gen(2, 200, 2, 64, 64)
    sign = torch.ones_like(x['gv'], dtype=torch.int8)
    g_s, o_s, S_s = run_grads(functools.partial(chunk_kronlstm, sign=sign), x, torch.bfloat16, 'vector', 'E')
    g_u, o_u, S_u = run_grads(chunk_kronlstm, x, torch.bfloat16, 'vector', 'E')
    assert torch.equal(o_s, o_u) and torch.equal(S_s, S_u)
    for name in g_u:
        assert torch.equal(g_s[name], g_u[name]), name


@pytest.mark.parametrize('dtype', ['fp32', 'bf16'])
def test_chunk_signed_gate_varlen(dtype):
    """The parity resets at every document: a packed signed run equals the signed oracle per document."""
    cu = [0, 15, 64, 64, 500, 513, 1000]
    x = gen_packed(cu, 2, 128, 128)
    sign = gen_sign(x, 'vector')
    dt = DTYPES[dtype]
    with torch.no_grad():
        o, S = chunk_kronlstm(*(x[n].to(dt) for n in ('q', 'k', 'v', 'gv', 'beta')), initial_state=x['S0'].float(),
                              output_final_state=True, allow_neg_eigval=True, target_gate=x['ct'].float(),
                              cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device), sign=sign)
    for i, (s, e) in enumerate(zip(cu[:-1], cu[1:])):
        if s == e:
            assert torch.equal(S[i:i + 1], x['S0'][i:i + 1].float())
            continue
        o_ref, S_ref = naive_recurrent_kronlstm(
            *(x[n][:, s:e] for n in ('q', 'k', 'v', 'gv', 'beta')), initial_state=x['S0'][i:i + 1],
            output_final_state=True, allow_neg_eigval=True, target_gate=x['ct'][:, s:e], sign=sign[:, s:e])
        assert_close(f'o[{i}]', o_ref, o[:, s:e], TOL[dt])
        assert_close(f'ht[{i}]', S_ref, S[i:i + 1], TOL[dt])


@pytest.mark.parametrize('gate', ['vector', 'scalar'])
def test_fused_recurrent_signed_gate(gate):
    x = gen(2, 100, 2, 64, 64)
    sign = gen_sign(x, gate)
    args = [x[n] for n in ('q', 'k', 'v', GATES[gate], 'beta')]
    kw = dict(initial_state=x['S0'], output_final_state=True, allow_neg_eigval=True, target_gate=x['ct'], sign=sign)
    o_ref, S_ref = naive_recurrent_kronlstm(*args, **kw)
    o, S = fused_recurrent_kronlstm(*(t.float() for t in args), **{**kw, 'initial_state': x['S0'].float(),
                                                                  'target_gate': x['ct'].float()})
    assert_close('o', o_ref, o, TOL[torch.float32])
    assert_close('ht', S_ref, S, TOL[torch.float32])


def test_layer_signed_gate_runs_and_packs():
    """The layer with a signed gate: a packed batch equals its documents run one by one."""
    from fla.layers import KronLSTM
    torch.manual_seed(0)
    layer = KronLSTM(hidden_size=256, k_head_dim=64, v_head_dim=64, num_k_heads=4, num_v_heads=4,
                     gate='signed_sigmoid2', allow_neg_eigval=True).to(device)
    # a spread of negative gates: flip half the channels' bias, as Complex KDA's 'spread' init does
    with torch.no_grad():
        layer.dt_bias.mul_(torch.where(torch.rand_like(layer.dt_bias) < 0.5, -1.0, 1.0))
    cu = [0, 37, 100, 230]
    h = torch.randn(1, cu[-1], 256, device=device)
    y_packed = layer(h, cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device))[0]
    y_sep = torch.cat([layer(h[:, s:e])[0] for s, e in zip(cu[:-1], cu[1:])], 1)
    assert_close('y', y_sep, y_packed, 1e-4)


def test_layer_signed_gate_init_is_complex_kdas():
    """The signed forget gate starts as Complex KDA's does: A = 1, f = +exp(-dt) (no channel saturated),
    and a live gradient into the gate's pre-activation on every channel."""
    from fla.layers import KronLSTM
    from fla.layers.complex_kda_layer import signed_gate
    torch.manual_seed(0)
    layer = KronLSTM(hidden_size=256, k_head_dim=64, v_head_dim=64, num_k_heads=4, num_v_heads=4,
                     gate='signed_sigmoid2', allow_neg_eigval=True)
    assert torch.equal(layer.A_log, torch.zeros_like(layer.A_log))
    z = torch.zeros(1, 1, 4, 64, requires_grad=True)
    sign, g = signed_gate(z, layer.A_log, layer.dt_bias, layer.lower_bound)
    assert (sign > 0).all()
    f = g.exp()
    assert f.min() > 0.85 and f.max() < 0.9995  # exp(-dt), dt in [1e-3, 1e-1]
    g.sum().backward()
    assert (z.grad.abs() > 1e-5).all()


def test_layer_neg_eigval_init_bias():
    """With the bias, s * beta (s = 2) starts at mean ~0.5 and mostly in [0, 1], as the s = 1 layer's beta does;
    the bias is a learnt, weight-decay-free parameter, on the erase gate only."""
    from fla.layers import KronLSTM
    torch.manual_seed(0)
    layer = KronLSTM(hidden_size=768, k_head_dim=128, v_head_dim=128, num_k_heads=6, num_v_heads=6,
                     gate='signed_sigmoid2', allow_neg_eigval=True, neg_eigval_init_bias=True)
    x = F.rms_norm(torch.randn(4096, 768), (768,))
    with torch.no_grad():
        eff = 2 * layer.b_proj(x).sigmoid()
    assert abs(eff.mean().item() - 0.5) < 0.05 and (eff <= 1).float().mean() > 0.95
    assert layer.b_proj.bias._no_weight_decay and layer.c_proj.bias is None
    plain = KronLSTM(hidden_size=768, k_head_dim=128, v_head_dim=128, num_k_heads=6, num_v_heads=6,
                     neg_eigval_init_bias=True)
    assert plain.b_proj.bias is None


def test_layer_unit_write():
    """unit_write: under s = 2 the op gets c / 2, so the write s * c / 2 = c stays in [0, 1] by construction;
    otherwise the layers are identical."""
    import fla.layers.kronlstm as mod
    from fla.layers import KronLSTM
    torch.manual_seed(0)
    kw = dict(hidden_size=128, k_head_dim=32, v_head_dim=32, num_k_heads=4, num_v_heads=4, allow_neg_eigval=True)
    unit = KronLSTM(unit_write=True, **kw).to(device)
    full = KronLSTM(**kw).to(device)
    full.load_state_dict(unit.state_dict())
    h = torch.randn(1, 96, 128, device=device)
    calls, orig = [], mod.chunk_kronlstm

    def spy(**a):
        calls.append(a['target_gate'].detach().clone())
        return orig(**a)
    mod.chunk_kronlstm = spy
    try:
        unit(h)
        full(h)
    finally:
        mod.chunk_kronlstm = orig
    assert torch.allclose(calls[0] * 2, calls[1]) and calls[0].max() <= 0.5
    # no effect without the extended range
    assert not KronLSTM(unit_write=True, **{**kw, 'allow_neg_eigval': False}).unit_write


def test_layer_lowrank_output_gate():
    """The low-rank output gate has Complex KDA's shape (hidden -> v_head_dim -> value_dim) and starts as the full
    one: sigmoid(3) for every token (CPU: the gate projections only)."""
    from fla.layers import KronLSTM
    from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
    kw = dict(hidden_size=256, k_head_dim=64, v_head_dim=64, num_k_heads=4, num_v_heads=4)
    torch.manual_seed(0)
    low = KronLSTM(output_gate_rank='lowrank', **kw)
    full = KronLSTM(**kw)
    ckda = ComplexKimiDeltaAttention(hidden_size=256, num_heads=4, head_dim=64, output_gate="lowrank")
    shapes = lambda m: [tuple(p.shape) for p in m.g_proj.parameters()]  # noqa: E731
    assert shapes(low) == shapes(ckda)
    h = torch.randn(1, 64, 256)
    with torch.no_grad():
        assert torch.equal(low.g_proj(h), full.g_proj(h))
        assert torch.allclose(low.g_proj(h), torch.full_like(low.g_proj(h), 3.0))
    with pytest.raises(ValueError):
        KronLSTM(output_gate_rank='half', **kw)


def _spy_op(layer, h):
    """The op's inputs from one layer forward."""
    import fla.layers.kronlstm as mod
    calls, orig = [], mod.chunk_kronlstm

    def spy(**a):
        calls.append({k: (v.detach().clone() if torch.is_tensor(v) else v) for k, v in a.items()})
        return orig(**a)
    mod.chunk_kronlstm = spy
    try:
        out = layer(h)
    finally:
        mod.chunk_kronlstm = orig
    return calls[0], out


@pytest.mark.parametrize('opts', [dict(forget_angle=True), dict(forget_angle=True, erase_angle=True),
                                  dict(forget_angle=True, erase_angle=True, signed_write=True)])
def test_layer_polar_gates(opts):
    """Polar gates: the op gets f = m cos(theta) (sign gauge + log|f|), beta* = (1 - r cos(phi)) / 2 under s = 2,
    the write c (signed: 2 sigmoid - 1, halved for s = 2); at init each starts at the paper layer's value."""
    from fla.layers import KronLSTM
    kw = dict(hidden_size=128, k_head_dim=32, v_head_dim=32, num_k_heads=4, num_v_heads=4)
    torch.manual_seed(0)
    polar = KronLSTM(**kw, **opts).to(device)
    torch.manual_seed(0)
    paper = KronLSTM(**kw).to(device)
    paper.load_state_dict({k: v for k, v in polar.state_dict().items() if k in paper.state_dict()}, strict=False)
    h = torch.randn(1, 64, 128, device=device)
    a, _ = _spy_op(polar, h)
    b, _ = _spy_op(paper, h)
    # forget: log|f| = log m + log cos(theta0), all positive at init
    assert torch.allclose(a['g'], b['g'] + math.log(math.cos(math.pi * 0.05)), atol=1e-5)
    assert (a['sign'] > 0).all()
    if opts.get('erase_angle'):
        assert a['allow_neg_eigval'] and polar.unit_write
        # beta* ||a||^2 = 2 * beta_op = 1 - sigmoid(-z) cos(theta0) ~ sigmoid(z)
        assert torch.allclose(2 * a['beta'].float(), 1 - (1 - b['beta'].float()) * math.cos(math.pi * 0.05), atol=1e-3)
        if opts.get('signed_write'):  # c = 2 sigmoid(z + ln 3) - 1, z the paper gate's pre-activation
            c = 2 * torch.sigmoid(torch.logit(b['target_gate'].double()) + math.log(3.0)).float() - 1
            assert abs(c.mean().item() - 0.5) < 0.1
        else:
            c = b['target_gate']
        assert torch.allclose(a['target_gate'], 0.5 * c, atol=1e-5)
    # the angles learn: a flipped channel (theta past pi/2) gives sign -1 and the right magnitude
    with torch.no_grad():
        polar.f_angle_proj[-1].bias[:5] = 4.0  # theta = pi * sigmoid(4) = 3.08: cos = -0.998
    a2, _ = _spy_op(polar, h)
    sg = a2['sign'].reshape(*a2['sign'].shape[:-2], -1)
    assert (sg[..., :5] < 0).all() and (sg[..., 5:] > 0).all()
    polar.zero_grad()
    polar(h)[0].sum().backward()
    assert polar.f_angle_proj[-1].weight.grad.abs().sum() > 0
    if opts.get('erase_angle'):
        assert polar.b_angle_proj.weight.grad.abs().sum() > 0


def test_chunk_negative_target_gate():
    """A signed write target c in [-1, 1] (signed_write) through the chunk kernels, against the naive rule."""
    x = gen(2, 200, 2, 64, 64)
    x['ct'] = 2 * x['ct'] - 1
    ref = run_grads(naive_recurrent_kronlstm, x, torch.float64, 'vector', 'E')
    tri = run_grads(chunk_kronlstm, x, torch.float32, 'vector', 'E')
    assert_all_close(ref, tri, TOL[torch.float32])


def test_layer_forget_offset():
    """forget_offset subtracts a constant from the op's log-gate (KronLSTM) -- and equals the polar forget gate's
    init (cos(0.157): offset -log cos(pi * 0.05)) without the angle."""
    from fla.layers import KronLSTM
    kw = dict(hidden_size=128, k_head_dim=32, v_head_dim=32, num_k_heads=4, num_v_heads=4)
    off = -math.log(math.cos(math.pi * 0.05))
    torch.manual_seed(0)
    a = KronLSTM(forget_offset=off, **kw).to(device)
    torch.manual_seed(0)
    polar = KronLSTM(forget_angle=True, **kw).to(device)
    torch.manual_seed(0)
    plain = KronLSTM(**kw).to(device)
    a.load_state_dict(plain.state_dict())
    polar.load_state_dict({**polar.state_dict(), **plain.state_dict()})
    h = torch.randn(1, 64, 128, device=device)
    ga, gp, g0 = _spy_op(a, h)[0]['g'], _spy_op(polar, h)[0]['g'], _spy_op(plain, h)[0]['g']
    assert torch.allclose(ga, g0 - off, atol=1e-6) and torch.allclose(ga, gp, atol=1e-5)


def test_ckda_forget_offset():
    """Complex KDA's forget_offset: the op's log|alpha| shifted by the offset (clamped at the lower bound)."""
    import fla.layers.complex_kda_layer as mod
    torch.manual_seed(0)
    kw = dict(hidden_size=128, num_heads=4, head_dim=32)
    a = mod.ComplexKimiDeltaAttention(forget_offset=0.012, **kw).to(device)
    torch.manual_seed(0)
    b = mod.ComplexKimiDeltaAttention(**kw).to(device)
    b.load_state_dict(a.state_dict())
    calls, orig = [], mod.chunk_kda

    def spy(**k):
        calls.append(k['g'].detach().clone())
        return orig(**k)
    mod.chunk_kda = spy
    try:
        h = torch.randn(1, 64, 128, device=device)
        a(h)
        b(h)
    finally:
        mod.chunk_kda = orig
    assert torch.allclose(calls[0], (calls[1] - 0.012).clamp_min(a.lower_bound), atol=1e-6)
    with pytest.raises(ValueError):
        mod.ComplexKimiDeltaAttention(forget_offset=-1.0, **kw)


def test_layer_polar_forget_broad_init():
    """forget_angle_init='broad': the angle bias spread per channel, so f starts with both signs and
    a spread of |cos(theta)| (the magnitude gate unchanged); 'narrow' keeps theta0 = 0.157 everywhere."""
    from fla.layers import KronLSTM
    kw = dict(hidden_size=128, k_head_dim=32, v_head_dim=32, num_k_heads=4, num_v_heads=4, forget_angle=True)
    torch.manual_seed(0)
    broad = KronLSTM(**kw, forget_angle_init='broad').to(device)
    torch.manual_seed(0)
    narrow = KronLSTM(**kw).to(device)
    bias = broad.f_angle_proj[-1].bias
    cos = torch.cos(math.pi * torch.sigmoid(bias.float()))
    assert cos.min() < -0.5 and cos.max() > 0.5 and cos.abs().max() <= math.cos(math.pi * 0.05) + 1e-5
    assert torch.allclose(narrow.f_angle_proj[-1].bias, torch.full_like(bias, math.log(0.05 / 0.95)))
    h = torch.randn(1, 64, 128, device=device)
    a, _ = _spy_op(broad, h)
    assert (a['sign'] < 0).any() and (a['sign'] > 0).any()
    with pytest.raises(ValueError):
        KronLSTM(**kw, forget_angle_init='wide')
