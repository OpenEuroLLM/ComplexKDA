# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Context parallelism for KronLSTM (`fla/ops/kronlstm/cp.py`): the token stream split into
`world` contiguous pieces, the state relayed around the scans, against one process over the whole
stream (whose kernels `tests/ops/test_kronlstm.py` checks against the naive recurrence): outputs and
every input gradient, unsigned and with the signed forget gate, documents cut by rank boundaries,
aligned with them, and one document over every rank.

NCCL with one GPU per rank when there are enough GPUs, else gloo with every rank on GPU 0."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from fla.ops.cp import build_cp_context
from fla.ops.kronlstm import chunk_kronlstm

H, K, V = 2, 64, 64


def _free_port():
    with socket.socket() as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def _data(lengths, signed, dtype, device):
    torch.manual_seed(0)
    T = sum(lengths)
    d = dict(device=device, dtype=torch.float32)
    x = dict(
        q=F.normalize(torch.randn(1, T, H, K, **d), dim=-1).to(dtype),
        k=F.normalize(torch.randn(1, T, H, K, **d), dim=-1).to(dtype),
        v=F.normalize(torch.randn(1, T, H, V, **d), dim=-1).to(dtype),
        g=-(torch.rand(1, T, H, K, **d) * 0.9 + 0.1) * 0.3,
        beta=(torch.rand(1, T, H, **d) * 0.9 + 0.05).to(dtype),
        ct=torch.sigmoid(torch.randn(1, T, H, **d) * 3.0),
        do=torch.randn(1, T, H, V, **d).to(dtype),
    )
    x['sign'] = torch.where(torch.rand(1, T, H, K, device=device) < 0.3, -1, 1).to(torch.int8) if signed else None
    return x


def _run(x, sl, **kw):
    names = ('q', 'k', 'v', 'g', 'beta', 'ct')
    leaves = {n: x[n][:, sl].detach().clone().requires_grad_(True) for n in names}
    o, _ = chunk_kronlstm(leaves['q'], leaves['k'], leaves['v'], leaves['g'], leaves['beta'],
                          allow_neg_eigval=True, target_gate=leaves['ct'],
                          sign=x['sign'][:, sl] if x['sign'] is not None else None, **kw)
    (o.float() * x['do'][:, sl].float()).sum().backward()
    return o.detach(), {n: t.grad for n, t in leaves.items()}


def _worker(rank, world, port, lengths, signed, dtype, tol, errors):
    try:
        os.environ.update(MASTER_ADDR='localhost', MASTER_PORT=str(port))
        multi_gpu = torch.cuda.device_count() >= world
        dist.init_process_group('nccl' if multi_gpu else 'gloo', rank=rank, world_size=world)
        device = torch.device('cuda', rank if multi_gpu else 0)
        torch.cuda.set_device(device)
        x = _data(lengths, signed, dtype, device)
        T = sum(lengths)
        cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long, device=device)
        o_ref, g_ref = _run(x, slice(None), cu_seqlens=cu)
        ctx = build_cp_context(cu, group=dist.group.WORLD)
        n = T // world
        sl = slice(rank * n, (rank + 1) * n)
        o, g = _run(x, sl, cp_context=ctx)

        def rel(a, b):
            return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()

        errs = {'o': rel(o, o_ref[:, sl]), **{f'd{k}': rel(g[k], g_ref[k][:, sl]) for k in g}}
        bad = {k: e for k, e in errs.items() if not e < tol}
        if bad:
            errors.put(f'rank {rank}: {bad}')
    except Exception as e:  # noqa: BLE001
        errors.put(f'rank {rank}: {type(e).__name__}: {e}')
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


CASES = [
    pytest.param(2, [300, 212], id='cp2-cut'),
    pytest.param(2, [256, 256], id='cp2-aligned'),
    pytest.param(4, [100, 600, 50, 274], id='cp4-mixed'),
    pytest.param(4, [1024], id='cp4-one-doc'),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@pytest.mark.parametrize('signed', [False, True], ids=['unsigned', 'signed'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
@pytest.mark.parametrize(('world', 'lengths'), CASES)
def test_cp_kronlstm(world, lengths, dtype, signed):
    ctx = mp.get_context('spawn')
    errors = ctx.Queue()
    tol = 2e-3 if dtype == torch.float32 else 3e-2
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, lengths, signed, dtype, tol, errors))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=1800)
    msgs = []
    while not errors.empty():
        msgs.append(errors.get())
    assert not msgs, '\n'.join(msgs)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
