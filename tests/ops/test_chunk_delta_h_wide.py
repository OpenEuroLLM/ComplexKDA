# Wide-key (K > 256) state recurrence: against the register-tile kernels where both
# apply, and chunk_kda beyond K = 256 against the naive recurrence.

import pytest
import torch
import torch.nn.functional as F

from fla.ops.common import chunk_delta_h as base
from fla.ops.common.chunk_delta_h_wide import wide_bwd_dhu, wide_fwd_h
from fla.ops.kda import chunk_kda
from fla.ops.kda.naive import naive_recurrent_kda
from fla.utils import device

pytestmark = pytest.mark.skipif(device != "cuda", reason="needs CUDA")


def _cumsum_chunks(step, chunk=64):
    B, T, H, K = step.shape
    pad = (-T) % chunk
    s = F.pad(step, (0, 0, 0, 0, 0, pad)).view(B, -1, chunk, H, K).cumsum(2)
    return s.view(B, T + pad, H, K)[:, :T].contiguous()


@pytest.mark.parametrize("K", [128, 256])
@pytest.mark.parametrize("use_gk", [True, False])
def test_wide_kernels_match_register_kernels(K, use_gk):
    torch.manual_seed(0)
    B, T, H, V = 2, 200, 2, 64
    k = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).bfloat16()
    q = F.normalize(torch.randn(B, T, H, K, device=device), dim=-1).bfloat16()
    w = (torch.randn(B, T, H, K, device=device) * 0.1).bfloat16()
    u = torch.randn(B, T, H, V, device=device).bfloat16()
    gk = _cumsum_chunks(-F.softplus(torch.randn(B, T, H, K, device=device)) * 0.05) if use_gk else None
    h0 = torch.randn(B, H, K, V, device=device) * 0.1
    ref = base.chunk_gated_delta_rule_fwd_h.__wrapped__(
        k=k, w=w, u=u, gk=gk, initial_state=h0, output_final_state=True)
    out = wide_fwd_h(k, w, u, gk=gk, initial_state=h0, output_final_state=True)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)
    do = torch.randn(B, T, H, V, device=device).bfloat16()
    dv = (torch.randn(B, T, H, V, device=device) * 0.1).bfloat16()
    dht = torch.randn(B, H, K, V, device=device) * 0.1
    ref = base.chunk_gated_delta_rule_bwd_dhu.__wrapped__(
        q=q, k=k, w=w, do=do, dv=dv, gk=gk, h0=h0, dht=dht, scale=K**-0.5)
    out = wide_bwd_dhu(q, k, w, do, dv, gk=gk, h0=h0, dht=dht, scale=K**-0.5)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("K", [384, 512])
def test_chunk_kda_wide_keys_match_naive(K):
    torch.manual_seed(1)
    B, T, H, V = 1, 128, 2, 64
    q = torch.randn(B, T, H, K, device=device)
    k = torch.randn(B, T, H, K, device=device)
    v = torch.randn(B, T, H, V, device=device)
    g = -F.softplus(torch.randn(B, T, H, K, device=device)) * 0.2
    beta = torch.rand(B, T, H, device=device)
    dy = torch.randn(B, T, H, V, device=device)
    grads = []
    for path in ("chunk", "naive"):
        xs = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
        qq, kk = F.normalize(xs[0], dim=-1), F.normalize(xs[1], dim=-1)
        if path == "chunk":
            o, _ = chunk_kda(qq.bfloat16(), kk.bfloat16(), xs[2].bfloat16(), xs[3], xs[4],
                             scale=K**-0.5, state_v_first=False)
        else:
            o, _ = naive_recurrent_kda(qq, kk, xs[2], xs[3], xs[4], scale=K**-0.5)
        o.float().backward(dy)
        grads.append([o.detach().float()] + [x.grad for x in xs])
    for a, b in zip(*grads):
        assert (a.float() - b.float()).norm() <= 3e-2 * b.float().norm()


def test_wide_keys_need_the_k_major_layout():
    q = torch.randn(1, 64, 1, 512, device=device).bfloat16()
    with pytest.raises(ValueError, match="state_v_first"):
        chunk_kda(q, q, q[..., :64], torch.zeros(1, 64, 1, 512, device=device),
                  torch.rand(1, 64, 1, device=device), state_v_first=True)
