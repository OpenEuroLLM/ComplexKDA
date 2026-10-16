# -*- coding: utf-8 -*-
"""Context parallel through the KDA / Complex KDA *layers* (``cp_context`` in the layer kwargs).

The full token stream (several sequences, flattened to B = 1 with cu_seqlens) runs through the
layer once without CP; then every rank runs its contiguous slice with the CP context. Outputs,
input gradients and (all-reduced) parameter gradients must match. Complex KDA exercises the
sign gauge, whose running product must carry across ranks (fla.ops.cp.sign).
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fla.ops.cp import build_cp_context


def _worker(rank, world_size, layer_name, lengths, port):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
        from fla.layers.kda import KimiDeltaAttention

        torch.manual_seed(0)
        H, D = 4, 64
        cls = ComplexKimiDeltaAttention if layer_name == "complex_kda" else KimiDeltaAttention
        kwargs = dict(hidden_size=H * D, head_dim=D, num_heads=H, mode="chunk")
        if layer_name == "complex_kda":
            kwargs["allow_neg_eigval"] = True
        layer = cls(**kwargs).cuda()
        if layer_name == "complex_kda":
            with torch.no_grad():  # make the sign gauge matter: a mix of negative decays
                layer.dt_bias.copy_(torch.randn_like(layer.dt_bias))
        T = sum(lengths)
        assert T % world_size == 0
        cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), device="cuda")
        x = torch.randn(1, T, H * D, device="cuda")
        dy = torch.randn(1, T, H * D, device="cuda")

        # reference: the whole stream, no CP
        xr = x.clone().requires_grad_(True)
        out_ref = layer(xr, cu_seqlens=cu)[0]
        out_ref.backward(dy)
        ref_grads = {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}
        layer.zero_grad()

        # CP: this rank's contiguous slice of the stream
        n = T // world_size
        sl = slice(rank * n, (rank + 1) * n)
        ctx = build_cp_context(cu, group=dist.group.WORLD, conv1d_kernel_size=layer.conv_size)
        xl = x[:, sl].clone().requires_grad_(True)
        out = layer(xl, cp_context=ctx)[0]
        out.backward(dy[:, sl])
        for p in layer.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)

        torch.testing.assert_close(out, out_ref[:, sl], rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(xl.grad, xr.grad[:, sl], rtol=5e-3, atol=5e-3)
        for name, p in layer.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, ref_grads[name], rtol=1e-2, atol=1e-2,
                                           msg=lambda m, name=name: f"{name}: {m}")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("layer_name", ["kda", "complex_kda"])
@pytest.mark.parametrize("world_size,lengths", [
    (2, [300, 500, 224]),           # a sequence cut at the rank boundary
    (4, [1800, 100, 148]),          # the first sequence spans three ranks
])
def test_cp_layer_matches_full_stream(layer_name, world_size, lengths):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} GPUs required")
    port = 29600 + world_size + (10 if layer_name == "complex_kda" else 0)
    mp.start_processes(_worker, args=(world_size, layer_name, lengths, port), nprocs=world_size,
                       join=True, start_method="spawn")
