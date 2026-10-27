# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Layers on a GPU that is not the current device (a model spread over GPUs, e.g. HF device_map)
must give the same outputs and gradients as on the current device. A Triton kernel launched
without input_guard runs on the current device: an error, or with peer access on, reads of the
other GPU's memory (wrong or NaN outputs that depend on what ran before)."""

import pytest
import torch

from fla.layers.complex_kda_layer import ComplexKimiDeltaAttention
from fla.layers.kronlstm import KronLSTM

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")

# the ambitious main run's CKDA layer at a smaller width; signed_sigmoid2 runs the signed gate kernel
LAYERS = {
    "complex_kda": lambda: ComplexKimiDeltaAttention(
        hidden_size=256, num_heads=2, head_dim=128, allow_neg_eigval=True, gate="signed_sigmoid2",
        gate_init_style="shipped", lower_bound=-5.0, output_gate="lowrank", drop_silu=True, conv_size=4),
    "kronlstm": lambda: KronLSTM(hidden_size=256, k_head_dim=64, v_head_dim=64, num_k_heads=4, num_v_heads=4),
}


def _run(layer, x, cu_seqlens):
    x = x.detach().requires_grad_()
    kw = {"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}
    out = layer(hidden_states=x, **kw)[0]
    out.float().pow(2).mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in layer.named_parameters() if p.grad is not None}
    return out.detach(), x.grad.detach(), grads


@pytest.mark.parametrize("name", list(LAYERS))
@pytest.mark.parametrize("varlen", [False, True])
def test_layer_off_current_device(name, varlen):
    torch.manual_seed(0)
    ref_layer = LAYERS[name]().to("cuda:0", torch.bfloat16)
    off_layer = LAYERS[name]().to("cuda:1", torch.bfloat16)
    off_layer.load_state_dict(ref_layer.state_dict())
    T = 1536  # several chunks and > 1024 (the HF-export NaN began at position 1024)
    x = torch.randn(1, T, 256, dtype=torch.bfloat16, device="cuda:0")
    cu0 = torch.tensor([0, 700, T], dtype=torch.int32, device="cuda:0") if varlen else None
    torch.cuda.set_device(0)
    ref = _run(ref_layer, x, cu0)
    off = _run(off_layer, x.to("cuda:1"), cu0.to("cuda:1") if varlen else None)
    assert torch.cuda.current_device() == 0
    torch.testing.assert_close(off[0].cpu(), ref[0].cpu(), atol=0, rtol=0)
    # gradients: bf16 tolerances (atomic accumulations in the backward are not bit-deterministic);
    # a kernel on the wrong device is far outside them
    torch.testing.assert_close(off[1].cpu().float(), ref[1].cpu().float(), atol=1e-3, rtol=2e-2)
    assert off[2].keys() == ref[2].keys()
    for n in ref[2]:
        torch.testing.assert_close(off[2][n].cpu().float(), ref[2][n].cpu().float(), atol=1e-3, rtol=2e-2,
                                   msg=lambda m, n=n: f"{n}: {m}")
