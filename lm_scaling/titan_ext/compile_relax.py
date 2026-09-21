"""Let torchtitan compile a block that contains an ``fla`` mixer.

torchtitan compiles each TransformerBlock with ``fullgraph=True``
(``torchtitan/models/llama3/infra/parallelize.py:252``, and the qwen3_custom
path follows it). An fla block cannot satisfy that: fla marks three hot
entry points with ``@torch.compiler.disable`` --

    fla/modules/conv/triton/ops.py   causal_conv1d_fwd
    fla/ops/kda/chunk.py:203
    fla/ops/kda/gate.py:332

-- and a *disabled* call inside a fullgraph region is a hard error
(``torch._dynamo.exc.Unsupported: Skip calling torch.compiler.disable()'d
function``), not a graph break. So the linear arms were running with the model
in eager while the dense baseline got the full treatment, which is worth 35% at
302M and 21% at 1.7B. That is not a fair comparison between arms, and it leaves
the largest measured lever on the table.

Relaxing to ``fullgraph=False`` turns each disabled call back into an ordinary
graph break. The Triton kernels stay in eager -- they are hand-written and were
never going to be compiled -- while the MLP, the norms, the residuals and the
projection epilogues around them still fuse. That is exactly the part that made
attention faster.

Why patch ``torch.compile`` rather than reimplement the parallelize function:
torchtitan's ordering is load-bearing (activation checkpointing, then compile,
then ``fully_shard``), and a copy of those ~200 lines would drift from
titan-oellm's. The patch is scoped to a single synchronous call during model
setup and restores the original in a ``finally``.

The alternative real fix is to re-register fla's three ops as
``torch.library.custom_op`` with meta kernels, which would make them opaque but
traceable and let fullgraph succeed. That belongs in fla, not here.
"""

from __future__ import annotations

import contextlib
import functools

import torch


@contextlib.contextmanager
def relaxed_fullgraph():
    """Force ``fullgraph=False`` on every ``torch.compile`` call made inside."""
    original = torch.compile

    @functools.wraps(original)
    def patched(*args, **kwargs):
        kwargs["fullgraph"] = False
        return original(*args, **kwargs)

    torch.compile = patched
    try:
        yield
    finally:
        torch.compile = original


def relaxing(parallelize_fn):
    """Wrap a train spec's ``parallelize_fn`` so its block compile can break."""

    @functools.wraps(parallelize_fn)
    def wrapper(*args, **kwargs):
        with relaxed_fullgraph():
            return parallelize_fn(*args, **kwargs)

    return wrapper
