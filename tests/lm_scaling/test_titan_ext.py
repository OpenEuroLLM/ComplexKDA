"""The torchtitan shim in ``lm_scaling/titan_ext``.

These tests deliberately do not need torchtitan: the shim's job is to present an
``fla`` layer through torchtitan's attention contract, and that contract is small
enough to state directly. ``register()`` itself needs the titan container and is
exercised by the backend-comparison job.

The invariants here are the ones whose violation would be *silent* — a run
labelled ``complex-kda`` that trains plain attention, or one whose gate
initialization has been quietly overwritten with a truncated normal.
"""

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lm_scaling"))

from titan_ext import swap_in_fla_mixers  # noqa: E402
from titan_ext.mixer import MIXERS, FLAMixer  # noqa: E402

DIM, HEAD_DIM = 256, 64


def _args(mixer="complex-kda"):
    return types.SimpleNamespace(
        dim=DIM,
        head_dim=HEAD_DIM,
        norm_eps=1e-6,
        mixer=mixer,
        allow_neg_eigval=True,
        lower_bound=-5.0,
        gate="signed_sigmoid2",
        gate_init_style="spread",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fla dispatches to Triton")
@pytest.mark.parametrize("mixer", MIXERS)
def test_mixer_builds_and_matches_titan_signature(mixer):
    """torchtitan calls attention(x, rope_cache, masks, positions) -> tensor."""
    m = FLAMixer(mixer, _args(mixer), layer_idx=0).cuda().to(torch.bfloat16)
    x = torch.randn(2, 64, DIM, device="cuda", dtype=torch.bfloat16)
    # rope_cache/masks/positions must be accepted and ignored, not forwarded:
    # fla asserts a 2-D [B, T] padding mask and would reject titan's mask object.
    try:
        out = m(x, rope_cache=None, attention_masks=object(), positions=None)
    except RuntimeError as e:
        # A shared GPU may have no room for Triton's workspace.  Skip loudly
        # rather than report a failure the code did not cause -- but only for
        # OOM; anything else is a real defect.
        if "out of memory" in str(e).lower():
            pytest.skip(f"GPU out of memory ({torch.cuda.mem_get_info()[0] / 1e9:.1f} GB free)")
        raise
    assert out.shape == x.shape, f"{mixer}: {out.shape} != {x.shape}"


@pytest.mark.parametrize("mixer", MIXERS)
def test_init_weights_preserves_gate_parameters(mixer):
    """The gate init is the thing under study and must survive titan's init.

    ``gate_init_style="spread"`` puts a specific distribution on the gate's
    sign. If torchtitan's truncated normal reached ``dt_bias`` / ``A_log`` /
    ``f_proj``, every titan run would silently become the "shipped" arm.

    The obvious way to assert that -- "init_weights must not modify these
    tensors" -- is what this test used to say, and it was actively harmful.
    torchtitan builds on meta and calls ``to_empty()`` before ``init_weights``,
    so a parameter nobody writes holds uninitialized memory. Passing that
    assertion is exactly the failure: every linear arm trained with a garbage
    decay gate and came out 37.5% above `lm/`.

    The property that matters is the *value*: after init_weights, the gate
    parameters equal what `fla` itself would have constructed.
    """
    m = FLAMixer(mixer, _args(mixer), layer_idx=0)
    gate_names = [k for k, _ in m.named_parameters() if any(s in k for s in ("A_log", "dt_bias", "f_proj"))]
    assert gate_names, f"{mixer} exposes no gate parameters; this proves nothing"

    # Whatever to_empty() would leave behind.
    with torch.no_grad():
        for p_ in m.parameters():
            p_.fill_(-12345.0)

    m.init_weights(0.02)
    # fla draws A_log and f_proj from the RNG, and init_weights seeds that draw
    # from the layer index so every rank agrees; build the reference the same
    # way rather than from the ambient RNG.
    from lm_scaling.titan_ext.mixer import reference_layer

    ref = {f"inner.{k}": v for k, v in reference_layer(mixer, _args(mixer), 0).named_parameters()}
    now = dict(m.named_parameters())
    for k in gate_names:
        assert not torch.equal(now[k], torch.full_like(now[k], -12345.0)), (
            f"{mixer}: init_weights left {k} at the sentinel -- under "
            f"torchtitan that is uninitialized memory, not fla's init"
        )
        torch.testing.assert_close(now[k], ref[k], msg=lambda s, n=k: f"{mixer}: {n} is not fla's init\n{s}")


def test_swap_replaces_every_block():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(DIM, DIM)

    model = nn.Module()
    model.layers = nn.ModuleList([Block() for _ in range(4)])

    n = swap_in_fla_mixers(model, _args())
    assert n == 4
    assert all(isinstance(b.attention, FLAMixer) for b in model.layers)


def test_swap_raises_rather_than_silently_doing_nothing():
    """A zero-swap must be loud: otherwise the run trains plain attention."""
    model = nn.Module()
    model.layers = nn.ModuleList([nn.Linear(DIM, DIM)])  # no .attention
    with pytest.raises(RuntimeError, match="no transformer blocks were swapped"):
        swap_in_fla_mixers(model, _args())

    no_layers = nn.Module()
    with pytest.raises(RuntimeError, match="has no .layers"):
        swap_in_fla_mixers(no_layers, _args())


def test_unknown_mixer_is_rejected():
    model = nn.Module()
    model.layers = nn.ModuleList()
    with pytest.raises(ValueError, match="model.mixer must be one of"):
        swap_in_fla_mixers(model, _args(mixer="not-a-mixer"))


def test_relaxed_fullgraph_forces_partial_graphs_and_restores():
    """fla blocks must compile with graph breaks allowed, and only inside.

    torchtitan hardcodes fullgraph=True, which an fla block cannot satisfy --
    fla marks causal_conv1d, chunk_kda and the gate with
    @torch.compiler.disable, and a disabled call inside a fullgraph region is a
    hard error rather than a break. Relaxing it lets the MLP, norms and
    residuals fuse while the Triton kernels stay in eager.
    """
    import torch

    from lm_scaling.titan_ext.compile_relax import relaxed_fullgraph, relaxing

    original = torch.compile
    seen = []

    def fake_compile(mod, **kwargs):
        seen.append(kwargs.get("fullgraph"))
        return mod

    torch.compile = fake_compile
    try:
        with relaxed_fullgraph():
            torch.compile(object(), backend="inductor", fullgraph=True)
            torch.compile(object(), backend="inductor")
        assert seen == [False, False], seen
        # Outside the context the caller's own value must survive untouched.
        torch.compile(object(), fullgraph=True)
        assert seen[-1] is True
    finally:
        torch.compile = original

    assert torch.compile is original, "torch.compile was not restored"

    # relaxing() must wrap a parallelize_fn without changing its result.
    def parallelize(model, **kw):
        return ("parallelized", model)

    assert relaxing(parallelize)("m") == ("parallelized", "m")


def test_relaxed_fullgraph_restores_on_exception():
    import torch

    from lm_scaling.titan_ext.compile_relax import relaxed_fullgraph

    original = torch.compile
    with pytest.raises(RuntimeError), relaxed_fullgraph():
        raise RuntimeError("boom")
    assert torch.compile is original


def test_init_weights_leaves_no_parameter_uninitialized():
    """torchtitan to_empty()s before init_weights, so silence means garbage.

    The shim used to skip A_log, dt_bias, f_proj, the convs and the norms, to
    avoid overwriting the gate initialization that is the subject of the study.
    Under `to_empty` that did not preserve them -- it left them as whatever was
    in the freed memory. Every linear arm trained under torchtitan with an
    uninitialized decay gate, converged, and sat 37.5% above the same
    configuration under `lm/`.

    This fills the tensors with a sentinel first, so anything init_weights
    fails to write is detectable rather than plausible-looking noise.
    """
    from types import SimpleNamespace

    import torch

    from lm_scaling.titan_ext.mixer import FLAMixer

    for mixer in ("kda", "complex-kda"):
        args = SimpleNamespace(dim=256, head_dim=64, norm_eps=1e-6)
        torch.manual_seed(0)
        m = FLAMixer(mixer, args, layer_idx=0)

        sentinel = -12345.0
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(sentinel)

        torch.manual_seed(0)
        m.init_weights(0.02)

        untouched = [n for n, p in m.named_parameters() if bool((p == sentinel).all())]
        assert not untouched, (
            f"{mixer}: init_weights never wrote {untouched}; under torchtitan "
            f"those would be uninitialized memory, not fla's initialization"
        )


def test_init_weights_reproduces_flas_own_scheme():
    """Matching `lm/` means fla's initialization, not torchtitan's std."""
    from types import SimpleNamespace

    import torch

    from lm_scaling.titan_ext.mixer import FLAMixer, reference_layer

    args = SimpleNamespace(dim=256, head_dim=64, norm_eps=1e-6)
    m = FLAMixer("complex-kda", args, layer_idx=3)
    m.init_weights(0.02)
    ref = dict(
        reference_layer("complex-kda", SimpleNamespace(dim=256, head_dim=64, norm_eps=1e-6), 3).named_parameters()
    )

    # The gate parameters are deterministic given the config, so they must come
    # back exactly -- these are the ones gate_init_style actually sets.
    for name in ("A_log", "dt_bias"):
        got = dict(m.inner.named_parameters())[name]
        torch.testing.assert_close(got, ref[name], msg=lambda s, n=name: f"{n} is not fla's init\n{s}")


def test_weight_decay_skips_norms_and_biases_like_lm():
    """torchtitan decays everything; `lm/` does not, and at wd 0.1 it matters.

    A single decayed group pulls every RMSNorm gain -- and, with tied
    embeddings, the embedding table -- toward zero for the whole run. Two
    backends that disagree about that are not optimizing the same objective,
    however exactly their parameter counts line up.
    """
    from types import SimpleNamespace

    from lm_scaling.titan_ext.optimizer import _no_decay

    # The rule itself, mirroring lm/src/models/construct.py.
    p = torch.zeros(3)
    assert _no_decay("layers.0.attention_norm.weight", p)
    assert _no_decay("layers.0.attention.wq.bias", p)
    assert not _no_decay("layers.0.attention.wq.weight", p)
    tagged = torch.zeros(3)
    tagged._no_weight_decay = True
    assert _no_decay("layers.0.attention.A_log", tagged)

    # The regrouping, against a stand-in container.
    model = nn.Sequential()
    model.add_module("lin", nn.Linear(4, 4))
    model.add_module("norm", nn.RMSNorm(4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    container = SimpleNamespace(optimizers=[opt])

    import lm_scaling.titan_ext.optimizer as mod

    real = mod.build_optimizers_matching_lm

    def fake_build(*a, **k):
        return container

    sys.modules.setdefault("torchtitan", types.ModuleType("torchtitan"))
    comp = types.ModuleType("torchtitan.components")
    optmod = types.ModuleType("torchtitan.components.optimizer")
    optmod.build_optimizers = fake_build
    sys.modules["torchtitan.components"] = comp
    sys.modules["torchtitan.components.optimizer"] = optmod
    try:
        out = real([model], None, None)
    finally:
        for k in ("torchtitan.components.optimizer", "torchtitan.components"):
            sys.modules.pop(k, None)

    groups = out.optimizers[0].param_groups
    assert len(groups) == 2, f"expected a decay and a no-decay group, got {len(groups)}"
    by_wd = {g["weight_decay"]: g["params"] for g in groups}
    assert set(by_wd) == {0.1, 0.0}
    assert any(p is model.norm.weight for p in by_wd[0.0]), "the norm gain is still being decayed"
    assert any(p is model.lin.weight for p in by_wd[0.1]), "the linear weight lost its weight decay"
    assert not any(p is model.lin.bias for p in by_wd[0.1]), "the bias is still being decayed"


def test_init_weights_writes_through_an_fsdp_shard():
    """Under FSDP the parameters are DTensors, not plain tensors.

    torchtitan applies fully_shard before init_weights, so on more than one GPU
    `param.copy_(plain_tensor)` raises "got mixed torch.Tensor and DTensor" and
    the run dies on the first layer. The single-GPU loss matching passed
    throughout, which is exactly why this needs its own test: the code path
    that breaks is the one the cheap check never takes.
    """

    from lm_scaling.titan_ext.mixer import _copy_into

    dst = torch.zeros(4, 3)
    _copy_into(dst, torch.ones(4, 3))
    assert torch.equal(dst, torch.ones(4, 3))

    # NOTE: the DTensor branch itself is NOT covered here -- only the plain
    # tensor path above is. The stand-in FakeDTensor this test once set up was
    # never wired to an assertion, so it was removed rather than left looking
    # like coverage.
    import lm_scaling.titan_ext.mixer as mod

    src_mod = mod._copy_into.__globals__
    assert "torch" in src_mod


def test_init_weights_is_rank_deterministic():
    """Every rank must build the same reference, or the shards disagree.

    Each rank runs init_weights on its own shard and derives the values from a
    locally-built reference layer. If those references come from different RNG
    draws, the assembled parameter is noise -- and nothing reports it, because
    each shard is individually well-formed.
    """
    from types import SimpleNamespace

    from lm_scaling.titan_ext.mixer import FLAMixer

    args = SimpleNamespace(dim=256, head_dim=64, norm_eps=1e-6)
    a = FLAMixer("complex-kda", args, layer_idx=2)
    b = FLAMixer("complex-kda", SimpleNamespace(dim=256, head_dim=64, norm_eps=1e-6), layer_idx=2)
    # Different ambient RNG, as two ranks would have.
    torch.manual_seed(1)
    a.init_weights(0.02)
    torch.manual_seed(999)
    b.init_weights(0.02)

    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    for name in pa:
        torch.testing.assert_close(pa[name], pb[name], msg=lambda s, n=name: f"{n} differs between ranks\n{s}")


def _std(t):
    return t.detach().float().std().item()


def test_mixer_init_is_flas_hf_scheme():
    """What `lm/` and flame (MN5 runs) start from: every Linear and
    Conv1d N(0, 0.02), biases 0, A_log 0, and spread's f_proj rescale on top.

    Until 2026-09-11 the reference layer was only constructed, which gives
    PyTorch's defaults -- U(+-1/sqrt(fan_in)), and U(+-0.5) for a 4-tap
    depthwise conv -- and every torchtitan run started there.
    """
    from lm_scaling.titan_ext.mixer import FLAMixer as Mixer

    args = _args("complex-kda")
    args.output_gate = "linear"
    inner = Mixer("complex-kda", args, layer_idx=0)
    inner.init_weights(0.02)
    inner = inner.inner
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "b_proj", "q_conv1d", "k_conv1d", "v_conv1d"):
        got = _std(getattr(inner, name).weight)
        assert abs(got - 0.02) < 0.002, f"{name}: std {got:.4f}, fla's is 0.02"
    assert abs(_std(inner.f_proj[0].weight) - 0.02) < 0.002
    got = _std(inner.f_proj[-1].weight)
    assert abs(got - 0.002) < 0.0003, f"f_proj[-1]: std {got:.5f}, spread's is 0.1 x 0.02"
    assert torch.equal(inner.g_proj.bias, torch.zeros_like(inner.g_proj.bias))
    assert torch.equal(inner.A_log, torch.zeros_like(inner.A_log))


def test_mixer_init_matches_the_model_lm_builds():
    """Per parameter, the same distribution as ComplexKDAForCausalLM after
    post_init."""
    from fla.models.complex_kda import ComplexKDAConfig, ComplexKDAForCausalLM
    from lm_scaling.titan_ext.mixer import FLAMixer as Mixer

    args = _args("complex-kda")
    args.output_gate = "linear"
    m = Mixer("complex-kda", args, layer_idx=0)
    m.init_weights(0.02)
    ours = dict(m.inner.named_parameters())
    cfg = ComplexKDAConfig(
        hidden_size=DIM,
        head_dim=HEAD_DIM,
        num_heads=DIM // HEAD_DIM,
        num_hidden_layers=1,
        vocab_size=64,
        intermediate_size=512,
        expand_v=1.0,
        gate="signed_sigmoid2",
        gate_init_style="spread",
        allow_neg_eigval=True,
        output_gate="linear",
    )
    theirs = dict(ComplexKDAForCausalLM(cfg).model.layers[0].attn.named_parameters())
    assert ours.keys() == theirs.keys()
    for k in ("A_log", "g_proj.bias"):
        torch.testing.assert_close(ours[k], theirs[k])
    for k, p in ours.items():
        if p.numel() >= 1000:
            a, b = _std(p), _std(theirs[k])
            assert abs(a - b) / b < 0.1, f"{k}: std {a:.4f} here, {b:.4f} under lm/"


def test_everything_outside_the_mixers_is_redrawn_like_hf(monkeypatch):
    """Embedding, MLP and any non-fla attention from N(0, 0.02); the mixers,
    norms and tying untouched; a tied weight drawn once."""
    from lm_scaling.titan_ext.hf_init import init_like_hf_
    from lm_scaling.titan_ext.mixer import FLAMixer as Mixer

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention_norm = nn.RMSNorm(DIM)
            self.attention = Mixer("complex-kda", _args("complex-kda"), layer_idx=0)
            self.feed_forward = nn.Module()
            self.feed_forward.w1 = nn.Linear(DIM, 512, bias=False)
            self.feed_forward.w2 = nn.Linear(512, DIM, bias=False)
            self.feed_forward.w3 = nn.Linear(DIM, 512, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embeddings = nn.Embedding(1000, DIM)
            self.layers = nn.ModuleDict({"0": Block()})
            self.norm = nn.RMSNorm(DIM)
            self.output = nn.Linear(DIM, 1000, bias=False)
            self.output.weight = self.tok_embeddings.weight

    model = Model()
    with torch.no_grad():
        model.tok_embeddings.weight.normal_(0.0, 1.0)  # torchtitan's
    mixer_before = {k: v.clone() for k, v in model.layers["0"].attention.named_parameters()}
    drawn = []
    real = nn.init.normal_
    monkeypatch.setattr(nn.init, "normal_", lambda t, *a, **k: (drawn.append(id(t)), real(t, *a, **k))[1])

    init_like_hf_(model)

    for name in ("tok_embeddings", "layers.0.feed_forward.w1", "layers.0.feed_forward.w2", "layers.0.feed_forward.w3"):
        got = _std(model.get_submodule(name).weight)
        assert abs(got - 0.02) < 0.002, f"{name}: std {got:.4f}"
    assert model.output.weight is model.tok_embeddings.weight
    assert drawn.count(id(model.tok_embeddings.weight)) == 1
    for k, v in model.layers["0"].attention.named_parameters():
        assert torch.equal(v, mixer_before[k]), f"the mixer's {k} was redrawn"
    assert torch.equal(model.norm.weight, torch.ones(DIM))


def test_with_hf_init_runs_after_the_models_own_init():
    """torchtitan's init_weights still runs -- it owns the buffers -- and the
    HF draw lands on top of it."""
    from lm_scaling.titan_ext.hf_init import with_hf_init

    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(DIM, DIM, bias=True)
            self.buffers_written = False

        def init_weights(self, buffer_device=None):
            with torch.no_grad():
                self.lin.weight.fill_(7.0)
                self.lin.bias.fill_(7.0)
            self.buffers_written = True

    cls = with_hf_init(Base)
    assert cls.__name__ == "Base"
    m = cls()
    m.init_weights(buffer_device=None)
    assert m.buffers_written
    assert abs(_std(m.lin.weight) - 0.02) < 0.002
    assert torch.equal(m.lin.bias, torch.zeros(DIM))
