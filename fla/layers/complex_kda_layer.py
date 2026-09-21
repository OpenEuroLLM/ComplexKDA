# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KimiDeltaAttention with a signed decay gate: alpha in [-1, 1] instead of (0, 1].

Diff against fla/layers/kda.py, which this file otherwise mirrors:

  1. `gate` selects the decay parameterisation (2 unsigned, 2 signed).
  2. The +-1 part of a signed gate is carried as a running sign pushed onto q/k
     (the gauge); the kernels themselves run on |alpha|. On the Triton path the
     sign is handed to the op as `sign=` and applied inside, in KDA's own
     l2norm epilogue (fla/ops/kda/gauge.py); the reference path gauges in torch.
  3. `backend` adds a pure-torch CPU path, since fla's kernels are Triton-only.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.utils import get_layer_cache, repad_hidden_states, unpad_hidden_states, update_layer_cache
from fla.modules import FusedRMSNormGated, ShortConvolution
from fla.ops.kda import chunk_kda, fused_recurrent_kda

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


# ---------------------------------------------------------------------------
# gate registry
#
#   name              alpha range   activation
#   "softplus"        (0, 1]        -exp(A_log) * softplus(u)   [shipped default]
#   "sigmoid"         (0, 1]        lower_bound * sigmoid(A * u)
#   "signed_sigmoid2" [-1, 1]       2*sigmoid(u) - 1, as tanh(u/2)
#   "signed_tanh"     [-1, 1]       tanh(u)
# ---------------------------------------------------------------------------

GATES = ("softplus", "sigmoid", "signed_sigmoid2", "signed_tanh")


def is_signed(gate: str) -> bool:
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    return gate.startswith("signed_")


def safe_gate_ok(gate: str) -> bool:
    """Whether g is bounded in [lower_bound, 0], enabling the kernel's M=16 path.
    False only for "softplus", which is unbounded below."""
    return gate != "softplus"


def compute_gate(gate, z, A_log=None, dt_bias=None, lower_bound=-5.0):
    """name -> (sign int8, log|alpha| fp32). sign is None for the unsigned gates,
    which is what tells the caller to skip the gauge."""
    if gate not in GATES:
        raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
    if gate.startswith("signed_"):
        return signed_gate(z, A_log, dt_bias, lower_bound, activation=gate[len("signed_") :])
    u = z.float()
    if dt_bias is not None:
        u = u + dt_bias.view(*([1] * (z.dim() - 2)), *z.shape[-2:])
    A = A_log.float().exp().view(*([1] * (z.dim() - 2)), -1, 1) if A_log is not None else 1.0
    if gate == "softplus":
        return None, -A * F.softplus(u)
    return None, lower_bound * torch.sigmoid(A * u)


def signed_gate(z, A_log=None, dt_bias=None, lower_bound: float = -5.0, activation: str = "sigmoid2"):
    """One pre-activation -> (sign, log|alpha|) for alpha in [-1, 1].

    |alpha| = eps + (1 - eps)*|a|, eps = exp(lower_bound), a = tanh(u/2) or tanh(u).
    "sigmoid2" is 2*sigmoid(u)-1 evaluated as tanh(u/2): the literal spelling
    cancels catastrophically near u = 0, and the sign is a comparison against
    zero, so it would be decided by rounding rather than by u. sign shares z with
    the magnitude, so it is locally constant and detaching it is exact.
    """
    eps = math.exp(lower_bound)
    u = z.float()
    if dt_bias is not None:
        u = u + dt_bias.view(*([1] * (z.dim() - 2)), *z.shape[-2:])
    if A_log is not None:
        u = A_log.float().exp().view(*([1] * (z.dim() - 2)), -1, 1) * u
    if activation == "sigmoid2":
        a = torch.tanh(0.5 * u)
    elif activation == "tanh":
        a = torch.tanh(u)
    else:
        raise ValueError(f"activation must be 'sigmoid2' or 'tanh', got {activation!r}")
    s = torch.where(a.detach() < 0, -1, 1).to(torch.int8)
    return s, (eps + (1.0 - eps) * a.abs()).log()


def signed_gate_init(dt, lower_bound: float = -5.0, activation: str = "sigmoid2"):
    """dt_bias giving alpha = +exp(-dt) at step 0. Targets exp(-dt) rather than
    1 - dt so the signed gates match the unsigned ones bit for bit."""
    eps = math.exp(lower_bound)
    target = ((torch.exp(-dt) - eps) / (1 - eps)).clamp(1e-7, 1 - 1e-7)
    inv = torch.atanh(target)
    return 2.0 * inv if activation == "sigmoid2" else inv


def gate_init(gate, dt, lower_bound=-5.0):
    """dt_bias init inverting each gate's own forward, so all four start at the
    same alpha = exp(-dt). Only "softplus" may use the shipped inv_softplus;
    reusing it for "sigmoid" is wrong by a factor of |lower_bound|."""
    if gate.startswith("signed_"):
        return signed_gate_init(dt, lower_bound, gate[len("signed_") :])
    if gate == "sigmoid":
        p = (dt / abs(lower_bound)).clamp(1e-7, 1 - 1e-7)
        return torch.log(p) - torch.log1p(-p)
    return dt + torch.log(-torch.expm1(-dt))


def init_dt_bias(gate, gate_dim=None, lower_bound=-5.0, gate_init_style="shipped", dt=None):
    """Whole-layer dt_bias. Single source of truth: the layer's __init__ and the
    HF _init_weights both call this, so `gate_init_style` cannot silently apply
    in one path and not the other."""
    if dt is None:
        dt = torch.exp(torch.rand(gate_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(
            min=1e-4
        )
    init = gate_init(gate, dt, lower_bound)
    if is_signed(gate) and gate_init_style == "spread":
        # Same |alpha| as "shipped", sign flipped on half the channels: the
        # activation is odd, so sign and magnitude do not trade off.
        init = init * torch.where(torch.rand_like(init) < 0.5, -1.0, 1.0)
    return init


def _beta_spread_init_params(logit_std: float, quadrature_points: int = 1024) -> tuple[float, float]:
    """Return the weight scale and bias that jointly match standard beta's variance and modes at 0.5/1.5."""
    if logit_std <= 0:
        return 1.0, math.log(3.0)

    q = (torch.arange(quadrature_points, dtype=torch.float64) + 0.5) / quadrature_points
    z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
    target_std = torch.sigmoid(logit_std * z).std(correction=0).item()
    log3 = math.log(3.0)

    def solve_bias(spread_std: float) -> float:
        def mode_residual(bias: float) -> float:
            return log3 - bias * math.tanh(bias * log3 / spread_std**2) - 0.5 * spread_std**2

        lo, hi = 0.0, log3 + 2.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if mode_residual(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    lo, hi = 1e-6, math.sqrt(2.0 * log3) * (1.0 - 1e-6)
    for _ in range(64):
        spread_std = 0.5 * (lo + hi)
        bias = solve_bias(spread_std)
        beta = 2.0 * torch.sigmoid(spread_std * z - bias)
        if beta.std(correction=0).item() < target_std:
            lo = spread_std
        else:
            hi = spread_std
    spread_std = 0.5 * (lo + hi)
    return spread_std / logit_std, solve_bias(spread_std)


@torch.no_grad()
def _init_beta_projection_spread_(projection: nn.Linear) -> None:
    if projection.bias is None:
        raise ValueError("beta spread initialization requires a projection bias")
    if projection.weight.device.type == "meta":
        return

    logit_std = projection.weight.float().square().sum(dim=-1).mean().sqrt().item()
    weight_scale, bias = _beta_spread_init_params(logit_std)
    projection.weight.mul_(weight_scale)
    signs = torch.ones(projection.out_features, device=projection.bias.device, dtype=projection.bias.dtype)
    signs[:projection.out_features // 2] = -1
    signs = signs[torch.randperm(projection.out_features, device=projection.bias.device)]
    projection.bias.copy_(signs * bias)


# ---------------------------------------------------------------------------
# gauge: carry the +-1 part of alpha as a running sign on q/k
# ---------------------------------------------------------------------------

def running_sign(s: torch.Tensor, cu_seqlens: torch.Tensor | None = None) -> torch.Tensor:
    """P_t = prod_{u<=t} s_u along dim 1, as int8. Integer parity prefix sum:
    exact at any length, and carries no autograd graph (the sign has no
    gradient). Resets at sequence starts."""
    bits = (s < 0).to(torch.int32)
    par = bits.cumsum(dim=1)
    if cu_seqlens is not None:
        starts = cu_seqlens[:-1]
        idx = torch.repeat_interleave(starts, cu_seqlens[1:] - starts)
        par = par - (par[:, idx] - bits[:, idx])
    return torch.where(par & 1 == 1, -1, 1).to(torch.int8)


class _ApplySign(torch.autograd.Function):
    """x * P, keeping P as int8 rather than letting `mul` upcast it."""

    @staticmethod
    def forward(ctx, x, P):
        ctx.save_for_backward(P)
        return x * P.to(x.dtype)

    @staticmethod
    def backward(ctx, go):
        (P,) = ctx.saved_tensors
        return go * P.to(go.dtype), None


def apply_sign(x, P):
    return _ApplySign.apply(x, P)


def ungauge_state(ht, P_last, state_v_first: bool, head_k_dim: int | None = None):
    """S_T = Diag(P_T) S~_T, on whichever axis holds K.

    state_v_first=True stores [N, HV, V, K], i.e. K LAST, despite fla's docstring
    saying otherwise; head_k_dim turns a silent wrong-axis bug into an assert.
    """
    if ht is None or P_last is None:
        return ht
    if P_last.ndim != ht.ndim - 1:
        # A gauge one rank too high broadcasts into a state with a spurious
        # leading axis instead of failing, which is how the varlen boundary
        # read (P[0, ...], not P[:, ...]) silently cached a batch-1 state.
        raise AssertionError(
            f"gauge rank mismatch: state {tuple(ht.shape)} takes a gauge of {ht.ndim - 1} dims, got {tuple(P_last.shape)}."
        )
    axis = -1 if state_v_first else -2
    if head_k_dim is not None and ht.shape[axis] != head_k_dim:
        raise AssertionError(
            f"state layout mismatch: state_v_first={state_v_first} implies K on "
            f"axis {axis}, but state shape {tuple(ht.shape)} has {ht.shape[axis]} "
            f"there, not head_k_dim={head_k_dim}."
        )
    P = P_last.to(ht.dtype)
    return ht * (P.unsqueeze(-2) if state_v_first else P.unsqueeze(-1))


def _triton_available() -> bool:
    """Stricter than `import triton` succeeding: fla imports fine on CPU and
    fails at kernel launch."""
    try:
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


HAS_TRITON = _triton_available()


def check_chunk_span(g, chunk_size, dtype=torch.float32):
    """naive_chunk_kda exponentiates over a whole block before masking, so past a
    span of ~88 in fp32 the forward looks finite and the backward is NaN."""
    limit = 88.0 if dtype == torch.float32 else 709.0
    span = float(g.detach().min().abs()) * chunk_size
    if span > limit:
        raise ValueError(
            f"naive_chunk_kda would overflow: exponent span |log alpha|*chunk = "
            f"{span:.0f} exceeds {limit:.0f}. Use backend='naive_recurrent', or "
            f"reduce chunk_size below {limit / max(span / chunk_size, 1e-9):.0f}."
        )


def naive_kda_call(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, chunkwise=False, chunk_size=16):
    """fla.ops.kda.naive. q/k must arrive l2-normalised and gauged; g is log|alpha|."""
    from fla.ops.kda.naive import naive_chunk_kda, naive_recurrent_kda

    if not chunkwise:
        return naive_recurrent_kda(
            q, k, v, g, beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state
        )
    check_chunk_span(g, chunk_size, q.dtype)
    T = q.shape[1]
    if T % chunk_size:
        raise ValueError(f"naive_chunk_kda needs T ({T}) divisible by chunk_size ({chunk_size})")
    return naive_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )


class ShortConvolutionCPU(nn.Conv1d):
    """Causal depthwise conv1d + activation. Subclasses nn.Conv1d exactly as
    fla's ShortConvolution does, so the parameter names match and a checkpoint
    is portable between the CPU and Triton paths."""

    def __init__(self, hidden_size, kernel_size=4, bias=False, activation="silu"):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=bias)
        if activation not in (None, "silu", "swish"):
            raise ValueError(f"unsupported activation {activation!r}")
        self.hidden_size, self.activation = hidden_size, activation

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **kwargs):
        B, T, D = x.shape
        h = x.transpose(1, 2)
        if cache is not None:
            h = torch.cat([cache, h], dim=-1)[:, :, -(T + self.kernel_size[0] - 1) :]
            pad = self.kernel_size[0] - 1 - (h.shape[-1] - T)
            if pad > 0:
                h = F.pad(h, (pad, 0))
        else:
            h = F.pad(h, (self.kernel_size[0] - 1, 0))
        new_cache = h[:, :, -(self.kernel_size[0] - 1) :] if output_final_state else None
        y = self._conv_forward(h, self.weight, self.bias)[:, :, :T].transpose(1, 2)
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        return y, new_cache


class FusedRMSNormGatedCPU(nn.Module):
    """rms(x) * weight * act(g). The gate is applied AFTER normalising, per the
    Triton kernel."""

    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish"):
        super().__init__()
        if activation not in ("swish", "silu", "sigmoid"):
            raise ValueError(f"Unsupported activation: {activation}")
        self.eps, self.activation = eps, activation
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

    def forward(self, x, g, **kwargs):
        dt = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        gf = g.float()
        y = y * (torch.sigmoid(gf) if self.activation == "sigmoid" else gf * torch.sigmoid(gf))
        return y.to(dt)


# DISPATCH ON THE TENSOR, NOT ON THE MACHINE. Choosing here on `HAS_TRITON`
# alone was wrong in the one case that matters: a box that HAS a GPU but is
# handed CPU tensors. `HAS_TRITON` is `torch.cuda.is_available()`, so on such a
# box the Triton norm is installed and a CPU forward dies inside the kernel
# launch with "Pointer argument (at 0) cannot be accessed from Triton (cpu
# tensor?)" -- which is what `backend='naive_recurrent'` exists to avoid. The
# `backend` switch covers the recurrence and nothing else; the gated norm and
# the short convolution are Triton too, and they are reached whatever it says.
#
# The subclasses below keep the Triton path for CUDA tensors and fall back to
# the pure-torch forward for everything else. They SUBCLASS rather than wrap, so
# the parameters, their names and the state dict are exactly the Triton
# classes': a checkpoint stays portable in both directions, which is the
# property `ShortConvolutionCPU` was written to preserve.
if HAS_TRITON:

    class FusedRMSNormGatedAuto(FusedRMSNormGated):
        """Triton on CUDA, `FusedRMSNormGatedCPU`'s arithmetic elsewhere."""

        def forward(self, x, g, **kwargs):
            if x.device.type == "cuda":
                return super().forward(x, g, **kwargs)
            return FusedRMSNormGatedCPU.forward(self, x, g, **kwargs)

    class ShortConvolutionAuto(ShortConvolution):
        """Likewise for the causal depthwise convolution.

        The arithmetic is spelled out here rather than delegated to
        `ShortConvolutionCPU.forward`, because the two classes build their
        `nn.Conv1d` DIFFERENTLY: fla's carries `padding=(kernel_size - 1,)` and
        the CPU one `padding=(0,)`, pre-padding by hand instead. Borrowing the
        CPU forward onto an fla instance therefore pads twice -- once explicitly
        and once inside `_conv_forward` -- and shifts the output. It is not a
        small error: measured against a hand-rolled depthwise causal
        convolution, the borrowed version was off by 1.6 where fla's kernel
        agrees to 1e-7. So `padding=0` is passed explicitly below, which makes
        this correct whatever the instance was constructed with.
        """

        def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, **kwargs):
            if x.device.type == "cuda":
                return super().forward(
                    x, cache=cache, output_final_state=output_final_state,
                    cu_seqlens=cu_seqlens, **kwargs)
            if cu_seqlens is not None:
                raise NotImplementedError(
                    "the CPU fallback does not handle cu_seqlens: the varlen "
                    "metadata would have to reach the convolution, and ignoring "
                    "it would run the sequences into each other")
            T, width = x.shape[1], self.kernel_size[0]
            h = x.transpose(1, 2)
            if cache is not None:
                h = torch.cat([cache, h], dim=-1)[:, :, -(T + width - 1):]
                pad = width - 1 - (h.shape[-1] - T)
                if pad > 0:
                    h = F.pad(h, (pad, 0))
            else:
                h = F.pad(h, (width - 1, 0))
            new_cache = h[:, :, -(width - 1):] if output_final_state else None
            y = F.conv1d(h, self.weight, self.bias, stride=1, padding=0,
                         dilation=1, groups=self.groups)[:, :, :T].transpose(1, 2)
            if self.activation in ("silu", "swish"):
                y = F.silu(y)
            return y, new_cache

    FusedRMSNormGated, ShortConvolution = FusedRMSNormGatedAuto, ShortConvolutionAuto
else:  # the layer below then reads as kda.py does
    FusedRMSNormGated, ShortConvolution = FusedRMSNormGatedCPU, ShortConvolutionCPU


def _identity(x):
    return x


class ComplexKimiDeltaAttention(nn.Module):
    """KDA with a signed decay gate. Args are KimiDeltaAttention's, plus:

    Args:
        gate (str):
            One of ``GATES``; selects the whole decay parameterisation.
            Default: ``"signed_sigmoid2"``.
        gate_init_style (str):
            Only affects signed gates. ``"shipped"`` starts every channel at
            alpha > 0, leaving training to discover the negative ones;
            ``"spread"`` starts half of them negative at the same |alpha|.
            Default: ``"shipped"``.
        output_gate (str):
            Shape of the output forget gate. ``"lowrank"`` is fla's and Kimi
            Linear's factored ``hidden -> head_v_dim -> value_dim`` pair;
            ``"linear"`` is the single full-rank map Kimi K3 uses. It sits
            downstream of the recurrence, so it does not interact with the
            signed decay gate. Default: ``"lowrank"``.
        beta_init_style (str):
            The beta projection initialization. ``"standard"`` keeps the existing bias-free initialization.
            ``"spread"`` initializes two beta populations with modes at 0.5 and 1.5 and standard deviations matching
            standard beta initialization. Spread initialization requires ``allow_neg_eigval=True``. Default: ``"standard"``.
        backend (str):
            ``"auto"`` (kernel if Triton can launch, else ``naive_recurrent``),
            ``"kernel"``, ``"naive_recurrent"`` or ``"naive_chunk"``.
        conv_silu (str):
            Subset of "qkv" whose short convs keep silu. Defaults to all three;
            narrows drop_silu per projection so a single path can be ablated.
        drop_silu (bool):
            Replace the q/k/v silu with the identity, in BOTH the short-conv
            branch (``activation=None``) and the no-conv branch. silu is bounded
            below at ~-0.278, so it pushes q/k/v toward the non-negative orthant;
            k is the DIRECTION of the rank-1 update ``(I - beta k k^T)``, so that
            restricts which update directions are reachable. Default: ``False``
            (shipped behaviour).
        drop_key_silu (bool):
            Replace only the key silu with the identity, retaining silu on the
            query and value projections. Ignored when ``drop_silu=True``.
            Default: ``False``.

    ``safe_gate`` is not an argument: it is derived from ``gate``.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        gate: str = "signed_sigmoid2",
        drop_silu: bool = False,
        drop_key_silu: bool = False,
        conv_silu: str = "qkv",
        gate_init_style: str = "shipped",
        output_gate: str = "lowrank",
        beta_init_style: str = "standard",
        backend: str = "auto",
        lower_bound: float = -5.0,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> ComplexKimiDeltaAttention:
        super().__init__()

        if gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {gate!r}")
        if gate_init_style not in ("shipped", "spread"):
            raise ValueError(f"gate_init_style must be 'shipped' or 'spread', got {gate_init_style!r}")
        if output_gate not in ("lowrank", "linear"):
            raise ValueError(f"output_gate must be 'lowrank' or 'linear', got {output_gate!r}")
        if beta_init_style not in ("standard", "spread"):
            raise ValueError(f"beta_init_style must be 'standard' or 'spread', got {beta_init_style!r}")
        if beta_init_style == "spread" and not allow_neg_eigval:
            raise ValueError("beta_init_style='spread' requires allow_neg_eigval=True")
        if backend not in ("auto", "kernel", "naive_recurrent", "naive_chunk"):
            raise ValueError(f"unknown backend {backend!r}")
        if backend == "kernel" and not HAS_TRITON:
            raise ValueError("backend='kernel' needs a launchable Triton install")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound}")

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.gate = gate
        self.act = _identity if drop_silu else F.silu
        self.k_act = _identity if drop_silu or drop_key_silu else F.silu
        self.gate_init_style = gate_init_style
        self.beta_init_style = beta_init_style
        self.backend = backend
        self.safe_gate = safe_gate_ok(gate)
        self.lower_bound = lower_bound
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."
        if self.num_v_heads > self.num_heads and is_signed(gate):
            warnings.warn(
                "signed gate under GVA expands q/k to num_v_heads (the gauge is per "
                "value head but q/k are shared), losing the GVA memory saving.",
                stacklevel=2,
            )

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if any(c not in "qkv" for c in conv_silu):
            raise ValueError(f"conv_silu must be a subset of 'qkv', got {conv_silu!r}")
        # Which projections' convs keep silu. drop_silu stays the global switch;
        # conv_silu narrows it per projection, so k can be ablated on its own --
        # k is the DIRECTION of the rank-1 update, and silu (bounded below at
        # ~-0.278) biases it toward the non-negative orthant.
        self.conv_silu = "" if drop_silu else conv_silu
        if drop_key_silu:
            self.conv_silu = self.conv_silu.replace("k", "")

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu" if "q" in self.conv_silu else None,
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu" if "k" in self.conv_silu else None,
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu" if "v" in self.conv_silu else None,
            )

        self.gate_dim = int(self.num_v_heads * self.head_k_dim)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=beta_init_style == "spread")
        if beta_init_style == "spread":
            _init_beta_projection_spread_(self.b_proj)

        if self.safe_gate:
            self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        else:
            self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(init_dt_bias(gate, self.gate_dim, lower_bound, gate_init_style))
        if is_signed(gate) and gate_init_style == "spread":
            with torch.no_grad():
                self.f_proj[-1].weight.mul_(0.1)
        self.dt_bias._no_weight_decay = True

        # Output forget gate, the ONLY structural difference between the arms
        # of this study and Kimi Linear. "lowrank" factors hidden -> head_v_dim
        # -> value_dim, which ties the gate's rank to one head's width;
        # "linear" is the single full-rank map Kimi K3 uses instead. The decay
        # gate alpha is untouched either way -- the change is downstream of the
        # recurrence -- so the signed parameterisation applies identically to
        # both, which is what makes them comparable at all.
        if output_gate == "lowrank":
            self.g_proj = nn.Sequential(
                nn.Linear(hidden_size, self.head_v_dim, bias=False),
                nn.Linear(self.head_v_dim, self.value_dim, bias=True),
            )
        else:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=True)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        backend = self.backend
        if backend == "auto":
            backend = "kernel" if HAS_TRITON else "naive_recurrent"
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training and backend == "kernel":
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = kwargs.get("cu_seqlens")
        hidden_states, indices, cu_seqlens = unpad_hidden_states(hidden_states, cu_seqlens, attention_mask, q_len)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.act(self.q_proj(hidden_states))
            k = self.k_act(self.k_proj(hidden_states))
            v = self.act(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states)

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        sign, g = compute_gate(self.gate, g, self.A_log, self.dt_bias, self.lower_bound)

        # GVA: the gauge is per value head but q/k are shared across the group,
        # so q/k are expanded to HV before either path applies it. The kernel
        # asserts HV == H when `sign` is passed, so this cannot be skipped.
        if sign is not None and sign.shape[2] != q.shape[2]:
            r = sign.shape[2] // q.shape[2]
            q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
        # P is set only on the reference path; the kernel gauges internally.
        P = None

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        scale = self.head_k_dim**-0.5
        if backend == "kernel":
            # The ops take `sign` directly: the gauge rides KDA's own l2norm
            # epilogue (fla/ops/kda/gauge.py), which is free because the norm
            # already materialises the q/k the kernels read, and the final state
            # comes back un-gauged -- so nothing is done here.
            #
            # use_gate_in_kernel MUST stay False: the kernel's two fused gate
            # forms are monotone and neither sends |alpha| -> eps at the crossing.
            if mode == "chunk":
                o, recurrent_state = chunk_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    sign=sign,
                    scale=scale,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=True,
                    allow_neg_eigval=self.allow_neg_eigval,
                    safe_gate=self.safe_gate,
                    lower_bound=self.lower_bound,
                    state_v_first=True,
                    cu_seqlens=cu_seqlens,
                )
            elif mode == "fused_recurrent":
                o, recurrent_state = fused_recurrent_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    sign=sign,
                    scale=scale,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    use_qk_l2norm_in_kernel=True,
                    use_gate_in_kernel=False,
                    use_beta_sigmoid_in_kernel=True,
                    allow_neg_eigval=self.allow_neg_eigval,
                    lower_bound=self.lower_bound,
                    state_v_first=True,
                    cu_seqlens=cu_seqlens,
                )
            else:
                raise NotImplementedError(f"Not supported mode `{mode}`.")
            state_v_first = True
        else:
            # fla's reference applies neither l2norm nor the beta sigmoid, and
            # returns the state K-first, so all three are done here.
            if cu_seqlens is not None:
                raise NotImplementedError("variable-length batching needs the Triton backend")
            if sign is not None:
                P = running_sign(sign)
                q, k = apply_sign(q, P), apply_sign(k, P)
            qn = F.normalize(q.float(), dim=-1, eps=1e-6).to(q.dtype)
            kn = F.normalize(k.float(), dim=-1, eps=1e-6).to(k.dtype)
            bt = torch.sigmoid(beta.float()) * (2.0 if self.allow_neg_eigval else 1.0)
            o, recurrent_state = naive_kda_call(
                qn,
                kn,
                v,
                g.to(q.dtype),
                bt.to(q.dtype),
                scale=scale,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                chunkwise=(backend == "naive_chunk"),
            )
            state_v_first = False

        if P is not None and recurrent_state is not None:
            recurrent_state = ungauge_state(recurrent_state, P[:, -1], state_v_first=state_v_first,
                                            head_k_dim=self.head_k_dim)

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        o = repad_hidden_states(o, indices, batch_size, q_len)

        return o, None, past_key_values
