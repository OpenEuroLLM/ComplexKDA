# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Per-call KDA implementation selection, retained by the autograd context."""

import inspect
import os
from dataclasses import dataclass
from functools import wraps

from fla.utils import IS_NVIDIA_HOPPER


@dataclass(frozen=True)
class KDAImplementation:
    name: str
    wy_backward: object = None
    intra_backward: object = None
    paired_norm: bool = False
    fuse_norm: bool = False
    parallel_parity: bool = False


def resolve_implementation(name, q, v, use_qk_l2norm_in_kernel, cp_context):
    from fla.ops.kda.chunk_bwd import chunk_kda_bwd_wy_dqkg_fused
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra

    if name is None:
        from fla.ops.backends import _DISPATCH_DISABLED
        from fla.ops.kda.backends.tilelang import KDATileLangBackend
        if not q.is_cuda or cp_context is not None:
            return KDAImplementation('auto')
        name = 'triton'
        if (not _DISPATCH_DISABLED and q.shape[2] == v.shape[2]
                and KDATileLangBackend.is_enabled() and KDATileLangBackend.is_available()):
            name = 'tilelang'
    optimized = name in ('triton_optimized', 'hybrid_optimized')
    wy = inspect.unwrap(chunk_kda_bwd_wy_dqkg_fused)
    intra = inspect.unwrap(chunk_kda_bwd_intra)
    if optimized:
        from fla.ops.kda.backends.optimized.intra import chunk_kda_bwd_intra as intra
        from fla.ops.kda.backends.optimized.wy import chunk_kda_bwd_wy_dqkg_fused as wy
    if name in ('tilelang', 'hybrid_optimized'):
        from fla.ops.kda.backends.tilelang.chunk_bwd_dqkg import chunk_kda_bwd_wy_dqkg_fused_tilelang as wy
    fuse_norm = (
        optimized and use_qk_l2norm_in_kernel and IS_NVIDIA_HOPPER
        and q.shape[-1] <= 128 and q.shape[2] == v.shape[2] and cp_context is None
    )
    return KDAImplementation(name, wy, intra, optimized, fuse_norm, optimized)


def strict_selection(func):
    # copied Dynamo attributes can make compiler.disable unwrap past this guard.
    @wraps(func, updated=())
    def wrapper(*args, **kwargs):
        from fla.ops.backends import _DISPATCH_DISABLED
        from fla.ops.kda.backends.optimized import (
            KDABaselineTileLangBackend,
            KDAOptimizedHybridBackend,
            KDAOptimizedTritonBackend,
            KDATritonBackend,
        )
        selected = os.environ.get('FLA_KDA_BACKEND', 'auto')
        backends = {cls.name(): cls for cls in (
            KDATritonBackend, KDABaselineTileLangBackend, KDAOptimizedTritonBackend, KDAOptimizedHybridBackend,
        )}
        if selected != 'auto':
            if selected not in backends:
                raise ValueError(f'Unknown FLA_KDA_BACKEND={selected!r}; choose auto or {tuple(backends)}')
            if _DISPATCH_DISABLED:
                raise RuntimeError('Explicit FLA_KDA_BACKEND requires FLA_DISABLE_BACKEND_DISPATCH=0 before import')
            accepted, reason = backends[selected]().chunk_kda_verifier(*args, **kwargs)
            if not accepted:
                raise RuntimeError(f'KDA backend {selected!r} rejected the call: {reason}')
        elif any(os.environ.get(flag) == '1' for flag in (
            'FLA_EXPERIMENTAL_UNSIGNED_NORM_FUSION', 'FLA_EXPERIMENTAL_SIGNED_NORM_FUSION',
        )):
            raise RuntimeError('Select FLA_KDA_BACKEND=triton_optimized or hybrid_optimized instead of legacy fusion flags')
        return func(*args, **kwargs)
    return wrapper
