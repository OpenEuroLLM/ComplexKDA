# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Explicit baseline and optimized KDA routes."""

import os

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_NVIDIA_HOPPER


class KDAImplementationBackend(BaseBackend):
    package_name = None
    env_var = 'FLA_KDA_BACKEND'
    default_enable = False
    priority = -10
    selection_name = None

    @classmethod
    def name(cls):
        return cls.selection_name or cls.backend_type

    @classmethod
    def is_enabled(cls):
        return os.environ.get(cls.env_var) == cls.name()

    def chunk_kda_verifier(self, q, k, v, g, beta, *args, **kwargs):
        import inspect

        from fla.ops.kda.chunk import _chunk_kda_impl

        bound = inspect.signature(_chunk_kda_impl).bind(q, k, v, g, beta, *args, **kwargs)
        bound.apply_defaults()
        if not q.is_cuda or any(x.device != q.device for x in (k, v, g, beta)):
            return False, 'explicit KDA backends require CUDA tensors on one device'
        if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
            return False, 'expected matching four-dimensional q/k and four-dimensional v'
        if bound.arguments['cp_context'] is not None:
            return False, 'explicit KDA backends do not yet support context parallelism; use auto'
        if self.name() != 'triton' and v.shape[2] != q.shape[2]:
            return False, 'this KDA backend requires equal query and value head counts'
        if self.name().endswith('_optimized'):
            if not IS_NVIDIA_HOPPER:
                return False, 'optimized KDA backends are currently restricted to NVIDIA Hopper'
            if q.shape[-1] > 128:
                return False, 'optimized KDA backends currently require key dimension <= 128'
            if q.dtype not in (torch.float32, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
                return False, 'optimized KDA backends require matching FP32 or BF16 q/k/v'
        if self.name() in ('tilelang', 'hybrid_optimized'):
            from fla.ops.kda.backends.tilelang import KDATileLangBackend
            if not KDATileLangBackend.is_available():
                return False, 'TileLang and a usable CUDA compiler are required'
        return True, None

    def chunk_kda(self, *args, **kwargs):
        from fla.ops.kda.chunk import _chunk_kda_impl
        return _chunk_kda_impl(*args, **kwargs, _implementation=self.name())


class KDATritonBackend(KDAImplementationBackend):
    backend_type = 'triton'


class KDABaselineTileLangBackend(KDAImplementationBackend):
    backend_type = 'tilelang_baseline'
    selection_name = 'tilelang'


class KDAOptimizedTritonBackend(KDAImplementationBackend):
    backend_type = 'triton_optimized'


class KDAOptimizedHybridBackend(KDAImplementationBackend):
    backend_type = 'hybrid_optimized'
