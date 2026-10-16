# -*- coding: utf-8 -*-

"""The sign gauge of signed / Complex KDA under context parallelism.

``chunk_kda(sign=...)`` gauges q and k with the running sign product ``P_t = prod_{u<=t} s_u``
of each sequence. Computed per rank (rank-local cu_seqlens), it restarts at every rank
boundary, so the state handed from the previous rank and the gauge here would disagree. Only
the first local sequence can continue from earlier ranks; its prefix is the product of the
local parities of those ranks (each of which holds a whole segment of that sequence as its
last local sequence). One all-gather of an ``[H, K]`` int8 parity per rank.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from fla.ops.cp.context import FLACPContext


def cp_sign_prefix(p: torch.Tensor, cp_context: FLACPContext) -> torch.Tensor:
    """Fold the earlier ranks' sign parity into ``p`` ``[1, T_local, H, K]`` (rank-local running
    sign products, as from ``kda_sign_cumprod`` with the context's cu_seqlens)."""
    group = cp_context.group
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    # the parity of this rank's segment of its last local sequence: p at the last token
    last = p[0, -1].contiguous()
    gathered = [torch.empty_like(last) for _ in range(world)]
    dist.all_gather(gathered, last, group=group)
    n_prev = int(cp_context.pre_num_ranks or 0)
    if n_prev == 0:
        return p
    prefix = gathered[rank - n_prev]
    for j in range(rank - n_prev + 1, rank):
        prefix = prefix * gathered[j]
    first_end = int(cp_context.cu_seqlens_cpu[1]) if cp_context.cu_seqlens_cpu is not None \
        else int(cp_context.cu_seqlens[1])
    p = p.clone()
    p[:, :first_end] *= prefix.to(p.dtype)
    return p
