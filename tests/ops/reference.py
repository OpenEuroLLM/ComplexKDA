"""fp64 reference KDA recurrence for the signed-gate tests.

Deliberately a transcription of fla.ops.kda.naive.naive_recurrent_kda, with two
additions the signed tests need and the shipped reference does not have:

    g_sign        apply a SIGNED decay directly, alpha = g_sign * exp(g), so a
                  test can compare the gauged pipeline against the thing the
                  gauge is supposed to be equivalent to.
    use_qk_l2norm normalise q/k inside, standing in for the kernel's fused
                  use_qk_l2norm_in_kernel=True.

Kept in float throughout (the caller passes float64) so gradcheck is meaningful.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kda_recurrence_reference(
    q, k, v, g, beta,
    g_sign=None,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = False,
):
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    if use_qk_l2norm:
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    decay = g.exp()
    if g_sign is not None:
        decay = decay * g_sign

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state
    o = []
    for i in range(T):
        q_i, k_i, v_i, b_i = q[:, i], k[:, i], v[:, i], beta[:, i]
        S = S * decay[:, i][..., None]
        S = S + torch.einsum(
            'b h k, b h v -> b h k v',
            b_i[..., None] * k_i, v_i - (k_i[..., None] * S).sum(-2))
        o.append(torch.einsum('b h k, b h k v -> b h v', q_i, S))
    o = torch.stack(o, dim=1)
    return o, (S if output_final_state else None)
