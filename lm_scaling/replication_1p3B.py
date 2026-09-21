"""Generate `replication_1p3B.json`: the study's arms at the 1.3B geometry
other groups publish 100BT results at.

    CC=/usr/bin/gcc PYTHONPATH=.:lm_scaling python lm_scaling/replication_1p3B.py

Every 1.3B/100BT linear-attention comparison on FineWeb-Edu reports against
Transformer++ 1.3B -- Gated DeltaNet's Table 3 (arXiv:2412.06464) is the table
the others copy rows from. Its shape is fla-hub's `transformer-1.3B-100B`:

    hidden 2048, 24 layers, SwiGLU at hidden_ratio 4, vocabulary 32,000,
    UNTIED embeddings

`REFERENCE` is that config.json, transcribed; `target_N` is what fla builds
from it, so the count every arm is matched to is the published model's rather
than a number typed here.

Three things differ from the scaling ladder, and each is the published setup's
choice rather than ours:

**Untied embeddings and a 32,000 vocabulary**, 131M of the 1.36B parameters.
Both are the published model's; neither is a default of ours.

**head_dim 128, i.e. 16 heads.** fla's own KDAConfig defaults ARE this 1.3B
shape -- hidden 2048, 24 layers, 16 heads of 128, vocabulary 32,000, untied --
and Kimi Linear runs 128 throughout. The point is to be the KDA other groups
would build, with a state the size theirs has. The published Transformer++
runs 32 heads of 64. A transformer's head count moves no parameters, so `attn`
below is exactly the reference model's size; it is the target, not an arm the
campaign runs.

**Every arm is matched to the reference** by `param_match.match_ffn` in
multiples of 64, so the two members of a pair are identical to the parameter
-- which is what makes the pair a comparison of the gate and nothing else.

The arms are `param_match.ARCHS` verbatim: the layer, gate, init and hybrid
pattern the campaign runs. Changing one here would make "the layer works at
1.3B" a claim about a different layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import param_match as pm  # noqa: E402

TAG = "1.3B"
REFERENCE_ID = "fla-hub/transformer-1.3B-100B"
# config.json of REFERENCE_ID, in the keywords fla's TransformerConfig takes.
# max_position_embeddings is left out: it builds no parameters.
REFERENCE = dict(hidden_size=2048, num_hidden_layers=24, num_heads=32,
                 hidden_ratio=4, intermediate_size=None, vocab_size=32000,
                 tie_word_embeddings=False)
GEOMETRY = dict(d_model=2048, n_layers=24, n_heads=16, head_dim=128,
                vocab_size=32000, seq_len=4096, tie_word_embeddings=False)
# Every arm this geometry carries, and it must stay in step with
# `param_match.ARCHS`: an arm in the JSON and not here would be DROPPED the
# next time this is regenerated, silently. `attn` is not listed because it is
# not an arm -- `build()` always emits it as the parameter reference.
#
# The four are two pairs: a bounded sigmoid gate against the signed gate, each
# once as a pure linear stack and once inside the 3:1 hybrid, all at the
# shipped init and fla's factored (low-rank) output gate.
ARMS = (
    "kda-sig-lowrank", "ckda-shipped-lowrank",
    "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank",
)


def reference() -> tuple[int, int]:
    """(N, d_ffn) of the published Transformer++ 1.3B, as fla builds it."""
    import torch
    from transformers import AutoModelForCausalLM

    from fla.models import TransformerConfig

    cfg = TransformerConfig(**REFERENCE)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    return pm.count_params(cfg), int(model.model.layers[0].mlp.intermediate_size)


def _entry(arch: str, d_ffn: int, n: int, target: int) -> dict:
    out = {"d_ffn": d_ffn, "N": n, "delta_pct": 100.0 * (n - target) / target,
           "arch": pm.ARCHS[arch]["arch"], "mixer": pm.ARCHS[arch]["mixer"],
           "kwargs": dict(pm.ARCHS[arch]["kwargs"])}
    # A property of the stack, not a mixer kwarg -- see param_match.main.
    if "hybrid" in pm.ARCHS[arch]:
        out["hybrid"] = dict(pm.ARCHS[arch]["hybrid"])
    return out


def build() -> dict:
    g = GEOMETRY
    geo = dict(d_model=g["d_model"], n_heads=g["n_heads"], n_layers=g["n_layers"],
               head_dim=g["head_dim"], vocab_size=g["vocab_size"],
               seq_len=g["seq_len"], tie=g["tie_word_embeddings"])
    target, attn_ffn = reference()

    # At 16 heads instead of 32 our `attn` has to be the reference's size to
    # the parameter. If that stops holding, the target is no longer the
    # published model and every width below is matched to something else.
    ours = pm.n_params_of("attn", d_ffn=attn_ffn, **geo)
    if ours != target:
        raise SystemExit(f"attn at {g['n_heads']} heads builds {ours:,} parameters "
                         f"against the reference's {target:,}")

    archs = {"attn": _entry("attn", attn_ffn, target, target)}
    for arch in ARMS:
        ffn, n = pm.match_ffn(arch, target, multiple=64, **geo)
        archs[arch] = _entry(arch, ffn, n, target)
    return {"tag": TAG, **g, "target_N": target, "reference": REFERENCE_ID,
            "source": "lm_scaling/replication_1p3B.py, arms from param_match.ARCHS",
            "archs": archs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE / "replication_1p3B.json"))
    a = ap.parse_args()
    out = build()
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"{TAG}: d_model {out['d_model']}, {out['n_layers']} layers, "
          f"{out['n_heads']} heads of {out['head_dim']}, vocab {out['vocab_size']}, "
          f"untied; target {out['target_N']:,} ({REFERENCE_ID})")
    for arch, e in out["archs"].items():
        print(f"  {arch:<12} d_ffn {e['d_ffn']:<6} N={e['N']:>14,}  "
              f"{e['delta_pct']:+.3f}%  mixer={e['mixer']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
