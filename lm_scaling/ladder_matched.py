"""Generate `ladder_matched.json`: every arm's width at every rung of the
scaling ladder.

    CC=/usr/bin/gcc PYTHONPATH=.:lm_scaling python lm_scaling/ladder_matched.py

The counterpart to `replication_1p3B.py`, for the ladder rather than the 1.3B
fixed point. The rungs, the vocabulary, the context and the tying all come
from `ladder_spec.yaml`; the arms come from `param_match.ARCHS`. Nothing here
is typed twice.

WHAT "MATCHED" MEANS. The dense transformer at each rung IS the target: its
parameter count is what every linear-attention arm is resized to, by moving
the SwiGLU MLP's `intermediate_size` in multiples of 64 and leaving depth,
width and head geometry alone. The mixer block of a KDA-family layer does not
weigh what the attention it replaces weighs, so a comparison at fixed depth
and width is not a comparison at fixed N unless something absorbs the
difference.

The residual is what the 64-multiple grid cannot remove -- at 47M one step of
`d_ffn` is 1.9% of N, so under half a step is the best any arm can do, and
`delta_pct` records it per cell rather than hiding it.

Counting instantiates the real `fla` config on the meta device, so the number
is whatever the model actually allocates rather than an analytic form that can
drift from the implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import ladder as L  # noqa: E402
import param_match as pm  # noqa: E402


def build(archs=None) -> dict:
    archs = list(archs or pm.ARCHS)
    out = {}
    for r in L.SCALING:
        geo = dict(d_model=r["d_model"], n_heads=r["n_heads"], n_layers=r["n_layers"],
                   head_dim=L.HEAD_DIM, vocab_size=L.VOCAB_SIZE, seq_len=L.SEQ_LEN,
                   tie=True)
        # The dense arm at the paper's own d_ffn: the target, not a fit.
        target = pm.n_params_of("attn", d_ffn=r["d_ffn"], **geo)
        entry = {"d_model": r["d_model"], "n_heads": r["n_heads"],
                 "n_layers": r["n_layers"], "head_dim": L.HEAD_DIM,
                 "tie_word_embeddings": True, "target_N": target,
                 "paper_params": r["paper_params"], "archs": {}}
        for arch in archs:
            if arch == "attn":
                ffn, n = r["d_ffn"], target
            else:
                ffn, n = pm.match_ffn(arch, target, multiple=64, **geo)
            e = {"d_ffn": ffn, "N": n, "delta_pct": 100.0 * (n - target) / target,
                 # Written, not implied: the torchtitan flavors are generated
                 # from these, so a change here reaches the backend or nothing
                 # does.
                 "arch": pm.ARCHS[arch]["arch"], "mixer": pm.ARCHS[arch]["mixer"],
                 "kwargs": dict(pm.ARCHS[arch]["kwargs"])}
            # A property of the STACK, not a mixer kwarg: which layers get a
            # mixer at all. titan_ext reads it when it registers the flavor.
            if "hybrid" in pm.ARCHS[arch]:
                e["hybrid"] = dict(pm.ARCHS[arch]["hybrid"])
            entry["archs"][arch] = e
        out[r["tag"]] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE / "ladder_matched.json"))
    a = ap.parse_args()
    out = build()
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    hdr = f"{'rung':>6} {'arch':>34} {'d_ffn':>7} {'params':>14} {'delta':>8}"
    print(hdr)
    print("-" * len(hdr))
    for tag, entry in out.items():
        for arch, e in entry["archs"].items():
            print(f"{tag:>6} {arch:>34} {e['d_ffn']:>7} {e['N']:>14,} {e['delta_pct']:>7.2f}%")
        print()
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
