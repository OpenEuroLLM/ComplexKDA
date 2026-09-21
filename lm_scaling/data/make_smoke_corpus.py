"""A throwaway Megatron corpus, so the anchor's wiring is tested before 12 hours of GPU.

The four anchor arms are the expensive part of this study and every one of
their failure modes -- a validator reading the wrong file, a flavor that builds
the wrong geometry, a scheduler key torchtitan ignores -- is visible in the
first thirty steps. What is not visible in thirty steps is a real loss, which
is why the tokens here are random: this checks that the machinery runs, not
what it produces.

    python lm_scaling/data/make_smoke_corpus.py --out /path/smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lm_scaling.data.megatron_writer import MegatronIndexedWriter  # noqa: E402

# GPT-2's real vocabulary, so the embedding indices a smoke run produces are in
# the same range the corpus will use and an off-by-one in vocab padding still
# shows up here.
VOCAB = 50257


def build(out: Path, train_tokens: int, valid_tokens: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, total in (("train", train_tokens), ("valid", valid_tokens)):
        written = 0
        with MegatronIndexedWriter(out / name) as w:
            while written < total:
                n = min(int(rng.integers(200, 4000)), total - written)
                w.add(rng.integers(0, VOCAB, size=n, dtype=np.uint16))
                written += n
        print(f"{out / name}: {written:,} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tokens", type=int, default=40_000_000)
    ap.add_argument("--valid-tokens", type=int, default=2_000_000)
    a = ap.parse_args()
    build(Path(a.out), a.train_tokens, a.valid_tokens)


if __name__ == "__main__":
    main()
