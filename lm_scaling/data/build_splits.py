"""Assemble the paper's train/valid split from per-shard tokenized output.

`prepare_paper_corpus.py tokenize` writes one `.bin`/`.idx` pair per parquet
shard, which is right for a resumable job array and wrong for a trainer that
takes a single prefix. This concatenates them and carves off the validation
set.

Two choices follow the paper's `condor.sh` rather than convenience:

* **75M rows**, i.e. the first 15 shards at 5M rows each -- their
  `--nrows_tokenize=75000000`. The full corpus is 160M rows; training on all
  of it would be a different experiment.
* **10M validation tokens** -- their `--n_tokens_valid=10000000`, taken from
  the tail of that subset.

What this cannot reproduce is *which* 10M tokens they held out:
`experiments.download_or_tokenize_data` is not in this repo, so the boundary is
a guess. A different 10M-token sample of the same corpus differs by sampling
noise, which at that size is small -- but "small" is not "zero", and any
comparison to their published number should say so.

Concatenation is a byte copy plus an index rebuild: document lengths carry
over unchanged and pointers shift by the running byte offset, so nothing is
re-tokenized.

    python lm_scaling/data/build_splits.py --tokdir DIR --out DIR --shards 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from megatron_indexed import MegatronIndexedReader  # noqa: E402
from lm_scaling.data.megatron_writer import MegatronIndexedWriter  # noqa: E402

VALID_TOKENS = 10_000_000
PAPER_SHARDS = 15          # 15 x 5M rows = the paper's 75M


def build(tokdir: Path, out: Path, n_shards: int, valid_tokens: int) -> None:
    prefixes = sorted({p.with_suffix("") for p in tokdir.glob("*.idx")})
    if not prefixes:
        raise SystemExit(f"no tokenized shards under {tokdir}")
    if len(prefixes) < n_shards:
        raise SystemExit(
            f"only {len(prefixes)} shards tokenized, need {n_shards}; "
            f"the array job is not finished"
        )
    use = prefixes[:n_shards]
    out.mkdir(parents=True, exist_ok=True)

    readers = [MegatronIndexedReader(p) for p in use]
    total = sum(int(r.total_tokens) for r in readers)
    print(f"{len(use)} shards, {sum(len(r) for r in readers):,} documents, "
          f"{total / 1e9:.2f}B tokens")
    if total <= valid_tokens:
        raise SystemExit("corpus smaller than the requested validation split")

    # Walk backwards to find where the last `valid_tokens` begin, so the split
    # falls on a document boundary rather than mid-document.
    seen = 0
    cut = None
    for si in range(len(readers) - 1, -1, -1):
        r = readers[si]
        for di in range(len(r) - 1, -1, -1):
            seen += int(r.sequence_lengths[di])
            if seen >= valid_tokens:
                cut = (si, di)
                break
        if cut:
            break
    print(f"validation = shard {cut[0]} document {cut[1]} onward "
          f"({seen:,} tokens, requested {valid_tokens:,})")

    for name, keep in (("train", False), ("valid", True)):
        w = MegatronIndexedWriter(out / name)
        for si, r in enumerate(readers):
            if keep and si < cut[0]:
                continue
            if not keep and si > cut[0]:
                continue
            lo = cut[1] if (keep and si == cut[0]) else 0
            hi = cut[1] if (not keep and si == cut[0]) else len(r)
            for di in range(lo, hi):
                w.add(r.get(di))
        w.finalize()
        print(f"  {name}: {w.n_documents:,} documents, {w.n_tokens / 1e9:.3f}B tokens "
              f"-> {out / name}.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shards", type=int, default=PAPER_SHARDS,
                    help="how many shards to use (15 = the paper's 75M rows)")
    ap.add_argument("--valid-tokens", type=int, default=VALID_TOKENS)
    a = ap.parse_args()
    build(a.tokdir, a.out, a.shards, a.valid_tokens)


if __name__ == "__main__":
    main()
