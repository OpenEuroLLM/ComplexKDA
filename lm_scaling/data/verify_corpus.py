"""Read a Megatron corpus's own header and check it is what the config says.

    python lm_scaling/data/verify_corpus.py fwedu_1p3B

Standard library only, on purpose: this runs on a login node, where the point
is to answer the question without a container or an environment.

Why it exists. `prefix` and `valid_prefix` are strings, and a string can point
at a corpus that has moved, been rebuilt with a different tokenizer, or been
truncated by a job that died mid-copy -- and all three still train. The token
count in the header is the cheapest thing that distinguishes them, and the
config records it, so a mismatch is a hard failure rather than a loss curve
that looks slightly odd months later.

It has already caught one class of this: the oellm-autoexp configs name
/e/scratch/projectnucleus/... for the held-out split, which this account
cannot read at all.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

# Megatron's MMapIndexedDataset dtype codes.
ITEMSIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 4, 7: 8, 8: 2}
DTYPE = {1: "uint8", 2: "int8", 3: "int16", 4: "int32", 5: "int64",
         6: "float32", 7: "float64", 8: "uint16"}
MAGIC = b"MMIDIDX\x00\x00"


class CorpusError(RuntimeError):
    pass


def read_header(prefix: str) -> dict:
    """Documents, tokens and dtype, from the .idx header and the .bin size."""
    idx, bin_ = Path(prefix + ".idx"), Path(prefix + ".bin")
    for p in (idx, bin_):
        if not p.exists():
            raise CorpusError(f"{p} does not exist")
        if not os.access(p, os.R_OK):
            raise CorpusError(f"{p} is not readable by this account")
    with idx.open("rb") as f:
        if f.read(9) != MAGIC:
            raise CorpusError(f"{idx} is not a Megatron indexed dataset")
        version = struct.unpack("<Q", f.read(8))[0]
        code = struct.unpack("<B", f.read(1))[0]
        # Megatron writes len(sizes) then len(doc_idx), and doc_idx carries a
        # trailing sentinel -- so the second number is documents + 1, and
        # reporting it as the document count is off by one in a way that looks
        # like a real difference between two corpora.
        documents = struct.unpack("<Q", f.read(8))[0]
        doc_idx_len = struct.unpack("<Q", f.read(8))[0]
    if code not in ITEMSIZE:
        raise CorpusError(f"{idx} has unknown dtype code {code}")
    return {"version": version, "dtype": DTYPE[code], "documents": documents,
            "doc_idx_len": doc_idx_len,
            "tokens": bin_.stat().st_size // ITEMSIZE[code]}


def check(prefix: str, expect_tokens: int, seq_len: int, need_windows: int,
          label: str) -> list[str]:
    problems = []
    try:
        head = read_header(prefix)
    except CorpusError as exc:
        return [f"{label}: {exc}"]

    if expect_tokens and head["tokens"] != expect_tokens:
        problems.append(
            f"{label}: header says {head['tokens']:,} tokens, the config says "
            f"{expect_tokens:,}. The corpus has moved, been rebuilt or been "
            "truncated; all three still train.")
    # uint16 is not decoration: it caps the vocabulary at 65,536, and a corpus
    # written with a 256k tokenizer would be int32 and read as garbage pairs.
    if head["dtype"] != "uint16":
        problems.append(f"{label}: dtype is {head['dtype']}, not uint16 -- "
                        "this is not the GPT-NeoX-20B pretokenization")
    windows = head["tokens"] // (seq_len + 1)
    if windows < need_windows:
        problems.append(f"{label}: {windows:,} windows at seq {seq_len}, "
                        f"fewer than the {need_windows:,} asked for")
    print(f"{label}: {head['documents']:,} docs, {head['tokens']:,} tokens "
          f"({head['dtype']}), {windows:,} windows at {seq_len}")
    return problems


def main(argv=None) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import plan as P

    experiment = (argv or sys.argv[1:] or ["fwedu_1p3B"])[0]
    cfg = P.load(experiment)
    d = cfg.data

    problems = check(d.prefix, d.train_tokens, d.seq_len, 1, "train")
    problems += check(d.valid_prefix, d.valid_tokens, d.seq_len,
                      d.eval_sequences, "valid")
    if problems:
        print("\n" + "\n".join("  " + p for p in problems), file=sys.stderr)
        return 1
    print("\nboth corpora match the config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
