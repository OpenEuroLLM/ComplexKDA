"""Write Megatron ``IndexedDataset`` (``.bin``/``.idx``) files.

The mirror of `data/megatron_indexed.MegatronIndexedReader`, so anything
written here is readable by all three backends without conversion. Format
(version 1) is documented in that module; the two must agree byte for byte, and
`tests/lm/test_megatron_writer.py` asserts a round trip rather than trusting
that they do.

uint16 is the only sensible dtype for a GPT-2 corpus: the vocabulary is 50257,
so ids fit in 16 bits, and the alternative int32 doubles a 100 GB corpus for
nothing.
"""

from __future__ import annotations

import contextlib
import os
import struct
from pathlib import Path

import numpy as np

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_CODES = {
    np.uint8: 1, np.int8: 2, np.int16: 3, np.int32: 4,
    np.int64: 5, np.float64: 6, np.float32: 7, np.uint16: 8,
}


class MegatronIndexedWriter:
    """Append documents, then ``finalize()`` to emit the index.

    Lengths and byte offsets are accumulated in memory: a 50B-token corpus of
    ~90M documents costs ~1 GB of bookkeeping, against the 100 GB of tokens
    being streamed to disk, so there is no reason to be clever about it.
    """

    def __init__(self, prefix: str | Path, dtype=np.uint16):
        prefix = str(prefix)
        for suffix in (".bin", ".idx"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
        self.prefix = prefix
        self.dtype = np.dtype(dtype)
        if self.dtype.type not in _CODES:
            raise ValueError(f"unsupported dtype {dtype}")
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        # Held open for the writer's lifetime -- this class IS the context
        # manager (__exit__ closes it), so SIM115 does not apply.
        self._bin = open(f"{prefix}.bin", "wb")  # noqa: SIM115
        self._lengths: list[int] = []
        self._offset = 0          # running BYTE offset into the .bin
        self._pointers: list[int] = []

    def add(self, tokens) -> None:
        arr = np.asarray(tokens, dtype=self.dtype)
        if arr.ndim != 1:
            raise ValueError(f"expected a 1-D token array, got shape {arr.shape}")
        if arr.size == 0:
            return
        self._pointers.append(self._offset)
        self._lengths.append(arr.size)
        self._bin.write(arr.tobytes(order="C"))
        self._offset += arr.nbytes

    def add_many(self, docs, eos: int | None = None) -> None:
        """Append a whole batch with one concatenate and one write.

        `add` per document costs a numpy allocation, a `tobytes` copy and a
        file write EACH, and a shard is five million documents. That Python
        overhead is a large fraction of a tokenization task -- the tokenizer
        itself runs in Rust across every core, and the loop feeding it does
        not.

        `eos` is appended to each document if given, which is where the
        separator belongs: doing it as `ids + [eos]` in the caller builds a
        second Python list per document for no reason.
        """
        lengths = []
        total = 0
        for d in docs:
            n = len(d) + (1 if eos is not None else 0)
            if n == 0:
                continue
            lengths.append(n)
            total += n
        if not lengths:
            return
        buf = np.empty(total, dtype=self.dtype)
        i = 0
        for d in docs:
            n = len(d)
            if n == 0 and eos is None:
                continue
            if n:
                buf[i:i + n] = d
                i += n
            if eos is not None:
                buf[i] = eos
                i += 1
        off = self._offset
        for n in lengths:
            self._pointers.append(off)
            off += n * self.dtype.itemsize
        self._lengths.extend(lengths)
        self._bin.write(buf.tobytes(order="C"))
        self._offset = off

    @property
    def n_documents(self) -> int:
        return len(self._lengths)

    @property
    def n_tokens(self) -> int:
        return self._offset // self.dtype.itemsize

    def finalize(self) -> None:
        self._bin.close()
        n = len(self._lengths)
        with open(f"{self.prefix}.idx", "wb") as f:
            f.write(_INDEX_HEADER)
            f.write(struct.pack("<Q", 1))                       # version
            f.write(struct.pack("<B", _CODES[self.dtype.type]))
            f.write(struct.pack("<Q", n))                       # sequence count
            # document_count: one document per sequence here, so the boundary
            # array is 0..n. Megatron uses it to group sequences into documents;
            # our corpus has no such grouping, and the reader ignores it.
            f.write(struct.pack("<Q", n + 1))
            f.write(np.array(self._lengths, dtype=np.int32).tobytes(order="C"))
            f.write(np.array(self._pointers, dtype=np.int64).tobytes(order="C"))
            f.write(np.arange(n + 1, dtype=np.int64).tobytes(order="C"))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._bin.closed:
            return False
        if exc_type is None:
            self.finalize()
            return False
        # Do NOT finalize a shard whose writer raised. `finalize` is what
        # creates the .idx, and the .idx is exactly what the tokenizer's resume
        # check reads to decide a shard is done -- so writing one here would
        # mark a TRUNCATED shard complete, and every later run would skip it.
        # `build_splits` would then assemble a corpus quietly missing documents
        # with nothing raising anywhere.
        #
        # A hard kill (Slurm's SIGKILL at the wall clock, a node failure) never
        # reaches this method at all, and is safe for the same reason: no .idx,
        # so the shard is re-tokenized and the stale .bin is truncated by the
        # next open(..., "wb").
        self._bin.close()
        with contextlib.suppress(OSError):
            os.unlink(f"{self.prefix}.bin")
        return False
