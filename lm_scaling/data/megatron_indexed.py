"""Read Megatron-LM indexed datasets (``.bin`` / ``.idx``).

The corpus `data/prepare_fineweb_edu.py` stages is in Megatron's
``IndexedDataset`` format, and this reads it back without converting it: a
memory-mapped window view, which is what `titan_ext/megatron_data.py` hands
torchtitan and what `data/build_splits.py` splits.

Megatron's format rather than a framework's own is what makes "they read
identical tokens in identical order" a fact rather than an assumption --
any backend that can read these files reads the same curriculum, so a curve
mismatch between two of them is attributable.

Format (Megatron ``indexed_dataset.py``, version 1). The ``.idx`` file is::

    9 bytes   b"MMIDIDX\\x00\\x00"
    8 bytes   version, uint64, == 1
    1 byte    dtype code (1 uint8, 2 int8, 3 int16, 4 int32, 5 int64,
                          6 float64, 7 float32, 8 uint16)
    8 bytes   sequence_count, uint64
    8 bytes   document_count, uint64
    int32  * sequence_count   sequence lengths, in elements
    int64  * sequence_count   byte offsets into the .bin
    int64  * document_count   document boundaries

and the ``.bin`` is the concatenated token ids. Everything is memory-mapped, so
opening a 1.8 TB corpus costs nothing and each worker pages in only what it
touches.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_INDEX_HEADER = b"MMIDIDX\x00\x00"

_DTYPES = {
    1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
    5: np.int64, 6: np.float64, 7: np.float32, 8: np.uint16,
}


class MegatronIndexedReader:
    """Memory-mapped reader over one ``.bin`` / ``.idx`` pair."""

    def __init__(self, prefix: str | Path):
        prefix = str(prefix)
        for suffix in (".bin", ".idx"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
        self.prefix = prefix
        self.idx_path = f"{prefix}.idx"
        self.bin_path = f"{prefix}.bin"

        with open(self.idx_path, "rb") as f:
            header = f.read(9)
            if header != _INDEX_HEADER:
                raise ValueError(f"{self.idx_path} is not a Megatron index (bad magic {header!r})")
            (version,) = struct.unpack("<Q", f.read(8))
            if version != 1:
                raise ValueError(f"{self.idx_path}: unsupported index version {version}")
            (code,) = struct.unpack("<B", f.read(1))
            if code not in _DTYPES:
                raise ValueError(f"{self.idx_path}: unknown dtype code {code}")
            self.dtype = _DTYPES[code]
            (self.sequence_count,) = struct.unpack("<Q", f.read(8))
            (self.document_count,) = struct.unpack("<Q", f.read(8))
            offset = f.tell()

        idx = memoryview(np.memmap(self.idx_path, mode="r", order="C"))
        self.sequence_lengths = np.frombuffer(
            idx, dtype=np.int32, count=self.sequence_count, offset=offset
        )
        self.sequence_pointers = np.frombuffer(
            idx, dtype=np.int64, count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )

        self._bin = np.memmap(self.bin_path, dtype=self.dtype, mode="r")

    def __len__(self) -> int:
        return self.sequence_count

    def get(self, i: int) -> np.ndarray:
        """Token ids of document ``i``."""
        length = int(self.sequence_lengths[i])
        # Pointers are BYTE offsets; the memmap is typed, so convert to elements.
        start = int(self.sequence_pointers[i]) // self._bin.dtype.itemsize
        return self._bin[start:start + length]

    @property
    def total_tokens(self) -> int:
        return int(self.sequence_lengths.sum())


class PackedSequenceDataset(Dataset):
    """Fixed-length windows over a Megatron indexed corpus.

    Documents are concatenated in file order and cut into ``seq_len + 1`` token
    windows (the extra token is the shift target). Document boundaries are not
    respected -- that matches how the reference runs were trained, and
    ``intra_doc_masking`` is off throughout this study.

    The concatenation is expressed as a lookup into the flat token stream, so
    no tokens are copied and construction is O(number of documents), not
    O(tokens).
    """

    def __init__(self, prefix: str | Path, seq_len: int, num_tokens: int | None = None):
        self.reader = MegatronIndexedReader(prefix)
        self.seq_len = seq_len

        # `sequence_pointers` is already the cumulative BYTE offset of each
        # document, memory-mapped and sorted, so dividing by the item size gives
        # the cumulative TOKEN offset for free. Computing it instead as
        # `cumsum(sequence_lengths)` allocated a 5.3 GB int64 array and forced a
        # 2.7 GB read of the index at startup, per process, on a 665M-document
        # corpus -- for a number the file already stores.
        r = self.reader
        self._itemsize = r._bin.dtype.itemsize
        n = len(r)
        total = (
            int(r.sequence_pointers[-1]) // self._itemsize + int(r.sequence_lengths[-1])
            if n else 0
        )
        if num_tokens is not None:
            total = min(total, int(num_tokens))

        # Materialise only the document offsets this run can reach.
        #
        # searchsorted over the memory-mapped pointer array costs ~30 random
        # page faults across 5.3 GB of .idx on a shared filesystem -- measured
        # at 0.49 s PER WINDOW, i.e. 7.9 s per batch and 80 minutes of pure data
        # loading for a 611-step run. The offsets themselves are tiny: a run
        # reading 90M tokens touches ~190k documents, 1.5 MB of int64. Copying
        # that prefix into RAM turns each lookup into an in-memory binary search.
        #
        # Bounded by `total`, so an uncapped dataset over the full 665.6M
        # documents still only materialises what it will actually index.
        last_doc = int(np.searchsorted(
            r.sequence_pointers, total * self._itemsize, side="right"))
        last_doc = min(n, last_doc + 1)
        self.starts = np.array(r.sequence_pointers[:last_doc], dtype=np.int64)
        self._n_docs = last_doc
        self.total = total
        self.n = max(0, (total - 1) // seq_len)
        if self.n == 0:
            raise ValueError(
                f"{prefix}: {total} tokens is too few for even one window of "
                f"{seq_len + 1}"
            )

    def __len__(self) -> int:
        return self.n

    def _doc_start(self, doc: int) -> int:
        """Cumulative token offset at which document `doc` begins."""
        return int(self.starts[doc]) // self._itemsize

    def __getitem__(self, i: int) -> dict:
        start = i * self.seq_len
        want = self.seq_len + 1
        out = np.empty(want, dtype=np.int64)

        filled = 0
        # searchsorted over the memmapped byte offsets: `start` is in tokens, so
        # compare against tokens by scaling the target rather than the array.
        doc = int(np.searchsorted(self.starts, start * self._itemsize, side="right")) - 1
        doc = max(0, doc)
        pos = start - self._doc_start(doc)
        while filled < want and doc < self._n_docs:
            tokens = self.reader.get(doc)
            take = min(want - filled, len(tokens) - pos)
            if take > 0:
                out[filled:filled + take] = tokens[pos:pos + take]
                filled += take
            doc += 1
            pos = 0
        if filled < want:
            # Only the final window can run short; wrap to the start so every
            # window is full length rather than dropping the tail.
            out[filled:] = np.asarray(self.reader.get(0)[: want - filled])

        return {"input_ids": torch.from_numpy(out)}


def build_megatron_dataset(path: str | Path, seq_len: int, num_tokens: int | None = None):
    """Build a :class:`PackedSequenceDataset`, accepting a prefix or a directory.

    A directory is resolved to its single ``.bin``; if it holds several, the
    caller must name one, because silently picking the first would make the
    corpus depend on filesystem ordering.
    """
    path = Path(path)
    if path.is_dir():
        bins = sorted(path.glob("*.bin"))
        if not bins:
            raise FileNotFoundError(f"no .bin under {path}")
        if len(bins) > 1:
            raise ValueError(
                f"{path} holds {len(bins)} .bin files; name one explicitly "
                f"(e.g. {bins[0].with_suffix('')})"
            )
        path = bins[0].with_suffix("")
    return PackedSequenceDataset(path, seq_len, num_tokens)
