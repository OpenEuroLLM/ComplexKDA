"""Round-trip the Megatron ``.bin``/``.idx`` reader.

The corpus staged on JUPITER is 1.8 TB of Megatron-indexed tokens, so a
mis-read of the header would not announce itself — it would just train on
garbage that still looks like token ids. These tests write the format from the
spec and read it back, so the parser is checked against a known payload rather
than against itself.
"""

import struct

import numpy as np
import pytest
import torch

from lm_scaling.data.megatron_indexed import (
    MegatronIndexedReader,
    PackedSequenceDataset,
    build_megatron_dataset,
)

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def _write_indexed(tmp_path, docs, dtype=np.int32, code=4, name="data"):
    """Write documents in Megatron IndexedDataset v1 format."""
    prefix = tmp_path / name
    with open(f"{prefix}.bin", "wb") as f:
        for d in docs:
            f.write(np.asarray(d, dtype=dtype).tobytes(order="C"))

    lengths = np.array([len(d) for d in docs], dtype=np.int32)
    itemsize = np.dtype(dtype).itemsize
    pointers = np.zeros(len(docs), dtype=np.int64)
    running = 0
    for i, n in enumerate(lengths):
        pointers[i] = running
        running += int(n) * itemsize
    doc_idx = np.arange(len(docs) + 1, dtype=np.int64)

    with open(f"{prefix}.idx", "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", code))
        f.write(struct.pack("<Q", len(docs)))
        f.write(struct.pack("<Q", len(doc_idx)))
        f.write(lengths.tobytes(order="C"))
        f.write(pointers.tobytes(order="C"))
        f.write(doc_idx.tobytes(order="C"))
    return prefix


def test_reader_returns_each_document_verbatim(tmp_path):
    docs = [[1, 2, 3], [10, 11], [100, 101, 102, 103], [7]]
    prefix = _write_indexed(tmp_path, docs)

    r = MegatronIndexedReader(prefix)
    assert len(r) == len(docs)
    assert r.total_tokens == sum(len(d) for d in docs)
    for i, d in enumerate(docs):
        np.testing.assert_array_equal(r.get(i), np.asarray(d, dtype=np.int32))


@pytest.mark.parametrize("dtype,code", [(np.uint16, 8), (np.int32, 4), (np.int64, 5)])
def test_reader_handles_each_token_dtype(tmp_path, dtype, code):
    """Pointers are byte offsets; a wrong itemsize silently shifts every doc."""
    docs = [[1, 2, 3], [4, 5, 6, 7], [8]]
    prefix = _write_indexed(tmp_path, docs, dtype=dtype, code=code, name=f"d{code}")
    r = MegatronIndexedReader(prefix)
    for i, d in enumerate(docs):
        np.testing.assert_array_equal(r.get(i), np.asarray(d, dtype=dtype))


def test_reader_rejects_a_foreign_file(tmp_path):
    (tmp_path / "bad.idx").write_bytes(b"NOTANIDX" + b"\x00" * 64)
    (tmp_path / "bad.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="not a Megatron index"):
        MegatronIndexedReader(tmp_path / "bad")


def test_packing_concatenates_documents_in_order(tmp_path):
    """Windows must tile the flat token stream with no gaps or repeats."""
    docs = [list(range(0, 10)), list(range(10, 25)), list(range(25, 40))]
    prefix = _write_indexed(tmp_path, docs)

    seq_len = 8
    ds = PackedSequenceDataset(prefix, seq_len=seq_len)
    flat = np.concatenate([np.asarray(d) for d in docs])

    assert len(ds) == (len(flat) - 1) // seq_len
    for i in range(len(ds)):
        got = ds[i]["input_ids"]
        assert got.shape == (seq_len + 1,)
        expected = flat[i * seq_len: i * seq_len + seq_len + 1]
        np.testing.assert_array_equal(got.numpy(), expected)


def test_window_spanning_a_document_boundary(tmp_path):
    """The failure mode worth pinning: a window that starts mid-document."""
    docs = [[1, 2, 3], [4, 5, 6, 7, 8, 9]]
    prefix = _write_indexed(tmp_path, docs)
    ds = PackedSequenceDataset(prefix, seq_len=4)
    np.testing.assert_array_equal(ds[0]["input_ids"].numpy(), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(ds[1]["input_ids"].numpy(), [5, 6, 7, 8, 9])


def test_num_tokens_truncates(tmp_path):
    docs = [list(range(100))]
    prefix = _write_indexed(tmp_path, docs)
    full = PackedSequenceDataset(prefix, seq_len=10)
    short = PackedSequenceDataset(prefix, seq_len=10, num_tokens=50)
    assert len(short) < len(full)
    assert len(short) == (50 - 1) // 10


def test_build_from_directory_refuses_an_ambiguous_choice(tmp_path):
    """Picking the first .bin would make the corpus depend on file ordering."""
    _write_indexed(tmp_path, [list(range(20))], name="a")
    ds = build_megatron_dataset(tmp_path, seq_len=4)
    assert isinstance(ds, PackedSequenceDataset)

    _write_indexed(tmp_path, [list(range(20))], name="b")
    with pytest.raises(ValueError, match="holds 2 .bin files"):
        build_megatron_dataset(tmp_path, seq_len=4)


def test_dataset_output_matches_what_the_engine_expects(tmp_path):
    """`_move_to_device` slices batch["input_ids"] and needs int64 on CPU."""
    prefix = _write_indexed(tmp_path, [list(range(100))])
    item = PackedSequenceDataset(prefix, seq_len=8)[0]
    assert set(item) == {"input_ids"}
    assert item["input_ids"].dtype == torch.int64
