"""The writer and the reader must agree byte for byte.

They are two independent implementations of one on-disk format, written months
apart. If they drift, a corpus tokenized here becomes unreadable by the
trainer -- or worse, readable but misaligned, which would show up as a loss
curve nobody can explain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lm_scaling.data.megatron_indexed import MegatronIndexedReader  # noqa: E402
from lm_scaling.data.megatron_writer import MegatronIndexedWriter  # noqa: E402


def test_round_trip_preserves_every_document(tmp_path):
    rng = np.random.default_rng(0)
    docs = [rng.integers(0, 50257, size=n, dtype=np.uint16)
            for n in (1, 7, 2048, 4095, 300)]

    prefix = tmp_path / "corpus"
    with MegatronIndexedWriter(prefix) as w:
        for d in docs:
            w.add(d)
        assert w.n_documents == len(docs)
        assert w.n_tokens == sum(len(d) for d in docs)

    r = MegatronIndexedReader(prefix)
    assert len(r) == len(docs)
    for i, d in enumerate(docs):
        np.testing.assert_array_equal(r.get(i), d, err_msg=f"document {i}")


def test_offsets_are_bytes_not_elements(tmp_path):
    """The pointer array is a BYTE offset; reading it as elements halves it.

    uint16 makes the two differ by exactly 2x, which is the kind of bug that
    still returns plausible-looking tokens.
    """
    prefix = tmp_path / "c"
    with MegatronIndexedWriter(prefix) as w:
        w.add(np.arange(10, dtype=np.uint16))
        w.add(np.arange(100, 105, dtype=np.uint16))

    r = MegatronIndexedReader(prefix)
    assert int(r.sequence_pointers[1]) == 20, "second document starts at byte 20"
    np.testing.assert_array_equal(r.get(1), np.arange(100, 105))


def test_empty_documents_are_skipped(tmp_path):
    """A zero-length document would give two identical offsets and a 0 length,
    which the reader can express but nothing downstream expects."""
    prefix = tmp_path / "c"
    with MegatronIndexedWriter(prefix) as w:
        w.add(np.array([1, 2, 3], dtype=np.uint16))
        w.add(np.array([], dtype=np.uint16))
        w.add(np.array([4, 5], dtype=np.uint16))
    r = MegatronIndexedReader(prefix)
    assert len(r) == 2
    np.testing.assert_array_equal(r.get(1), [4, 5])


def test_uint16_is_enough_for_gpt2(tmp_path):
    """50257 < 65536, so the corpus is half the size int32 would make it."""
    prefix = tmp_path / "c"
    with MegatronIndexedWriter(prefix) as w:
        w.add(np.array([50256], dtype=np.uint16))
    r = MegatronIndexedReader(prefix)
    assert r.dtype == np.uint16
    assert int(r.get(0)[0]) == 50256


def test_split_builder_partitions_documents_exactly(tmp_path):
    """train + valid must be the corpus, with no document lost or duplicated.

    The split walks backwards to land on a document boundary, so an off-by-one
    would either drop a document between the two halves or put it in both --
    and a duplicated document in the validation set silently inflates the
    reported number, which is the one thing the anchor exists to compare.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from lm_scaling.data.build_splits import build

    rng = np.random.default_rng(0)
    tokdir = tmp_path / "tok"
    tokdir.mkdir()
    all_docs = []
    for shard in range(3):
        with MegatronIndexedWriter(tokdir / f"{shard:08d}") as w:
            for _ in range(20):
                d = rng.integers(0, 50257, size=int(rng.integers(5, 60)),
                                 dtype=np.uint16)
                all_docs.append(d)
                w.add(d)

    out = tmp_path / "splits"
    build(tokdir, out, n_shards=3, valid_tokens=200)

    train = MegatronIndexedReader(out / "train")
    valid = MegatronIndexedReader(out / "valid")
    assert len(train) + len(valid) == len(all_docs), "documents lost or duplicated"
    assert int(valid.total_tokens) >= 200, "validation split is short"

    rebuilt = [train.get(i) for i in range(len(train))] + \
              [valid.get(i) for i in range(len(valid))]
    assert len(rebuilt) == len(all_docs)
    for got, want in zip(rebuilt, all_docs):
        np.testing.assert_array_equal(got, want)


def test_split_refuses_an_unfinished_tokenization(tmp_path):
    """Building from a partial array job would silently train on less data."""
    import sys

    import pytest

    sys.path.insert(0, str(ROOT))
    from lm_scaling.data.build_splits import build

    tokdir = tmp_path / "tok"
    tokdir.mkdir()
    with MegatronIndexedWriter(tokdir / "00000000") as w:
        w.add(np.arange(50, dtype=np.uint16))

    with pytest.raises(SystemExit, match="not finished"):
        build(tokdir, tmp_path / "o", n_shards=15, valid_tokens=10)


def test_a_writer_that_raises_leaves_no_index(tmp_path):
    """A truncated shard must not look finished.

    The tokenizer's resume check is `does the .idx exist`. Finalizing on the
    way out of a failed `with` would write one for a partial shard, every later
    run would skip it, and build_splits would assemble a corpus quietly missing
    documents -- with nothing raising anywhere along the way.
    """
    prefix = tmp_path / "shard"
    with pytest.raises(RuntimeError), MegatronIndexedWriter(prefix) as w:
        w.add(np.arange(10, dtype=np.uint16))
        raise RuntimeError("parquet went bad half way through")

    assert not (tmp_path / "shard.idx").exists(), "the shard looks complete"
    assert not (tmp_path / "shard.bin").exists(), "a stale partial .bin remains"


def test_a_clean_exit_still_finalizes(tmp_path):
    prefix = tmp_path / "ok"
    with MegatronIndexedWriter(prefix) as w:
        w.add(np.arange(10, dtype=np.uint16))
    assert (tmp_path / "ok.idx").exists()
    assert len(MegatronIndexedReader(prefix)) == 1


def test_a_killed_shard_is_retokenized_not_trusted(tmp_path):
    """The hard-kill path: no __exit__ runs, so there is a .bin and no .idx.

    Slurm's wall-clock SIGKILL and a node failure both land here, which is the
    shape to expect during the pre-maintenance drain. Re-opening must truncate
    the stale .bin rather than append to it.
    """
    prefix = tmp_path / "killed"
    w = MegatronIndexedWriter(prefix)
    w.add(np.arange(100, dtype=np.uint16))
    del w  # no finalize, as if the process vanished

    assert not (tmp_path / "killed.idx").exists()
    with MegatronIndexedWriter(prefix) as w2:
        w2.add(np.arange(7, dtype=np.uint16))
    r = MegatronIndexedReader(prefix)
    assert len(r) == 1
    np.testing.assert_array_equal(r.get(0), np.arange(7))


def test_add_many_matches_add_document_for_document(tmp_path):
    """The bulk path is an optimisation; it must not be a second format.

    `add` per document costs an allocation, a copy and a write EACH, and a
    shard is five million documents. `add_many` does one of each per batch --
    but only if it produces byte-identical output, which is what this checks
    against the implementation it replaces.
    """
    rng = np.random.default_rng(3)
    docs = [rng.integers(0, 50257, size=n, dtype=np.uint16).tolist()
            for n in (5, 0, 2048, 1, 300)]
    eos = 50256

    one = tmp_path / "one"
    with MegatronIndexedWriter(one) as w:
        for d in docs:
            w.add(np.array(d + [eos], dtype=np.uint16))
    many = tmp_path / "many"
    with MegatronIndexedWriter(many) as w:
        w.add_many(docs, eos)

    assert (tmp_path / "one.bin").read_bytes() == (tmp_path / "many.bin").read_bytes()
    assert (tmp_path / "one.idx").read_bytes() == (tmp_path / "many.idx").read_bytes()
    r = MegatronIndexedReader(many)
    assert len(r) == len(docs)
    for i, d in enumerate(docs):
        np.testing.assert_array_equal(r.get(i), np.array(d + [eos], dtype=np.uint16))


def test_add_many_in_several_calls_matches_one_call(tmp_path):
    """Batching is the caller's choice, so the boundary must not show."""
    rng = np.random.default_rng(4)
    docs = [rng.integers(0, 50257, size=int(n), dtype=np.uint16).tolist()
            for n in rng.integers(1, 500, size=20)]

    whole = tmp_path / "whole"
    with MegatronIndexedWriter(whole) as w:
        w.add_many(docs, 50256)
    split = tmp_path / "split"
    with MegatronIndexedWriter(split) as w:
        w.add_many(docs[:7], 50256)
        w.add_many(docs[7:], 50256)
    assert (tmp_path / "whole.bin").read_bytes() == (tmp_path / "split.bin").read_bytes()
    assert (tmp_path / "whole.idx").read_bytes() == (tmp_path / "split.idx").read_bytes()


def test_add_many_without_eos_skips_empty_documents(tmp_path):
    """Matching `add`, which refuses a zero-length document."""
    prefix = tmp_path / "noeos"
    with MegatronIndexedWriter(prefix) as w:
        w.add_many([[1, 2], [], [3]])
    r = MegatronIndexedReader(prefix)
    assert len(r) == 2
    np.testing.assert_array_equal(r.get(1), np.array([3], dtype=np.uint16))
