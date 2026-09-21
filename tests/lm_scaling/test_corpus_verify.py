"""The corpus a config names must be the corpus it describes.

`prefix` is a string, and a string can point at a corpus that has moved, been
rebuilt with a different tokenizer, or been truncated by a copy that died --
and all three still train. The token count in the .idx header is the cheapest
thing that tells them apart, so the config records it and the check is hard.

Verified on JUPITER (2026-09-09):

    train  318,592,619,699 tokens, 665,613,726 docs, uint16, 77,762,416 windows
    valid  106,105,926,671 tokens, 221,617,471 docs, uint16, 25,898,444 windows
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))
sys.path.insert(0, str(_ROOT / "lm_scaling" / "data"))

import verify_corpus as V  # noqa: E402


def _write(prefix: Path, *, tokens: int, docs: int, code: int = 8,
           magic: bytes = V.MAGIC) -> str:
    itemsize = V.ITEMSIZE.get(code, 2)
    with (prefix.with_suffix(".idx")).open("wb") as f:
        f.write(magic)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", code))
        f.write(struct.pack("<Q", docs))
        f.write(struct.pack("<Q", docs + 1))
    (prefix.with_suffix(".bin")).write_bytes(b"\0" * (tokens * itemsize))
    return str(prefix)


def test_the_header_is_read_the_way_megatron_wrote_it(tmp_path):
    p = _write(tmp_path / "c", tokens=1000, docs=7)
    head = V.read_header(p)
    assert head["tokens"] == 1000
    # doc_idx carries a trailing sentinel, so the second count is docs + 1.
    # Reporting THAT as the document count is off by one in a way that reads
    # like a real difference between two corpora.
    assert head["documents"] == 7 and head["doc_idx_len"] == 8
    assert head["dtype"] == "uint16"


def test_a_truncated_corpus_is_caught(tmp_path):
    p = _write(tmp_path / "c", tokens=900, docs=7)
    problems = V.check(p, expect_tokens=1000, seq_len=4096, need_windows=0,
                       label="train")
    assert problems and "has moved, been rebuilt or been truncated" in problems[0]


def test_a_corpus_from_the_wrong_tokenizer_is_caught(tmp_path):
    """uint16 is not decoration: it caps the vocabulary at 65,536. The 256k
    pretokenization sits next to ours on JUPITER, is int32, and would be read
    as garbage pairs of tokens with no error anywhere."""
    p = _write(tmp_path / "c", tokens=1000, docs=7, code=4)  # int32
    problems = V.check(p, expect_tokens=1000, seq_len=4096, need_windows=0,
                       label="train")
    assert any("not uint16" in x for x in problems)


def test_a_split_too_small_for_the_evaluation_is_caught(tmp_path):
    p = _write(tmp_path / "c", tokens=4097 * 10, docs=2)
    problems = V.check(p, expect_tokens=4097 * 10, seq_len=4096,
                       need_windows=2560, label="valid")
    assert any("fewer than the 2,560" in x for x in problems)


def test_a_missing_or_unreadable_corpus_says_which(tmp_path):
    problems = V.check(str(tmp_path / "nope"), 0, 4096, 1, "valid")
    assert problems and "does not exist" in problems[0]


def test_something_that_is_not_a_megatron_index_is_caught(tmp_path):
    p = _write(tmp_path / "c", tokens=1000, docs=7, magic=b"NOTANIDX\x00")
    with pytest.raises(V.CorpusError, match="not a Megatron indexed dataset"):
        V.read_header(p)


def test_the_config_records_what_the_staged_corpus_reported():
    """Numbers read off the staged splits' own headers, not from a note. They
    switch on two checks: `verify_corpus.py` compares them with the files, and
    `plan.check` refuses a run longer than its corpus -- the reader wraps
    around and repeats it otherwise, silently training a second epoch under a
    single-epoch label. If the corpus is ever rebuilt these move together or
    the check fails."""
    import yaml

    d = yaml.safe_load((_ROOT / "lm_scaling" / "config" / "data" /
                        "fineweb_edu_llama2.yaml").read_text())
    assert d["train_tokens"] == 114_841_055_675
    assert d["valid_tokens"] == 16_367_254
    assert d["vocab_size"] == 32_000 and d["seq_len"] == 4096
    # 100BT is one pass with room to spare, which is what makes the data order
    # a curriculum rather than a repeat.
    assert d["train_tokens"] > 100e9
    # The evaluation is a small share of the split it is drawn from.
    assert d["eval_sequences"] * (d["seq_len"] + 1) < d["valid_tokens"]
