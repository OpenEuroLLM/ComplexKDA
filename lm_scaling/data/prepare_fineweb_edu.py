"""Stage FineWeb-Edu `sample-100BT` under the Llama-2 tokenizer as Megatron
`.bin`/`.idx` -- the corpus `config/experiments/fwedu_1p3B.yaml` trains on.

    # 1. JUWELS login node -- login nodes have the network -- into /p:
    HF_HOME=$HF python3 lm_scaling/data/prepare_fineweb_edu.py download --out $P
    # 2. JUWELS CPU nodes, one array task per parquet shard:
    sbatch --array=0-139%40 lm_scaling/slurm/fwedu_tokenize_juwels.sbatch
    # 3. JUPITER login node, which mounts /p and /e both: the shards to where
    #    training reads them, then the splits -- a byte copy and an index
    #    rebuild on one core:
    rsync -a $P/tokenized_llama2/ $E/tokenized_llama2/
    python3 lm_scaling/data/prepare_fineweb_edu.py build --out $E
    lm_scaling/container_run lm_scaling/data/verify_corpus.py fwedu_1p3B

Each inside its cluster's titan image (`apptainer exec`, PYTHONPATH at the
checkout). None of it needs a GPU, and JUPITER has no nodes without one, hence
JUWELS for the tokenizing -- about two minutes a shard on a large node.

The target is Gated DeltaNet's corpus (arXiv:2412.06464) as its code builds
it, and three choices decide whether this is that corpus.

**The documents.** HuggingFaceFW/fineweb-edu `sample/100BT` -- 140 shards,
97,270,686 documents -- at the revision scripts/mn5_stage_fineweb_edu.py pins,
so JUPITER and MareNostrum train on the same text.

**The tokenizer, and how a document ends.** Llama-2's, with `</s>` appended to
every document and no BOS in front. That is `Tokenizer.encode(text)` in GDN's
`lit_gpt/tokenizer.py`, whose defaults are bos=False, eos=True, called the way
Samba's `prepare_slimpajama.py` -- where GDN's data loader comes from -- calls
it. It is also our NeoX corpora's convention: `<|endoftext|>` after every
document and nothing before, on all of 1,000 Nemotron-CC documents sampled
from train and held-out. GDN's tokenizer runs SentencePiece; the HF fast
tokenizer used here gives the same ids on 600 FineWeb-Edu documents, and it is
the one that parallelises. Llama-2 splits every number into single digits.

The tokenizer's files are copied beside the corpus, into `tokenizer/`, and
that directory is what the compute nodes load. Not the Hub id: offline,
transformers 5.3 refuses a Hub id whose repo has no config.json -- this one
has none -- however complete the cache is, and every tokenize task would die
on it. It also leaves the tokenizer next to the data it made.

**The order.** The reader takes the corpus front to back, and 100BT is most of
its ~116BT, so the order IS the curriculum, and its tail is what the cosine
anneal fits. The train split interleaves the shards in blocks of documents,
drawn in a seeded random permutation of blocks, which holds the mix of all 140
shards constant along the whole run. Round-robin would not: the shards range
from 0.2 to 2.2 GB, so the small ones would run out early and the run would
end on the large ones.

The held-out split takes the last documents of EVERY shard, in proportion to
its size, shuffled. A validation rank scores one contiguous block of it, and
this makes any block a sample of the whole corpus rather than of one shard.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

REPO = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SUBDIR = "sample/100BT"
N_SHARDS = 140

# Llama-2's tokenizer from an ungated repo: the same vocabulary as the
# Llama-2-7b tokenizer and the same encodings on 600 FineWeb-Edu documents.
TOKENIZER = "hf-internal-testing/llama-tokenizer"
TOKENIZER_REVISION = "d02ad6cb9dd2c2296a6332199fa2fdca5938fef0"
TOKENIZER_DIR = "tokenizer"             # under --out; see the module docstring
VOCAB = 32000
EOS = 2                                 # </s>, after every document
HELLO = [15043, 3186, 29889]            # "Hello world." under Llama-2, no specials

# 2,560 validation windows of 4,097 tokens are 10.5M. The rest is room for
# every world size to take equal contiguous blocks (megatron_validate).
VALID_TOKENS = 16_000_000
BLOCK_DOCS = 512                        # ~0.6M tokens, about one step
SEED = 1234


def shard_files(out: Path) -> list[Path]:
    files = sorted((out / SUBDIR).glob("*.parquet"))
    if len(files) != N_SHARDS:
        raise SystemExit(f"{len(files)} parquet shards under {out / SUBDIR}, "
                         f"expected {N_SHARDS}; run `download` again")
    return files


def check_tokenizer(tok):
    """Refuse any tokenizer that is not the Llama-2 one, judged by what LOADED.

    The name is not enough: the tokenizer fla-hub ships beside its 1.3B
    checkpoints is also a 32,000-token LlamaTokenizer, and it is Mistral's.
    """
    got = list(tok(["Hello world."], add_special_tokens=False)["input_ids"][0])
    if len(tok) != VOCAB or got != HELLO or tok.eos_token_id != EOS:
        raise SystemExit(
            f"{TOKENIZER} did not load as Llama-2: {len(tok)} tokens, eos "
            f"{tok.eos_token_id}, 'Hello world.' -> {got}; expected {VOCAB}, "
            f"{EOS} and {HELLO}")
    return tok


def load_tokenizer(out: Path):
    """The tokenizer `download` copied beside the corpus, checked."""
    from transformers import AutoTokenizer

    path = out / TOKENIZER_DIR
    if not (path / "tokenizer.json").exists():
        raise SystemExit(f"no tokenizer at {path}; run `download` on a login node")
    return check_tokenizer(AutoTokenizer.from_pretrained(str(path)))


def download(out: Path, workers: int) -> None:
    from huggingface_hub import HfApi, snapshot_download

    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                      allow_patterns=[f"{SUBDIR}/*.parquet"], local_dir=str(out),
                      max_workers=workers)
    # Every size against the Hub's own record. A shard a dropped connection
    # truncated is only noticed otherwise when its tokenize task dies on it.
    want = {Path(f.path).name: f.size for f in HfApi().list_repo_tree(
        REPO, path_in_repo=SUBDIR, repo_type="dataset", revision=REVISION)
        if f.path.endswith(".parquet")}
    files = shard_files(out)
    bad = [f.name for f in files if f.stat().st_size != want.get(f.name)]
    if bad:
        raise SystemExit(f"{len(bad)} shards differ in size from the Hub's "
                         f"record: {bad[:5]}")
    print(f"{len(files)} shards, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB, "
          f"sizes match the Hub, {(time.perf_counter() - t0) / 60:.1f} min")

    # The tokenizer too, as files beside the corpus: the tokenize tasks run on
    # compute nodes, which cannot reach the Hub.
    snapshot_download(repo_id=TOKENIZER, revision=TOKENIZER_REVISION,
                      local_dir=str(out / TOKENIZER_DIR))
    load_tokenizer(out)
    print(f"{TOKENIZER}@{TOKENIZER_REVISION[:8]} in {out / TOKENIZER_DIR}")


def tokenize_shard(src: Path, prefix: Path, tok) -> tuple[int, int]:
    """One parquet shard to one `.bin`/`.idx`: each document's ids, then EOS.

    `add_special_tokens=False` because the Llama tokenizer would otherwise put
    a BOS in front -- GDN's corpus has none -- and the EOS is the writer's to
    append, once per document.
    """
    import pyarrow.parquet as pq

    from lm_scaling.data.megatron_writer import MegatronIndexedWriter

    with MegatronIndexedWriter(prefix) as w:
        # 20k rows a batch: the fast tokenizer parallelises ACROSS the batch.
        for batch in pq.ParquetFile(src).iter_batches(batch_size=20_000,
                                                      columns=["text"]):
            texts = [t for t in batch.column("text").to_pylist() if t]
            if texts:
                w.add_many(tok(texts, add_special_tokens=False)["input_ids"], EOS)
        return w.n_documents, w.n_tokens


def tokenize(out: Path, shard: int) -> None:
    src = shard_files(out)[shard]
    prefix = out / "tokenized_llama2" / src.stem
    if Path(f"{prefix}.idx").exists():
        # The writer only writes an .idx for a shard it finished, so a
        # resubmitted array skips exactly the shards that are done.
        print(f"{src.stem}: already tokenized")
        return
    tok = load_tokenizer(out)
    t0 = time.perf_counter()
    docs, toks = tokenize_shard(src, prefix, tok)
    print(f"{src.stem}: {docs:,} docs, {toks / 1e9:.3f}BT in "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def plan_split(lengths: list[np.ndarray], valid_tokens: int, block: int,
               seed: int) -> tuple[list[int], np.ndarray]:
    """Where each shard's held-out tail begins, and the order of train blocks.

    `cuts[s]`: shard s's documents from this index on are held out, about its
    proportional share of `valid_tokens`, and at least one of them. `order`:
    one entry per train block of `block` documents, naming the shard it comes
    from, in a seeded random permutation -- so every stretch of the train
    split mixes the shards in the proportions of the whole.
    """
    totals = np.array([float(l.sum()) for l in lengths])
    cuts = []
    for l, share in zip(lengths, valid_tokens * totals / totals.sum()):
        from_end = np.cumsum(l[::-1])
        held = int(np.searchsorted(from_end, share)) + 1
        cuts.append(len(l) - min(held, len(l) - 1))
    blocks = [-(-c // block) for c in cuts]
    order = np.repeat(np.arange(len(lengths)), blocks)
    np.random.default_rng(seed).shuffle(order)
    return cuts, order


def build(prefixes: list[Path], out: Path, valid_tokens: int = VALID_TOKENS,
          block: int = BLOCK_DOCS, seed: int = SEED) -> None:
    from megatron_indexed import MegatronIndexedReader
    from lm_scaling.data.megatron_writer import MegatronIndexedWriter

    if (out / "train.idx").exists() and (out / "valid.idx").exists():
        print(f"{out}: already built")
        return
    readers = [MegatronIndexedReader(p) for p in prefixes]
    lengths = [np.asarray(r.sequence_lengths, dtype=np.int64) for r in readers]
    cuts, order = plan_split(lengths, valid_tokens, block, seed)

    n_docs = sum(len(l) for l in lengths)
    held = [(s, i) for s, c in enumerate(cuts) for i in range(c, len(lengths[s]))]
    train_tokens = sum(int(l[:c].sum()) for l, c in zip(lengths, cuts))
    print(f"{len(readers)} shards, {n_docs:,} documents -> train "
          f"{train_tokens:,} tokens, valid {len(held):,} documents")

    # Every document lands exactly once, checked INSIDE each writer: the
    # writer deletes its .bin and writes no .idx when its block raises, so a
    # corpus that lost documents is never left looking finished.
    out.mkdir(parents=True, exist_ok=True)
    with MegatronIndexedWriter(out / "valid") as w:
        for j in np.random.default_rng(seed + 1).permutation(len(held)):
            s, i = held[j]
            w.add(readers[s].get(i))
        valid_docs, valid_toks = w.n_documents, w.n_tokens
        if valid_docs != len(held):
            raise SystemExit(f"valid holds {valid_docs} of {len(held)} documents")

    cursor = [0] * len(readers)
    t0 = time.perf_counter()
    with MegatronIndexedWriter(out / "train") as w:
        for n, s in enumerate(order, 1):
            lo = cursor[s]
            hi = cursor[s] = min(lo + block, cuts[s])
            w.add_many([readers[s].get(i) for i in range(lo, hi)])
            if n % 20_000 == 0:
                print(f"  {n:,}/{len(order):,} blocks, {w.n_tokens / 1e9:.1f}BT, "
                      f"{(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        if cursor != cuts or w.n_documents + valid_docs != n_docs:
            raise SystemExit(f"train holds {w.n_documents:,} + valid {valid_docs:,} "
                             f"of {n_docs:,} documents")
        train_docs, train_toks = w.n_documents, w.n_tokens
    print(f"train: {train_docs:,} documents, {train_toks:,} tokens\n"
          f"valid: {valid_docs:,} documents, {valid_toks:,} tokens\n"
          "record both token counts in config/data/fineweb_edu_llama2.yaml")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["download", "tokenize", "build"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, help="tokenize: the parquet shard, 0-139")
    ap.add_argument("--workers", type=int, default=16,
                    help="download: files fetched concurrently")
    a = ap.parse_args()

    # Before anything imports transformers, which reads it at import. One
    # shard per process wants the Rust tokenizer's threads -- off, a shard
    # tokenizes on one core of 288 (see prepare_paper_corpus.py).
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    if a.action == "download":
        download(a.out, a.workers)
    elif a.action == "tokenize":
        if a.shard is None:
            ap.error("tokenize needs --shard")
        tokenize(a.out, a.shard)
    else:
        prefixes = [a.out / "tokenized_llama2" / f.stem for f in shard_files(a.out)]
        missing = [p.name for p in prefixes if not Path(f"{p}.idx").exists()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(prefixes)} shards are not "
                             f"tokenized ({missing[:5]}); a partial corpus is a "
                             "different corpus")
        build(prefixes, a.out / "splits")


if __name__ == "__main__":
    main()
