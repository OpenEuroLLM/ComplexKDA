"""A synthetic dataloader, so torchtitan can be timed without staged data.

The speed comparison across the three backends has to isolate the training
step. The speed probes of this study and the Megatron one both feed
random token ids for exactly that reason: a run that waits on GPFS measures the
filesystem, and the three frameworks read different corpora through different
readers, so any data-path difference lands directly in the throughput number
and cannot be separated from the thing being measured.

Everything else stays torchtitan's -- Trainer, FSDP2 sharding, activation
checkpointing, `torch.compile`, loss and metrics -- because that is what the
comparison is about.

This is for benchmarking only. Loss-matching runs read the real corpus.
"""

from __future__ import annotations

import torch
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader


class _RandomTokens(IterableDataset):
    """Endless stream of uniform token ids.

    Uniform ids make the loss meaningless (it sits at ln(vocab) forever) and
    that is deliberate: this dataset must never be mistaken for a training run.
    The step time, which is what it exists to measure, does not depend on the
    values.
    """

    def __init__(self, seq_len: int, vocab_size: int, seed: int = 0):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            # seq_len + 1 so the trainer can split inputs/labels by shifting.
            ids = torch.randint(0, self.vocab_size, (self.seq_len + 1,), generator=g)
            yield {"input": ids[:-1]}, ids[1:]


def build_synthetic_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer=None,
    job_config=None,
    infinite: bool = True,
    **kwargs,
) -> ParallelAwareDataloader:
    """Drop-in for a train spec's ``build_dataloader_fn``."""
    seq_len = job_config.training.seq_len
    batch_size = job_config.training.local_batch_size
    vocab = getattr(job_config.model, "vocab_size", None) or 50304
    ds = _RandomTokens(seq_len, vocab, seed=dp_rank)
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
