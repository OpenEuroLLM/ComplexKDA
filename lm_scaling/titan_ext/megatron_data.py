"""Feed torchtitan the same Megatron `.bin`/`.idx` corpus the other two read.

The speed comparison runs on synthetic tokens (`synthetic.py`) so that nothing
measures GPFS. Loss matching needs the opposite: byte-identical inputs across
`lm/`, Megatron and torchtitan, because a difference in the *data* is
indistinguishable from a difference in the *model* once you are staring at two
loss curves that do not overlap.

So this wraps `lm_scaling/data/megatron_indexed.PackedSequenceDataset` -- already
the reader `lm/` uses, and reading the same files Megatron reads natively --
in torchtitan's dataloader contract. The packing (documents concatenated,
sliced into `seq_len + 1` windows, window `i` starting at `i * seq_len`) is
therefore identical by construction rather than by agreement, and each rank
takes a strided subset so the union over ranks is the whole corpus with no
overlap.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    # Only for the return annotation below; the runtime import stays inside
    # build_megatron_dataloader so this module imports without torchtitan.
    from torchtitan.components.dataloader import ParallelAwareDataloader

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


_MASK64 = 0xFFFFFFFFFFFFFFFF


def _mix(x: int, k: int) -> int:
    """SplitMix64's finalizer: a round function that is cheap and deterministic.

    Written out rather than taken from `random`, because every rank and every
    worker must agree on it exactly and must keep agreeing after a restart.
    """
    x = (x + 0x9E3779B97F4A7C15 + k) & _MASK64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


class _Permutation:
    """A seeded permutation of ``range(n)``, computed rather than stored.

    WHY NOT AN ARRAY. Megatron materialises a shuffle index; at 4096 tokens the
    training corpus is 77.8M windows, so int32 is 311 MB -- per rank and per
    worker, which is 28 copies on a four-GPU node. A Feistel network is a
    bijection on the same domain at O(1) time and no memory, and being a pure
    function of the index it survives restarts for free: the dataloader's state
    is still just a position, and the resume logic below is untouched.

    HOW. Four Feistel rounds over ``2*h`` bits, where ``2^(2h) >= n``, then
    cycle-walking (re-apply until the image lands under ``n``) which keeps the
    result a bijection on ``range(n)``. Four rounds is the standard choice for
    a strong pseudo-random permutation; the mixing here does not carry any
    security requirement, only decorrelation.
    """

    def __init__(self, n: int, seed: int, rounds: int = 4):
        if n <= 1:
            raise ValueError(f"nothing to permute: n={n}")
        self.n = n
        self.h = max(1, ((n - 1).bit_length() + 1) // 2)
        self.mask = (1 << self.h) - 1
        self.seed = int(seed) & _MASK64
        self.rounds = rounds

    def __call__(self, i: int) -> int:
        while True:
            lo, hi = i & self.mask, i >> self.h
            for k in range(self.rounds):
                hi, lo = lo, hi ^ (_mix(lo + (k << 40), self.seed) & self.mask)
            i = (hi << self.h) | lo
            if i < self.n:
                return i


class _ShardedWindows(IterableDataset):
    """Windows `rank, rank + world, rank + 2*world, ...` of the packed corpus.

    Strided rather than contiguous-blocked: a blocked split would give each rank
    a different region of the corpus, and at these budgets the regions are not
    interchangeable -- the corpus is not shuffled, so rank 0 would see only the
    first shard's documents for the whole run.
    """

    def __init__(self, prefix: str, seq_len: int, dp_rank: int, dp_world: int,
                 num_tokens: int | None = None, infinite: bool = True):
        from lm_scaling.data.megatron_indexed import build_megatron_dataset

        self.ds = build_megatron_dataset(prefix, seq_len, num_tokens)
        # SHUFFLING, off unless TITAN_EXT_DATA_SHUFFLE names a seed.
        #
        # Without it this walks the corpus in order, so a step's batch is a
        # contiguous slab of it. That is only harmless if neighbouring windows
        # are interchangeable, and they are not: the same model scores four
        # contiguous blocks of the held-out split 0.077 nats worse than a
        # uniform sample of it, measured. Adjacent
        # documents share source, length and topic, so a nominal batch of 64
        # carries far less gradient diversity than 64 independent draws, and
        # the training distribution drifts through the run instead of being
        # stationary. Megatron shuffles both documents and samples; this is the
        # difference that leaves our dense baseline 0.25 nats behind theirs.
        #
        # OPT-IN, because turning it on changes what every finished campaign
        # would produce, and those runs are already in the harvest.
        import os
        seed = os.environ.get("TITAN_EXT_DATA_SHUFFLE", "").strip()
        self._perm = _Permutation(len(self.ds), int(seed)) if seed else None
        self._shuffle_seed = int(seed) if seed else None
        self.rank, self.world = dp_rank, dp_world
        self.infinite = infinite
        self._index = dp_rank
        self._stride = dp_world
        # Set by load_state_dict: this copy carries on where a saved one stopped.
        self._restored = False

    def __iter__(self):
        n = len(self.ds)
        # With num_workers > 0 every worker runs this iterator, so without a
        # further split each worker would yield the SAME windows and the rank
        # would see each batch `num_workers` times. Interleave by worker id on
        # top of the rank stride: rank r worker w takes every
        # (world * num_workers)-th window starting at r * num_workers + w.
        info = torch.utils.data.get_worker_info()
        if info is not None:
            stride = self.world * info.num_workers
            start = self.rank * info.num_workers + info.id
        else:
            stride, start = self.world, self.rank
        # ...from its START only on a fresh run. torchdata restores each
        # worker's state through load_state_dict BEFORE this body first runs
        # (worker.py: restore dataset state, then build the iterator), and
        # assigning the start unconditionally threw that state away: every
        # restart and every cooldown read the corpus again from its first
        # window, with the checkpoint holding the right position all along.
        # Measured 2026-09-11: a job resumed from step 120 re-read step 1's
        # batches -- its loss changes correlate with the first run's 120 steps
        # earlier (r = 0.41) and not with the same steps (r = 0.11).
        if not self._restored:
            self._index = start
        elif self._stride != stride or self._index % stride != start:
            raise ValueError(
                f"dataloader state was saved reading every {self._stride}-th "
                f"window from window {self._index}, but this worker reads every "
                f"{stride}-th from {start}: resume with the dp_world and "
                f"num_workers the state was saved with")
        self._stride = stride
        while True:
            while self._index < n:
                i = self._index
                # Advanced BEFORE the yield, so a state taken while the
                # generator is suspended names the next window rather than the
                # one just handed out, which a resume would hand out again.
                self._index += self._stride
                ids = self.ds[self._perm(i) if self._perm else i]["input_ids"]
                # torchtitan wants (inputs, labels) with labels already shifted.
                yield {"input": ids[:-1]}, ids[1:]
            if not self.infinite:
                return
            self._index = self._index % self._stride

    # --- Stateful, so a WSD annealing continues the stream instead of
    # replaying it. Without these, torchdata logs "Neither dataset nor
    # iter(dataset) defines state_dict/load_state_dict" and every annealing
    # branching off a stable run restarts at window 0 -- re-training on tokens
    # the stable phase already consumed, which is exactly the repetition the
    # single-epoch budget is constructed to avoid. The warning is the only
    # sign; the loss curve looks entirely reasonable.
    def state_dict(self) -> dict:
        return {"index": self._index, "rank": self.rank, "world": self.world,
                "stride": self._stride, "shuffle_seed": self._shuffle_seed}

    def load_state_dict(self, state: dict) -> None:
        if state.get("world") != self.world:
            # Rank count changed between the stable run and the annealing, so a
            # saved stride no longer describes this rank's subsequence. Refuse
            # rather than silently resume onto someone else's windows.
            raise ValueError(
                f"dataloader state was saved with dp_world={state.get('world')} "
                f"but this run has {self.world}; the strided split no longer "
                f"lines up. Re-run the annealing at the stable run's GPU count."
            )
        saved = state.get("shuffle_seed")
        if saved != self._shuffle_seed:
            # A cooldown resuming a stable phase under a different order would
            # re-read tokens the stable phase consumed and skip others, with
            # nothing in the loss curve to show for it.
            raise ValueError(
                f"dataloader state was saved with shuffle seed {saved} but this "
                f"run has {self._shuffle_seed}; set TITAN_EXT_DATA_SHUFFLE to "
                f"the value the stable phase ran with")
        self._index = int(state["index"])
        self._stride = int(state.get("stride", self.world))
        self._restored = True


def build_megatron_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer=None,
    job_config=None,
    infinite: bool = True,
    **kwargs,
) -> ParallelAwareDataloader:
    """Drop-in for a train spec's ``build_dataloader_fn``.

    The corpus prefix comes from ``[training] dataset_path`` in the job config.
    """
    # Imported here, not at module scope: the dataset above is plain torch and
    # is unit-tested without torchtitan installed.
    from torchtitan.components.dataloader import ParallelAwareDataloader

    prefix = getattr(job_config.training, "dataset_path", None)
    if not prefix:
        raise ValueError(
            "set [training] dataset_path to the Megatron .bin/.idx prefix; "
            "without it torchtitan would read a different corpus from the "
            "other backends and the loss comparison would be meaningless"
        )
    ds = _ShardedWindows(
        prefix, job_config.training.seq_len, dp_rank, dp_world_size,
        infinite=infinite,
    )
    # Workers are not a tuning knob here, they are a correctness one. With the
    # default num_workers=0 the training process itself does the memmap reads,
    # so a single slow page fault on shared GPFS stalls that rank inside the
    # step -- and under FSDP the other three ranks sit in an all-gather waiting
    # for it. That is exactly how the first multi-GPU run died:
    #
    #   Watchdog caught collective operation timeout:
    #   WorkNCCL(SeqNum=1144, OpType=_ALLGATHER_BASE) ran for 100003 ms
    #
    # A single-GPU run has no collective to time out, which is why this only
    # appeared once the equivalence moved to four GPUs. `lm/` never hit it
    # because its own loader has always run four workers with prefetch.
    workers = int(getattr(job_config.training, "num_workers", 0) or 4)
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.local_batch_size,
        num_workers=workers,
        # NOT pin_memory. torchdata's StatefulDataLoader and this container's
        # torch (2.10.0a0 nightly) disagree about _pin_memory_loop's signature
        # -- "takes 4 positional arguments but 5 were given", and the pin
        # thread dies taking the run with it. The workers are what matter here
        # anyway: at ~0.7 M tok/s of uint16 the host-to-device copy is not
        # where the time goes, and `lm/` gets away with pin_memory only
        # because it uses a plain DataLoader.
        prefetch_factor=4 if workers else None,
        persistent_workers=bool(workers),
    )
