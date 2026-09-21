"""Validate on the paper's held-out split instead of torchtitan's C4 default.

`Validator.__init__` calls `build_text_validation_dataloader` directly -- not
through the train spec's `build_dataloader_fn` -- so swapping the training
loader for the Megatron reader leaves validation pointed at `c4_validation`,
downloaded from HuggingFace and tokenized with a tokenizer we have set to
`None`. That is not a subtle mismatch: it is the one number this whole anchor
exists to compare, measured on the wrong corpus.

Two things differ from the training loader, and both are about determinism
rather than throughput:

**Contiguous blocks, not a stride.** Training strides because the corpus is
unshuffled and a blocked split would hand rank 0 nothing but the first shard.
Validation averages over the whole split, so which rank sees which document
does not matter -- but the *count* does: `dist_mean` over per-rank means equals
the global mean only when every rank contributed the same number of tokens. A
block of `n // dp_world` gives exactly that; a stride gives ranks differing by
one window and a silently mis-weighted average.

**Restart at the block start every pass.** The training dataset is stateful so
that a WSD annealing resumes the stream. A validation loader with that property
evaluates a *different* 10M tokens at each call and the curve moves for reasons
that have nothing to do with the model.

**The same windows at any width.** Which text is scored must not depend on how
many ranks score it. It used to: each rank took its own block of the whole
split, so 4 ranks scored the four quarter starts and 128 ranks scored 128
starts. The upper ladders size each arm's width from its micro-batch ceiling,
so at 1.7B attention and the linear arms were validated on different text. See
`_ValidationWindows`.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

from torch.utils.data import IterableDataset

# The repo root, so `import lm_scaling.data...` works however this module was
# reached.
# `megatron_data.py` next door does the same and this one did not, which held
# only until something imported it from a script whose sys.path[0] is
# `lm_scaling/` rather than the root. Unit tests never saw it, because pytest
# puts the root on the path itself.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


class _ValidationWindows(IterableDataset):
    """This rank's share of a FIXED set of held-out windows -- the same set at
    any world size, scored from the top on every pass.

    THE SET: `BLOCKS` contiguous blocks spread evenly over the split, together
    holding exactly the windows the run scores (`validation.steps x
    local_batch_size x dp_world`). THE SHARE: that set, in order, cut into
    `dp_world` equal consecutive slices, one per rank -- equal, for the
    `dist_mean` reason in the module docstring.

    It used to be the rank's own block of the WHOLE split -- `n // dp_world`
    windows from `dp_rank * (n // dp_world)`, read from the start. That let the
    world size decide which text was scored: the four quarter starts at 4
    ranks, 128 starts at 128. The upper ladders size each arm's width from its
    micro-batch ceiling, so at 1.7B attention (16 nodes) and the linear arms
    (32 nodes) were scored on different text, and every comparison between
    them was also a comparison between corpora.

    `BLOCKS` is 4 because that is what every one-node job has scored since the
    lower ladder launched: one block per rank, at the quarter starts. At 4
    ranks this reproduces that selection window for window, so a running
    campaign resumed on this code keeps scoring the text it began on.
    Changing it changes the validation set of a campaign in flight.
    """

    BLOCKS = 4

    def __init__(self, prefix: str, seq_len: int, dp_rank: int, dp_world: int,
                 per_rank: int | None = None):
        from lm_scaling.data.megatron_indexed import build_megatron_dataset

        self.ds = build_megatron_dataset(prefix, seq_len, None)
        n = len(self.ds)
        if per_rank is None or per_rank <= 0:
            # `validation.steps = -1`, torchtitan's default: consume the rank's
            # whole block of the split. Nothing on the ladders runs this way;
            # kept working for a smoke run.
            block = n // dp_world
            if block == 0:
                raise ValueError(
                    f"validation split has {n} windows of {seq_len} tokens but "
                    f"dp_world={dp_world}; every rank must get at least one")
            self.indices = range(dp_rank * block, (dp_rank + 1) * block)
        else:
            total = per_rank * dp_world
            if total % self.BLOCKS:
                raise ValueError(
                    f"{total} validation windows do not split into "
                    f"{self.BLOCKS} equal blocks; make eval_sequences divisible "
                    f"by {self.BLOCKS} x mbs x world")
            block, stride = total // self.BLOCKS, n // self.BLOCKS
            if block > stride:
                raise ValueError(
                    f"{total} validation windows asked for, but {self.BLOCKS} "
                    f"blocks of a {n}-window split hold at most "
                    f"{stride * self.BLOCKS}")
            lo = dp_rank * per_rank
            self.indices = [(i // block) * stride + i % block
                            for i in range(lo, lo + per_rank)]
        self.start, self.per_rank = self.indices[0], len(self.indices)

    def __iter__(self):
        for i in self.indices:
            ids = self.ds[i]["input_ids"]
            yield {"input": ids[:-1]}, ids[1:]

    # StatefulDataLoader logs a warning and, worse, a checkpoint round-trip
    # would restore a mid-split position. There is no position to keep: every
    # pass covers the same windows in the same order by construction.
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        return None


def build_megatron_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer=None,
    job_config=None,
    infinite: bool = False,
    **kwargs,
):
    prefix = getattr(job_config.validation, "dataset_path", None)
    if not prefix:
        raise ValueError(
            "set [validation] dataset_path to the held-out .bin/.idx prefix; "
            "otherwise torchtitan validates on c4_validation and the number "
            "cannot be compared to the paper's"
        )
    from torchtitan.components.dataloader import ParallelAwareDataloader

    # Exactly the windows torchtitan will consume -- `steps` batches of
    # `local_batch_size` on every rank -- so the set is fixed by the run's
    # validation budget and not by the length of the split.
    steps = int(getattr(job_config.validation, "steps", -1) or -1)
    per_rank = steps * int(job_config.validation.local_batch_size) if steps > 0 else None
    ds = _ValidationWindows(prefix, job_config.validation.seq_len, dp_rank,
                            dp_world_size, per_rank=per_rank)
    # num_workers=0 on purpose. The split is ~20 MB of uint16 and page-cached
    # after the first pass, so workers buy nothing, and each one would need its
    # own sub-block to avoid re-yielding the same windows -- a duplication that
    # would show up only as a validation loss that is subtly wrong.
    return ParallelAwareDataloader(
        dataset=ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.validation.local_batch_size,
        num_workers=0,
    )


def validation_steps(prefix: str, seq_len: int, local_batch_size: int, dp_world: int) -> int:
    """How many validation steps consume the split exactly once.

    Exported so the config generator can write `[validation] steps` from the
    real file rather than from an estimate. torchtitan's own default, -1,
    warns about hangs from unequal sample counts across ranks; with a number
    computed here there is nothing to be unequal about.
    """
    from lm_scaling.data.megatron_indexed import build_megatron_dataset

    n = len(build_megatron_dataset(prefix, seq_len, None))
    return max(1, (n // dp_world) // local_batch_size)


@contextlib.contextmanager
def _megatron_validation_loader():
    """Rebind the loader torchtitan's `Validator.__init__` hard-codes.

    Scoped to the constructor call. Subclassing cannot help -- the base
    `__init__` builds the HuggingFace loader before any subclass gets control,
    and with `build_tokenizer_fn = None` that raises before we could replace
    it. Rebinding for the duration of the call leaves every other part of
    `Validator` (pipeline-parallel handling, the loss mean, the metrics hook)
    running the container's own code, which is the part worth not forking.
    """
    import torchtitan.components.validate as V

    original = V.build_text_validation_dataloader
    V.build_text_validation_dataloader = build_megatron_validation_dataloader
    try:
        yield
    finally:
        V.build_text_validation_dataloader = original


def build_megatron_validator(*args, **kwargs):
    import types

    from torchtitan.components.validate import build_validator

    with _megatron_validation_loader():
        validator = build_validator(*args, **kwargs)

    # torchtitan validates when `step % freq == 0`, and the final step almost
    # never is: at 114,441 steps with freq 2,288 the last validation lands at
    # 114,400 and the number the run is judged on is 41 steps stale. Tiny in
    # loss, but the end-of-decay value is the one compared against a published
    # figure, and "the last one we happened to take" is not that value.
    last = int(validator.job_config.training.steps)
    base = type(validator).should_validate

    def should_validate(self, step: int) -> bool:
        return step == last or base(self, step)

    validator.should_validate = types.MethodType(should_validate, validator)
    return validator
