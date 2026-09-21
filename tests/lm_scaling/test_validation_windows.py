"""Validation scores the same windows, and exactly `eval_sequences` of them,
at any world size.

Both halves were broken for any campaign running on more than one node:

  * the COUNT: `validation.steps` divided by one node's GPUs, so every job
    wider than one node scored 2x to 32x `eval_sequences`;
  * the TEXT: megatron_validate handed each rank its own block of the whole
    split, so the world size decided which text was scored -- at 1.7B,
    attention (16 nodes) and the linear arms (32 nodes) were validated on
    different corpora.

The lower ladder was exact on both counts because every job is one node, and
the fix must leave it window-for-window unchanged: it was running when this was
found, and its restarts re-import this code.
"""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "lm_scaling"))

import lm_scaling.data.megatron_indexed as MI  # noqa: E402
from lm_scaling.titan_ext.megatron_validate import _ValidationWindows, build_megatron_validation_dataloader  # noqa: E402

N = 25_898_444        # windows in the real held-out split at 4096
E = 2560              # eval_sequences


class _Split:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": torch.tensor([i, i + 1])}


@pytest.fixture(autouse=True)
def fake_split(monkeypatch):
    monkeypatch.setattr(MI, "build_megatron_dataset", lambda prefix, seq_len, n: _Split(N))


def _scored(world, per_rank):
    out = []
    for r in range(world):
        out.append(list(_ValidationWindows("x", 4096, r, world, per_rank=per_rank).indices))
    return out


def test_every_width_scores_the_same_windows():
    reference = None
    for world in (1, 2, 4, 8, 16, 32, 64, 128):
        per = _scored(world, E // world)
        union = sorted(w for rank in per for w in rank)
        assert len(union) == E and len(set(union)) == E, f"{world} ranks: duplicates or gaps"
        if reference is None:
            reference = union
        assert union == reference, f"{world} ranks score different text"


def test_four_ranks_score_exactly_what_the_running_ladder_scores():
    """The old selection at 4 ranks: rank r's first E/4 windows of its quarter
    of the split. The fix has to reproduce it window for window."""
    per = _scored(4, E // 4)
    for r in range(4):
        start = r * (N // 4)
        assert per[r] == list(range(start, start + E // 4)), f"rank {r} moved"


def test_every_rank_scores_the_same_number_of_windows():
    """`dist_mean` over per-rank means is the global mean only when every rank
    contributed the same count."""
    for world in (4, 32, 128):
        assert {len(r) for r in _scored(world, E // world)} == {E // world}


def test_the_batches_torchtitan_takes_are_whole():
    ds = _ValidationWindows("x", 4096, 3, 128, per_rank=20)
    got = [x["input"][0].item() for x, _ in ds]
    assert got == list(ds.indices) and len(got) == 20


# Only a width that is not a multiple of 4 can fail this: at 4k ranks the
# total is per_rank x 4k, which always splits. (4, 3) was the wrong case.
@pytest.mark.parametrize("world,per_rank", [(2, 3), (1, 2)])
def test_a_count_that_does_not_split_into_blocks_is_refused(world, per_rank):
    with pytest.raises(ValueError, match="equal blocks"):
        _ValidationWindows("x", 4096, 0, world, per_rank=per_rank)


def test_a_count_larger_than_the_split_is_refused(monkeypatch):
    monkeypatch.setattr(MI, "build_megatron_dataset", lambda *a: _Split(100))
    with pytest.raises(ValueError, match="hold at most"):
        _ValidationWindows("x", 4096, 0, 4, per_rank=40)


def test_steps_minus_one_still_reads_the_ranks_whole_block():
    ds = _ValidationWindows("x", 4096, 2, 4)
    assert ds.indices == range(2 * (N // 4), 3 * (N // 4))


def test_the_loader_passes_the_runs_own_validation_budget(monkeypatch):
    """The per-rank count comes from `validation.steps x local_batch_size`, the
    batches torchtitan will actually take -- not from the length of the split."""
    fake = types.ModuleType("torchtitan.components.dataloader")
    fake.ParallelAwareDataloader = lambda **kw: kw
    for name in ("torchtitan", "torchtitan.components"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "torchtitan.components.dataloader", fake)

    cfg = SimpleNamespace(validation=SimpleNamespace(
        dataset_path="x", seq_len=4096, local_batch_size=4, steps=5))
    got = build_megatron_validation_dataloader(dp_world_size=128, dp_rank=5, job_config=cfg)
    assert got["dataset"].per_rank == 20 and got["batch_size"] == 4


@pytest.fixture(scope="module")
def campaign():
    pytest.importorskip("hydra_staged_sweep", reason="needs the config layer")
    pytest.importorskip("monitor", reason="needs the config layer")
    import plan as P
    return P.plans("fwedu_1p3B")


def test_every_job_scores_exactly_eval_sequences(campaign):
    """At 32 ranks rather than one node's 4, which is where this broke: a
    `validation.steps` divided by `gpus_per_node` scores `eval_sequences x
    nodes`, exact on one node and eight times too many here."""
    for j in campaign:
        c, v = j.config, j.config.backend.titan.validation
        scored = int(v["steps"]) * int(v["local_batch_size"]) * int(c.aux["world_gpus"])
        assert scored == c.data.eval_sequences, (
            f"{c.job.name}: scores {scored} at {c.aux['world_gpus']} ranks")


def test_the_count_holds_at_any_width(campaign):
    """The same campaign resolved at another node count must still score
    exactly `eval_sequences` -- the property that makes two runs of different
    width comparable at all."""
    import plan as P

    for nodes in (4, 8, 16):
        for j in P.plans("fwedu_1p3B", [f"++aux.nodes={nodes}"]):
            c, v = j.config, j.config.backend.titan.validation
            scored = int(v["steps"]) * int(v["local_batch_size"]) * int(c.aux["world_gpus"])
            assert scored == c.data.eval_sequences, f"{nodes} nodes: {scored}"
