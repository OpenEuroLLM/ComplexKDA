"""fwedu_1p3B: the published 1.3B/100BT recipe, with the study's arms in it.

A replication is worth exactly as much as its comparison, and the comparison
holds only if this trains what the published table trained. So the recipe is
pinned against the numbers Gated DeltaNet states (arXiv:2412.06464) and the
ones its code uses where the paper is silent -- 0.5M-token batches at 4096,
4e-4 decaying by cosine to a tenth, a 1B-token warmup from zero, weight decay
on everything, `</s>` after every document -- rather than against another
generator, which would only prove the two agree.

Also pinned: the two things a port gets wrong without a word -- a geometry the
config and torchtitan read from different places, and a schedule that is not
the one its name says.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "lm_scaling"))
sys.path.insert(0, str(_ROOT / "lm_scaling" / "data"))

TABLE = json.loads((_ROOT / "lm_scaling" / "replication_1p3B.json").read_text())
ARMS = ("kda-sig-lowrank", "ckda-shipped-lowrank", "kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank")
TOK_PER_STEP = 128 * 4096  # "a batch size of 0.5M tokens"


# --- the geometry ------------------------------------------------------------


def test_the_pairs_are_identical_and_every_arm_is_matched():
    archs = TABLE["archs"]
    assert archs["attn"]["N"] == TABLE["target_N"]
    for a, b in (
        ("kda-sig-lowrank", "ckda-shipped-lowrank"),
        ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"),
    ):
        assert (archs[a]["d_ffn"], archs[a]["N"]) == (archs[b]["d_ffn"], archs[b]["N"]), (
            f"{a}/{b}: the pair is the comparison and must be matched exactly"
        )
    for arm in ARMS:
        assert abs(archs[arm]["delta_pct"]) < 1.0, arm
    for arm in ("kda-sig-hybrid-lowrank", "ckda-shipped-hybrid-lowrank"):
        assert archs[arm]["hybrid"]["attn_every"] == 4, "Kimi Linear's 3:1"
    assert (TABLE["vocab_size"], TABLE["tie_word_embeddings"]) == (32000, False)
    assert TABLE["n_heads"] * TABLE["head_dim"] == TABLE["d_model"]


def test_the_arms_are_the_ladders_arms():
    """Verbatim from ARCHS. A knob changed here would make "the layer works at
    1.3B" a claim about some other layer."""
    pytest.importorskip("fla", reason="param_match imports the fla layers")
    import param_match as pm

    for arm, entry in TABLE["archs"].items():
        assert entry["kwargs"] == pm.ARCHS[arm]["kwargs"], arm
        assert entry["mixer"] == pm.ARCHS[arm]["mixer"], arm
        assert entry.get("hybrid") == pm.ARCHS[arm].get("hybrid"), arm


def test_the_target_is_the_published_model_as_fla_builds_it():
    """`target_N` is fla's count of fla-hub's own Transformer++ config, and
    every N is what fla allocates at that width. One arm per pair; the pair
    test covers the other."""
    pytest.importorskip("fla", reason="counting builds fla models")
    import param_match as pm
    import replication_1p3B as R

    assert R.reference() == (TABLE["target_N"], TABLE["archs"]["attn"]["d_ffn"])
    geo = dict(
        d_model=TABLE["d_model"],
        n_heads=TABLE["n_heads"],
        n_layers=TABLE["n_layers"],
        head_dim=TABLE["head_dim"],
        vocab_size=TABLE["vocab_size"],
        seq_len=TABLE["seq_len"],
        tie=TABLE["tie_word_embeddings"],
    )
    for arm in ("attn", "kda-sig-lowrank", "kda-sig-hybrid-lowrank"):
        entry = TABLE["archs"][arm]
        assert pm.n_params_of(arm, d_ffn=entry["d_ffn"], **geo) == entry["N"], arm


def test_the_config_and_torchtitan_read_one_table():
    """The citations and the flavors both go through `geometries`, so a
    geometry one of them can see is one the other can."""
    pytest.importorskip("omegaconf", reason="resolvers registers with OmegaConf")
    import geometries
    import resolvers

    table = geometries.table()
    assert "1.3B" in table
    assert table["1.3B"]["archs"] == TABLE["archs"]
    for arm in ARMS:
        assert resolvers.matched("1.3B", arm, "d_ffn") == TABLE["archs"][arm]["d_ffn"]
    assert resolvers.rung("1.3B", "head_dim") == 128


def test_a_geometry_defined_twice_raises(tmp_path, monkeypatch):
    """Two files defining one tag would make a run's widths depend on which
    was read last."""
    import geometries

    names = (*geometries.FIXED_POINTS, "another_1p3B.json")
    for name in names:
        (tmp_path / name).write_text(json.dumps({**dict.fromkeys(geometries._SHARED, 0), "tag": "1.3B", "archs": {}}))
    # The ladder file is read first and must exist; an empty one keeps this
    # test about the collision it is named for.
    (tmp_path / geometries.LADDER).write_text("{}")
    monkeypatch.setattr(geometries, "_HERE", tmp_path)
    monkeypatch.setattr(geometries, "FIXED_POINTS", names)
    monkeypatch.setattr(geometries, "_table", None)
    with pytest.raises(ValueError, match="already defines"):
        geometries.table()


def test_torchtitan_registers_every_arm_at_1p3B():
    pytest.importorskip("torch", reason="titan_ext imports torch")
    from dataclasses import make_dataclass

    import lm_scaling.titan_ext as ext

    fields = [
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "hidden_dim",
        "vocab_size",
        "max_seq_len",
        "norm_eps",
        "rope_theta",
        "qk_norm",
        "depth_init",
        "mixer",
    ]
    names = list(dict.fromkeys(fields + list(ext.FLAModelArgs.__dataclass_fields__)))
    fla = ext.ladder_flavors(make_dataclass("A", [(f, object, None) for f in names]))
    dense = ext.ladder_flavors(make_dataclass("D", [(f, object, None) for f in fields[:-1]]))

    for arm in ARMS:
        flavor, entry = fla[f"{arm}-1.3B"], TABLE["archs"][arm]
        assert (flavor.dim, flavor.n_layers, flavor.n_heads, flavor.head_dim) == (2048, 24, 16, 128)
        assert (flavor.hidden_dim, flavor.vocab_size, flavor.mixer) == (entry["d_ffn"], 32000, entry["mixer"])
        assert flavor.attn_every == (4 if "hybrid" in entry else None), arm
        for name, value in entry["kwargs"].items():
            assert getattr(flavor, name) == value, f"{arm}: {name}"
    assert "attn-1.3B" in dense and "attn-1.3B" not in fla


# --- the optimizer -----------------------------------------------------------


def test_the_gdn_recipe_decays_every_parameter(monkeypatch):
    """GDN's AdamW takes model.parameters() as one group. The study exempts
    norms and biases; this experiment opts out of the exemption, and no other
    campaign may."""
    torch = pytest.importorskip("torch")
    import types

    import lm_scaling.titan_ext.optimizer as O

    class Container:
        def __init__(self, optimizers):
            self.optimizers = optimizers

    def build_optimizers(model_parts, optimizer_config, parallel_dims, ft_manager=None):
        # torchtitan's own: one group, decayed.
        params = [p for m in model_parts for p in m.parameters()]
        return Container([torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1)])

    stub = types.ModuleType("torchtitan.components.optimizer")
    stub.build_optimizers = build_optimizers
    for name in ("torchtitan", "torchtitan.components"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "torchtitan.components.optimizer", stub)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.norm = torch.nn.LayerNorm(4)

    def decays(model):
        groups = O.build_optimizers_matching_lm([model], None, None).optimizers[0].param_groups
        return sorted((len(g["params"]), g["weight_decay"]) for g in groups)

    monkeypatch.delenv(O.DECAY_ALL, raising=False)
    assert decays(Model()) == [(1, 0.1), (3, 0.0)], "the study's default exempts three"
    monkeypatch.setenv(O.DECAY_ALL, "1")
    assert decays(Model()) == [(4, 0.1)], "GDN's decays all four"


# --- the campaign ------------------------------------------------------------
#
# Planned against the measured ceilings: 4 for every arm (peak_mbs.yaml,
# gh200_seq4096_titan_g4), which is also the most this width can use.


@pytest.fixture(scope="module")
def ceilings():
    pytest.importorskip("hydra_staged_sweep", reason="needs the config layer")
    pytest.importorskip("monitor", reason="needs the config layer")


@pytest.fixture(scope="module")
def jobs(ceilings):
    import plan as P

    return P.plans("fwedu_1p3B")


def test_the_campaign_is_four_arms_and_passes_every_check(jobs):
    import plan as P

    assert sorted(j.config.aux["model"] for j in jobs) == sorted(ARMS)
    P.check(jobs)


def test_the_recipe_is_the_published_one(jobs):
    for j in jobs:
        c, a, t = j.config, j.config.aux, j.config.backend.titan
        assert int(a["tok_per_step"]) == TOK_PER_STEP
        assert int(t.training["global_batch_size"]) == 128
        assert (c.data.seq_len, c.data.vocab_size, c.data.tie_embeddings) == (4096, 32000, False)
        assert t.optimizer["lr"] == pytest.approx(4e-4)
        assert t.optimizer["weight_decay"] == pytest.approx(0.1)
        assert (t.optimizer["beta1"], t.optimizer["beta2"]) == (0.9, 0.95)
        assert int(t.debug["seed"]) == 3407
        assert c.slurm.env["TITAN_EXT_DECAY_ALL"] == "1"

        # 100BT, rounded up onto the 256-step grid and no further. Literal on
        # purpose: `oc.cdivi` returns 0 for a token count PyYAML left as a
        # string, and a zero-step campaign plans cleanly.
        steps = int(t.training["steps"])
        assert steps == 190_976
        assert 100e9 <= steps * TOK_PER_STEP < 100e9 + 256 * TOK_PER_STEP

        # A 1B-token warmup FROM ZERO, then cosine to lr / 10, no WSD tail --
        # GDN's get_lr.
        s = t.lr_scheduler
        assert 1e9 <= int(s["warm_steps"]) * TOK_PER_STEP < 1e9 + TOK_PER_STEP
        assert s["lr_min_absolute"] == 0
        assert s["main_decay_type"] == "cosine"
        assert s["main_decay_ratio"] == pytest.approx(0.1)
        assert int(s["cooldown_steps"]) == 0


def test_decaying_every_parameter_is_this_campaign_s_opt_out(ceilings):
    """The study exempts norms, biases and A_log from weight decay; this
    campaign opts out of the exemption because GDN's AdamW takes
    model.parameters() as one group. The opt-out has to be stated by the
    experiment, not inherited, or it would silently apply to everything that
    shares the base config."""
    import plan as P

    assert P.load("fwedu_1p3B").slurm.env["TITAN_EXT_DECAY_ALL"] == "1"
    # Stated by THIS experiment and by nothing it inherits from.
    exp = _ROOT / "lm_scaling" / "config" / "experiments"
    assert "TITAN_EXT_DECAY_ALL" in (exp / "fwedu_1p3B.yaml").read_text()
    for shared in ("base.yaml",):
        assert "TITAN_EXT_DECAY_ALL" not in (exp / shared).read_text(), shared


def test_the_toml_carries_numbers_not_strings(jobs):
    """`aux` is typed dict[str, str]. A value that reached the TOML through it
    as "0.1" would multiply a float by a string inside the scheduler."""
    import plan as P
    import tomllib

    sched = tomllib.loads(P.to_toml(jobs[0].config.backend.titan))["lr_scheduler"]
    for key in ("warm_steps", "cooldown_steps"):
        assert isinstance(sched[key], int), key
    assert isinstance(sched["main_decay_ratio"], float)


def test_every_arm_runs_at_one_width_and_scores_the_same_text(jobs):
    """megatron_validate hands rank r the r-th block of the held-out split, so
    the world size decides which text is scored."""
    assert {int(j.config.aux["world_gpus"]) for j in jobs} == {32}
    for j in jobs:
        c, a = j.config, j.config.aux
        v = c.backend.titan.validation
        assert int(v["steps"]) * int(a["mbs"]) * int(a["world_gpus"]) == c.data.eval_sequences, (
            "every rank, exactly eval_sequences"
        )
        assert int(a["mbs"]) * int(a["grad_accum"]) == 4


def test_the_final_step_is_validated_and_the_checkpoints_kept(jobs):
    for j in jobs:
        t = j.config.backend.titan
        steps, freq = int(t.training["steps"]), int(t.validation["freq"])
        assert steps % freq == 0 and steps // freq >= 32
        assert int(t.checkpoint["keep_latest_k"]) == 0
        assert 5e9 <= int(t.checkpoint["interval"]) * TOK_PER_STEP < 5e9 + 256 * TOK_PER_STEP


def test_each_job_builds_the_table_s_model(jobs):
    for j in jobs:
        m, arm = j.config.backend.titan.model, j.config.aux["model"]
        # `-nosilu` because the campaign states aux.drop_silu: SiLU has no
        # parameters, so dropping it is a separate flavor NAME rather than a
        # changed definition (titan_ext.ladder_flavors, resolvers.flavor).
        assert (m["name"], m["flavor"]) == ("fla_custom", f"{arm}-1.3B-nosilu")
        assert (int(m["hidden_dim"]), int(m["head_dim"]), int(m["vocab_size"])) == (
            TABLE["archs"][arm]["d_ffn"],
            128,
            32000,
        )
        assert m["enable_weight_tying"] is False
        assert list(j.config.backend.titan.compile["components"]) == ["loss"]
        assert "cosine100B" in j.config.job.name


def test_a_run_longer_than_its_corpus_is_refused(ceilings):
    """The reader wraps instead of failing, so a short corpus would train a
    second epoch under a single-epoch label."""
    import plan as P

    jobs = P.plans("fwedu_1p3B", ["data.train_tokens=1000000000"])
    with pytest.raises(SystemExit, match="wraps around"):
        P.check(jobs)


def test_a_constant_main_phase_without_a_cooldown_is_still_refused(ceilings):
    """The cosine exemption in plan.check is for a main phase that DECAYS.
    The same campaign with it held constant never anneals."""
    import plan as P

    jobs = P.plans("fwedu_1p3B", ["backend.titan.lr_scheduler.main_decay_type=const"])
    with pytest.raises(SystemExit, match="cooldown is 0"):
        P.check(jobs)


# --- the corpus --------------------------------------------------------------


class _Tokenizer:
    """The three things check_tokenizer and tokenize_shard ask of one."""

    def __init__(self, ids, n=32000, eos=2):
        self.ids, self.n, self.eos_token_id = ids, n, eos

    def __len__(self):
        return self.n

    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False, "a BOS would reach the corpus"
        return {"input_ids": [self.ids(t) if callable(self.ids) else self.ids for t in texts]}


def test_only_llama2_passes_the_tokenizer_check():
    pytest.importorskip("numpy")
    import prepare_fineweb_edu as F

    F.check_tokenizer(_Tokenizer(F.HELLO))
    for tok in (
        _Tokenizer([22557, 1526, 28723]),  # Mistral's, which fla-hub ships
        _Tokenizer(F.HELLO, n=50277),  # the right ids, another vocab
        _Tokenizer(F.HELLO, eos=0),
    ):  # an EOS that is not </s>
        with pytest.raises(SystemExit):
            F.check_tokenizer(tok)


def test_a_missing_tokenizer_copy_is_named_not_traced(tmp_path):
    """The tokenize tasks load the copy `download` put beside the corpus --
    transformers 5.3 will not load the Hub id offline. Without the copy the
    task should say which step was skipped, not fail inside transformers."""
    pytest.importorskip("numpy")
    import prepare_fineweb_edu as F

    with pytest.raises(SystemExit, match="run `download`"):
        F.load_tokenizer(tmp_path)


def test_a_document_is_its_ids_then_eos_and_no_bos(tmp_path):
    """GDN's `Tokenizer.encode(text)`: bos=False, eos=True. Empty rows are
    dropped rather than stored as a lone EOS."""
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("torch", reason="the Megatron reader lives in lm/")
    import prepare_fineweb_edu as F
    import pyarrow.parquet as pq

    from lm_scaling.data.megatron_indexed import MegatronIndexedReader

    pq.write_table(pa.table({"text": ["ab", "", "cde"]}), tmp_path / "s.parquet")
    tok = _Tokenizer(lambda t: [10 + ord(c) - ord("a") for c in t])
    assert F.tokenize_shard(tmp_path / "s.parquet", tmp_path / "tok" / "s", tok) == (2, 7)
    r = MegatronIndexedReader(tmp_path / "tok" / "s")
    assert [r.get(i).tolist() for i in range(len(r))] == [[10, 11, F.EOS], [12, 13, 14, F.EOS]]


def test_the_train_order_mixes_every_shard_along_the_whole_run():
    """Round-robin would exhaust the small shards early and end the run on the
    large ones. Each assertion below fails with probability ~1e-5 under a
    uniform permutation, so they hold for any seed."""
    np = pytest.importorskip("numpy")
    import prepare_fineweb_edu as F

    lengths = [np.full(n, 100) for n in (3000, 400, 1200, 8000, 600)]
    cuts, order = F.plan_split(lengths, valid_tokens=20_000, block=8, seed=F.SEED)

    for l, c in zip(lengths, cuts):
        assert 0 < c < len(l), "every shard gives to both splits"
    assert sum(int(l[c:].sum()) for l, c in zip(lengths, cuts)) >= 20_000
    assert [int((order == s).sum()) for s in range(5)] == [-(-c // 8) for c in cuts]

    fifths = np.array_split(order, 5)
    for s in range(5):
        assert all((f == s).any() for f in fifths), f"shard {s} missing from a fifth"
        assert np.flatnonzero(order == s).max() >= 0.8 * len(order), f"shard {s} ran out before the last fifth"

    _, again = F.plan_split(lengths, 20_000, 8, F.SEED)
    assert (again == order).all(), "the order is seeded"


def test_the_split_keeps_every_document_exactly_once(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch", reason="the Megatron reader lives in lm/")
    import prepare_fineweb_edu as F

    from lm_scaling.data.megatron_indexed import MegatronIndexedReader
    from lm_scaling.data.megatron_writer import MegatronIndexedWriter

    rng = np.random.default_rng(0)
    prefixes = []
    for s, n_docs in enumerate((300, 40, 120, 800, 60)):  # unequal, as real
        prefix = tmp_path / "tok" / f"{s:03d}"
        with MegatronIndexedWriter(prefix) as w:
            for d in range(n_docs):
                # Each document names itself -- shard, index -- then filler,
                # then the EOS every real one ends with.
                filler = rng.integers(3, 99, int(rng.integers(1, 40)))
                w.add([100 + s, 1000 + d, *filler.tolist(), F.EOS])
        prefixes.append(prefix)

    def docs(prefix):
        r = MegatronIndexedReader(prefix)
        return [tuple(int(t) for t in r.get(i)) for i in range(len(r))]

    F.build(prefixes, tmp_path / "splits", valid_tokens=2000, block=8, seed=1)
    train, valid = docs(tmp_path / "splits" / "train"), docs(tmp_path / "splits" / "valid")
    assert sorted(train + valid) == sorted(d for p in prefixes for d in docs(p))
    assert sum(map(len, valid)) >= 2000
    assert {d[0] for d in valid} == {100 + s for s in range(len(prefixes))}
    assert all(d[-1] == F.EOS for d in train + valid), "every document still ends in EOS"
