"""Submit a ladder into Megatron-LM, cell for cell with the torchtitan one.

    lm_scaling/megatron_ext/submit_ladder.py ladder            # print and price
    lm_scaling/megatron_ext/submit_ladder.py ladder --submit   # actually submit

PRINTING IS THE DEFAULT and needs no cluster: it resolves every cell's geometry,
width and price and prints them without touching slurm. There is no `--dry-run`
flag because there is nothing to opt out of. Submitting runs on the JUPITER
login node with the autoexp venv; it shells out to oellm-autoexp's
`scripts/run_autoexp.py` once per cell.

THE WHOLE SWEEP, which is the five groups below run in turn:

    for g in ladder ladder_beta ladder_chaincheck ladder_302M_upper ladder_medium
    do  lm_scaling/megatron_ext/submit_ladder.py $g
    done

THE CAMPAIGN IS 180 CELLS: 6 arms x 30 (rung, budget) pairs -- 9 + 4
single-stage from `CELLS`, 2 + 15 chained from `CHAINS`.
`lm_scaling/harvest/megatron_ladder.tsv` is what came back, one row per cell,
and `config/experiments/ladder_results.md` reads it.

More JOBS than cells get printed, and that is not a discrepancy: a chain is a
trunk plus one cooldown per budget, so its cells cost several jobs each. And
`ladder_chaincheck`'s two cells deliberately DUPLICATE two of `ladder`'s, at
the same hyperparameters through the other schedule -- it is the measurement of
whether chaining costs anything, so its results are a comparison and not two
more cells.

WHY NOT THEIR SWEEP DSL. oellm-autoexp expands sweeps through a hydra staged
DAG, and the thing we need -- a geometry that changes with BOTH the rung and
the arm, because the arms are parameter-matched by their MLP width -- is a
product with a lookup inside it. One process per cell is a page of code instead
of a fight with the expander, and each cell's overrides are then visible in its
own rendered sbatch.

WHAT IS HELD FIXED, and why each:

    beta2 0.95          asked for; the reference sweeps 0.99 at these rungs and
                        0.95 above 300M, and we want one value across the ladder
    biases off          asked for, outside the (s)KDA layers, which keep fla's
                        own -- the spec refuses add_bias_linear/add_qkv_bias
    rotary_base 10000   Megatron's default and the reference's; the smoke
                        config's 100000 is a deviation that is not in their
                        published runs
    20% linear cooldown the ladder's own WSD shape
    min_lr 1e-5         theirs and ours
    204,800 valid seqs  the reference's held-out set, which they specify as a
                        SEQUENCE count (their eval_iters 100 at gbs 2048). Our
                        cells run at gbs 32-128, so eval_iters is derived, not
                        fixed, or each cell would score a different-sized slice

GEOMETRY comes from ladder_matched.json, the same file the torchtitan flavors
are generated from, so the two ladders are parameter-matched by construction
rather than by agreement.

PRICES come from `lm_scaling/throughput.py` against the torchtitan campaign's
own measured token rates, which cover every arm at every rung here, times one
Megatron factor measured at 47M attention. The first draft of this file used
`190k tok/s/GPU scaled as 1/N` instead, which is wrong in both directions at
once: it under-prices 47M (the measurement is 278k) and over-prices 1.7B by
6x, because a 47M run is overhead-bound and a 1.7B one is not.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]

COMPLEX_KDA = os.environ.get("CKDA_REPO", "/e/project1/e-sta-openeurollm/poeppel1/complex_kda")
#: Our OWN checkout and our OWN venv, both inside the project. This ran off a
#: shared clone at ~/work/oellm-autoexp for a while, which is a checkout
#: nothing in this repo controls: its branch, its submodule commits and its
#: local edits all sit outside the project and change without us. Everything
#: this ladder depends on belongs under COMPLEX_KDA.
AUTOEXP = Path(os.environ.get("OELLM_AUTOEXP", f"{COMPLEX_KDA}/external/oellm-autoexp"))
VENV = Path(os.environ.get("AUTOEXP_VENV", str(AUTOEXP / "autoexp_venv"))) / "bin" / "python"
#: Our root config, versioned HERE and copied into the checkout at submit time.
#: hydra composes it against autoexp's own `config/` tree -- `defaults: ../base`
#: and `/container: ...` only resolve from inside it -- so it cannot simply be
#: pointed at from outside. Copying keeps the authority in this repo: the file
#: in the checkout is a build artefact and is overwritten every submission.
CONFIG_SRC = _HERE / "config" / "ckda_ladder_jupiter.yaml"
CONFIG_REL = "experiments/ckda/ckda_ladder_jupiter"
#: Set by --nodes to override nodes_for() for one submission. Module level
#: because chain_jobs is called from several places that have no argparse
#: namespace to thread it through.
FORCE_NODES = None

#: Where the runs land. autoexp reads $OUTPUT_DIR and falls back to ./output,
#: so leaving it to the shell puts a campaign wherever the submitting session
#: happened to point -- which it did once, scattering five smoke runs. Pinned
#: to the same root the torchtitan campaigns use, one directory per group.
OUTPUT_ROOT = os.environ.get(
    "CKDA_OUTPUT_ROOT", "/e/scratch/e-sta-openeurollm/poeppel1/complex_kda_runs")

#: The six arms. `qk` is Megatron's qk_layernorm; only the labelled attention
#: arm turns it on. `geom` names the arm whose matched MLP width to use --
#: the attention arms share `attn`'s, the hybrids their own.
ARMS = {
    "attn":                        dict(qk=False, geom="attn"),
    "attn-qknorm":                 dict(qk=True,  geom="attn"),
    "kda-sig-lowrank":             dict(qk=False, geom="kda-sig-lowrank",
                                        gate="sigmoid",         beta_max=1.0),
    "ckda-shipped-lowrank":        dict(qk=False, geom="ckda-shipped-lowrank",
                                        gate="signed_sigmoid2", beta_max=2.0),
    "kda-sig-hybrid-lowrank":      dict(qk=False, geom="kda-sig-hybrid-lowrank",
                                        gate="sigmoid",         beta_max=1.0,
                                        attn_every=4),
    "ckda-shipped-hybrid-lowrank": dict(qk=False, geom="ckda-shipped-hybrid-lowrank",
                                        gate="signed_sigmoid2", beta_max=2.0,
                                        attn_every=4),
}

#: The arm, as Megatron arguments.
#:
#: NO `--spec` AND NO ARM NAME. Megatron builds Complex KDA itself, through the
#: same mechanism it builds GatedDeltaNet and mLSTM with:
#: `experimental_attention_variant` says which mixer and `linear_attention_freq`
#: says where it goes. "Arm" is this study's word and means nothing there, so
#: what crosses the boundary is the axes the arm is made of.
#:
#: The two gate settings are what separate the signed arms from their
#: baselines: `signed_sigmoid2` puts alpha in (-1, 1) and `linear_beta_max=2.0`
#: puts beta in [0, 2]. Everything else is held across all six.
def arm_overrides(arm: str, n_layers: int) -> list[str]:
    """Megatron overrides for one arm, as `backend.megatron.*` settings."""
    spec = ARMS[arm]
    ov = [f"backend.megatron.qk_layernorm={'true' if spec['qk'] else 'false'}"]
    if "gate" not in spec:
        # The attention arms are plain Megatron; no variant, nothing to add.
        return ov

    # EVERY LAYER LINEAR, for the pure arms. `linear_attention_freq` marks
    # layer i as full attention when (i + 1) % freq == 0, so a freq larger than
    # the model is deep never marks one. It is derived from `n_layers` here and
    # not written as a literal, because the two have to move together: a rung
    # that grew past a hard-coded freq would quietly gain an attention layer.
    freq = spec.get("attn_every") or (n_layers + 1)
    ov += [
        "backend.megatron.experimental_attention_variant=complex_kda",
        f"backend.megatron.linear_attention_freq={freq}",
        f"backend.megatron.linear_gate_activation={spec['gate']}",
        f"backend.megatron.linear_beta_max={spec['beta_max']}",
        # Campaign-wide, not per arm: measured, dropping the q/k/v SiLU costs
        # the unsigned arms 0.015-0.021 nats and the signed arms nothing, so
        # setting it per arm would manufacture the comparison.
        "backend.megatron.linear_drop_qkv_silu=true",
    ]
    if spec.get("attn_every"):
        # A hybrid's full-attention layers are gated and NoPE, as in Kimi
        # Linear. Megatron's own attention is neither, and the gate alone is
        # hidden_size**2 parameters a layer -- so the default would build a
        # smaller model under the same name and break the parameter match.
        ov.append("backend.megatron.linear_hybrid_attention=gated_nope")
    return ov

#: (rung, tokens, global batch, lr) -- copied from the torchtitan ladders so
#: the cells line up one to one.
CELLS = {
    "ladder": [
        ("47M",  6e9,  32, 0.002), ("47M",  12e9, 64, 0.002), ("47M",  20e9, 64, 0.002),
        ("124M", 6e9,  64, 0.001), ("124M", 12e9, 64, 0.001), ("124M", 20e9, 64, 0.001),
        ("302M", 6e9,  64, 0.001), ("302M", 12e9, 64, 0.001), ("302M", 20e9, 128, 0.001),
    ],
    # 302M joined on 2026-09-17. The torchtitan ladder_beta it was copied from
    # stops at 124M, which left 302M the only rung with three budgets where
    # every other has five -- and 302M is where the advantage over qk-normed
    # attention peaks (-0.0546, the largest in the campaign), so it was the
    # rung carrying the most weight in alpha and the least support in D.
    #
    # b* at 302M/30BT and 302M/50BT is 128 by the same table the other rows
    # come from.
    "ladder_beta": [
        ("47M",  30e9, 64,  0.002), ("47M",  50e9, 128, 0.002),
        ("124M", 30e9, 128, 0.001), ("124M", 50e9, 128, 0.001),
    ],
}

#: The medium rungs, as CHAINS: (rung, global batch, lr, budgets sharing them).
#:
#: A chain is one constant-LR trunk plus one cooldown per budget, which is
#: WSD's whole economy -- run separately these fifteen cells are 118BT an arm,
#: chained they are 73BT. The grouping is not ours: a run cannot change its
#: batch halfway, and the reference's own b*(N, D) steps 64 -> 128 inside this
#: range, so budgets are grouped by the b* they share and each group is its own
#: chain. Copied cell for cell from `config/experiments/ladder_large.yaml`,
#: where the rows were derived from arXiv:2608.28308 Table 2 and snapped to
#: their grids; a test re-checks the two files against each other.
#:
#: 1.7B/6BT is a chain of one -- no other budget at that rung wants b* 64 --
#: and a chain of one is a plain single-stage WSD run, not a trunk with one
#: cooldown hanging off it.
CHAINS = {
    # DOES THE SCHEDULE ITSELF MATTER? One chain at 47M covering the two
    # budgets that share gbs 64, against the single-stage cells we already have
    # at exactly those budgets and hyperparameters.
    #
    # It should not matter. A cooldown branched at 0.8D and a standalone WSD run
    # to D see the same tokens under the same schedule shape, so with the
    # hyperparameters held they should land on the same loss.
    #
    # torchtitan measured chained 0.004-0.016 BELOW single-stage, and that is
    # very likely its in-order reader rather than the schedule: a 6BT run there
    # consumed one shard -- the one whose mean document length is 27% off the
    # corpus -- while a cooldown continued from the trunk's position much deeper
    # in. Megatron shuffles, so both arms of this comparison see statistically
    # the same data and the difference should disappear.
    #
    # If it does not disappear, `dchain` is real and every chained cell in the
    # medium rungs carries it. If it does, the offset in the lower ladder is
    # beta2 and biases alone, and dchain can be dropped from the fit.
    "ladder_chaincheck": [
        ("47M", 64, 0.002, [12e9, 20e9]),
    ],
    # 302M's upper budgets, chained. 20, 30 and 50BT all take b* 128, so they
    # are one b*-group and one trunk serves them: 40BT of trunk plus 16BT of
    # cooldowns against 80BT run apart, a 30% saving.
    #
    # 20BT is NOT in this chain. It already exists as a single-stage cell from
    # `ladder`, and re-running it chained would spend 20BT to replace a result
    # we have. The cost is that 302M ends up with three single-stage budgets
    # and two chained ones; `dchain` carries that, and the chaincheck at 47M
    # measures whether it is worth carrying at all.
    "ladder_302M_upper": [
        ("302M", 128, 0.001, [30e9, 50e9]),
    ],
    "ladder_medium": [
        ("588M", 64,  0.0005, [6e9, 12e9]),
        ("588M", 128, 0.0005, [20e9, 30e9, 50e9]),
        ("983M", 64,  0.0005, [6e9, 12e9]),
        ("983M", 128, 0.0005, [20e9, 30e9, 50e9]),
        ("1.7B", 64,  0.0005, [6e9]),
        ("1.7B", 128, 0.0005, [12e9, 20e9, 30e9, 50e9]),
    ],
}

#: The cooldown's share of a cell's budget, for the chains built here. Single
#: -stage cells get theirs from `aux.decay_fraction` in the root config, which
#: holds the same number; the two describe one schedule shape and
#: tests/lm_scaling/test_ladder_recipe.py checks they still agree.
COOLDOWN_FRAC = 0.2

#: Largest micro-batch that FITS at each rung. The value actually used is the
#: largest divisor of gbs/dp no bigger than this -- a cell with gbs 32 on four
#: GPUs can take at most 8, and Megatron makes up any remainder with gradient
#: accumulation.
MBS_CAP = {"47M": 16, "124M": 16, "302M": 8, "588M": 4, "983M": 4, "1.7B": 2}
#: The SMALLEST node count each rung fits in, from the torchtitan campaign's
#: micro-batch probes. A floor, not the answer: `nodes_for` scales up from here
#: until the job fits the wall.
MIN_NODES = {"47M": 1, "124M": 1, "302M": 1, "588M": 2, "983M": 2, "1.7B": 4}
#: The booster's QOS caps a job at 12 hours. Size for 10, because the estimate
#: is a rate times a token count and carries no allowance for startup, dataset
#: index building, or a slow node.
#:
#: This is not a refinement. At one node per rung, 53 of the campaign's 252
#: jobs ran past 12 hours -- a fifth of it, including 50BT cells at 27 hours --
#: and every one would have been killed at the wall with nothing to show.
WALL_TARGET_H = 10.0
MAX_NODES = 32
#: Margin over the estimate when asking SLURM for a time limit, and the floor
#: and ceiling around it.
#:
#: ASKING FOR 12 HOURS ON EVERY JOB IS NOT FREE. The backfill scheduler holds
#: capacity for the next large job, and a 12-hour request cannot fit in the
#: window before it starts -- so 252 one-to-eight-node jobs sat behind a
#: 512-node job with 2,227 nodes idle, none of them able to backfill. A 47M/6BT
#: cell needs about 90 minutes; asking for twelve hours buys nothing and costs
#: the whole gap.
#:
#: The margin was 2.5x, set before anything had been measured. 114 completed
#: cooldowns since then put the estimator's error at: median actual/predicted
#: 0.85, p90 1.00, max 1.70 -- and, the number that matters for sizing a wall,
#: WORST ABSOLUTE OVERSHOOT +0.35 h. The large ratios are all at 47M, where a
#: job is minutes long and fixed startup cost dominates; nowhere does the
#: estimate miss by more than 21 minutes of clock.
#:
#: So the margin is additive, not multiplicative: 0.75 h of slack plus 25% of
#: the estimate. At a 1-hour cell that is an hour of headroom against a 0.35 h
#: worst case; at the 8.75-hour 1.7B/50BT cooldown it is 2.9 hours. A flat
#: 2.5x asked 10 hours for a 4-hour job and could not backfill anywhere.
TIME_MARGIN = 1.25
TIME_SLACK_H = 0.75
TIME_FLOOR_H, TIME_CAP_H = 2, 12
#: A cooldown whose last logged learning rate is still above this never
#: annealed, whatever else its bookkeeping says. Same value the harvester
#: uses to reject such a run's loss.
LR_ANNEALED = 1e-4
SEQ = 4096
GPUS_PER_NODE = 4

#: Megatron's rate over torchtitan's, at the one cell where both were measured:
#: 47M attention, micro-batch 8, four GH200s -- Megatron 278,213 tok/s/GPU
#: against torchtitan's 414,854. Applied at every rung, which is the
#: PESSIMISTIC direction: 47M is overhead-bound, where a framework's fixed
#: costs matter most, so the true factor at 588M and above should be closer to
#: 1 and these estimates should come in high rather than low.
MEGATRON_FACTOR = 278_213 / 414_854
#: The regime the rates were measured in. `token_rate` refuses to fall back to
#: another one, so a typo here is a missing price and not a silent 2.3x error.
REGIME = "gh200_seq4096_titan_g4"
#: Arms with no rate of their own, and the arm whose rate stands in. qk_norm
#: adds one RMSNorm over head_dim 64 per q and k -- far below the noise on a
#: throughput measurement -- so attention's rate is the right stand-in, and
#: naming it here keeps `throughput.py`'s own proxy table about architectures
#: rather than about this ladder's labels.
RATE_ARCH = {"attn-qknorm": "attn"}


def price(arm, rung, tokens, gbs, mbs):
    """GPU-hours for one cell, or None where nothing was measured."""
    sys.path.insert(0, str(_REPO / "lm_scaling"))
    from throughput import token_rate

    rate = token_rate(RATE_ARCH.get(arm, arm), rung, mbs, prefer=REGIME)
    return None if not rate else tokens / (rate * MEGATRON_FACTOR) / 3600


def nodes_for(arm, rung, tokens, gbs):
    """Fewest nodes that finish inside `WALL_TARGET_H`, and divide the batch.

    Scaling out is not free: a wider job gives each rank a smaller share of the
    global batch, which caps the micro-batch lower and costs throughput per
    GPU. `price` sees that through the micro-batch, so the search buys wall
    clock with GPU-hours knowingly rather than by accident.
    """
    options = []
    for n in range(MIN_NODES[rung], MAX_NODES + 1):
        dp = n * GPUS_PER_NODE
        if gbs % dp:
            continue
        h = price(arm, rung, tokens, gbs, micro_batch(rung, gbs, dp))
        if h is None:
            return n          # unmeasured: take the narrowest that divides
        options.append((n, h, h / dp))
    if not options:
        return MIN_NODES[rung]

    # CHEAPEST THAT FITS THE WALL -- not the first that fits, and not the
    # widest.
    #
    # Scaling out shrinks the per-rank share of the batch, which caps the
    # micro-batch, which costs throughput. Past a point a wider job is BOTH
    # more expensive and no faster: ckda-shipped-lowrank at 302M/50BT ran the
    # search to 32 nodes, where gbs 128 over 128 ranks leaves a micro-batch of
    # 1 -- 1,750 GPU-h and still 13.7h of wall, against 276 GPU-h in 8 nodes
    # for the same cell on a sibling arm. The first draft returned the first
    # width under the target and, failing that, the widest tried; both pick
    # that pathology.
    fits = [o for o in options if o[2] <= WALL_TARGET_H]
    if fits:
        return min(fits, key=lambda o: o[1])[0]
    # Nothing fits: take the fastest, and among ties the cheapest.
    return min(options, key=lambda o: (round(o[2], 2), o[1]))[0]


def _exit_mins(wall, margin_min=15):
    """Minutes of training to allow inside a `HH:MM:SS` wall, leaving room to
    write a checkpoint. 15 minutes covers a 1.7B save with the optimizer."""
    hh, mm, _ss = (int(x) for x in wall.split(":"))
    return max(5, hh * 60 + mm - margin_min)


def time_limit(hours):
    """`HH:MM:SS` for an estimated wall-clock, with margin."""
    import math

    h = min(TIME_CAP_H, max(TIME_FLOOR_H,
                            math.ceil(TIME_MARGIN * hours + TIME_SLACK_H)))
    return f"{h:02d}:00:00"


def micro_batch(rung, gbs, dp):
    """Largest micro-batch that fits AND divides the per-rank share."""
    per_rank = gbs // dp
    return max(m for m in range(1, MBS_CAP[rung] + 1) if per_rank % m == 0)


def geometry():
    return json.load(open(_REPO / "lm_scaling" / "ladder_matched.json"))


#: Sequences in the held-out set, from the reference: "a common held-out set
#: containing 204,800 sequences (0.838BT)". They reach it with eval_iters 100
#: at gbs 2048; we divide instead, so every cell scores the SAME sequences --
#: Megatron's valid shuffle is a permutation of the whole split seeded by
#: --seed alone, so the first 204,800 of it do not depend on the batch size.
VALID_SEQS = 204_800


def cell_overrides(arm, rung, tokens, gbs, lr, table, group, smoke=False,
                   nodes=None):
    """`nodes` is passed by a CHAIN, which must run every one of its jobs at
    one world size: the torch-format checkpoint and the data iterator state
    both depend on it, so a segment that resumed at a different width would
    either refuse to load or silently restart the stream."""
    e = table[rung]
    spec = ARMS[arm]
    ffn = e["archs"][spec["geom"]]["d_ffn"]
    nodes = nodes or nodes_for(arm, rung, tokens, gbs)
    dp = nodes * GPUS_PER_NODE
    if gbs % dp:
        raise SystemExit(
            f"{arm} {rung} {tokens/1e9:.0f}BT: gbs {gbs} is not divisible by "
            f"dp {dp}; fix MIN_NODES")
    mbs = micro_batch(rung, gbs, dp)
    if VALID_SEQS % gbs:
        raise SystemExit(
            f"{arm} {rung}: gbs {gbs} does not divide {VALID_SEQS}; the cell "
            f"would score a different held-out slice than its neighbours")
    eval_iters = VALID_SEQS // gbs
    # An unmeasured cell gets the cap: no estimate, no trimming.
    est = price(arm, rung, tokens, gbs, mbs)
    wall = time_limit(est / (nodes * GPUS_PER_NODE)) if est else f"{TIME_CAP_H}:00:00"
    # One evaluation, at the end. Megatron always validates after the last
    # iteration, and sizing the valid dataset off eval_interval means a small
    # interval would ask the split for far more samples than it holds.
    eval_interval, save_interval = 10 ** 6, 10 ** 6
    tail = []
    if smoke:
        eval_iters, eval_interval = 5, 50
        wall = "00:20:00"
        tail = [
            "backend.megatron.train_iters=40",
            "backend.megatron.lr_warmup_iters=5",
        ]
    return [
        f"backend.megatron.hidden_size={e['d_model']}",
        f"backend.megatron.num_layers={e['n_layers']}",
        f"backend.megatron.num_attention_heads={e['n_heads']}",
        f"backend.megatron.num_query_groups={e['n_heads']}",
        f"backend.megatron.kv_channels={e['head_dim']}",
        f"backend.megatron.ffn_hidden_size={ffn}",
        f"backend.megatron.micro_batch_size={mbs}",
        f"backend.megatron.lr={lr}",
        "backend.megatron.adam_beta2=0.95",
        "backend.megatron.add_qkv_bias=false",
        "backend.megatron.add_bias_linear=false",
        "backend.megatron.rotary_base=10000",
        *arm_overrides(arm, e["n_layers"]),
        # A PRIVATE DATASET INDEX CACHE PER CELL.
        #
        # Megatron caches the document, sample and shuffle indices under one
        # path keyed by the blend. Cells that share a corpus, batch and budget
        # share that key -- which is most of this ladder, since the six arms at
        # a rung differ only in the mixer -- so starting them together has
        # several processes writing one file and the readers finding it half
        # written:
        #
        #     EOFError: No data left in file   (numpy.load, document_index)
        #
        # It killed attn-qknorm_302M_6BT three minutes in, and the branch's own
        # multilingual campaign carries the same fix with the same reasoning.
        #
        # The cost is that every cell rebuilds its indices instead of sharing
        # them. The order is unaffected: the shuffle is a permutation of the
        # epoch seeded by --seed, so rebuilding it reproduces it exactly.
        f"backend.megatron.data_cache_path={OUTPUT_ROOT}/{group}/.index/{arm}_{rung}_{tokens/1e9:g}BT_gbs{gbs}",
        f"backend.megatron.eval_interval={eval_interval}",
        f"backend.megatron.eval_iters={eval_iters}",
        f"backend.megatron.save_interval={save_interval}",
        f"++backend.env.CKDA_ARM={arm}",
        f"++backend.env.PYTHONPATH=.:submodules/Megatron-LM:{COMPLEX_KDA}",
        "++backend.env.TRITON_LIBCUDA_PATH=/opt/nvidia/lib",
        f"aux.gbs={gbs}",
        f"aux.tokens={int(tokens)}",
        # THE BUDGET BELONGS IN THE NAME. job.name is
        # ckda_<tag>_gbs<gbs>_lr<lr>_<stage> and base_output_dir follows it, so
        # without the budget the 6, 12 and 20BT cells of a rung that share a
        # batch and a learning rate all resolve to ONE directory: one --save,
        # one latest_checkpointed_iteration.txt, one tensorboard. Measured: 54
        # cells produced 30 directories, and with --load pointing at the same
        # path a 12BT cell could resume from the 6BT cell's checkpoint.
        f"aux.additional_tag={arm}_{rung}_{tokens/1e9:g}BT",
        f"aux.experiment_group={group}",
        f"slurm.sbatch.nodes={nodes}",
        f"slurm.sbatch.time={wall}",
        # SAVE BEFORE THE WALL, NOT AFTER IT. SLURM kills at the limit with no
        # useful grace -- the sbatch asks for no `--signal`, and KillWait is
        # tens of seconds, which is not a 1.7B write -- so a job that overruns
        # loses every iteration it ran. That is what happened to a fifth of
        # the first campaign. Megatron compares elapsed training time against
        # this every iteration and, if it is past, saves and exits cleanly.
        #
        # This does NOT rescue a job that would have finished: the wall is
        # 2.5x the estimate, so a healthy cell exits at about 40% of it and
        # never looks at this number. A cell that reaches it was going to be
        # killed anyway, and now leaves a checkpoint instead of nothing.
        f"backend.megatron.exit_duration_in_mins={_exit_mins(wall)}",
    ] + tail


def _set(ov, key, value):
    """Set `key` in an override list, replacing any entry already there.

    Hydra refuses a list carrying the same key twice, and appending to a list
    somebody else built is exactly how that happens. Every stage-specific
    change goes through here rather than through `+=`.
    """
    out = [o for o in ov if o.split("=", 1)[0] != key]
    out.append(f"{key}={value}")
    return out


def _iters(tokens, gbs):
    return -(-int(tokens) // (SEQ * gbs))


def trunk_reached(ckpt):
    """The iteration this chain's trunk has already been trained to, or 0.

    A resubmitted trunk segment is NOT harmless. stable0 carries no ckpt_step,
    so `--load <dir>` takes `latest` -- which after an earlier run is PAST its
    own train_iters. Megatron then decides training is done, trains nothing,
    and saves on exit, leaving the chain's checkpoint carrying a
    consumed-samples count from the wrong iteration. The cooldown that resumes
    from it derives its scheduler step from that count, never reaches the
    anneal, and reports a stable-phase loss as an endpoint.

    That is how attn_588M_decay12B and attn_983M_decay12B both failed: the
    trunk segments each logged zero iterations and saved anyway.
    """
    f = Path(ckpt) / "latest_checkpointed_iteration.txt"
    try:
        return int(f.read_text().strip())
    except Exception:
        return 0


def branch_view(ckpt, branch, make=False):
    """A private load directory whose tracker names exactly `branch`.

    ckpt_step IS NOT ENOUGH. It selects which iter_ directory to read, but the
    iteration Megatron then adopts comes from `state_dict['iteration']` after a
    fallback path that re-reads latest_checkpointed_iteration.txt -- so a
    cooldown only lands on its branch when the branch happens to BE the
    trunk's latest. Branch off any earlier segment and it loads the final
    trunk state instead, finds it past train_iters, and goes straight to
    "[after training is done]" having trained nothing. It does not fail: it
    logs "loading checkpoint ... at iteration 18312" and then "successfully
    loaded ... at iteration 76295" twelve lines later, exits 0, and leaves a
    directory that looks like a run. 48 of 86 queued cooldowns were doing this.

    So give each cooldown a directory of its own holding one relative symlink
    to the iteration it wants and a tracker naming that iteration. Then the
    fallback is not a fallback -- latest IS the branch -- and the result no
    longer depends on which trunk segment happened to write last.
    """
    ckpt = Path(ckpt)
    # NOT inside the trunk directory. A view is a throwaway; the trunk beside
    # it is 1.1 TiB that cost the campaign weeks. Keeping them in one place
    # means any future "clean up the views" is one glob away from deleting
    # iter_* as well, so they live in a sibling tree and the trunk directory
    # holds nothing but trunk checkpoints.
    d = ckpt.parent.parent / "ckpt_views" / ckpt.name / f"branch_{branch}"
    if make:
        it = f"iter_{branch:07d}"
        d.mkdir(parents=True, exist_ok=True)
        link = d / it
        if not link.is_symlink():
            link.symlink_to(os.path.relpath(ckpt / it, d))
        (d / "latest_checkpointed_iteration.txt").write_text(f"{branch}\n")
    return str(d)


def cooldown_ckpt(group, arm, rung, gbs, d):
    """Where one cooldown's final weights land.

    A sibling of the trunk tree, not a child of it, for the reason in
    `branch_view`: the trunk directory must contain only trunk checkpoints so
    that nothing aimed at cooldown output can reach them.
    """
    return (f"{OUTPUT_ROOT}/{group}/ckpt_decay/{arm}_{rung}_gbs{gbs}"
            f"/decay{d/1e9:g}B")


def cooldown_ckpt_legacy(group, arm, rung, gbs, d):
    """Where cooldowns saved before the split. Read, never written."""
    return (f"{OUTPUT_ROOT}/{group}/ckpt/{arm}_{rung}_gbs{gbs}"
            f"/decay{d/1e9:g}B")


def has_weights(group, arm, rung, gbs, d):
    """True only if this cell's ENDPOINT weights are on disk.

    Not merely "a checkpoint exists". A cooldown that loaded the wrong
    iteration trains nothing and then saves anyway, at whatever iteration it
    thinks it is at -- so the directory fills with a checkpoint from the
    trunk's final step, and a presence test reads that as a finished cell and
    skips the rerun. Checking the tracker against the iteration this cell is
    supposed to END at makes the test say what it means.
    """
    want = _iters(d, gbs)
    # AND THE RUN THAT WROTE THEM HAD TO ANNEAL. Weights at the right
    # iteration are not enough: a cooldown that resumed a trunk checkpoint
    # carrying a stale consumed_train_samples sits in the stable phase for its
    # whole run, never decays the learning rate, and then saves perfectly good
    # bookkeeping around a model that is not an endpoint. Three 983M/12BT
    # cells looked finished that way -- their losses were rejected by the
    # harvester for exactly this reason while their weights were accepted.
    runs = sorted(Path(f"{OUTPUT_ROOT}/{group}").glob(
        f"ckda_{arm}_{rung}_decay{d/1e9:g}B_gbs{gbs}_*"))
    for run in runs:
        logs = sorted(run.glob("slurm-*.log"))
        if not logs:
            continue
        lrs = re.findall(r"learning rate: ([0-9.E+-]+)", logs[-1].read_text())
        if lrs and float(lrs[-1]) > LR_ANNEALED:
            return False
    for where in (cooldown_ckpt, cooldown_ckpt_legacy):
        f = Path(where(group, arm, rung, gbs, d),
                 "latest_checkpointed_iteration.txt")
        try:
            if int(f.read_text().strip()) == want:
                return True
        except Exception:
            pass
    return False


def chain_jobs(arm, rung, gbs, lr, budgets, table, group, make_views=False):
    """One chain's jobs, in submission order.

    Yields `(tag, overrides, waits_for, tokens)` where `waits_for` is the tag
    of the job whose checkpoint this one resumes from, or None.

    THE TRUNK IS SEGMENTED, one segment per budget, and each segment stops
    exactly at a branch point. The obvious alternative -- run the trunk once
    and checkpoint periodically -- needs a period fine enough that rounding a
    branch point down to it wastes little, and at 1.7B a period that fine is
    hundreds of checkpoints and terabytes. Segmenting gives exactly one
    checkpoint per budget, at exactly the iteration wanted, and costs nothing:
    a segment resumes where the last one stopped, so the trunk is still run
    once end to end.

    THE COOLDOWN'S DECAY BEGINS AT ITS BRANCH by construction:
    lr_wsd_decay_iters is `iters - branch` and Megatron starts the WSD anneal
    at `train_iters - lr_wsd_decay_iters`. No iterations are re-run.
    """
    budgets = sorted(budgets)
    ckpt = f"{OUTPUT_ROOT}/{group}/ckpt/{arm}_{rung}_gbs{gbs}"
    marks = []
    for d in budgets:
        it = _iters(d, gbs)
        cool = int(COOLDOWN_FRAC * it)
        marks.append((d, it, cool, it - cool))

    # ONE WIDTH FOR THE WHOLE CHAIN, sized by its longest single job. The
    # segments differ in length by up to 3x, so sizing each on its own would
    # have a cooldown resume a trunk checkpoint at a different world size.
    prev, longest = 0, 0
    for _d, _it, cool, branch in marks:
        longest = max(longest, (branch - prev) * gbs * SEQ, cool * gbs * SEQ)
        prev = branch
    width = FORCE_NODES or nodes_for(arm, rung, longest, gbs)

    if len(budgets) == 1:
        d, it, cool, branch = marks[0]
        ov = cell_overrides(arm, rung, d, gbs, lr, table, group)
        yield f"{arm}_{rung}_single{d/1e9:g}B", ov, None, d
        return

    reached = trunk_reached(ckpt)
    prev_branch = 0
    for i, (d, it, cool, branch) in enumerate(marks):
        if branch <= reached:
            # Already trained past this segment's end. Submitting it would
            # load `latest`, train nothing, and re-save -- corrupting the
            # checkpoint the cooldowns depend on.
            prev_branch = branch
            continue
        ov = cell_overrides(arm, rung, d, gbs, lr, table, group, nodes=width)
        ov = _set(ov, "backend.megatron.train_iters", branch)
        ov = _set(ov, "backend.megatron.lr_wsd_decay_iters", 0)
        ov = _set(ov, "backend.megatron.save", ckpt)
        ov = _set(ov, "backend.megatron.load", ckpt)
        # train_iters changes from segment to segment, and the scheduler
        # asserts its loaded shape matches unless told otherwise. With
        # lr_wsd_decay_iters 0 the schedule is constant after warmup whatever
        # the horizon, so overriding changes nothing but the assert.
        ov = _set(ov, "backend.megatron.override_opt_param_scheduler", "true")
        # A cheap sanity eval, not the ladder's. The 204,800-sequence set is
        # for the cooldowns, which are the cells that get reported.
        ov = _set(ov, "backend.megatron.eval_iters", 10)
        ov = _set(ov, "aux.additional_tag", f"{arm}_{rung}_stable{i}")
        if i:
            # THE SAME BRANCH VIEW THE COOLDOWNS USE, and for the same reason.
            # A trunk segment resuming with ckpt_step=prev_branch against a
            # tracker that has moved on loads the LATEST state, finds its
            # train_iters already passed, trains nothing -- and saves anyway,
            # at its own iteration, carrying the consumed_train_samples of the
            # checkpoint it actually read. That is how four iter_0036622 came
            # to record 18312 x gbs: the weights are right, the counter is a
            # segment behind, and a cooldown branching there sits in the
            # stable phase for its whole run and never anneals.
            # RESUME FROM THE NEWEST CHECKPOINT THIS SEGMENT MAY USE, which
            # is not always its branch. exit_signal_handler does save on
            # scancel -- 16 seconds for a 983M optimizer state, against the
            # grace period I assumed was too short -- so a cancelled segment
            # leaves an iter_ of its own partway through. Restarting such a
            # segment from prev_branch throws that away: 4078 iterations in
            # the one case that prompted this.
            #
            # Strictly between prev_branch and this segment's branch. A
            # tracker at or past `branch` belongs to a LATER segment and
            # loading it is the bug this whole file has been chasing.
            resume = (reached if prev_branch <= reached < branch
                      else prev_branch)
            ov = _set(ov, "backend.megatron.load",
                      branch_view(ckpt, resume, make=make_views))
            ov = _set(ov, "backend.megatron.ckpt_step", resume)
            ov = _set(ov, "backend.megatron.exit_on_missing_checkpoint", "true")
        yield (f"stable{i}_{arm}_{rung}_gbs{gbs}", ov,
               (f"stable{i-1}_{arm}_{rung}_gbs{gbs}"
                if i and marks[i - 1][3] > reached else None),
               (branch - prev_branch) * gbs * SEQ)
        prev_branch = branch

    for i, (d, it, cool, branch) in enumerate(marks):
        ov = cell_overrides(arm, rung, d, gbs, lr, table, group, nodes=width)
        ov = _set(ov, "backend.megatron.train_iters", it)
        ov = _set(ov, "backend.megatron.lr_wsd_decay_iters", cool)
        ov = _set(ov, "backend.megatron.lr_warmup_iters", 0)
        ov = _set(ov, "backend.megatron.override_opt_param_scheduler", "true")
        # ALWAYS THE BRANCH. A cooldown cannot resume its own mid-anneal save,
        # tempting as it looks: exit_signal_handler does write one on scancel,
        # but cooldowns save with no_save_optim -- that is what keeps a 1.7B
        # endpoint at 3.5 GB instead of 30 -- so the checkpoint has no
        # optimizer state and loading it dies with
        #
        #     KeyError: 'optimizer'
        #
        # on every rank, three minutes in, at ANY width. Resuming one would
        # mean saving the optimizer for all 78 of them, an order of magnitude
        # of disk, to salvage the occasional cancelled anneal. Restarting from
        # the branch is 20% of a cell and correct.
        ov = _set(ov, "backend.megatron.load",
                  branch_view(ckpt, branch, make=make_views))
        ov = _set(ov, "backend.megatron.ckpt_step", branch)
        # Without this a missing checkpoint starts from scratch and reports a
        # loss, which is the one failure mode that does not look like one.
        ov = _set(ov, "backend.megatron.exit_on_missing_checkpoint", "true")
        # COOLDOWNS KEEP THEIR WEIGHTS, MODEL ONLY.
        #
        # These are the cells we report, and the only checkpoints worth
        # evaluating downstream -- a trunk checkpoint is a model stopped at
        # constant learning rate, mid-anneal, which is not what any benchmark
        # should be run against. They were `save=null` on the argument that a
        # cooldown produces a number and not a model, and that cost us every
        # final checkpoint in the campaign.
        #
        # Nothing ever resumes a cooldown, so the optimizer state is dead
        # weight: dropping the Adam moments and the fp32 master copy takes a
        # 1.7B checkpoint from about 30 GB to about 3.5 GB, which is the
        # difference between 1.2 TB and 130 GB across the 78 of them.
        ov = _set(ov, "backend.megatron.save",
                  cooldown_ckpt(group, arm, rung, gbs, d))
        ov = _set(ov, "backend.megatron.no_save_optim", "true")
        ov = _set(ov, "backend.megatron.no_save_rng", "true")
        ov = _set(ov, "aux.additional_tag", f"{arm}_{rung}_decay{d/1e9:g}B")
        # No dependency when the trunk segment it branches from is already
        # trained and therefore not submitted: the checkpoint is on disk, and
        # exit_on_missing_checkpoint makes an absent one fail loudly.
        yield (f"decay{d/1e9:g}B_{arm}_{rung}_gbs{gbs}", ov,
               (f"stable{i}_{arm}_{rung}_gbs{gbs}" if branch > reached else None),
               cool * gbs * SEQ)


def harvested_cells(group):
    """`{(arm, rung, budget_bt)}` already present in the harvest.

    --resume used to consult the QUEUE alone, which cannot see the state that
    actually matters: a chain that is neither queued nor complete. Both the
    588M and the 983M gbs128 chains sat in exactly that state for hours --
    skipped at refill because a segment was live, never refilled once it
    finished, and invisible to a resume that only asks what is running. Ten
    cells were unreachable and nothing reported it.
    """
    try:
        r = subprocess.run([sys.executable, str(_HERE / "harvest_megatron.py"),
                            group], capture_output=True, text=True, timeout=300)
    except Exception:
        return set()
    out = set()
    for line in r.stdout.splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 3:
            out.add((f[0], f[1], int(f[2])))
    return out


def queued_names():
    """Job names this user already has in the queue.

    Submitting 174 chained jobs takes minutes, one run_autoexp process per
    job, and an interruption partway leaves a set that is uneven rather than
    short -- two arms complete, one half done, three missing, and the chains
    among them broken. That happened twice in one afternoon, and cleaning it
    up cost more than the submission. With this, re-running the same command
    finishes the set instead of duplicating it.
    """
    try:
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h",
                            "-o", "%j"], capture_output=True, text=True,
                           timeout=60)
    except Exception:
        return set()
    return {l.strip() for l in r.stdout.splitlines() if l.strip()}


def queued_ids():
    """`{job name: SLURM id}` for this user's queue.

    `--only` resubmits one job out of a chain whose other members are still
    queued. Dropping its dependency would be wrong twice over: the trunk it
    branches from may not have written that iteration yet, and
    exit_on_missing_checkpoint would then fail it after it had waited in the
    queue. Keeping the dependency needs the id of a job THIS process never
    submitted, which is what this recovers.
    """
    try:
        r = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h",
                            "-o", "%j %i"], capture_output=True, text=True,
                           timeout=60)
    except Exception:
        return {}
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def job_name(ov):
    """The job.name our config builds, for comparison against the queue."""
    def v(key):
        for o in ov:
            if o.startswith(key + "="):
                return o.split("=", 1)[1]
        return ""
    return (f"ckda_{v('aux.additional_tag')}_gbs{v('aux.gbs')}"
            f"_lr{v('backend.megatron.lr')}_stable")


def submit(ov, config_name, label, waits_for_id=None):
    """Hand one job to run_autoexp; return its SLURM id, or None."""
    if waits_for_id:
        # sbatch_extra_directives is the only place autoexp passes a raw
        # directive through; its `sbatch` block is a declared dataclass and
        # will not take an undeclared `dependency` key.
        ov = ov + [f'++slurm.sbatch_extra_directives='
                   f'["--dependency=afterok:{waits_for_id}"]']
    # --yes: run_autoexp asks for confirmation above 100 GPU-h, and a prompt
    # read from a closed stdin is an abort, not a wait -- which is how the
    # whole 1.7B rung "failed to submit" with no error but the prompt itself.
    # The campaign as a whole is priced before anything is sent; this flag
    # declines to re-litigate that per job.
    cmd = [str(VENV), "scripts/run_autoexp.py", "--config-name", config_name,
           "--submit-and-exit", "--no-monitor", "--yes", *ov]
    env = dict(os.environ, PYTHONPATH=".", OUTPUT_DIR=OUTPUT_ROOT)
    r = subprocess.run(cmd, cwd=AUTOEXP, env=env, capture_output=True, text=True)
    ids = re.findall(r"runtime_job_id='(\d+)'", r.stdout + r.stderr)
    if ids:
        print(f"  {label:<44s} {ids[-1]}"
              + (f"  after {waits_for_id}" if waits_for_id else ""))
        return ids[-1]
    print(f"  {label:<44s} FAILED: {r.stdout[-300:]} {r.stderr[-300:]}")
    return None


def run_chains(a, table, group) -> int:
    """Submit the chained ladder, trunk segments and cooldowns together."""
    total, unpriced, n = 0.0, [], 0
    chains = [c for c in CHAINS[a.ladder] if a.rungs is None or c[0] in a.rungs]
    for arm in a.arms:
        for rung, gbs, lr, budgets in chains:
            jobs = list(chain_jobs(arm, rung, gbs, lr, budgets, table, group,
                                   make_views=a.submit))
            if a.only:
                jobs = [j for j in jobs if a.only in j[0]]
                if not jobs:
                    continue
            # BOTH HALVES, the same test as the per-job skip below. Checking
            # only the harvested loss skipped whole chains whose cells have a
            # number but no endpoint weights -- which, after every cooldown ran
            # under save=null, is all of them.
            if a.submit and not a.only and all(
                    (arm, rung, round(b / 1e9)) in a.done
                    and has_weights(group, arm, rung, gbs, b)
                    for b in budgets):
                print(f"  {arm:<30s} {rung:>5s} gbs{gbs:<4d} all "
                      f"{len(budgets)} cell(s) done, skipping chain")
                continue
            if a.submit and not a.only and any(job_name(ov) in a.already
                                               for _t, ov, _w, _k in jobs):
                # ALL OR NOTHING per chain. A segment already in the queue has
                # an id this process never saw, so its cooldowns could not be
                # given a dependency on it -- they would run against whatever
                # checkpoint happened to exist. Chains resume as a unit.
                print(f"  {arm:<30s} {rung:>5s} gbs{gbs:<4d} chain already "
                      f"queued, skipping whole chain")
                continue
            ids = {}
            names = {t: job_name(o) for t, o, _w, _k in
                     chain_jobs(arm, rung, gbs, lr, budgets, table, group)}
            # chain_jobs is re-run here only for the tag->name map; views were
            # already made by the call above, so make_views stays False.
            for tag, ov, waits, tokens in jobs:
                # --only DOES NOT MEAN --force. Without this, `--only decay`
                # resubmits every cooldown in the chain including the ones
                # already harvested, because the harvested-chain check above
                # is deliberately skipped when --only is set.
                # A CELL IS FINISHED WHEN IT HAS BOTH A LOSS AND WEIGHTS.
                # Harvested-alone was the old test, and it is what would now
                # skip every one of the 79 cooldowns that ran under save=null
                # -- the exact set we are trying to rebuild. Requiring the
                # checkpoint too makes --resume self-healing: a cell missing
                # either half comes back.
                m = re.search(r"decay([\d.]+)B", tag)
                if a.submit and a.resume and m and \
                        (arm, rung, round(float(m.group(1)))) in a.done and \
                        has_weights(group, arm, rung, gbs,
                                    float(m.group(1)) * 1e9):
                    print(f"  {arm:<30s} {tag:<36s} harvested, has weights")
                    continue
                # Per-job, not per-chain, under --only: a chain whose 30BT
                # cooldown is already queued still has 6, 12 and 20BT cells
                # that need rebuilding.
                if a.submit and a.only and job_name(ov) in a.already:
                    print(f"  {arm:<30s} {tag:<36s} already queued")
                    continue
                n += 1
                mbs = int([o for o in ov
                           if "micro_batch_size" in o][0].split("=")[1])
                h = price(arm, rung, tokens, gbs, mbs)
                if h is None:
                    unpriced.append(f"{arm} {rung}")
                else:
                    total += h
                if not a.submit:
                    print(f"  {arm:<30s} {tag:<36s} "
                          f"{tokens/1e9:>5.1f}BT" + (f"  after {waits}" if waits
                                                     else ""))
                    continue
                if waits and waits not in ids:
                    if a.only:
                        # Prefer the queued job over the checkpoint on disk:
                        # if the trunk segment is still waiting to run, its
                        # branch iteration does not exist yet.
                        queued = a.queued.get(names.get(waits, ""))
                        if queued:
                            print(f"  {tag}: after queued {waits} ({queued})")
                            ids[waits] = queued
                        else:
                            waits = None   # trunk already done, on disk
                    else:
                        print(f"  {tag}: SKIPPED, {waits} did not submit")
                        continue
                ids[tag] = submit(ov, a.config_name, f"{arm} {tag}",
                                  ids.get(waits))
                if ids[tag] is None:
                    del ids[tag]
    print(f"\n{n} jobs, {total:,.0f} GPU-hours at the torchtitan campaign's "
          f"measured rates x{MEGATRON_FACTOR:.2f} for Megatron")
    if unpriced:
        print(f"  unmeasured: {', '.join(sorted(set(unpriced)))}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ladder", choices=sorted(set(CELLS) | set(CHAINS)))
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS))
    ap.add_argument("--rungs", nargs="*", default=None)
    ap.add_argument("--submit", action="store_true",
                    help="actually submit; without it this only prints and prices")
    ap.add_argument("--only", metavar="SUBSTRING",
                    help="submit only the chain jobs whose tag contains this, "
                         "e.g. --only decay12B. Dependencies on jobs that are "
                         "NOT submitted are dropped, so the job resumes from "
                         "the checkpoint already on disk -- safe because every "
                         "resuming job carries exit_on_missing_checkpoint, so "
                         "an absent checkpoint fails loudly instead of "
                         "silently training from scratch.")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose job name is already in the queue, so "
                         "an interrupted submission can be finished by re-running")
    ap.add_argument("--smoke", action="store_true",
                    help="40 iterations of the 47M/6BT cell per arm, 20-minute "
                         "limit -- proves each arm's spec builds and descends")
    ap.add_argument("--nodes", type=int, default=None, metavar="N",
                    help="force this width instead of nodes_for(), to clear a "
                         "critical path. GPU-hours are flat with width until "
                         "the micro-batch has to shrink, and then they are "
                         "not: at 983M/gbs128, 16 nodes costs 6%% more than 4 "
                         "and 32 nodes costs 96%% more. A chain widened "
                         "part-way through resumes a checkpoint written at "
                         "another data-parallel size, which the legacy "
                         "optimizer path reshards (dp_zero_gather_scatter); "
                         "check_checkpoint_args enforces only TP and PP.")
    ap.add_argument("--config-name", default=CONFIG_REL)
    a = ap.parse_args(argv)

    global FORCE_NODES
    FORCE_NODES = a.nodes
    group = "megatron_arm_smoke" if a.smoke else f"megatron_{a.ladder}"
    a.already = queued_names() if (a.submit and (a.resume or a.only)) else set()
    a.done = harvested_cells(group) if (a.resume and a.submit) else set()
    a.queued = queued_ids() if (a.only and a.submit) else {}
    if a.resume and a.submit:
        print(f"  resume: {len(a.already)} job(s) queued, "
              f"{len(a.done)} cell(s) already harvested")
    if a.submit:
        src = _HERE / "config" / f"{Path(a.config_name).name}.yaml"
        if not src.is_file():
            raise SystemExit(f"no config {src} for --config-name "
                             f"{a.config_name}")
        dst = AUTOEXP / "config" / f"{a.config_name}.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        print(f"  config  {src.name} -> {dst}")
    table = geometry()
    if a.ladder in CHAINS:
        if a.smoke:
            raise SystemExit("--smoke is for the single-stage ladders; a chain "
                             "cannot be smoke-tested in 40 iterations")
        return run_chains(a, table, group)
    cells = [c for c in CELLS[a.ladder] if a.rungs is None or c[0] in a.rungs]
    if a.smoke:
        cells = cells[:1]
    total_gpu_h = 0.0
    unpriced = []
    n = 0
    for arm in a.arms:
        for rung, tokens, gbs, lr in cells:
            ov = cell_overrides(arm, rung, tokens, gbs, lr, table, group,
                                smoke=a.smoke)
            n += 1
            # GPU-hours do not depend on how many GPUs share the work -- four
            # GPUs finish in a quarter of the wall time and bill the same --
            # so there is no node factor here.
            mbs = int([o for o in ov if "micro_batch_size" in o][0].split("=")[1])
            spent = 40 * gbs * SEQ if a.smoke else tokens
            h = price(arm, rung, spent, gbs, mbs)
            if h is None:
                unpriced.append(f"{arm} {rung}")
            else:
                total_gpu_h += h
            if not a.submit:
                print(f"  {arm:<30s} {rung:>5s} {tokens/1e9:>3.0f}BT gbs{gbs:<4d} "
                      f"lr{lr} nodes{nodes_for(arm, rung, tokens, gbs)}")
                continue
            if job_name(ov) in a.already:
                print(f"  {arm:<30s} {rung:>5s} {tokens/1e9:>3.0f}BT  already queued")
                continue
            if (arm, rung, round(tokens / 1e9)) in a.done:
                print(f"  {arm:<30s} {rung:>5s} {tokens/1e9:>3.0f}BT  already harvested")
                continue
            submit(ov, a.config_name,
                   f"{arm} {rung} {tokens/1e9:.0f}BT")
    print(f"\n{n} cells, {total_gpu_h:,.0f} GPU-hours at the torchtitan "
          f"campaign's measured rates x{MEGATRON_FACTOR:.2f} for Megatron")
    if unpriced:
        print(f"  {len(unpriced)} cell(s) NOT in that total, unmeasured: "
              + ", ".join(sorted(set(unpriced))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
