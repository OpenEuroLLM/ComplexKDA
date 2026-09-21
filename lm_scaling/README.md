# `lm_scaling` — the language-modelling experiments

Two campaigns, and they run on different backends:

* **the scaling ladder** — 6 arms across six rungs (47M to 1.7B) and up to five
  token budgets, on Nemotron-CC. How the arms scale against a tuned
  transformer. It runs on **Megatron-LM**, through `megatron_ext/` and the
  pinned stack in [`megatron_stack.lock`](megatron_stack.lock); its root config
  is [`megatron_ext/config/`](megatron_ext/config/) and the record of what has
  finished is `harvest/megatron_ladder.tsv`.
* **the 1.3B replication** — 4 arms at 1.3B on 100B tokens of FineWeb-Edu under
  Gated DeltaNet's published recipe, evaluated downstream against the tables
  other groups report in. It runs on **torchtitan**, through `titan_ext/` and
  [`config/experiments/fwedu_1p3B.yaml`](config/experiments/fwedu_1p3B.yaml).
  Described in full below.

An earlier torchtitan ladder is superseded and is not part of this release: the
two frameworks disagreed by 0.049 nats on identical configuration, and the
ladder moved to Megatron to remove that term. The arm table both use
(`ladder_matched.json`) is shared, which is what made the move checkable.

## Reproducing the paper's numbers without a cluster

The expensive results are **committed**, so every table and figure in the paper
regenerates from this checkout alone — no cluster, no GPU, no downloads:

```
lm_scaling/reproduce_tables.sh --check   # rewrite tex/ in a scratch dir and diff
lm_scaling/reproduce_tables.sh           # rewrite tex/ in place
```

`--check` is the claim in executable form: if it reports no differences, every
number in `tex/` is exactly what the data in `harvest/` produces.

Nearly all of its runtime is the scaling fits, each with a bootstrap over it —
tens of minutes at the paper's settings, since the holdout refits everything a
second time on the reduced ladder. To exercise the whole path quickly instead:

```
STARTS=40 BOOT_STARTS=2 THREE_REPS=20 REPS=20 lm_scaling/reproduce_tables.sh --check
```

That runs every script and writes every file in a couple of minutes, and it
will report differences in the fit tables — fewer restarts and fewer replicates
is a different computation. The well-determined numbers survive it (the shared
irreducible loss comes out at 1.1852 either way); the intervals and the
poorly-determined parameters do not. It checks that the pipeline runs, not that the numbers
match.

What it runs, and what each part reads:

| what | from | command |
|---|---|---|
| **the scaling result: two fits, largest cell held out** | the same | `scaling_holdout.py --form {skaling,chinchilla}` |
| the holdout figures | `tex/scaling_holdout*.json` | `holdout_plot.py` |
| scaling figures | the same | `scaling_plots.py --out tex` |
| the ladder's validation losses | the same | `megatron_ext/compare_table.py --latex` |
| ladder downstream pairing | `harvest/ladder_downstream/` (180 cells) | `ladder_downstream.py --latex PATH` |
| 1.3B downstream tables | `harvest/downstream/` (4 arms) | `eval_table.py --step 190976 --convention gdn2` |
| 1.3B RULER needles | `harvest/ruler/` (20 files) | `niah_table.py --format latex --step 190976` |
| 1.3B recall suite | `harvest/recall/` | `eval_table.py ... --with-recall` |
| the gate spectrum | `harvest/gate_spectrum/` (13 files) | `gate_spectrum_plot.py --latex`, `--out tex` |
| the architecture table | `ladder_matched.json`, `replication_1p3B.json` | `arch_table.py` |

Every one of these defaults to the committed data, so they run with no
arguments. Each also takes `--results`/`--data`/`--tsv` pointing at a fresh run,
so a reproduction can be diffed against ours cell by cell. Numbers will differ in
the last digit or two — different node counts, a renamed arm, ordinary
nondeterminism — and the seed-noise floor of this pipeline is ~0.002 nats,
measured.

`tex/` holds the generated tables and figures exactly as the paper uses them;
every file says at the top which command wrote it.

## The 1.3B / 100BT FineWeb-Edu experiment

| | |
|---|---|
| **arms** | `kda-sig-lowrank`, `ckda-shipped-lowrank`, and their 3:1 hybrids |
| **geometry** | 2048 wide, 24 layers, 16 heads of 128, vocab 32,000, untied |
| **recipe** | AdamW 4e-4, wd 0.1, clip 1.0, cosine to lr/10 after a 1B-token warm-up, 0.5M-token batches, 4,096-token sequences, seed 3407 |
| **budget** | 190,976 steps = 100.1 BT, one pass, no cooldown tail |
| **as run** | 8 nodes × 4 GH200 per arm on JUPITER, ~3,300 GPU-hours total |
| **result** | [`config/experiments/fwedu_1p3B_results.md`](config/experiments/fwedu_1p3B_results.md) |

The experiment is a **pairing**. Within a pair the two arms differ in the decay
gate and in nothing else — same width, same data, same order, same seed, same
schedule — so a difference between them is attributable to the gate. The
baseline is the bounded sigmoid gate (`kda-sig`), not fla's softplus, because
against it a signed-vs-positive comparison changes the gate's *sign* only.

**What the result supports, and what it does not.** All four arms beat every
published row of both comparison tables. Scored without a BOS token, the
signed arm *trails* the downstream average by 0.04 points on the pure pair and
leads by 0.45 on the hybrid, while *trailing* on held-out loss in both (0.0029
and 0.0014 nats). At n = 1 per arm neither difference is
resolvable, and the same pairing measured across a companion scaling ladder
goes the other way. The supported claim is that the signed gate **matches** the
baseline at this scale. The results document states this at length; please keep
the two tables together, which is why `eval_table.py --format paper` emits both.

---

## What is here

```
config/               the campaigns, as configs — see config/README.md
ladder_spec.yaml      the ladder's rungs, budgets and fitted laws, as data
ladder.py             the loader over it
ladder_matched.py     generates ladder_matched.json — every arm's width per rung
harvest/              THE COMMITTED RESULTS (see the table at the top)
  megatron_ladder.tsv     180 ladder cells: arm, rung, budget, batch, lr, steps, loss
  ladder_downstream/      the same 180 cells, scored on the 9-task suite
  downstream/             the four 1.3B arms at step 190,976
  ruler/                  the 1.3B RULER needle results
  recall/                 Based's recall suite at the same checkpoints
                          (all three scored WITHOUT a BOS token, which is how
                           this corpus was tokenised; each result carries a
                           .meta.json recording the convention)
  gate_spectrum/          the transition spectrum along two 1.3B runs
tex/                  the generated tables and figures, as the paper uses them
reproduce_tables.sh   regenerate all of tex/ from harvest/; --check diffs instead

scaling_fit.py        the forms, the fitters and the loader -- the library the
                      two below fit through. Its own --latex tables are
                      in-sample and are not part of the paper
scaling_fit_variants.py  the same ladder under two treatments of E, with
                      bootstrap intervals (isolated / pinned). IN SAMPLE, and
                      its tables are not shipped -- the paper reports the
                      holdout below, which reuses these fits as its control
scaling_holdout.py    those two fits again with the largest cell HELD OUT, and
                      the error there -- the paper's scaling result
holdout_plot.py       its three panels, from the JSON the above writes
scaling_plots.py      the Chinchilla-style figures
ladder_downstream.py  the baseline-vs-complex pairing over the ladder
arch_table.py         the architecture table: geometry per rung, and the recipe
plot_style.py         one matplotlib style for every figure here
reference/            Ajroldi et al.'s published losses, and the fit over them
niah_table.py         renders the RULER table
stage_ruler.py        builds the RULER samples (once, before eval_ruler.sbatch)
eval_ruler.sbatch     scores the needles
stage_recall.py       builds Based's recall suite; eval_recall.sbatch scores it
recall_table.py       renders it on its own (eval_table.py --with-recall joins it)
recall_failures.py    what a recall miss actually looks like, per sample
eval_ladder.sbatch    scores the ladder cells downstream
gate_spectrum.py      measures the transition spectrum of a checkpoint
gate_spectrum.sbatch  runs that over every checkpoint of a run
gate_spectrum_plot.py draws it -- six figures and a table
harvest_runs.py       a torchtitan campaign's finished runs, as one row each
collect_results.py    gathers the evaluation JSONs a campaign produced
titan_ext/            the torchtitan integration: fla layers as a train spec
megatron_ext/         the Megatron backend: the ladder's submitter, its
                      harvester, its config and a login-node smoke (see below)
data/                 staging FineWeb-Edu as Megatron .bin/.idx, and reading it
eval_tasks/           task overrides for the three datasets that need them
slurm/, templates/    the tokenize job, and the sbatch template

plan.py               expand a campaign into jobs and price it, without submitting
submit.py             write each job's three files, register them, optionally monitor
watch.py              what a running campaign is doing, in one screen
schema.py             the typed config classes
resolvers.py          the citations: ${matched:}, ${rung:}, ${peak_mbs:}, ${flavor:}
geometries.py         every geometry a campaign can name, as one table
param_match.py        what each arm IS, and the parameter matching
replication_1p3B.py   generates replication_1p3B.json — the arm table
throughput.py         measured throughput, for wall-clock estimates
titan_mbs_probe.py    measure the micro-batch ceiling of each arm

export_hf.py          a torchtitan DCP checkpoint -> a transformers directory
check_export_loss.py  score the export on the run's own validation windows
eval_downstream.py    lm-eval-harness over the published task suite
eval_table.py         render the results as the published tables
eval_fwedu.sbatch     export + check + evaluate, one arm per GPU

container_run         run something inside the container
login_run             run something in the login venv (the one that can reach Slurm)
make_login_venv.sh    build that venv
make_megatron_stack.sh  clone + pin the Megatron backend's external stack
megatron_stack.lock   the commits that stack is pinned to
                      Two of the commits it pins are ours, on a branch named
                      feat/complex-kda in each upstream repository
sync_to_cluster.sh    push this checkout to a cluster
```

Two measurement tables sit beside the code and are **data, not settings**:
`peak_mbs.yaml` (the largest micro-batch measured to fit) and `token_rate.yaml`
(measured tokens/s/GPU). `throughput.yaml` records TFLOP/s for reporting only —
pricing goes through the token rate, because converting between them needs a
FLOPs-per-token convention and getting it wrong is a silent 3×. A missing entry
in any of them **raises**; nothing is interpolated.

### A note on two names you will see in comments

* **`lm/`** is this study's first trainer — a plain FSDP2 PyTorch loop — which
  is not part of this release. Comments citing it are explaining why a value is
  what it is ("matching `lm/`", "loss matching against `lm/` sat 4% apart"); the
  behaviour they describe is recorded where it matters.
* **the scaling ladder** is the companion experiment (six arms across 47M–302M
  and 6–20BT). It is released separately. It is cited here only where its
  measurements bear on how this campaign's numbers may be read — above all in
  the downstream caveat, which is the reason this release does not claim a
  downstream win.

---

## Requirements

Five things, and they install in four different places. The split is not
incidental — it is what the two wrapper scripts exist to encode, and the top row
is separated from the rest by design:

| what | where it lives | who needs it |
|---|---|---|
| numpy, scipy, matplotlib, pyyaml, `scienceplots` | **anywhere** | regenerating the tables and figures |
| `hydra_staged_sweep`, `slurm_gen`, `monitor` | the **login venv** (`.venv`) | planning, submitting, monitoring |
| torchtitan + titan-oellm | the **container image** | the training job |
| `fla` (this checkout) | `PYTHONPATH`, ahead of the image | the training job |
| tilelang | the **container overlay** | the training job, on Hopper only |

**The first row is the only one a reader needs.** `reproduce_tables.sh`
rebuilds every table and figure in `tex/` from the committed measurements, and
it deliberately depends on nothing below that row — no Slurm, no container, no
orchestration layer, no GPU. Everything the campaigns resolved at submission
time and the tables still report ships as data: the ladder's recipe is read back
out of `megatron_ext/submit_ladder.py`, and the FineWeb-Edu column's out of
`campaign_recipes.json`.

**Two environments, split by who needs Slurm.** `sbatch` is a host binary
talking to the Slurm controller over a socket and configuration the image does
not carry, so `apptainer exec … submit.py --run` would write every job file,
register them, and then submit none — failing after the part that looks like
the work. Planning and submitting therefore run in the login venv
(`login_run`), and anything that builds a model runs in the container
(`container_run`). A training job needs neither wrapper: the rendered sbatch
enters the image itself.

### 1. The orchestration libraries

Not on PyPI. `make_login_venv.sh` builds `.venv` and does exactly this:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install --upgrade \
    pip compoconf omegaconf "hydra-core>=1.3" networkx tomli-w pyyaml
.venv/bin/python3 -m pip install --force-reinstall --no-deps \
    git+https://github.com/kpoeppel/hydra_staged_sweep \
    git+https://github.com/kpoeppel/slurm_gen \
    git+https://github.com/kpoeppel/monitor
```

**`--force-reinstall --no-deps` is not optional on the three git packages.**
They are pinned to a BRANCH, not a version, so their version string does not
move between commits and `--upgrade` alone decides they are already satisfied:
the install prints nothing, exits 0, and the venv keeps the old code.

The script then checks the venv can actually *resolve a campaign* and *find
`sbatch`*, rather than merely import — a venv that imports everything and
cannot see the Slurm binaries is the same failure one layer down. It also
asserts torch was **not** pulled in, because the day one of these tools grows a
torch import is the day this venv silently needs 3 GB.

```bash
lm_scaling/make_login_venv.sh
```

### 2. torchtitan and titan-oellm

The training backend. `config/backend/titan.yaml` launches
`python -m torchtitan.train`, reads its config through titan-oellm's
`titan_oellm.configs.sci_job_config`, and registers this repo's train spec via
`custom_import: lm_scaling.titan_ext`.

Both ship **inside the apptainer image** named in
`config/container/jupiter_titan.yaml` (`titan_0.2.1_aarch64_*.sif`), which is
how the published runs got them, and neither is installed by anything here.
To build an environment yourself instead:

```bash
pip install git+https://github.com/pytorch/torchtitan        # the trainer
pip install git+https://github.com/OpenEuroLLM/titan-oellm   # the job config + qwen3_custom spec
```

Pin both. `titan_ext` swaps the mixer into titan-oellm's `qwen3_custom` spec
and copies its `reference_layer`, so it is coupled to that spec's shape — and
torchtitan has renamed device-mesh dimensions under it before, which surfaces
as `KeyError: Invalid mesh_dim_names` at startup rather than as a version
complaint.

**`fla` must come first on `PYTHONPATH`.** The image ships its own
flash-linear-attention, which has neither the signed gate nor the knobs the
arms set. Every rendered job puts this checkout ahead of it
(`slurm.env.PYTHONPATH`); `container_run` and `login_run` do the same
interactively. Get this wrong and the run trains a different layer without a
word.

### 3. tilelang — required on Hopper, not preferred

`data/make_overlay.sh` builds the apptainer overlay that carries the
orchestration libraries into the container, and adds tilelang. It takes the
image and the overlay path from `config/container/<name>.yaml`, defaulting to
this cluster's titan config:

```bash
bash lm_scaling/data/make_overlay.sh                          # <cluster>_titan
CKDA_CONTAINER=other bash lm_scaling/data/make_overlay.sh      # any other container config
```

Training needs exactly one thing from that overlay, and only on GH200: the
image ships Triton 3.6.0, and `fla` **refuses** the gated chunk backward in
that band —

```
RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect
results for gated chunk_bwd_dqkwg. Please upgrade Triton to >= 3.7.1 or
install tilelang
```

— which is a correctness guard, not a performance one. `FLA_TILELANG: "1"` in
`config/slurm/jupiter.yaml` turns it on. On Ampere the guard never fires and
the overlay is orchestration-only.

### Megatron-LM

The name appears in two unrelated roles, and it is worth separating them.

**As a data format**, Megatron needs nothing installed.
`data/megatron_indexed.py` reads the `.bin`/`.idx` files
`prepare_fineweb_edu.py` writes — a few hundred lines of `struct` and `mmap`
with no Megatron dependency at all. That is how the 1.3B campaign, which trains
on torchtitan, uses the word.

**As a training backend**, Megatron-LM runs the scaling ladder: all 180 cells
in `harvest/megatron_ladder.tsv`, submitted by
[`megatron_ext/submit_ladder.py`](megatron_ext/submit_ladder.py). The ladder is
on a second backend because the two frameworks did not agree: at 47M/6BT a
matched torchtitan run lands 0.049 nats behind Megatron on the same machine,
corpus and hyperparameters, after every configuration difference we could find
had been measured and set.

**Complex KDA is a first-class attention variant in the fork**, alongside
GatedDeltaNet and mLSTM: `experimental_attention_variant=complex_kda` selects
the mixer and `linear_attention_freq` places it, which is how Megatron already
expresses a hybrid (Qwen3-Next is `gated_delta_net` at frequency 4). The beta
half of the model needs nothing new — `linear_beta_max=2.0` is exactly the
extended Householder range, and the fork already documents it as such. The gate
fields carry the alpha half.

An "arm" is therefore a combination of settings rather than a name Megatron
knows; `submit_ladder.arm_overrides` is the mapping, and
`tests/lm_scaling/test_megatron_variant.py` checks that the variant builds the
same layer `param_match` specifies — same class, parameter count, state-dict
and gate settings — because two tables that must agree is exactly the
arrangement that goes wrong quietly.

The second commit is autoexp's three mirrors of Megatron's argument table, which
the fields have to pass through.

Those are **two ordinary commits**, one in each repository, on a branch named
`feat/complex-kda`, and the lockfile pins them like everything else:

```
oellm-autoexp           6cdc977a  on top of bd5a54b4 (hybrid_exp)
submodules/Megatron-LM  40a52158  on top of 514b31b3
```

Every field defaults to the underlying layer's own value, so both commits are
inert unless the variant is selected — which is what makes them reasonable to
carry upstream. Tensor parallelism is refused rather than ignored: the fla
layer's projections are plain `nn.Linear`, so a larger TP would replicate the
mixer while sharding the rest of the layer.

To check the variant end to end on one GPU of a login node, with no
allocation and no slurm:

```bash
lm_scaling/megatron_ext/smoke_variant.sh          # ten iterations of each arm
```

It takes its flags from `submit_ladder.arm_overrides`, so it exercises the
settings the ladder submits rather than a copy of them. It is worth having
because Megatron dispatches on the variant in more than one place — the model
spec and the FLOP accounting — and a run whose model is built correctly still
dies after the first step if only the first was taught about it. That is
exactly what happened on the first attempt here.

Until those branches are pushed, the two commits resolve only in a local
checkout. `make_megatron_stack.sh --verify` says so in as many words rather
than letting a fresh clone fail on `reference is not a tree`, and the notice
disappears by itself once the branches exist on the remote — nothing in this
repository needs editing when that happens.

Its stack is pinned by [`megatron_stack.lock`](megatron_stack.lock) and
materialised by one script:

```bash
bash lm_scaling/make_megatron_stack.sh            # clone, check out, install
bash lm_scaling/make_megatron_stack.sh --verify   # check a tree, change nothing
bash lm_scaling/make_megatron_stack.sh --relock   # adopt what is checked out
```

It clones `oellm-autoexp` at the locked commit into `external/` (gitignored —
what belongs in git is the pin, not another project's tree), initialises its
submodules, **checks every resolved commit against the lock**, then builds
autoexp's own venv and installs the three orchestration libraries at their
locked commits. `--verify` is the half worth running in anger: a pin that is
never checked is a comment.

Two things it works around, both of which otherwise cost an afternoon:

* **Three of autoexp's four submodules are declared with `git@github.com:`
  URLs**, so a clone without a GitHub SSH key fails on repositories that are
  all public, and the error names SSH rather than the real problem. The script
  rewrites them to HTTPS with `-c url.…insteadOf`, scoped to its own commands
  so nothing lands in your global git config.
* **`hybrid_exp` is a branch, and branches move.** The lock pins a commit; the
  branch is only how to fetch it cheaply. Refreshing a pin is `--relock`, a
  deliberate act, never a side effect of setting the tree up.

**Every repository in that stack is public** (checked 2026-09-17), so the
software side is reinstallable in principle by anyone:

| repository | licence | role |
|---|---|---|
| [OpenEuroLLM/oellm-autoexp](https://github.com/OpenEuroLLM/oellm-autoexp) `hybrid_exp` | Apache-2.0 | the launcher; `scripts/run_autoexp.py` renders and submits each cell |
| [OpenEuroLLM/NVIDIA-Megatron-LM](https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM) | NVIDIA's own | the trainer, an OpenEuroLLM fork of Megatron-LM 0.18 |
| [OpenEuroLLM/titan-oellm](https://github.com/OpenEuroLLM/titan-oellm) | Apache-2.0 | also a submodule here; the torchtitan job config |
| [kpoeppel/hydra_staged_sweep](https://github.com/kpoeppel/hydra_staged_sweep), [slurm_gen](https://github.com/kpoeppel/slurm_gen), [monitor](https://github.com/kpoeppel/monitor) | Apache-2.0 | the orchestration layer, shared with step 1 |

**The container image is the part that is not public.** Megatron runs from its
own image (fla + liger + mamba-ssm), built internally and living beside the
titan one under `$CONTAINER_CACHE_DIR`. Rebuilding it from public sources is
the one step of this stack that nothing here automates — and it is the real
limit on reproducing the ladder elsewhere, not repository access.

Its **overlay**, though, is the same script rather than a new one:
`make_overlay.sh` takes the image and the overlay path from
`config/container/<name>.yaml`, so adding that container config is the whole
job.

```bash
CKDA_CONTAINER=jupiter_megatron bash lm_scaling/data/make_overlay.sh
```

That container config is not here yet, because the code that would mount it is
not either. **The ladder's own Megatron config is**, in
[`megatron_ext/config/`](megatron_ext/config/): it is this study's file rather
than a borrowed one, so it belongs in the repository that publishes the numbers
it produced, and it is readable on its own even before the Python that submits
it arrives. What it composes against is pinned by the lockfile above.

Everything else the Megatron path needs is already here too: the corpus
staging, the reader, the arm table, and `throughput.py` — which prices the
ladder's cells against this campaign's measured token rates.

### Another cluster

Everything site-specific is in `config/cluster/`, `config/slurm/` and
`config/container/`; add an entry for your machine and point
`config/autoexp.yaml` at it. Two files carry site values that are *not* config
and must be edited by hand: `eval_fwedu.sbatch` (its `--account`, `--partition`
and `--output` directives, and the `REPO`/`RUNS`/`OUT` defaults, all
overridable by environment except the directives) and
`slurm/fwedu_tokenize_juwels.sbatch`. The files here are the ones the published
runs used, unedited.

---

## Re-running it

### 1. Stage the corpus

FineWeb-Edu `sample-100BT` under the **Llama-2** tokenizer, `</s>` after every
document and no BOS — Gated DeltaNet's convention, not fla-hub's (whose
tokenizer beside those checkpoints is Mistral's). 140 parquet shards in, ~116BT
of Megatron `.bin`/`.idx` out, of which 100BT is one pass.

```bash
# 1. a node with network, into $P:
HF_HOME=$HF python3 lm_scaling/data/prepare_fineweb_edu.py download --out $P
# 2. CPU nodes, one array task per shard (~2 min each):
sbatch --array=0-139%40 lm_scaling/slurm/fwedu_tokenize_juwels.sbatch
# 3. move the shards where training reads them, then build the splits:
rsync -a $P/tokenized_llama2/ $E/tokenized_llama2/
python3 lm_scaling/data/prepare_fineweb_edu.py build --out $E
lm_scaling/container_run lm_scaling/data/verify_corpus.py fwedu_1p3B
```

Point `FWEDU_CORPUS` at `$E` (see `config/data/fineweb_edu_llama2.yaml`). The
`verify_corpus.py` step is not optional bookkeeping: it reads the staged files'
own headers and checks them against the token counts the config states, and
`plan.check` refuses a run longer than its corpus — the reader would otherwise
wrap around and silently train the beginning twice.

The train order is one seeded interleave of all 140 shards. Since the run is a
single pass, **the order is the curriculum** and its tail is what the cosine
anneals into.

### 2. Plan, and read the plan

```bash
lm_scaling/login_run lm_scaling/plan.py fwedu_1p3B
```

Four jobs, 190,976 steps each, mbs 4, no gradient accumulation, ~3,306 GPU-h.
`plan.check` runs first and refuses a campaign that would not be the experiment
it claims to be — a batch torchtitan rejects at startup, an output directory
the compute nodes cannot see, a final step that is never validated.

To look at what would be submitted without submitting it:

```bash
lm_scaling/login_run lm_scaling/submit.py fwedu_1p3B --out /tmp/fwedu
```

Three files per job: `titan.toml` (the backend's own config), the `.sbatch`,
and `config.yaml` — the resolved campaign config, archived beside the run. The
third is not a convenience; a number whose config cannot be read back has no
provenance.

### 3. Run

```bash
lm_scaling/login_run lm_scaling/submit.py fwedu_1p3B --write --run
lm_scaling/login_run lm_scaling/watch.py  fwedu_1p3B --loop 60
```

`--run` starts the monitor, which resubmits on wall-clock timeout and on the
launch flakes that are known to be transient, and deliberately **not** on OOM.
Each arm takes about three 12-hour allocations. `titan_ext/wall.py` writes one
more checkpoint about three minutes before the allocation ends, so a requeue
loses minutes rather than a checkpoint interval.

Checkpoints land every ~5BT and all twenty are kept, so the run can be
evaluated along training and not only at its end.

### 4. Evaluate

#### Loading a checkpoint

The published weights load as they are. `OpenEuroLLM/complex-kda-1.3B-100B`
carries `model_type: complex_kda` and bundles its own
`configuration_complex_kda.py` / `modeling_complex_kda.py`, so it loads with
`trust_remote_code=True` and no `fla` installed at all. Checked against this
package: both routes build `ComplexKDAForCausalLM` with 1,362,630,016
parameters and identical state-dict keys and shapes — the CKDA row of
`tex/arch_table.tex` (1362.63M).

**Older exports are a different matter.** The architecture was called Signed
KDA, and this package registers `complex_kda` only — deliberately, with no
alias, so that a checkpoint and the code loading it cannot disagree silently.
Anything exported before the rename (every directory under the cluster's
`eval_fwedu/export/`, for instance) fails at load:

```
ValueError: The checkpoint you are trying to load has model type `signed_kda`
but Transformers does not recognize this architecture.
```

Re-export it with `export_hf.py`, which stamps the current names. If you only
have the HuggingFace directory, two fields in its `config.json` are the whole
difference — everything else in the file is unchanged:

```json
"architectures": ["ComplexKDAForCausalLM"],
"model_type": "complex_kda"
```

Verified against a real pre-rename export: after that edit
`AutoModelForCausalLM` builds `ComplexKDAForCausalLM`, 1.363B parameters, with
`ComplexKimiDeltaAttention` as the mixer.

```bash
STEP=190976 sbatch lm_scaling/eval_fwedu.sbatch
lm_scaling/eval_table.py --step 190976 --convention gdn2
lm_scaling/eval_table.py --step 190976 --convention gdn
lm_scaling/eval_table.py --step 190976 --convention gdn2 --format paper \
    > lm_scaling/tex/fwedu_1p3B.tex
```

One arm per GPU: export from the DCP checkpoint, check the export, then the
task suite. Four things on this path fail quietly and each is guarded:

* **Check the export against the run's own validation loss**
  (`check_export_loss.py`, run automatically). A downstream number cannot tell a
  correct export from a `w1`/`w3` swap or a stray SiLU — those cost 0.2–1.0 nats
  and still produce plausible accuracies. Checkpoint steps and validation steps
  coincide exactly at the final step, where the comparison is exact.
* **Never load an export with `AutoModelForCausalLM`.** Another registered
  `model_type` can win and build a layer with different defaults — identical
  parameter counts, no error. `export_hf.load_exported()` constructs the classes
  directly.
* **Score at the model's own 4,096-token context.** A 2,048 window inflates
  WikiText perplexity by a consistent 5%, which is larger than the effects being
  compared.
* **Three tasks need the overrides in `eval_tasks/`** — social_iqa, boolq and
  winogrande name script-based repos that recent `datasets` refuses while
  reporting a *Hub connection* error, which points at the network rather than at
  the loader.

`--convention` matters: `gdn` is Gated DeltaNet's Table 3 (eight accuracies,
ARC-c and HellaSwag normalized, no OpenBookQA) and `gdn2` is Gated DeltaNet-2's
Table 2 (nine accuracies, ARC-c plain). The same published model prints 38.39 in
one and 35.15 in the other; reading one paper's metric into the other's column
moves a row by ~3 points, several times the gaps being compared. The
transcriptions in `published_1p3B.json` and `published_gdn2_1p3B.json` are
verified by reproducing each row's own printed average, and a test enforces it.

### Re-measuring the micro-batch ceiling

`peak_mbs.yaml` and `token_rate.yaml` were measured for exactly these arms at
exactly this geometry. If you change an arm, the stack, or the GPU, re-probe
rather than assume:

```bash
lm_scaling/login_run lm_scaling/titan_mbs_probe.py fwedu_1p3B --out DIR -o ++aux.peak=4
lm_scaling/login_run lm_scaling/titan_mbs_probe.py fwedu_1p3B --out DIR --collect
```

Each candidate is a short run of the campaign's **own** job with the micro-batch
overridden, so what is measured is what will run.

---

## Tests

```bash
pytest tests/lm_scaling/                        # config, planner, tables: no GPU
lm_scaling/container_run -m pytest tests/lm_scaling/   # same, inside the image
```

They are guards against the failures this pipeline has actually had, not
coverage: that the recipe is the published one to the number, that the config
and torchtitan read one geometry table, that the arm table matches its own
generator, that the four arms build the models the table names, and that the
published rows reproduce their own printed averages.
