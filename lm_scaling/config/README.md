# `config/` — the campaign, as one config

Every job this project submits is a point in a config expanded by
[`hydra_staged_sweep`], rendered by [`slurm_gen`] and watched by [`monitor`].
There is no Python that builds an experiment.

```
autoexp.yaml          the root: which backend, container, cluster, corpus
backend/titan.yaml    torchtitan, nested exactly as its TOML is
container/            what to enter and what to mount into it
cluster/              GPU type, where runs are written
slurm/                the cluster: sbatch directives, environment, srun
job/                  identity, output paths, what the monitor restarts on
data/                 corpus, tokenizer, context, held-out split
overlay/              named override bundles (`-o @probe`, `-o @mbs`)
sweep/                sweep shapes
experiments/
  base.yaml           the launch command, shared by every campaign
  fwedu_1p3B.yaml     1.3B/100BT on FineWeb-Edu, the published recipe, 4 arms
  fwedu_1p3B_results.md   what it produced
```

**This tree configures the torchtitan campaign only.** The scaling ladder runs
on Megatron-LM, whose configs compose against `oellm-autoexp`'s own tree rather
than this one; its root config is in `../megatron_ext/config/` and its stack is
pinned by `../megatron_stack.lock`.

Orchestration runs in the login venv, training in the container, and the split
between them is `sbatch`: it is a host binary talking to the Slurm controller,
and the container does not have it.

```
lm_scaling/login_run     lm_scaling/plan.py   fwedu_1p3B            # expand and price
lm_scaling/login_run     lm_scaling/plan.py   fwedu_1p3B --toml /tmp/out
lm_scaling/login_run     lm_scaling/submit.py fwedu_1p3B --out /tmp/fwedu
lm_scaling/container_run lm_scaling/data/verify_corpus.py fwedu_1p3B
```

Build the login venv with `lm_scaling/make_login_venv.sh` and the container
overlay with `lm_scaling/data/make_overlay.sh`. Training needs neither:
torchtitan and titan_oellm are in the image, and `fla` comes from this
checkout on `PYTHONPATH`.

## Why it is shaped like this

Earlier rounds of this study failed three times, and every failure was one
number living in more than one place: micro-batch ceilings in three files
(three runs OOM'd five minutes in), arm definitions in three files (the arm
named `kda` was really `kda-neg`), and one run's output directory built twice
(36 jobs died on `FileNotFoundError`).

So numbers are **cited, not copied**. `${matched:...}`, `${rung:...}` and
`${peak_mbs:...}` read one table and stop, and each RAISES on a missing entry
rather than returning something plausible. The arithmetic that combines them
is in the YAML — `oc.divi`, `oc.cdivi`, `oc.eval` — where it can be read and
overridden, not hidden inside a resolver that decides.

`aux` is untyped scratch: the intermediates a config derives on its way to
filling in the typed sections. Validation belongs on those — `compoconf`
rejects an unknown or missing key in `job`, `slurm`, `data` and
`backend.titan` at load.

## Cluster specifics are config, not template

`templates/cluster.sbatch` is 20 lines and contains nothing about JUPITER. The
container image, its bind mounts, the `nslookup` retry for `MASTER_ADDR`, the
node-local `TRITON_CACHE_DIR`, the absent `LD_LIBRARY_PATH` — all of it is in
`slurm/jupiter.yaml` and `container/jupiter_titan.yaml`, where it can be
overridden for one debugging run and where it shows up in the config archived
beside each run.

To run this somewhere else, add a `cluster/`, `slurm/` and `container/` entry
for that machine and point `autoexp.yaml` at them; nothing in
`experiments/fwedu_1p3B.yaml` is site-specific. The files here are the ones
the published runs used, unedited — see the top-level
[`lm_scaling/README.md`](../README.md) for what each of them assumes.

[`hydra_staged_sweep`]: https://github.com/kpoeppel/hydra_staged_sweep
[`slurm_gen`]: https://github.com/kpoeppel/slurm_gen
[`monitor`]: https://github.com/kpoeppel/monitor
