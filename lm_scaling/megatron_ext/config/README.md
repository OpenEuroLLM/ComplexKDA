# `megatron_ext/config/` — the Megatron ladder's root config

`ckda_ladder_jupiter.yaml` is **this study's** Megatron config, not a borrowed
one, which is why it lives in this repository rather than in the
`oellm-autoexp` checkout it runs through. `submit_ladder.py` copies it into
that checkout at submit time, so it is edited here and never there — a config
edited inside `external/` is a config that no commit records and that the next
sync overwrites.

It is here **ahead of the code that reads it**. The Python half of the Megatron
integration ships with the scaling-ladder release; the config ships now because
it is the part that says what the experiment *is*, and a released number whose
configuration cannot be read back has no provenance.

## What it composes against

The `defaults:` list reaches into `oellm-autoexp`'s own config tree:

```yaml
defaults:
  - ../base                                       # autoexp's base experiment
  - /backend: megatron_torchrun_localaddr         # autoexp's backend group
  - /container: jupiter_liger_mamba               # the image with fla in it
  - /slurm: jupiter
  - /backend/megatron: [base, data_nemonicco_jupiter]
  - /sweep: none
  - override /job: default
```

So this file alone does not resolve — it is the *delta* against a pinned
autoexp checkout, which is exactly what
[`../../megatron_stack.lock`](../../megatron_stack.lock) exists to fix in
place. Get that checkout with:

```bash
bash lm_scaling/make_megatron_stack.sh
```

Reading the file without the checkout still tells you what the ladder runs:
every line that differs from the reference's own scaling config is annotated
with why it differs.

## Two things to know before changing it

**The corpus is not this release's corpus.** The 1.3B FineWeb-Edu campaign and
this ladder are different experiments: the ladder trains on Nemotron-CC, whose
paths appear under `backend.megatron`. Nothing here reads
`config/data/fineweb_edu_llama2.yaml`.

**Geometry, batch, learning rate and token budget are not in this file.** They
arrive as per-cell overrides from `submit_ladder.py`, generated from the same
parameter-matched table the torchtitan arms are built from, so the two ladders
are matched by construction rather than by agreement. The values present here
are a default point that has to *compose*, not a cell anyone runs.
