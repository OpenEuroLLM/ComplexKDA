# fwedu_1p3B — results

The downstream numbers for `fwedu_1p3B.yaml`: four arms at 1.3B parameters on
100B tokens of FineWeb-Edu, all four finished at step 190,976 on 2026-09-13.

**EVERY NUMBER BELOW IS THE `--context-prefix none` MEASUREMENT** (2026-09-21).
The runs that produced this file before that date prefixed every scored context
with `<s>`, because this harness tokenised with specials on and the fwedu
tokenizer copy sets `add_bos_token: True` -- a token `prepare_fineweb_edu.py`
never writes (5 occurrences in 200M tokens of the corpus) and lm-eval's own
HFLM would have stripped. Those numbers were measured and are not published;
this table is the no-BOS one. The prefix did NOT cancel
in the paired delta and did not even move it consistently: on the nine
common-sense tasks the baselines gained about twice what the signed arms did
(recurrent +0.52 -> -0.04), and on the recall suite the signed arms gained more
(+0.48 the other way). There is no correction factor -- it is measured per
suite.

**Tables are generated, not transcribed.** Regenerate with:

```
# Both are the defaults, so the flags below are only to be explicit about
# WHICH data a table came from; `eval_table.py` with no --results reads these.
R="--results lm_scaling/harvest/downstream"
RR="--recall-results lm_scaling/harvest/recall"

lm_scaling/eval_table.py $R $RR --step 190976 --convention gdn2 --with-recall   # this file's first table
lm_scaling/eval_table.py $R --step 190976 --convention gdn                      # the second
lm_scaling/eval_table.py $R $RR --step 190976 --convention gdn2 --format paper \
    > lm_scaling/tex/fwedu_1p3B.tex   # THE PAPER FILE: both tables, drop-in
lm_scaling/eval_table.py $R $RR --step 190976 --convention gdn2 --format latex --with-recall \
    > lm_scaling/tex/fwedu_1p3B_table_gdn2_recall.tex   # 9 tasks + avg + recall
lm_scaling/eval_table.py $R --step 190976 --convention gdn2 --format latex \
    > lm_scaling/tex/fwedu_1p3B_table_gdn2.tex   # the comparison table alone
lm_scaling/eval_table.py $R --step 190976 --convention gdn --format latex \
    > lm_scaling/tex/fwedu_1p3B_table_gdn.tex    # the same, older convention
```

RULER needle retrieval — Gated DeltaNet-2's **Table 3**, the long-context one —
is a separate job, because it GENERATES rather than scores and its samples are
synthesised per context length:

```
lm_scaling/container_run lm_scaling/stage_ruler.py     # ONCE, on a login node
STEP=190976 sbatch lm_scaling/eval_ruler.sbatch        # both recurrent arms
STEP=190976 ARMS="kda-sig-hybrid-lowrank ckda-shipped-hybrid-lowrank" \
    LENGTHS_LONG=1024,2048,4096 sbatch lm_scaling/eval_ruler.sbatch   # hybrids
lm_scaling/niah_table.py --step 190976 --pairs         # the table

ARMS=kda-sig-lowrank,ckda-shipped-lowrank,kda-sig-hybrid-lowrank,ckda-shipped-hybrid-lowrank
lm_scaling/niah_table.py --results lm_scaling/harvest/ruler --step 190976 \
    --format latex --arms $ARMS > lm_scaling/tex/fwedu_1p3B_niah_nobos.tex  # THE PAPER FILE

# `fwedu_1p3B_niah.tex` is the SUPERSEDED `<s>`-prefixed table, kept because the
# single-needle cells moved by up to 18 points between the two conventions and
# that movement is itself evidence (see the S-NIAH section below). Regenerate it
```

The result JSONs are pulled off the cluster into `lm_scaling/harvest/ruler/`,
which is why the LaTeX can be regenerated without a cluster session. The table
needs `booktabs`; cells the hybrids were not evaluated at print `---`, never
`0.0`, and a test pins that.

The recall-intensive suite (SQuAD-completion, SWDE, FDA) is a third job, for
the same reason — it generates:

```
lm_scaling/container_run lm_scaling/stage_recall.py    # ONCE, on a login node
STEP=190976 sbatch lm_scaling/eval_recall.sbatch       # all four arms, ~26 min
lm_scaling/recall_table.py --step 190976 --pairs       # the table

lm_scaling/recall_table.py --results lm_scaling/harvest/recall --step 190976 \
    --format latex > lm_scaling/tex/fwedu_1p3B_recall.tex   # THE PAPER FILE
```

To print the recall columns INSIDE the gdn/gdn2 table instead of beside it, add
`--with-recall` to `eval_table.py`. `Avg.` stays over the common-sense
accuracies — it is the number every published row is quoted at — and the suite
gets its own `Rec.` average, blank on the published rows:

```
lm_scaling/eval_table.py --step 190976 --convention gdn2 --with-recall
```

**That join is refused when the two sides were tokenised differently.** Each
run records its `--context-prefix` in a `.meta.json` sidecar (missing = the old
`bos`), and a table whose common-sense columns carry a leading `<s>` and whose
recall columns do not is not one measurement of one model. `--force-mixed-prefix`
overrides it for looking at, never for publishing.

Two things about that table are not choices this file is free to make. The grid
is the paper's and is **not rectangular** (S-NIAH-1 and -2 to 8K, S-NIAH-3 and
MK-NIAH-1 to 4K), and the rows are **recurrent only** — the published hybrid
block uses 2K sliding-window attention where ours put full NoPE attention every
fourth layer, which is the same objection this file already raises about
Table 2's hybrids.

`lm_scaling/tex/` is in git; `lm_scaling/results/` is not (a bare `results`
rule in `.gitignore`), which is why the generated tables live in the one and not
the other. The paper file carries the
comparison table AND the pairing (Table 2), because the first has the signed arm
ahead on downstream accuracy and that is not a claim this study can make — see
"What may not be claimed" below.

Results live at `/e/scratch/e-sta-openeurollm/poeppel1/eval_fwedu/results/*_step190976.json`.

## What was measured, and how

* **Recipe**: Gated DeltaNet's, which Gated DeltaNet-2 also uses — AdamW, peak LR
  4e-4, weight decay 0.1, clipping 1.0, cosine to 0.1x peak after a 1B-token
  warm-up, 0.5M-token batches, 4,096-token sequences, Llama-2 tokenizer, seed 3407.
* **Arms**: the MN5 reference configuration. The baseline is the bounded sigmoid gate
  (`kda-sig`), not fla's softplus; the signed arm starts from the shipped init;
  both take fla's factored (low-rank) output gate; neither has SiLU on q/k/v. The
  hybrids keep gated NoPE attention every 4th layer (3:1).
* **Scoring**: lm-eval-harness at the model's own 4,096-token context — the
  window the published rows were measured at. A 2,048 window inflates WikiText
  perplexity by a consistent 5%.
* **Every export was checked against its own run's validation loss** before being
  scored (`check_export_loss.py`), which is what separates a correct export from
  a w1/w3 swap or a stray SiLU: those cost 0.2–1.0 nats and still produce
  plausible accuracies.

## Held-out validation loss (our split — comparable between our arms and nobody else)

| arm | final val loss | held-out ppl |
|---|---|---|
| `kda-sig-lowrank` | 2.0287 | 7.604 |
| `ckda-shipped-lowrank` | 2.0316 | 7.626 |
| `kda-sig-hybrid-lowrank` | 2.0181 | 7.524 |
| `ckda-shipped-hybrid-lowrank` | 2.0195 | 7.535 |

The baseline is ahead in both pairs, by 0.0029 (recurrent) and 0.0014 (hybrid)
nats. The ladder's measured seed-noise floor is ~0.002 nats, so both differences
sit at or below it.

## Downstream, Gated DeltaNet-2 convention (9 accuracies, ARC-c plain, OBQA included)

### fwedu_1p3B at step 190,976 (gdn2 convention: 9 accuracies, ARC-c acc)

| Model | Wiki. | LMB. | LMB. | PIQA | Hella. | Wino. | ARC-e | ARC-c | OBQA | SIQA | BoolQ | Avg. | SQuAD | SWDE | FDA | Rec. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *Recurrent models* | | | | | | | | | | | | | | | | |
| Mamba-2 | 16.79 | 12.38 | 45.24 | 72.58 | 55.51 | 55.33 | 70.68 | 35.26 | 31.00 | 40.63 | 60.19 | 51.82 | -- | -- | -- | -- |
| Gated DeltaNet | 16.40 | 11.89 | 49.62 | 72.31 | 56.50 | 56.75 | 68.81 | 35.15 | 30.20 | 40.53 | 58.78 | 52.07 | -- | -- | -- | -- |
| KDA | 16.81 | 11.68 | 48.13 | 72.09 | 55.75 | 55.72 | 70.83 | 35.92 | 30.40 | 40.99 | 60.67 | 52.28 | -- | -- | -- | -- |
| Mamba-3 (SISO) | 16.30 | 12.99 | 45.06 | 72.31 | 55.58 | 56.20 | 70.45 | 34.56 | 31.00 | **41.76** | 55.90 | 51.42 | -- | -- | -- | -- |
| Mamba-3 (MIMO) | 16.45 | 11.66 | 47.82 | 72.36 | 56.49 | 55.78 | 72.38 | 38.07 | 30.00 | 40.89 | 57.74 | 52.39 | -- | -- | -- | -- |
| Gated DeltaNet-2 | 15.90 | 11.41 | 48.09 | 72.80 | 56.84 | 57.85 | 72.43 | 38.23 | **31.60** | 40.58 | 59.54 | 53.11 | -- | -- | -- | -- |
| KDA, bounded gate (ours) | **15.73** | 10.53 | 50.46 | 72.85 | **59.32** | **61.33** | 73.23 | 38.40 | 28.80 | 41.35 | **61.10** | **54.09** | 38.17 | **49.59** | 30.40 | 39.39 |
| CKDA (ours) | 15.78 | **10.08** | **51.66** | **73.12** | 59.23 | 58.88 | **73.99** | **39.08** | 28.60 | 41.61 | 60.34 | 54.06 | **39.68** | 48.42 | **31.49** | **39.86** |
| *Attention or hybrid models* | | | | | | | | | | | | | | | | |
| Transformer (hybrid) | 19.22 | 13.72 | 48.32 | 70.21 | 56.12 | 55.85 | 69.23 | 33.84 | 25.00 | 39.74 | 59.42 | 50.86 | -- | -- | -- | -- |
| Mamba-2 (hybrid) | 17.46 | 11.29 | 48.05 | 71.47 | 57.52 | 56.17 | 70.50 | 34.73 | 29.80 | 40.35 | 59.31 | 51.99 | -- | -- | -- | -- |
| Gated DeltaNet (hybrid) | 16.00 | 10.82 | 48.71 | 70.06 | 57.50 | 56.83 | 70.41 | 35.15 | 30.60 | 40.97 | 60.00 | 52.25 | -- | -- | -- | -- |
| KDA (hybrid) | 16.01 | 10.66 | 49.21 | 71.06 | 56.89 | 57.77 | 71.59 | 35.07 | 30.00 | 40.53 | 62.03 | 52.68 | -- | -- | -- | -- |
| Mamba-3 (SISO, hybrid) | 15.54 | 10.65 | 49.19 | 71.01 | 58.75 | 57.30 | 70.54 | 36.35 | 32.00 | 41.20 | 57.86 | 52.69 | -- | -- | -- | -- |
| Mamba-3 (MIMO, hybrid) | 15.81 | 10.92 | 49.82 | 71.98 | 58.19 | 57.06 | 70.54 | **38.48** | 29.40 | 40.99 | 57.98 | 52.72 | -- | -- | -- | -- |
| Gated DeltaNet-2 (hybrid) | 15.62 | 10.43 | 50.90 | 72.20 | 58.46 | 58.56 | 71.89 | 36.69 | **33.00** | 41.50 | 62.57 | 53.97 | -- | -- | -- | -- |
| KDA + attn 3:1 (ours) | **15.04** | 9.93 | 51.60 | **73.88** | 59.57 | **60.62** | **72.69** | 37.20 | 27.00 | 42.43 | 60.31 | 53.92 | **42.69** | **64.81** | **64.79** | **57.43** |
| CKDA + attn 3:1 (ours) | 15.34 | **9.90** | **52.57** | 73.56 | **59.68** | 60.14 | 72.56 | 36.86 | 28.20 | **42.58** | **63.15** | **54.37** | 37.77 | 63.91 | 58.89 | 53.52 |

Perplexities lower is better, accuracies (%) higher; **bold** is the best value in the column within its block. Published rows: Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention, arXiv:2605.22791v1, Table 2.

- recurrent: signed − baseline = **-0.04** points of average accuracy, wiki ppl +0.05, LAMBADA ppl -0.45, held-out loss +0.0029 (our split; lower is better).
- hybrid: signed − baseline = **+0.45** points of average accuracy, wiki ppl +0.30, LAMBADA ppl -0.03, held-out loss +0.0014 (our split; lower is better).

## Downstream, Gated DeltaNet convention (8 accuracies, ARC-c normalized)

### fwedu_1p3B at step 190,976 (gdn convention: 8 accuracies, ARC-c acc_norm)

| Model | Wiki. | LMB. | LMB. | PIQA | Hella. | Wino. | ARC-e | ARC-c | SIQA | BoolQ | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *Recurrent models* | | | | | | | | | | | |
| RetNet | 19.08 | 17.27 | 40.52 | 70.07 | 49.16 | 54.14 | 67.34 | 33.78 | 40.78 | 60.39 | 52.02 |
| HGRN2 | 19.10 | 17.69 | 39.54 | 70.45 | 49.53 | 52.80 | 69.40 | 35.32 | 40.63 | 56.66 | 51.79 |
| Mamba | 17.92 | 15.06 | 43.98 | 71.32 | 52.91 | 52.95 | 69.52 | 35.40 | 37.76 | **61.13** | 53.12 |
| Mamba2 | 16.56 | 12.56 | 45.66 | 71.87 | 55.67 | 55.24 | 72.47 | 37.88 | 40.20 | 60.13 | 54.89 |
| DeltaNet | 17.71 | 16.88 | 42.46 | 70.72 | 50.93 | 53.35 | 68.47 | 35.66 | 40.22 | 55.29 | 52.14 |
| Gated DeltaNet | 16.42 | 12.17 | 46.65 | 72.25 | 55.76 | 57.45 | 71.21 | 38.39 | 40.63 | 60.24 | 55.32 |
| KDA, bounded gate (ours) | **15.73** | 10.53 | 50.46 | 72.85 | **59.32** | **61.33** | 73.23 | 40.96 | 41.35 | 61.10 | 57.57 |
| CKDA (ours) | 15.78 | **10.08** | **51.66** | **73.12** | 59.23 | 58.88 | **73.99** | **42.15** | **41.61** | 60.34 | **57.62** |
| *Attention or hybrid models* | | | | | | | | | | | |
| Transformer++ | 18.53 | 18.32 | 42.60 | 70.02 | 50.23 | 53.51 | 68.83 | 35.10 | 40.66 | 57.09 | 52.25 |
| Samba | 16.13 | 13.29 | 44.94 | 70.94 | 53.42 | 55.56 | 68.81 | 36.17 | 39.96 | 62.11 | 54.00 |
| Gated DeltaNet-H1 | 16.07 | 12.12 | 47.73 | 72.57 | 56.53 | 58.40 | 71.75 | **40.10** | 41.40 | **63.21** | 56.40 |
| Gated DeltaNet-H2 | 15.91 | 12.55 | 48.76 | 72.19 | 56.88 | 57.77 | 71.33 | 39.07 | 41.91 | 61.55 | 56.18 |
| KDA + attn 3:1 (ours) | **15.04** | 9.93 | 51.60 | **73.88** | 59.57 | **60.62** | **72.69** | 39.51 | 42.43 | 60.31 | 57.57 |
| CKDA + attn 3:1 (ours) | 15.34 | **9.90** | **52.57** | 73.56 | **59.68** | 60.14 | 72.56 | 39.42 | **42.58** | 63.15 | **57.96** |

Perplexities lower is better, accuracies (%) higher; **bold** is the best value in the column within its block. Published rows: Gated Delta Networks: Improving Mamba2 with Delta Rule, arXiv:2412.06464, Table 3.

- recurrent: signed − baseline = **+0.05** points of average accuracy, wiki ppl +0.05, LAMBADA ppl -0.45, held-out loss +0.0029 (our split; lower is better).
- hybrid: signed − baseline = **+0.38** points of average accuracy, wiki ppl +0.30, LAMBADA ppl -0.03, held-out loss +0.0014 (our split; lower is better).

## The pairing — signed minus baseline

**recurrent** — `ckda-shipped-lowrank` minus `kda-sig-lowrank`:

| task | baseline | signed | Δ | se | z |
|---|---|---|---|---|---|
| lmb.acc | 50.46 | 51.66 | +1.20 | 0.98 | +1.22 |
| piqa | 72.85 | 73.12 | +0.27 | 1.46 | +0.19 |
| hella | 59.32 | 59.23 | -0.09 | 0.69 | -0.13 |
| wino | 61.33 | 58.88 | -2.45 | 1.94 | -1.26 |
| arc-e | 73.23 | 73.99 | +0.76 | 1.28 | +0.59 |
| arc-c | 38.40 | 39.08 | +0.68 | 2.01 | +0.34 |
| obqa | 28.80 | 28.60 | -0.20 | 2.86 | -0.07 |
| siqa | 41.35 | 41.61 | +0.26 | 1.58 | +0.16 |
| boolq | 61.10 | 60.34 | -0.76 | 1.21 | -0.63 |
| **avg9** | 54.09 | 54.06 | **-0.04** | 0.56 | **-0.07** |
| squad | 38.17 | 39.68 | +1.51 | 1.26 | +1.19 |
| swde | 49.59 | 48.42 | -1.17 | 2.12 | -0.55 |
| fda | 30.40 | 31.49 | +1.09 | 1.97 | +0.55 |
| **rec.avg** | 39.39 | 39.86 | **+0.48** | 1.05 | **+0.45** |
| wiki ppl | 15.73 | 15.78 | +0.05 | — | — |
| lmb ppl | 10.53 | 10.08 | -0.45 | — | — |
| held-out loss | 2.0287 | 2.0316 | +0.0029 | — | — |

**hybrid** — `ckda-shipped-hybrid-lowrank` minus `kda-sig-hybrid-lowrank`:

| task | baseline | signed | Δ | se | z |
|---|---|---|---|---|---|
| lmb.acc | 51.60 | 52.57 | +0.97 | 0.98 | +0.99 |
| piqa | 73.88 | 73.56 | -0.33 | 1.45 | -0.22 |
| hella | 59.57 | 59.68 | +0.11 | 0.69 | +0.16 |
| wino | 60.62 | 60.14 | -0.47 | 1.94 | -0.24 |
| arc-e | 72.69 | 72.56 | -0.13 | 1.29 | -0.10 |
| arc-c | 37.20 | 36.86 | -0.34 | 1.99 | -0.17 |
| obqa | 27.00 | 28.20 | +1.20 | 2.83 | +0.42 |
| siqa | 42.43 | 42.58 | +0.15 | 1.58 | +0.10 |
| boolq | 60.31 | 63.15 | +2.84 | 1.20 | +2.37 |
| **avg9** | 53.92 | 54.37 | **+0.45** | 0.55 | **+0.80** |
| squad | 42.69 | 37.77 | -4.93 | 1.27 | -3.89 |
| swde | 64.81 | 63.91 | -0.90 | 2.03 | -0.44 |
| fda | 64.79 | 58.89 | -5.90 | 2.07 | -2.86 |
| **rec.avg** | 57.43 | 53.52 | **-3.91** | 1.05 | **-3.71** |
| wiki ppl | 15.04 | 15.34 | +0.30 | — | — |
| lmb ppl | 9.93 | 9.90 | -0.03 | — | — |
| held-out loss | 2.0181 | 2.0195 | +0.0014 | — | — |

## Along training — the KDA baseline

**THIS TABLE IS STILL THE `<s>`-PREFIXED MEASUREMENT**, and it is the one place
in this file that is. Only step 190,976 was re-evaluated under
`--context-prefix none`; the three intermediate checkpoints were not, and a
trajectory assembled from two conventions would be worse than one assembled
from the wrong one. That is why its final row reads 15.77 / 56.9 where the
tables above read 15.73 / 57.6 for the same arm and step -- the difference is
the prefix, not the model. Re-run
`STEPS="48640 97280 155648 190976" CONTEXT_PREFIX=none sbatch
lm_scaling/eval_fwedu.sbatch` to put it on one convention.

| step | tokens | wiki ppl | lmb ppl | avg8 |
|---|---|---|---|---|
| 48,640 | 25.5 BT | 20.77 | 17.33 | 52.6 |
| 97,280 | 51.0 BT | 18.23 | 14.04 | 54.3 |
| 155,648 | 81.6 BT | 16.26 | 10.96 | 56.8 |
| 190,976 | 100.1 BT | 15.77 | 10.53 | 56.9 |

A published row is the final step of a full budget; an intermediate checkpoint of
a cosine run is not a smaller run, because its learning rate has not annealed.
At 51 BT this arm read 18.23 WikiText against a published 16.42 and looked behind;
it finished at 15.77, ahead of every recurrent row in either paper.

## RULER needle retrieval — Gated DeltaNet-2's Table 3

Recurrent arms, step 190,976, 500 samples a cell, job 1839715 (1h56m, two arms
in parallel on one node). Regenerate with `lm_scaling/niah_table.py --step
190976 --pairs`; published rows transcribed in `published_gdn2_niah_1p3B.json`.

```
                            S-NIAH-1              S-NIAH-2           S-NIAH-3      MK-NIAH-1
                        1K    2K    4K    8K    1K    2K    4K    8K    1K    2K    4K    1K    2K    4K
KDA, bounded (ours)   49.6  75.0  94.2  69.8  91.6 100.0  97.0  20.8  10.6  44.8   9.8  50.0  26.6  25.8
Complex KDA (ours)    100.0 100.0 100.0  87.8 100.0 100.0  88.4  33.0  94.4  88.0  51.4  56.0  47.6  37.8
KDA (published)      100.0 100.0  99.2  70.6 100.0 100.0  89.0  30.6  77.4  63.2  26.2  54.0  44.2  28.0
Gated DeltaNet-2     100.0 100.0 100.0  97.8 100.0 100.0  93.0  39.2  92.0  89.8  31.8  72.6  51.4  37.8
signed - bounded     +50.4 +25.0  +5.8 +18.0  +8.4  +0.0  -8.6 +12.2 +83.8 +43.2 +41.6  +6.0 +21.0 +12.0
```

**The signed arm is a credible recurrent model on this table**: level with
published KDA on S-NIAH-2 (88.4/33.0 against 89.0/30.6), ahead of it throughout
S-NIAH-3, and level with Gated DeltaNet-2 at MK-NIAH-1 4K. That reading stands
without reference to our own baseline.

**The comparison is paired.** `prepare_niah` fixes RANDOM_SEED = 42, so both
arms answer the same 500 questions in the same order at every length; the
per-cell binomial se (2.2 points at p = 0.5) is therefore conservative for the
difference, as it was on the ladder.

**The bounded-gate row is the anomaly, and it is not the harness.** No published
recurrent model scores below 99.0 at S-NIAH-1 1K; ours reads 49.6. Three things
say the measurement is sound and the arm is not:

* the signed arm reproduces the published shape closely at every length, on the
  same harness, prompts, tokenizer and decoder;
* our bounded arm lands in the published range where it is healthy — 94.2/69.8
  against published KDA's 99.2/70.6 at S-NIAH-1 4K/8K. Five points at 4K is
  still ~5 se at n = 500, so this is "the same regime", not "the same number";
* its failures are **non-monotonic in length** — S-NIAH-3 runs 10.6 → 44.8 →
  9.8, S-NIAH-1 *improves* 49.6 → 75.0 → 94.2 as context grows. Published rows
  decay smoothly. Erratic ordering is a broken behaviour, not weaker retrieval.

Sampled generations agree: on S-NIAH-1 the bounded arm recites the haystack
(`":\nThe grass is green. The sky is blue..."`) and on S-NIAH-3 it continues the
question instead of answering. It is healthy on S-NIAH-2, the one task with an
essay haystack and a numeric answer. That is an output-behaviour failure, not a
retrieval one.

### The hybrids, and the interaction that decides how this reads

Jobs 1841900 (1K/2K/4K) and 1874149 (the 8K column, `ALLOW_EXTRAPOLATION=1`).
The 8K cells are past the 4,096 tokens these attention layers trained at, so
`eval_downstream.py` requires `--allow-extrapolation` for them: their attention
is NoPE and has no position table to exceed, which is the reason to expect the
extension, but it still had to be measured.

```
                            S-NIAH-1              S-NIAH-2           S-NIAH-3      MK-NIAH-1
                        1K    2K    4K    8K    1K    2K    4K    8K    1K    2K    4K    1K    2K    4K
KDA + attn 3:1       100.0 100.0 100.0 100.0 100.0  99.2  99.4  68.2  99.0  97.8  96.2  73.2  61.2  67.0
CKDA + attn 3:1       99.2  95.6  79.4  40.2  94.4  96.8  87.8  60.6  95.4  88.6  80.6  94.4  95.0  94.4
signed - bounded      -0.8  -4.4 -20.6 -59.8  -5.6  -2.4 -11.6  -7.6  -3.6  -9.2 -15.6 +21.2 +33.8 +27.4
                                   mean -4.20 over 14 cells; signed ahead in 3/14
```

**NoPE attention extends.** `KDA + attn 3:1` holds **100.0 at 8K on S-NIAH-1**,
at double its training context, and reads 68.2 on S-NIAH-2 there against 39.2
for the best published recurrent row and 33.0 for our own best recurrent arm.
Nothing here collapses at the training boundary, which is what a positional
table would have produced. It is still a length-generalisation result and not a
row of the published table: those hybrids use 2K sliding-window attention,
which extends by construction.

**The 8K column widens the hybrid gate gap rather than closing it.** The signed
hybrid decays 99.2 → 95.6 → 79.4 → 40.2 on S-NIAH-1 while the bounded one does
not move off 100.0, so the worst single cell in the pairing is −59.8.

### What those single-needle cells actually measure (2026-09-19)

**They measure whether the model answers, not whether it retrieves.** Read the
samples with `lm_scaling/niah_failures.py` (job 1883912/1883913, four arms,
four tasks, n=200 a cell, dumps under `eval_ruler_ablate`):

```
                                 scored  answered  wrong  correct|answered
S-NIAH-1 @4K  kda-sig-hybrid     100.0%    100.0%      0            100.0%
              ckda-shipped-hyb    77.5%     77.5%      0            100.0%
S-NIAH-3 @1K  kda-sig-lowrank      9.0%      9.5%      1             94.7%
S-NIAH-1 @1K  kda-sig-lowrank     50.0%     50.0%      0            100.0%
```

Across S-NIAH-1/2/3, at both lengths, for all four arms, the wrong-needle count
is 0–10 out of 200 and `correct | answered` is 90–100%. **No arm ever retrieves
the wrong needle on a single-needle task.** Every difference between these cells
is a difference in how often the model emits an answer at all; the failures are
a degenerate loop that restates the primer, uniform across needle depth.

So the bounded arm's "collapse" is not a retrieval deficit: at S-NIAH-3 1K it
answers 9.5% of the time and is right 94.7% of those. That retires `drop_silu`
as the suspect for it, and retires the reading above that the bounded hybrid is
the campaign's strongest arm -- it is the most COMPLIANT arm, answering on every
one of 200 prompts in every cell.

**MK-NIAH-1 is the only task here that measures retrieval.** Every arm answers
94–100% of the time and wrong needles run 13–129 per 200, so the score means
what it appears to. The ordering there is consistent and has no crossover:

```
correct | answered        @1K     @4K
CKDA + attn 3:1          93.5%   90.5%
KDA + attn 3:1           72.0%   64.0%
CKDA (recurrent)         58.5%   34.2%
KDA (recurrent)          44.0%   31.7%
```

The signed gate is ahead in BOTH stacks, and attention helps both. The apparent
gate crossover between the stacks was an artifact of answer rates on tasks that
cannot discriminate retrieval.

**Open**: whether those answer rates are a property of the models or of the
prompt. RULER's template ends at its answer prefix with nothing after it
("...mentioned in the provided text is"), exactly as NVIDIA's own does, and all
200 of our generations spend their first token emitting ":" there. Jobs
1884459–62 re-run the low-answer-rate cells with the primer ending "is:" and
"is: " to see how much of an answer rate is the last two characters. Those
variants depart from the published protocol and are tagged `_diag`, so they can
never enter this table.

**Attention alone repairs the bounded arm.** Same gate, same `drop_silu`, same
recipe: S-NIAH-1 1K goes 49.6 → 100.0 on adding full NoPE attention every fourth
layer. So the deficit lives in the RECURRENT PATH, and it is not a global defect
of sigmoid-gated arms — the sigmoid-gated hybrid carries `drop_silu` too and is
flawless.

**The gate contrast is task-dependent, not a uniform effect.** Read across both
stacks:

* S-NIAH-1 and S-NIAH-3: signed far ahead without attention (+50.4, +83.8),
  behind with it (−20.6, −15.6). The advantage does not survive attention.
* S-NIAH-2: signed slightly behind at 4K in both stacks (−8.6, −11.6).
* **MK-NIAH-1: signed ahead in BOTH stacks**, and by more with attention.
  Signed-hybrid reads 94.4/95.0/94.4 — flat in length where every other arm and
  every published row decays (best published, GDN-2: 72.6/51.4/37.8).

The reading this supports is that the signed gate's robust contribution is
**multi-key retrieval**, and that its large single-needle advantage in the
recurrent stack is compensation for a deficit rather than added capability —
attention removes the deficit and the advantage with it. That is the same shape
as the 47M SiLU result (two ways to supply one kind of flexibility; having both
buys nothing), with attention in SiLU's role.

## Recall-intensive tasks — SQuAD-completion, SWDE, FDA

All four arms, step 190,976, job 1919495 (25 minutes, four arms in parallel on
one node), scored with `--context-prefix none`. Based's recall suite (Arora et
al., arXiv:2402.18668) as lm-eval ships it: a document, an answer prefix, and a
score of whether the gold value is contained in the generation. Nothing in these
can be answered from parameters — the FDA key–value pairs and the SWDE relations
are in the prompt or nowhere — which is why they sit beside the common-sense
table rather than inside it.

**No published rows.** None of the papers this campaign is compared against
reports this suite at 1.3B/100BT FineWeb-Edu, so this table is ours-versus-ours
and the only comparison it licenses is between our own arms.

| Model | SQuAD | SWDE | FDA | Avg. |
|---|---|---|---|---|
| *Recurrent models* | | | | |
| KDA, bounded gate (ours) | 38.2 | 49.6 | 30.4 | 39.4 |
| CKDA (ours) | 39.7 | 48.4 | 31.5 | 39.9 |
| *Attention or hybrid models (3:1 full NoPE attention)* | | | | |
| KDA + attn 3:1 (ours) | 42.7 | 64.8 | 64.8 | 57.4 |
| CKDA + attn 3:1 (ours) | 37.8 | 63.9 | 58.9 | 53.5 |

Contains-match accuracy (%); n = 2,984 / 1,111 / 1,102 documents. Every arm
answers the same documents, so the per-task se below — computed as if the two
samples were independent — is **conservative** for the difference, the same way
the ladder's was.

| pair | task | baseline | signed | Δ | se | z |
|---|---|---|---|---|---|---|
| recurrent | SQuAD | 38.17 | 39.68 | +1.51 | 1.26 | +1.19 |
| recurrent | SWDE | 49.59 | 48.42 | −1.17 | 2.12 | −0.55 |
| recurrent | FDA | 30.40 | 31.49 | +1.09 | 1.97 | +0.55 |
| recurrent | **avg3** | 39.39 | 39.86 | **+0.48** | 1.05 | +0.45 |
| hybrid | SQuAD | 42.69 | 37.77 | −4.93 | 1.27 | −3.89 |
| hybrid | SWDE | 64.81 | 63.91 | −0.90 | 2.03 | −0.44 |
| hybrid | FDA | 64.79 | 58.89 | −5.90 | 2.07 | −2.86 |
| hybrid | **avg3** | 57.43 | 53.52 | **−3.91** | 1.05 | −3.71 |

**Attention buys in-context recall, and the amount tracks document length.**
For the bounded gate, adding full NoPE attention every fourth layer moves FDA
30.4 → 64.8 (+34.4, median prompt 2,429 tokens), SWDE 49.6 → 64.8 (+15.2,
median 1,308) and SQuAD 38.2 → 42.7 (+4.5, median 290). That ordering is the
signature of the mechanism rather than a coincidence of three tasks: the longer
the document that has to be held, the more a fixed-size state loses and the
more attention recovers. It is the same conclusion the RULER hybrid block
reached, on natural documents instead of synthesised haystacks.

**The recurrent pair is a dead heat**: +0.48 points of avg3 at z = +0.45, and
no single task reaches 1.2 se. The two tokenisation conventions below bracket
it at −0.09 and +0.48 — a one-token change to the prompt moves this pair by
more than the pair differs — which is what a tie looks like when it is measured
twice.

**The hybrid pair goes to the bounded gate**, by 3.91 points of avg3 at z =
−3.7, carried by SQuAD (−4.93) and FDA (−5.90) with SWDE level. In sign this
agrees with the RULER single-needle hybrid cells (mean −4.20 over 14) and
disagrees with MK-NIAH-1, where the signed hybrid led by +21 to +34. Unlike
those cells it is not an answer-rate artifact — see below.

**What the z does NOT say.** The se here is over DOCUMENTS, not over seeds.
n = 1 per arm: these are two particular training runs, and −3.91 ± 1.05 means
"these two models differ on these documents", not "this gate costs 3.9 points".
The campaign has no seed replicates at 1.3B, and the ladder is where a claim
about the gate would have to be settled.

### The tokenisation convention, and what it moved

These rows are scored with `--context-prefix none` — a bare context, lm-eval's
own convention, and the one this corpus was written with
(`add_special_tokens=False`, `</s>` appended per document).

**The first pass of this table (job 1917940) was not.** It predates that flag,
when this harness tokenised with specials on; the fwedu tokenizer copy sets
`add_bos_token: True`, so every document was silently prefixed with `<s>` — a
token the corpus contains 5 times in 200M. Those numbers are kept under
not published, and every run now records its convention in a
`.meta.json` sidecar beside its results.

Re-measuring under the correct convention moved the table by tenths and changed
no conclusion:

| | recurrent Δavg3 | hybrid Δavg3 | attention gain (bounded) |
|---|---|---|---|
| `bos` (job 1917940) | −0.09 (z −0.09) | −4.34 (z −4.11) | +17.4 |
| `none` (job 1919495) | +0.48 (z +0.45) | −3.91 (z −3.71) | +18.0 |

The hybrid result and the attention gradient are unchanged. The recurrent pair
flips sign between the two, which is the honest way to read it: it is a tie
either way, and it is smaller than a single leading token.

### These are not answer rates

The RULER single-needle cells turned out to measure whether a model emits an
answer at all rather than whether it retrieves, which retired an earlier
reading of this file. The same question has to be asked here, because
`contains_score` scores a degenerate generation as 0 exactly as
`string_match_all` does.

It does not carry this table. Job 1919990 re-ran all four arms at n = 200 a
task with `--log-samples`, under the same `--context-prefix none` as the table,
and `lm_scaling/recall_failures.py --results DIR` reads the dump (job 1918778
is the same pass under the old `bos` convention; the rates differ by under a
point, so nothing below rests on which was used):

```
STEP=190976 LIMIT=200 LOG_SAMPLES=1 TAG=_diag sbatch lm_scaling/eval_recall.sbatch
lm_scaling/recall_failures.py --results <the eval_recall results dir> --show 3
```

The `_diag` tag is what keeps this out of the table — `recall_table.py` refuses
to merge any file carrying it, because a 200-document pass merged over a
2,984-document one would silently coarsen a number whose se is quoted from the
full split. For the same reason **the percentages below are not the table's**:
they are the first 200 documents of each task, so SQuAD reads 33.0 here and
38.3 there. Only the columns beside each other are being compared.

"Answered" is a generation that offers a value rather than echoing the prompt
back or coming back empty:

```
                                          scored  answered  echo  correct|answered
SQuAD    KDA, bounded gate                 33.5%     98.5%   1.5%           34.0%
         CKDA                              32.5%     95.5%   4.5%           34.0%
         KDA + attn 3:1                    33.0%    100.0%   0.0%           33.0%
         CKDA + attn 3:1                   23.0%    100.0%   0.0%           23.0%
SWDE     KDA, bounded gate                 49.5%     99.0%   0.0%           50.0%
         CKDA                              45.5%     98.5%   0.0%           46.2%
         KDA + attn 3:1                    47.5%     99.5%   0.0%           47.7%
         CKDA + attn 3:1                   49.0%     99.5%   0.0%           49.2%
FDA      KDA, bounded gate                 32.0%     83.5%  15.5%           38.3%
         CKDA                              32.5%     83.0%  14.5%           39.2%
         KDA + attn 3:1                    59.0%     97.0%   2.5%           60.8%
         CKDA + attn 3:1                   55.0%     97.0%   1.5%           56.7%
```

Answer rates are 95–100% everywhere except the two recurrent arms on FDA, and
`correct | answered` tracks the raw score within a few points throughout. **The
differences are wrong values, not missing ones.** The decisive cell is the
hybrid SQuAD gap: both arms answer 200 of 200, and the bounded gate is right
33.0% of the time against the signed gate's 23.0%. Nothing about compliance can
explain that.

This is the opposite of the S-NIAH picture, and the task format predicts it:
RULER's template ends mid-sentence at an answer prefix with nothing after it,
while these documents end at a key and a colon inside a page of other key–value
pairs the model has just read.

**One real echo effect, and it is symmetric.** The two recurrent arms restate
the prompt on ~15% of FDA documents — the longest in the suite, median 2,429
tokens — where their hybrids do so on ~2%. So part of the +33.5 that attention
buys on FDA is a compliance gain rather than a recall gain. It moves the
recurrent-vs-hybrid comparison, not the gate comparison: the two recurrent arms
echo at 15.5% and 14.5%, the two hybrids at 2.0% and 1.5%, so it cancels within
each pair.

## Caveats, stated rather than hidden

* **"Our KDA" is not literally either paper's KDA**: it is `kda-sig` — bounded
  sigmoid gate, fla's low-rank output gate, no SiLU on q/k/v, fla's HF init with
  A_log = 0. Same family, different details.
* **The hybrids are different designs**: GDN2's interleaves 2K sliding-window
  attention; ours is full gated NoPE attention every 4th layer. Our perplexity
  edge in the hybrid pair is plausibly that.
* **OpenBookQA is our weak column** — 26.8–29.0 against 30–33 published, in all
  four arms. Our `acc_norm` there is ~40, so it is a length-bias effect on the
  unnormalized metric rather than a broken task. It costs ~0.5 of the 9-task
  average.
* **n = 1 per arm.** No seed replicates at this scale; differences of one point
  in a downstream average are not resolvable.
* The validation loss above is on our own held-out FineWeb-Edu split and is
  **not** comparable to anyone else's number.

## What may not be claimed

**A downstream advantage for the signed gate.** Scored without a BOS token it
*trails* by 0.04 points on the pure pair and leads by 0.45 on the hybrid, at
n = 1 — and the same pairing run across the Megatron ladder's sixty paired
cells reads **−0.020 points, se 0.084, negative in 31 of 60**
(`ladder_results.md`). Held-out loss disagrees with the accuracy ordering here too, in both
pairs. Two of the three measurements go the other way; the one that does not is
the smallest sample.

The recall suite makes that a fourth measurement, and it is the sharpest of
them: the recurrent pair ties (+0.48 points of avg3) and the hybrid pair trails
by 3.91 points at z = −3.7, with the answer-rate explanation ruled out. It is
still n = 1 per arm — the se is over documents, not seeds — but it is now the
only 1.3B measurement in this file with an effect several times its own error,
and it points away from the signed gate.

**And it is not BoolQ.** BoolQ is the noisiest column in this suite — cell-to-cell
sd 3.1–3.5 points on the ladder — so it is the first thing to suspect. It does
not carry this: at 1.3B it moves −1.77 (pure) and +3.27 (hybrid), and dropping it
leaves +0.81 and +0.56 rather than +0.52 and +0.86. Dropping it on the ladder
leaves −0.30 and −0.34. Both measurements survive the obvious excuse and still
disagree, which is what makes the disagreement worth stating.

**Any claim about retrieval that rests on a SINGLE-NEEDLE cell.** Those cells
do not measure retrieval on these arms — no arm ever fetches a wrong needle on
S-NIAH-1/2/3, so the cell is an answer rate (see "What those single-needle cells
actually measure"). The paired differences they produce are real numbers about
real models and say nothing about retrieval: +22.77 in the recurrent stack,
−4.20 in the hybrid one, sign flipping per task, all of it answer-rate.

**What IS supported** is the multi-key comparison, the one cell family where
every arm answers and wrong needles are common: the signed gate leads in both
stacks (+6.0/+21.0/+12.0 recurrent, +21.2/+33.8/+27.4 hybrid) and by 21.5–26.5
points of `correct | answered`. There the ordering is consistent, has no
crossover between stacks, and is far outside the ±3.5 points n = 200 allows.

**That support does not extend to natural-document recall.** MK-NIAH-1 is
synthesised multi-key retrieval; SQuAD/SWDE/FDA are the same capability asked
for on real text, at 20–30x the sample size, and there the signed gate ties in
the recurrent stack and loses by 3.91 points in the hybrid one. Both
measurements pass the answer-rate test, so the disagreement is between two
sound measurements of related things, not between a finding and an artifact.
What survives both is narrower than the MK-NIAH paragraph alone reads: the
signed gate's multi-key advantage is established on RULER's synthetic format
and is *not* established as an advantage on documents.

The recurrent difference is also not cleanly attributable to the gate.
`drop_silu: true` is a campaign-wide setting, and the 47M measurement in
`silu_47M` shows SiLU is worth 0.015–0.021 to the *sigmoid* arms and nothing to
the *signed* ones — the same axis this difference lies along. No 1.3B arm exists
with SiLU on, so nothing here separates "the signed gate fixes a retrieval
failure" from "dropping SiLU cost the sigmoid arm something needle retrieval
depends on".

The hybrids narrow that second story without settling it: the sigmoid-gated
hybrid carries `drop_silu` too and is flawless on the single-needle tasks, so
whatever SiLU's removal costs is confined to the RECURRENT path and is covered
when attention is present. It does not make sigmoid-gated arms broken in
general.

Note also that this makes **four measurements pointing different ways**:
downstream avg9 puts the arms within half a point at 1.3B, the ladder's sixty
paired cells put the signed arm 0.020 *behind*, RULER-recurrent puts it 22.77
*ahead*, and RULER-hybrid puts it back at +0.72 with the sign flipped on three
tasks of four. No single "the signed gate is better/worse" sentence survives all
four, which is itself the result worth reporting.

**The supported claim is that the signed gate matches the bounded-gate baseline
and does not degrade it**, at 1.3B on a second corpus, against a published table
every arm of which both of ours beat. That is what `lm_scaling/tex/fwedu_1p3B.tex` says in
the caption of its second table, so the tables cannot travel without it.
