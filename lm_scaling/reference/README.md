# The reference study's own numbers

`oellm_loss_post_annealing.csv` is `data/loss_all_exps_post_annealing.csv` from
github.com/OpenEuroLLM/dense_english_scaling_laws, unmodified. 909 runs: a
sweep over learning rate, batch size, beta2 and seed at six model sizes
(50M-1.7B) and nine budgets (6-300BT), each scored on their common held-out set
of 204,800 sequences.

## Why it is vendored

`scaling_fit.PAPER` held constants transcribed from the paper's Table 3, and
neither row reproduces these measurements under the forms in `scaling_fit.py`:

    PAPER[chinchilla] minus their data:  mean +0.9830  sd 0.2747
    PAPER[skaling]    minus their data:  mean +0.1054  sd 0.0221

Refitting this CSV with our own grid solver lands on alpha 0.270, beta 0.250,
E 1.462 (rmse 0.018) against their published alpha 0.2807, beta 0.2534,
E 1.418 -- so their exponents are right, their data is right, our solver is
right, and the A/B constants are in a convention we do not have (N and D are
probably not in raw units there). A reference you cannot evaluate is not a
reference, so the fit below is computed FROM the data and the transcription is
kept only as a marker.

## What it says about our ladder

Our attention arm sits +0.2665 nats above their best-HP run at the same (N, D),
sd 0.0167 over 25 matched cells. That is not hyperparameters: at 983M/50BT
their best is lr 5e-4, gbsz 128, beta2 0.95, which is our configuration
exactly, and the gap there is still +0.2464. Their best learning rates track
ours at nearly every rung, and at 302M and above their best beta2 is 0.95 like
ours.

The offset is near-constant across 36x in N and 8.3x in D. A training
deficiency grows or shrinks with scale; a flat offset is a measurement or
systematic difference.

QK_LAYERNORM IS ONE TERM OF IT AND IT IS SMALL. Theirs is on and our `attn` arm
has it off, so the ladder carries `attn-qknorm` as its own arm to price the
difference rather than argue about it: over the 30 cells both arms ran,
qk_layernorm is worth **+0.0277 nats, sd 0.0042** (range +0.0194 to +0.0382),
computable from `harvest/megatron_ladder.tsv`. That is a tenth of the +0.2665
offset and it does not grow with scale either. Whatever the rest is, it is not
this. The remaining candidate we have not isolated is the framework itself,
Megatron against torchtitan.

Until that is closed, the ladder's absolute losses are comparable to our own
arms and to nothing else, which is what config/data/nemotron_neox.yaml already
says. The results that survive it are the within-cell ones -- the hybrid's
x1.22-1.30 equivalent-data advantage and the hybrid-over-pure constant of
-0.0186 +- 0.002 -- because those need no baseline outside the cell.
