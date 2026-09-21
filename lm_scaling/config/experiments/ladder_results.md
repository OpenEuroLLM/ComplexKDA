# The scaling ladder — results

Six arms x six rungs (47M to 1.7B) x five budgets (6-50BT) on Nemotron-CC under
the GPT-NeoX tokenizer, trained with **Megatron-LM**. 180 cells, all finished.

An earlier torchtitan ladder is superseded and is not in this release: the two
frameworks disagreed by 0.049 nats on identical configuration, and the ladder
moved to Megatron to remove that term. Where a number here differs from one in
an earlier draft, this is the one to trust.

**Everything below regenerates from committed data**, no cluster needed:

```
lm_scaling/scaling_fit.py                    # the fits, from harvest/megatron_ladder.tsv
lm_scaling/scaling_fit.py --latex            # ... as the paper's tables
lm_scaling/scaling_plots.py                  # the figures
lm_scaling/ladder_downstream.py              # the pairing, from harvest/ladder_downstream/
```

## Held-out validation loss, every cell

| cell | attn | attn-qknorm | KDA-sig | CKDA | KDA-sig-hybrid | CKDA-hybrid |
|---|---|---|---|---|---|---|
| 47M/6BT | 2.9844 | 2.9463 | 2.9247 | 2.9267 | 2.9126 | 2.9167 |
| 47M/12BT | 2.9109 | 2.8833 | 2.8664 | 2.8671 | 2.8522 | 2.8560 |
| 47M/20BT | 2.8782 | 2.8504 | 2.8403 | 2.8397 | 2.8250 | 2.8257 |
| 47M/30BT | 2.8578 | 2.8335 | 2.8215 | 2.8220 | 2.8063 | 2.8075 |
| 47M/50BT | 2.8271 | 2.8045 | 2.7974 | 2.7962 | 2.7809 | 2.7814 |
| 124M/6BT | 2.7471 | 2.7188 | 2.6869 | 2.6880 | 2.6718 | 2.6719 |
| 124M/12BT | 2.6799 | 2.6541 | 2.6212 | 2.6220 | 2.6072 | 2.6083 |
| 124M/20BT | 2.6389 | 2.6124 | 2.5837 | 2.5850 | 2.5705 | 2.5708 |
| 124M/30BT | 2.6074 | 2.5851 | 2.5576 | 2.5581 | 2.5442 | 2.5462 |
| 124M/50BT | 2.5682 | 2.5488 | 2.5286 | 2.5288 | 2.5146 | 2.5154 |
| 302M/6BT | 2.6126 | 2.5789 | 2.5243 | 2.5258 | 2.5146 | 2.5174 |
| 302M/12BT | 2.5382 | 2.5044 | 2.4505 | 2.4550 | 2.4446 | 2.4437 |
| 302M/20BT | 2.4942 | 2.4560 | 2.4077 | 2.4089 | 2.3956 | 2.3967 |
| 302M/30BT | 2.4598 | 2.4287 | 2.3768 | 2.3784 | 2.3651 | 2.3667 |
| 302M/50BT | 2.4160 | 2.3882 | 2.3441 | 2.3453 | 2.3323 | 2.3335 |
| 588M/6BT | 2.5209 | 2.4892 | 2.4496 | 2.4532 | 2.4430 | 2.4424 |
| 588M/12BT | 2.4346 | 2.4051 | 2.3626 | 2.3673 | 2.3556 | 2.3555 |
| 588M/20BT | 2.3805 | 2.3534 | 2.3143 | 2.3191 | 2.3051 | 2.3049 |
| 588M/30BT | 2.3436 | 2.3172 | 2.2775 | 2.2815 | 2.2676 | 2.2683 |
| 588M/50BT | 2.3032 | 2.2770 | 2.2386 | 2.2417 | 2.2284 | 2.2295 |
| 983M/6BT | 2.4551 | 2.4282 | 2.3905 | 2.3963 | 2.3810 | 2.3850 |
| 983M/12BT | 2.3682 | 2.3403 | 2.3009 | 2.3059 | 2.2920 | 2.2959 |
| 983M/20BT | 2.3114 | 2.2844 | 2.2484 | 2.2532 | 2.2369 | 2.2405 |
| 983M/30BT | 2.2721 | 2.2464 | 2.2098 | 2.2141 | 2.1981 | 2.2017 |
| 983M/50BT | 2.2298 | 2.2070 | 2.1686 | 2.1722 | 2.1571 | 2.1608 |
| 1.7B/6BT | 2.4001 | 2.3725 | 2.3359 | 2.3402 | 2.3284 | 2.3299 |
| 1.7B/12BT | 2.3100 | 2.2829 | 2.2496 | 2.2507 | 2.2399 | 2.2433 |
| 1.7B/20BT | 2.2488 | 2.2218 | 2.1882 | 2.1898 | 2.1795 | 2.1828 |
| 1.7B/30BT | 2.2077 | 2.1812 | 2.1477 | 2.1494 | 2.1395 | 2.1429 |
| 1.7B/50BT | 2.1630 | 2.1374 | 2.1046 | 2.1061 | 2.0971 | 2.1002 |

## Mean against the dense baseline

```
attn                           mean vs attn: +0.0000 nats over 30 cells
  attn-qknorm                    mean vs attn: -0.0277 nats over 30 cells
  kda-sig-lowrank                mean vs attn: -0.0617 nats over 30 cells
  ckda-shipped-lowrank           mean vs attn: -0.0594 nats over 30 cells
  kda-sig-hybrid-lowrank         mean vs attn: -0.0728 nats over 30 cells
  ckda-shipped-hybrid-lowrank    mean vs attn: -0.0709 nats over 30 cells
```

**The family beats both attention baselines at every rung.** qk-normed attention
is 0.028 nats better than plain attention -- which is why it is in the ladder at
all, as the stronger baseline -- and the KDA family is a further 0.031 to 0.044
below that. The hybrids lead everywhere.

**Complex KDA against the bounded-gate baseline is a wash in loss, slightly
favouring the baseline**: +0.0023 nats in the pure pair (t +6.71 over 30 cells)
and +0.0018 in the hybrid pair (t +6.35). Small, consistent, and in the
baseline's favour. Downstream it is level: -0.027 points of 9-task average over
all 60 paired cells (se 0.084, negative in 31). The place the two arms separate
is long-context retrieval, and that is measured at 1.3B rather than here -- see
[`fwedu_1p3B_results.md`](fwedu_1p3B_results.md).

## What the fit says

`scaling_fit.py` fits the reference's loss law, L = E + A/N^alpha + B/D^beta,
jointly across arms with one shape and one offset per arm. Over the 180 cells it
gives alpha 0.49 and beta 0.42 against the reference's 0.48 and 0.35, and orders
the arms by E exactly as the per-cell losses do. The arms share a shape and
differ by an offset; that is the claim the ladder supports.
