# gate_spectrum — results

What the state transition's eigenvalues actually are, in the four `fwedu_1p3B`
arms, at every one of their 22 checkpoints, plus the ladder's own 124M/6BT
signed arm at its 4. 

**Tables and figures are generated, not transcribed.** Regenerate with:

```
lm_scaling/gate_spectrum_plot.py --results DIR --markdown        # this file's tables
lm_scaling/gate_spectrum_plot.py --results DIR --out lm_scaling/tex   # the figures
```

The unit throughout is one **transition** — a single (token, value head), i.e.
one application of M. The JSON calls this `frac_transition_any_complex`; the
first sweep's files say `frac_step_any_complex` for the same thing, renamed
because "step" reads as a training step in a document about training. Both keys
are written now and the readers accept either.

Raw per-checkpoint JSON and the eigenvalue dumps behind the complex-plane
figure are COMMITTED, at `lm_scaling/harvest/gate_spectrum/` --
`spectrum_titan_*.json` and `eigs_*_step*.npz`. They are what the figures in
this document were drawn from, so redrawing them needs no cluster:

```
lm_scaling/gate_spectrum_plot.py --out lm_scaling/tex
lm_scaling/gate_spectrum_plot.py --latex > lm_scaling/tex/gate_spectrum.tex
```

A job writes them to `lm_scaling/results/` on JUPITER; those are the files that
were copied here.

## What is measured

At one (token, value head) the recurrence applies

```
M = (I - beta k k^T) Diag(alpha),   ||k|| = 1
```

over `head_k_dim` dimensions. `gate_spectrum.py` samples such transitions from
real validation batches — alpha and beta are ACTIVATIONS, so this is an average
over data, per layer — builds M, and eigendecomposes it in float64.

Each factor alone does something different, which is why the split is the
measurement:

| | what it can produce |
|---|---|
| `alpha < 0` alone | negative REAL eigenvalues, no rotation |
| `beta > 1` alone | one negative real eigenvalue, along k |
| both together | COMPLEX pairs, i.e. rotation |

`tests/lm/test_gate_spectrum.py` pins this against closed forms: the 2x2 case
`beta = 2, alpha = (1, -1)` is a planar rotation by phi to 1e-12
(`appendix_wfa_min.tex`, `lem:flip`), and `alpha > 0` can never rotate because
`M` is then similar to the symmetric `D^{1/2}(I - beta k k^T)D^{1/2}`.

## The control, which is the reason to believe the rest

`kda-sig-*` runs `allow_neg_eigval: False`, so beta is in (0,1) and alpha > 0:
every eigenvalue must be real and positive, by the inertia argument above. Over
44 independent checkpoints both unsigned arms report **0.00% negative and 0.00%
complex, at every one**. The measurement is not manufacturing structure.

## What happened

**1. The shipped init starts with no negative alpha at all.** `A_log = 0` and
`dt_bias` in [3.0, 7.6], so `u = z + dt_bias` and a channel only turns negative
when `f_proj(x) < -5`. At init the minimum over all channels and tokens is
-0.08: the signed arm begins life as plain KDA and has to learn its way out.

**2. Weight decay collapses the gate bias, and explains essentially all of it.**
`fwedu_1p3B` sets `TITAN_EXT_DECAY_ALL=1` (following Gated DeltaNet's
`pretrain.py`, which hands `model.parameters()` to AdamW as one group), so
`dt_bias` is decayed at wd 0.1 despite carrying `_no_weight_decay`. Decoupled
decay alone predicts `5.29 * exp(-wd * sum lr_t)`; the first checkpoint matches
that to **1.00**, and the gradient only ever mounts a weak restoring force,
reaching 1.5x by the end.

**3. It does not propagate.** `alpha < 0` reaches ~5.9% by step 68,096 — 36% of
the way — and then sits flat while `dt_bias` falls a further 4x. The share is
regulated; the bias is not what regulates it. `f_proj` absorbs the change.

The ladder says the same thing from the other side. It does NOT set
`DECAY_ALL`, so `dt_bias` keeps its exemption and does not move at all
(5.29 -> 5.334 over 23k steps, a 1% drift upward) — and its signed arm still
reaches `alpha < 0` at 4.67%, with 7.3% of its (token, head) transitions rotating. **Two runs
whose gate bias differs by 45x land within a point of each other on the quantity
that bias supposedly controls.**

| run | dt_bias | alpha<0 | beta>1 | rotating (token, head) |
|---|---|---|---|---|
| ladder 124M/6BT, no wd on the gate | 5.334 | 4.67% | 18.8% | 7.3% |
| fwedu 1.3B/100BT, wd 0.1 on the gate | 0.119 | 5.89% | 42.5% | 26.5% |

The rotating shares differ far more than the alpha shares do, and the beta
column is why: these two runs are not matched on beta at all.

**4. What keeps moving is beta.** 25% -> 42.5% above 1 across the run, still
climbing at the end, and that — not alpha — is what lifts the rotating share
from 14% to 26.5%. If there is a late-training story here it is a beta story.

**4b. Layer 0's beta is not monotone, and nothing else is like it.** It opens at
96% of heads above 1 at 5BT, *falls* to 43% by 35BT, then climbs back to 83% by
100BT; the hybrid's layer 0 does the same (96% -> 57% -> 87%). Every other layer
rises steadily from ~20% to ~45%. Its alpha<0 meanwhile rises monotonically
throughout. This is what made layer 0 read bright-dim-bright in the depth
figure. Worth a second look before it is published — a non-monotone first layer
is also what a settling embedding looks like.

**5. Rotation is a bottom-of-the-stack phenomenon.** Layer 0 carries it: 82.8%
of its heads reflect and 74.6% of its (token, head) transitions have a complex
pair, against 10-28% for layers 13-23. The hybrid is more extreme still, its L0 saturating
near 85%.

**6. The rotations are large in angle and small in modulus.** Median |arg lambda|
is 2.07 rad (118 degrees), q99 3.10 — near pi. But the complex eigenvalues have
median modulus 0.082 and q99 0.427, so a rotating plane is a fast-alternating
direction forgotten within a step or two, not a long-lived oscillator. **Those
two facts belong in the same sentence**; the first alone reads as a much
stronger claim than the data supports.

**7. A head rotates in at most one plane, and that is structural.** `M` is a
rank-one update of a diagonal, which admits at most one conjugate pair however
many channels are negative — measured, the mean is exactly 2.00 complex
eigenvalues per rotating transition at every checkpoint, every layer, both arms.
So the unit to quote is the (token, head) transition: **26.5% of them rotate**
at the end of training, 32.0% in the hybrid, and up to 83.6% in layer 0. Of the
64 planes a head could in principle turn, it turns one.

## Figures

Written to `lm_scaling/tex/` by the command above, as `.pdf` (tracked) and
`.png` (generated alongside for viewing; `*.png` is gitignored), with
`gate_spectrum.tex` carrying the table and the figure environments.

* [`gate_spectrum_trajectory.pdf`](../../tex/gate_spectrum_trajectory.pdf) —
  dt_bias against the no-gradient decay curve; the two factors; the outcome.
  The unsigned arms are drawn as the flat zero they must be.
* [`gate_spectrum_depth.pdf`](../../tex/gate_spectrum_depth.pdf) — layers down,
  tokens across, colour is the share of transitions that rotate.
* [`gate_spectrum_gates.pdf`](../../tex/gate_spectrum_gates.pdf) — **the gates
  as distributions, not shares**: log density of alpha and beta against training
  (layers pooled) and against layer (final checkpoint), each column normalised
  on its own. Bins are merged where the sample is thin: beta has one value per
  transition against alpha's `head_dim`, so an 81-bin per-layer beta histogram
  holds three counts a bin and renders as a picture of the sample size.
* [`gate_spectrum_by_layer.pdf`](../../tex/gate_spectrum_by_layer.pdf) — every
  layer's trajectory at once, coloured by depth. Where the layer-0 beta U-turn
  below is visible.
* [`gate_spectrum_position.pdf`](../../tex/gate_spectrum_position.pdf) — the five
  statistics against position in the window, log axis. Sharp in the first ~64
  tokens, flat after.
* [`gate_spectrum_plane.pdf`](../../tex/gate_spectrum_plane.pdf) — the
  eigenvalues themselves, unsigned beside signed. Top row the complex plane with
  the unit circle and the rotating population inset at its own scale; bottom row
  the marginal in Re(lambda) on shared axes, which is where the result is
  actually legible: **the bounded-gate arm's histogram stops dead at 0** and the
  signed arm's runs to -1. Needs an `EIGS=` run (jobs 1792926-27).

## The numbers

### dt_bias over training (median across layers of each layer's median)

| tokens (B) | step | Complex KDA | KDA, bounded gate | Complex KDA + attn 3:1 | KDA + attn 3:1 | wd alone |
|---|---|---|---|---|---|---|
| 5.1 | 9,728 | 3.736 | -4.382 | 3.743 | -4.393 | 3.726 |
| 10.2 | 19,456 | 2.549 | -3.002 | 2.564 | -3.007 | 2.535 |
| 15.3 | 29,184 | 1.768 | -2.080 | 1.773 | -2.078 | 1.739 |
| 20.4 | 38,912 | 1.246 | -1.465 | 1.248 | -1.464 | 1.208 |
| 25.5 | 48,640 | 0.896 | -1.053 | 0.899 | -1.052 | 0.852 |
| 30.6 | 58,368 | 0.660 | -0.776 | 0.663 | -0.776 | 0.614 |
| 35.7 | 68,096 | 0.501 | -0.587 | 0.502 | -0.588 | 0.452 |
| 40.1 | 76,440 | 0.405 | -0.460 | 0.392 | -0.458 | 0.355 |
| 40.8 | 77,824 | 0.391 | -0.420 | 0.379 | -0.409 | 0.342 |
| 45.9 | 87,552 | 0.316 | -0.371 | 0.316 | -0.370 | 0.265 |
| 51.0 | 97,280 | 0.262 | -0.307 | 0.261 | -0.306 | 0.211 |
| 56.1 | 107,008 | 0.224 | -0.261 | 0.222 | -0.261 | 0.174 |
| 61.2 | 116,736 | 0.196 | -0.227 | 0.193 | -0.227 | 0.146 |
| 66.3 | 126,464 | 0.175 | -0.202 | 0.173 | -0.202 | 0.127 |
| 71.4 | 136,192 | 0.160 | -0.184 | 0.157 | -0.184 | 0.113 |
| 76.5 | 145,920 | 0.148 | -0.170 | 0.145 | -0.170 | 0.102 |
| 81.0 | 154,460 | 0.140 | -0.160 | 0.136 | -0.160 | 0.096 |
| 81.6 | 155,648 | 0.139 | -0.155 | 0.134 | -0.152 | 0.095 |
| 86.7 | 165,376 | 0.132 | -0.152 | 0.129 | -0.151 | 0.089 |
| 91.8 | 175,104 | 0.126 | -0.146 | 0.124 | -0.145 | 0.085 |
| 96.9 | 184,832 | 0.122 | -0.141 | 0.120 | -0.140 | 0.082 |
| 100.1 | 190,976 | 0.119 | -0.138 | 0.117 | -0.137 | 0.080 |

### Occupancy and spectrum, signed arms

| tokens (B) | step | Complex KDA: a<0 / b>1 / rotating | Complex KDA + attn 3:1: a<0 / b>1 / rotating |
|---|---|---|---|
| 5.1 | 9,728 | 3.03% / 25.1% / 14.0% | 3.11% / 28.6% / 16.5% |
| 10.2 | 19,456 | 3.72% / 26.3% / 14.8% | 3.79% / 29.5% / 17.5% |
| 15.3 | 29,184 | 4.35% / 25.0% / 14.6% | 4.20% / 29.7% / 18.1% |
| 20.4 | 38,912 | 4.78% / 24.5% / 14.1% | 4.66% / 29.8% / 18.1% |
| 25.5 | 48,640 | 5.24% / 25.2% / 15.0% | 5.06% / 30.1% / 18.6% |
| 30.6 | 58,368 | 5.57% / 25.6% / 15.0% | 5.30% / 31.6% / 18.9% |
| 35.7 | 68,096 | 5.83% / 26.7% / 15.5% | 5.49% / 32.3% / 19.6% |
| 40.1 | 76,440 | 5.87% / 27.6% / 16.5% | 5.62% / 34.4% / 20.9% |
| 40.8 | 77,824 | 5.83% / 27.6% / 16.2% | 5.70% / 34.5% / 22.1% |
| 45.9 | 87,552 | 5.90% / 29.2% / 17.0% | 5.77% / 35.2% / 22.4% |
| 51.0 | 97,280 | 5.92% / 30.9% / 18.2% | 5.57% / 37.4% / 23.4% |
| 56.1 | 107,008 | 6.06% / 31.8% / 18.8% | 5.76% / 38.3% / 24.0% |
| 61.2 | 116,736 | 6.21% / 33.4% / 19.8% | 5.76% / 39.5% / 25.2% |
| 66.3 | 126,464 | 6.04% / 34.9% / 21.3% | 5.80% / 41.3% / 27.5% |
| 71.4 | 136,192 | 6.04% / 35.8% / 22.0% | 5.77% / 42.9% / 27.9% |
| 76.5 | 145,920 | 5.86% / 37.1% / 23.1% | 5.74% / 43.7% / 29.5% |
| 81.0 | 154,460 | 5.97% / 39.1% / 24.1% | 5.66% / 45.6% / 30.6% |
| 81.6 | 155,648 | 5.94% / 38.8% / 24.2% | 5.60% / 45.6% / 30.2% |
| 86.7 | 165,376 | 5.93% / 40.0% / 24.8% | 5.64% / 46.4% / 31.0% |
| 91.8 | 175,104 | 5.88% / 41.1% / 25.5% | 5.67% / 47.4% / 31.6% |
| 96.9 | 184,832 | 5.81% / 41.7% / 25.9% | 5.58% / 48.3% / 32.1% |
| 100.1 | 190,976 | 5.89% / 42.5% / 26.5% | 5.51% / 48.8% / 32.0% |

### Final checkpoint, every arm

| arm | layers | a<0 | b>1 | mean beta | median abs alpha | rotating (token, head) | angle q50 | modulus q50 |
|---|---|---|---|---|---|---|---|---|
| Complex KDA | 24 | 5.89% | 42.5% | 0.911 | 0.9048 | 26.5% | 2.07 rad | 0.082 |
| KDA, bounded gate | 24 | 0.00% | 0.0% | 0.531 | 0.9321 | 0.0% | -- | -- |
| Complex KDA + attn 3:1 | 18 | 5.51% | 48.8% | 0.983 | 0.9077 | 32.0% | 2.09 rad | 0.093 |
| KDA + attn 3:1 | 18 | 0.00% | 0.0% | 0.581 | 0.9284 | 0.0% | -- | -- |

## Does position in the sequence explain it?

Raised by a collaborator: the sample is uniform over the 4,096-token window,
and at t = 0 there is no state to forget, so gates near 1 there would mean
something different from gates near 1 at t = 4000. Measured (job 1810642,
`PER_CALL=512`, first and last checkpoint of `ckda-shipped-lowrank`):

`ckda-shipped-lowrank` at step 190,976:

| position | n | alpha<0 | beta>1 | rotating | median abs alpha | mean alpha range |
|---|---|---|---|---|---|---|
| 0 | 8 | 10.06% | 87.5% | 75.0% | 0.4902 | 1.2316 |
| 1-7 | 75 | 7.61% | 64.0% | 48.0% | 0.7646 | 1.2925 |
| 8-63 | 719 | 6.30% | 52.2% | 32.0% | 0.8759 | 1.2599 |
| 64-511 | 5,437 | 5.72% | 43.6% | 26.9% | 0.9022 | 1.2553 |
| 512-2047 | 18,213 | 5.85% | 43.0% | 26.7% | 0.8974 | 1.2555 |
| 2048-4095 | 24,700 | **5.82%** | **41.7%** | **25.6%** | **0.9019** | 1.2542 |
| pooled | 49,152 | **5.83%** | **42.6%** | **26.3%** | **0.8997** | 1.2550 |

**The effect is real, confined, and points the other way.** The first ~64
positions are sharply different -- position 0 rotates in 75% of its transitions
against 25.6% at the end of the window, and its median |alpha| is 0.49 against
0.90. But those positions are 1.6% of the sample, so the pooled figure is the
late-window figure: 5.83% against 5.82%, 26.3% against 25.6%. Sampling only the
end of the sequence would move the headline numbers by hundredths of a point.

And the direction is the opposite of the concern. Early positions are MORE
aggressive, not more "stuck at 1": more negative alpha, more reflection, more
rotation, smaller |alpha|. Dropping them would lower the shares slightly, not
raise them.

**The mass near alpha = 1 is a training-time fact, not a position fact.** At 5BT
the median |alpha| is 0.963 and at 100BT it is 0.900; the init is 0.988. The
model starts barely forgetting and learns to forget more, which is what the
density panel shows.

## Caveats

* The ladder-vs-fwedu comparison confounds weight decay with scale AND budget
  (124M/6BT against 1.3B/100BT). It shows the bias is not what sets the share;
  it does NOT isolate what weight decay is worth. One 124M/6BT run with
  `TITAN_EXT_DECAY_ALL=1` would, for about 9 GPU-hours.
* **The data is a small fixed slice of the held-out split**, taken from the
  head of the validation iterator with no shuffle: 4 batches x
  `local_batch_size` sequences, i.e. 65,536 tokens (16 windows) for the fwedu
  arms and 131,072 (64 windows) for the ladder. The same text at every
  checkpoint, which is what makes the trajectory paired rather than resampled --
  but a narrow slice, so nothing text-dependent is averaged away.
* **The ladder comparison is on a different corpus too.** Nemotron for the
  ladder, FineWeb-Edu for fwedu, on top of the difference in scale and budget.
  The alpha<0 agreement survives all three, which is the point, but the list of
  differences is longer than "weight decay".
* **Per-layer precision is 3 points, pooled is under 1.** `per_call` 64 x 4
  batches = 256 transitions per layer per checkpoint, so a per-layer share near
  0.4 has a binomial se of ~3 points -- that is the visible jitter in the
  by-layer curves, and a 3-point gap between two middle layers is not a result.
  alpha<0 is tighter (256 x 128 channel values, though correlated within a
  transition), and pooling 24 layers brings the shares to ~0.5-1 point.
* **The share of EIGENVALUES in a conjugate pair is not reported, because it is
  not a measurement.** `M` is a rank-one update of a diagonal, so it admits at
  most ONE conjugate pair per head — the conditional mean is exactly 2.00
  complex eigenvalues per rotating transition, at every checkpoint of every
  layer of both arms. That makes the per-eigenvalue share equal to
  `frac_step_any_complex * 2/head_dim` exactly (agreeing to 1e-7 over 44
  checkpoints), with a ceiling of 2/128 = 1.56%. Quoting it as "0.41% complex"
  reads as "rotation is vanishingly rare" when it means "a quarter of the
  maximum the parameterisation permits". It stays in the JSON and is out of
  every table and figure.
* The complex/real split is a numerical judgement at the bottom. Reported at
  three thresholds on |Im| / spectral radius (1e-8, 1e-6, 1e-4); all three agree
  to 0.01 points here, so the threshold is not load-bearing.

## Reproducibility

The measurement is deterministic: the windows come from the head of the
validation split in order and the sampler is seeded, so two independent
submissions of the same sweep agree to **0.00e+00** on every share at every
checkpoint (jobs 1792234 and 1796693, 22 checkpoints). Differences between
checkpoints are the model moving, not the sample.

## Reproduce

Only to MEASURE A NEW RUN, or to re-check the committed numbers -- the figures
above regenerate from `harvest/gate_spectrum/` without any of this.

```
CFG=<run>/titan.toml sbatch lm_scaling/gate_spectrum.sbatch                 # all checkpoints
CFG=<run>/titan.toml STEPS="9728 190976" EIGS=lm_scaling/results/eigs_<arm> \
    sbatch lm_scaling/gate_spectrum.sbatch                                  # + raw spectrum
```
