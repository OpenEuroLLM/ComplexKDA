# Finite-group state tracking

This directory contains the state-tracking experiments from the CKDA paper. Given a sequence of arbitrary elements of a finite group, a model must predict the cumulative product after every input. The experiments train on short words and evaluate at substantially greater lengths, making them a controlled test of whether a learned recurrent transition preserves the group state rather than merely fitting the training horizon.

## Task and model

For a group $G$, the input is an integer tensor of shape `batch × sequence_length`; each integer indexes one of the $|G|$ group elements. The target has the same shape and contains the running products $y_t=x_t x_{t-1}\cdots x_1$. Tokens are embedded to `batch × sequence_length × 48`, passed through one sequence layer, and decoded to logits of shape `batch × sequence_length × |G|`; training uses next-state cross-entropy and evaluation reports accuracy over the final quarter of each sequence.

The standard paper configuration uses one layer, hidden size 48, 12 heads of dimension 16, no short convolution, and a `48 → 192 → |G|` MLP readout. Models train for 60,000 updates with batch size 1,024, Muon on hidden matrices, AdamW on the remaining parameters, and a scheduled length curriculum of `4, 6, 8, 16, 32`. Results are evaluated at every length through 512 and reported as accuracy scaled so that random choice is zero and perfect prediction is one.

The four KDA settings independently vary the channel-wise gate range and Householder rate range:

| Driver name | Gate range | Rate range |
| --- | --- | --- |
| `a01_b01` | $\alpha\in[0,1]$ | $\beta\in[0,1]$ |
| `a01_b02` | $\alpha\in[0,1]$ | $\beta\in[0,2]$ |
| `a11_b01` | $\alpha\in[-1,1]$ | $\beta\in[0,1]$ |
| `a11_b02` | $\alpha\in[-1,1]$ | $\beta\in[0,2]$ |

## Installation

Install the repository and the plotting dependency from the repository root:

```bash
# NVIDIA example
pip install -e ".[cuda,benchmark]"
pip install -r requirements-experiments.txt

# CPU or Apple Silicon example
pip install -e ".[cpu,benchmark]"
pip install -r requirements-experiments.txt
```

The reference recurrence works on CPU and Apple Silicon. The complete paper sweep is computationally expensive; use the smoke mode locally before launching the full configuration.

## Reproduce

Run a short end-to-end check of all four KDA settings on $S_3$, $S_4$, and $A_5$, plus the theory-initialized $A_5$ model:

```bash
python group_word_problems/reproduce_state_tracking.py --mode smoke --device cpu
```

Run the paper configuration with three seeds per setting:

```bash
python group_word_problems/reproduce_state_tracking.py --mode paper --device cuda
```

Use `--tasks s3,s4`, `--variants a11_b02`, or `--skip-theory-a5` to select a smaller subset. Add `--include-deltaproduct2` for the two-Householder baseline. The driver refuses to append to existing result files; choose a new `--output-root` for a separate run.

Outputs are written to `artifacts/state_tracking/`:

- `results/{task}_{variant}.jsonl`: one row per seed, including the complete length-generalization curve and training configuration.
- `results/aggregates.json`: every per-seed row plus unrounded distributional summaries and the paper's explicit best-seed statistic.
- `checkpoints/`: trained state dictionaries referenced by the result rows.
- `figures/state_tracking.pdf`: scaled accuracy against sequence length for every completed setting.
- `environment.json` and `run_manifest.json`: source, software/hardware environment, device request, and selected run suite.

Every new result row includes a full-precision training trace, the unrounded evaluation curve, exact parameter counts, and all optimizer settings. Use [`evidence/aggregate_results.py`](../evidence/aggregate_results.py) to retain every seed and report both distributional summaries and the paper's explicit best-seed selection.

The checked best-of-three paper values at lengths 32, 128, and 512 are recorded in [`reference_state_tracking_metrics.json`](reference_state_tracking_metrics.json). They show that both range extensions are necessary for the strongest learned extrapolation on $S_3$ and $S_4$. Randomly initialized models do not learn $A_5$ in this setup; the separately reported theory-initialized model is smaller, uses a different schedule, and is therefore evidence about optimization rather than a controlled initialization-only ablation.

## Additional entry points

- [`train_wordproblem.py`](train_wordproblem.py) contains the group construction, data generator, models, training loop, curricula, evaluation, and lower-level CLI.
- [`plot_state_tracking.py`](plot_state_tracking.py) plots outputs produced by the reproduction driver.
- [`a5_exact_tracker.py`](a5_exact_tracker.py) verifies the exact four-dimensional quaternion construction without training.
- [`plot_complex_kda_interpretability.py`](plot_complex_kda_interpretability.py) analyzes learned gates, keys, and transition spectra from a trained $S_3$ or $S_4$ checkpoint.
- [`plot_a5_interpretability.py`](plot_a5_interpretability.py) compares a theory-initialized $A_5$ checkpoint with the quaternion construction.
- [`evaluate_s3_phase_control.py`](evaluate_s3_phase_control.py) performs radial and phase-reversal interventions on a learned complex mode.

Generated checkpoints, JSONL logs, and figures are excluded from version control. The reproduction driver records every effective hyperparameter and checkpoint path in its output so that derived analyses remain auditable.
