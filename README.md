# ComplexKDA
[![Paper](https://img.shields.io/static/v1?label=Paper&message=2609.24797&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2609.24797)


> **Paper:** https://arxiv.org/abs/2609.24797
>
> **Authors:** Julien Siems, Riccardo Grazzi, Korbinian Pöppel, Jaisidh Singh, Arber Zela, Timur Carstensen, Jenia Jitsev, Frank Hutter, Volkan Cevher, Antonio Orvieto, Aaron Klein

<img width="997" height="697" alt="image" src="https://github.com/user-attachments/assets/d243e734-3176-43f6-beb4-e98cb113a2c0" />

This repository is the reproducibility artifact for **Complex Kimi Delta Attention (CKDA)**. It contains the CKDA implementation built on [Flash Linear Attention (FLA)](https://github.com/fla-org/flash-linear-attention), together with the finite-group state-tracking and periodic-waveform experiments used in the paper.

CKDA extends the ranges of the KDA recurrence parameters to permit signed state transitions: the channel-wise gate satisfies **α ∈ [−1, 1]**, and the Householder rate satisfies **β ∈ [0, 2]**. The experiments isolate when these extended ranges enable persistent reflections and rotations that ordinary nonnegative transitions cannot represent directly.


## Contents

| Component | Description |
| --- | --- |
| [`fla/`](fla/) | FLA kernels, layers, and model implementations, including CKDA. |
| [`group_word_problems/`](group_word_problems/) | Single-layer finite-group state tracking on S₃, S₄, and A₅. |
| [`audio_toys/`](audio_toys/) | Single-layer periodic waveform continuation, baselines, spectral analysis, plots, and WAV export. |
| [`lm_scaling/`](lm_scaling/) | The language-modelling experiments: the 1.3B / 100BT FineWeb-Edu runs (torchtitan) and the six-rung scaling ladder (Megatron-LM), with corpus staging, both backends' integrations, the campaign configs, the downstream and RULER evaluations, the scaling-law fits, and the measured results those tables are made from. |
| [`evidence/`](evidence/) | Claim-to-artifact index, archived audio measurements and arrays, provenance manifests, and unrounded aggregation tools. |
| [`tests/group_word_problems/`](tests/group_word_problems/) | Focused tests for the state-tracking task and exact A₅ construction. |
| [`tests/audio_toys/`](tests/audio_toys/) | Focused tests for waveform generation, causality, and transition initialization. |
| [`tests/lm_scaling/`](tests/lm_scaling/) | Focused tests for the campaign config, the arm table, the torchtitan shim, and the result tables. |

The detailed protocols are documented in the [state-tracking guide](group_word_problems/README.md), the [audio guide](audio_toys/README.md) and the [language-modelling guide](lm_scaling/README.md). The [evidence index](evidence/README.md) distinguishes archived measurements from reproducibility support and lists remaining gaps explicitly.

The language-modelling results are the expensive ones, so they are committed
rather than only described — 180 scaling-ladder cells with their downstream
scores, the four 1.3B arms, the RULER runs and the gate spectrum, all under
`lm_scaling/harvest/`. Every table and figure the paper draws from them
regenerates from this checkout, with no cluster and no GPU:

```bash
lm_scaling/reproduce_tables.sh --check   # regenerate into a scratch dir and diff
```

If that reports no differences, each number in `lm_scaling/tex/` is exactly what
the committed measurements produce. Re-running the training that produced those
measurements is a separate matter and needs a cluster; the
[language-modelling guide](lm_scaling/README.md) covers it.

## Installation

Python 3.10 or newer is recommended. Clone the repository and install the backend appropriate for your machine:

```bash
git clone git@github.com:automl/ComplexKDA.git
cd ComplexKDA

# NVIDIA CUDA
pip install -e ".[cuda,benchmark]"

# CPU or Apple Silicon
pip install -e ".[cpu,benchmark]"

# Plotting dependencies used by both experiment packages
pip install -r requirements-experiments.txt
```

See [`INSTALL.md`](INSTALL.md) for the complete FLA backend matrix, including ROCm, Intel XPU, and Ascend NPU.

## Quick verification

The smoke configurations run on CPU and exercise both experiment packages end to end:

```bash
python group_word_problems/reproduce_state_tracking.py --mode smoke --device cpu
python audio_toys/reproduce_periodic_waveform.py --mode smoke --device cpu
```

Run the focused tests with:

```bash
pytest -q \
  tests/layers/test_complex_kda_layer.py \
  tests/group_word_problems/test_state_tracking.py \
  tests/audio_toys/test_periodic_waveform.py \
  tests/test_evidence.py
python evidence/verify_archive.py
```

The language-modelling tests need no GPU and run in well under a minute:

```bash
pytest -q tests/lm_scaling
```


## Optional kernel implementations

The default route retains the existing backend-selection policy; the additional optimizations are opt-in. On NVIDIA Hopper, select `FLA_KDA_BACKEND=triton_optimized` or `FLA_KDA_BACKEND=hybrid_optimized`; the hybrid requires TileLang. Explicit `triton` and `tilelang` selections retain the baseline routes. See the [backend guide](fla/ops/kda/backends/optimized/README.md) for supported inputs, reproducibility settings, and validation commands.


## Finite-group state tracking

Given arbitrary elements of a finite group, a model predicts the cumulative product after every token. The paper setup trains a single sequence layer on words of length at most 32 and evaluates length generalization through 512.

Run the complete three-seed configuration on a CUDA device:

```bash
python group_word_problems/reproduce_state_tracking.py \
  --mode paper \
  --device cuda
```

The driver trains the four KDA range combinations, evaluates S₃, S₄, and A₅, and generates the comparison figure. Use `--tasks`, `--variants`, or `--skip-theory-a5` to select a smaller subset. The exact commands, tensor shapes, curriculum, optimizer partition, and interpretation of the theory-initialized A₅ run are given in [`group_word_problems/README.md`](group_word_problems/README.md).

Compact checked values are included in [`reference_state_tracking_metrics.json`](group_word_problems/reference_state_tracking_metrics.json). Raw training logs and checkpoints are deliberately not stored in Git.

## Periodic waveform continuation

The audio experiment is a controlled phase-preserving extrapolation task. Each model receives an eight-frame half-bar cue from a randomly circularly shifted synthetic groove; all later inputs are zero, and the model regresses the subsequent waveform. Every architecture uses one sequence-mixing layer with comparable width.

Run the paper configuration with:

```bash
python audio_toys/reproduce_periodic_waveform.py \
  --mode paper \
  --device cuda
```

The driver trains the four KDA range variants, causal Transformer, and GRU; evaluates short, maximum-training, and extrapolation horizons; measures the CKDA transition spectrum; generates the paper figures; and exports listening WAVs. Additional positional-encoding and DeltaProduct-2 controls are available through `--include-controls`.

Compact checked values are included in [`reference_periodic_waveform_metrics.json`](audio_toys/reference_periodic_waveform_metrics.json). The task intentionally overfits circular shifts of one fixed groove, so it diagnoses phase-preserving dynamics rather than general-purpose audio generation. The GRU remains the strongest model on this particular task.

## Generated outputs

By default, reproduction artifacts are written below `artifacts/`:

```text
artifacts/
├── state_tracking/
│   ├── checkpoints/
│   ├── figures/
│   ├── results/
│   ├── environment.json
│   └── run_manifest.json
└── periodic_waveform/
    ├── arrays/
    ├── audio/
    ├── checkpoints/
    ├── figures/
    ├── results/
    ├── environment.json
    └── run_manifest.json
```

These newly generated directories are ignored by Git. The curated files under [`evidence/results/`](evidence/results/) are explicit exceptions: they archive the original waveform result rows and prediction arrays, but not checkpoints, generated audio, or unavailable historical state-tracking and language-model logs.

## Reproducibility notes

- The public reproduction drivers pass all numerical settings explicitly and refuse to append silently to existing result files.
- Paper runs use Muon for internal two-dimensional sequence-layer parameters and AdamW for adapters and remaining parameters. Language Model runs use AdamW only for comparability.
- CUDA uses optimized FLA kernels where available. CPU and Apple Silicon use the pure-PyTorch recurrent path and are suitable for smoke testing.
- Reference metrics are single-run or best-of-three summaries as stated in each experiment guide; consult those guides before drawing statistical conclusions.

### The scaling ladder's external stack

The scaling ladder runs through [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [oellm-autoexp](https://github.com/OpenEuroLLM/oellm-autoexp), pinned by commit in
[`lm_scaling/megatron_stack.lock`](lm_scaling/megatron_stack.lock) and fetched
on demand — the trees themselves are not vendored here. Two of the pinned
commits are ours, on a branch named `feat/complex-kda` in each OpenEuroLLM
repository: one adds the `complex_kda` attention variant to Megatron-LM, the
other carries its fields through oellm-autoexp's config layer. Both default to
the underlying layer's own values, so they are inert unless the variant is
selected.

```bash
lm_scaling/make_megatron_stack.sh            # build the stack from the pins
lm_scaling/make_megatron_stack.sh --verify   # names any pin no remote carries
```

Nothing in this repository depends on which branch those commits sit on: the
lockfile pins commit hashes, so `--verify` is the authority on whether a clone
can reach them.

## FLA attribution

This repository includes a snapshot of [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) so the experiments and modified CKDA layer can be reproduced from one checkout. The upstream project provides the efficient kernels, model infrastructure, tests, and multi-backend support on which this artifact is based.

## License

The code is released under the [MIT License](LICENSE). Existing FLA source files retain their original copyright and attribution notices.
