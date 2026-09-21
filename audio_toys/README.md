# Periodic waveform continuation

This directory contains the audio experiment reported in the paper. It is deliberately a controlled memorization-and-extrapolation diagnostic: every model is trained on random circular shifts of one fixed synthetic two-bar groove. It tests whether a single sequence layer can infer phase from a short waveform cue and preserve that phase after its inputs become zero; it is not presented as a general audio-generation benchmark.

## Task

The source is a 32-step groove at 124 BPM. Each sixteenth note is represented by 64 mono waveform samples, giving a fixed target of shape `32 × 64`. A training example selects one of the 32 circular phases, presents the first eight frames (half a bar, approximately 0.97 seconds), and replaces every later input frame with zeros. The input and regression target both have shape `batch × sequence_length × 64`, and mean-squared error is optimized only after the observed cue.

The paper configuration uses one sequence-mixing layer, hidden size 128, eight heads of dimension 16, a shared `128 → 512 → 64` GELU decoder, and no short convolution. Training uses one seed, 2,500 updates, batch size 64, a scheduled sequence-length curriculum of `12, 16, 24, 40, 72, 136`, Muon for internal two-dimensional sequence-layer weights, and AdamW for input/output adapters and remaining parameters. Evaluation enumerates all 32 phases at lengths 40, 136, and 264; length 264 is almost twice the maximum training length.

## Installation

Create a Python 3.10 or newer environment and install this repository with the backend appropriate for your machine. The plotting scripts additionally use SciencePlots.

```bash
# NVIDIA example
pip install -e ".[cuda,benchmark]"
pip install -r requirements-experiments.txt

# CPU or Apple Silicon example
pip install -e ".[cpu,benchmark]"
pip install -r requirements-experiments.txt
```

The experiment has a pure-PyTorch recurrent fallback and can run with `--device cpu` or `--device mps`; CUDA uses the optimized FLA path when available. The full control suite is slow on CPU and MPS, especially DeltaProduct-2's reference recurrence.

## Reproduce

Run a small end-to-end check first. It trains every main-paper architecture for 20 updates and verifies result serialization, checkpoint reload, audio rendering, dense evaluation, transition analysis, and both paper figures.

```bash
python audio_toys/reproduce_periodic_waveform.py --mode smoke --device cpu
```

Run the exact main-paper configuration with:

```bash
python audio_toys/reproduce_periodic_waveform.py --mode paper --device cuda
```

Use `--device mps` on Apple Silicon or omit `--device` for automatic selection. Add `--include-controls` to train the RoPE, fixed-base RoPE, Selective RoPE, no-position Transformer, and the gated/ungated DeltaProduct-2 initialization controls. All numerical settings are passed explicitly by the reproduction driver; in particular, it uses the native/spread KDA initialization reported in the paper rather than the exploratory endpoint initialization available in the lower-level training script.

Outputs are written to `artifacts/periodic_waveform/`:

- `results/models.jsonl`: configuration, parameter count, timing, and length-40/136/264 metrics for each model.
- `results/aggregates.json`: every per-seed row plus unrounded mean, sample standard deviation, minimum, and maximum.
- `results/dense_metrics.json`: waveform MSE at the sequence lengths used in the numerical plot.
- `results/complex_kda_spectrum.json`: eigenvalues and the complex-mode intervention for CKDA.
- `results/deltaproduct2_spectrum.json`: the zero-input two-reflection spectrum when `--include-controls` is used.
- `checkpoints/`: one state dictionary per trained model.
- `audio/`: mono predictions and stereo target-left/model-right comparisons.
- `arrays/`: compressed normalized inputs, targets, predictions, and phase indices for every circular phase.
- `environment.json` and `run_manifest.json`: source, software/hardware environment, device request, and selected run suite.
- `figures/groove_waveform_main.pdf`: the composite main-paper figure.
- `figures/groove_waveform_appendix.pdf`: the remaining KDA range ablations.

The checked reference numbers are in [`reference_periodic_waveform_metrics.json`](reference_periodic_waveform_metrics.json). At length 264, seed-0 CKDA reaches 38.06 dB SNR, the causal Transformer reaches 2.82 dB, and the GRU reaches 73.06 dB. These are single-seed, single-sequence results and should be interpreted as evidence about phase-preserving dynamics, not comparative general-purpose audio quality.

## Entry points

- [`train_groove_waveform.py`](train_groove_waveform.py) defines the task, models, curriculum, optimization, evaluation, checkpoints, and WAV export.
- [`plot_groove_waveform_paper_figures.py`](plot_groove_waveform_paper_figures.py) generates the main and appendix figures.
- [`analyze_groove_state_spectrum.py`](analyze_groove_state_spectrum.py) measures the learned KDA transition spectrum and removes complex modes from the post-cue state.
- [`plot_groove_waveform_transformer_positions.py`](plot_groove_waveform_transformer_positions.py) evaluates Transformer positional controls.
- [`plot_groove_deltaproduct2.py`](plot_groove_deltaproduct2.py) compares the DeltaProduct-2 gate and initialization controls.
- [`analyze_deltaproduct2_transition.py`](analyze_deltaproduct2_transition.py) measures the learned two-reflection transition spectrum and checks causality.

The waveform trainer reuses the deterministic groove synthesizer and control utilities in [`make_techno_loop.py`](make_techno_loop.py) and [`train_groove.py`](train_groove.py). These are support modules for the released experiment; the reproduction driver above remains the stable public entry point.

## Audio samples and generated artifacts

Generated checkpoints, bulk audio, plots, and ordinary result directories are intentionally excluded from version control. The paper's compact raw metrics, all-phase arrays, spectrum report, and checkpoint hashes are archived under [`evidence/results/periodic_waveform/`](../evidence/results/periodic_waveform/). The listening samples accompanying the submission are available from the [anonymous audio folder](https://www.dropbox.com/scl/fo/otpiegwjo6t2w9yonm0uc/ABtNxwvazCXP35_dH6FV8wM?rlkey=zf4igaux3crkcbgm6qztfbfs8&st=v1yp3tcw&dl=0).

## Lower-level commands

The driver is the authoritative reproduction interface. For development, individual architectures can be trained with `train_groove_waveform.py --variant MODEL`; run `python audio_toys/train_groove_waveform.py --help` for the complete set of KDA, GRU, Transformer, and DeltaProduct-2 controls. Do not compare ad-hoc runs without matching the hidden size, head configuration, optimizer partition, schedule, curriculum, seed, and evaluation phases recorded in `models.jsonl`.
