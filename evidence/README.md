# Evidence index

This directory distinguishes archived measurements from scripts that make future runs auditable. A reproduction script is not treated as evidence that an unavailable run occurred.

## Archived evidence

- [`results/periodic_waveform/models.jsonl`](results/periodic_waveform/models.jsonl) contains the original full-precision seed-0 result rows for the four KDA settings, GRU, causal Transformer, and the later-discarded non-causal control. Each row includes the effective architecture, trainable parameter count, optimizer and scheduler settings, curriculum transitions, evaluation metrics, runtime, and original checkpoint path.
- [`results/periodic_waveform/dense_metrics.json`](results/periodic_waveform/dense_metrics.json) contains the full-precision 18-length evaluation used for the numerical curve.
- [`results/periodic_waveform/complex_kda_spectrum.json`](results/periodic_waveform/complex_kda_spectrum.json) contains per-head gates, rates, eigenvalues, spectral radii, and the complex-mode state intervention.
- [`results/periodic_waveform/arrays/`](results/periodic_waveform/arrays/) contains targets and predictions for every circular phase and every main model through length 264. Its manifest records SHA-256 hashes for both arrays and the source checkpoints; checkpoints are not committed. [`verify_archive.py`](verify_archive.py) checks each committed hash and recomputes the three reported waveform MSE values from the arrays.
- [`results/sign_gauge_cpu.json`](results/sign_gauge_cpu.json) records output, final-state, and backward comparisons between independent direct-signed and sign-gauged float64 recurrences. CUDA kernel coverage remains in [`tests/ops/test_complex_kda.py`](../tests/ops/test_complex_kda.py); no archived CUDA test report is claimed.
- [`results/fineweb_edu_15b/config_manifest.json`](results/fineweb_edu_15b/config_manifest.json) records every released language-model configuration and its exact parameter count. This manifest belongs to the FineWeb-Edu recipe.
- [`lm_scaling/harvest/`](../lm_scaling/harvest/) holds the language-model results themselves: the 180 annealed endpoints of the Megatron-LM scaling ladder with their held-out validation loss, the same 180 cells scored on the nine-task downstream suite, the four 1.3B FineWeb-Edu arms at step 190,976, ten RULER needle-retrieval runs, and the transition spectrum of two 1.3B runs. Every table and figure in `lm_scaling/tex/` is derived from these by a script in this repository; `lm_scaling/reproduce_tables.sh --check` regenerates them all and diffs, and `tests/lm_scaling/test_harvest_reproduces.py` checks the cheap half of that in the normal test run. These are results and not runs: checkpoints, per-step logs and run directories are not committed.

Run the unrounded aggregation script with:

```bash
python evidence/aggregate_results.py --kind audio --inputs evidence/results/periodic_waveform/models.jsonl --output /tmp/audio_aggregate.json
python evidence/aggregate_results.py --kind state --inputs artifacts/state_tracking/results/*.jsonl --output /tmp/state_aggregate.json
python evidence/verify_archive.py
```

## Reproduction support added to the pipelines

New audio runs record full-precision per-seed training traces, total and trainable parameter counts, all optimizer fields, and compressed all-phase prediction arrays. New state-tracking runs record full-precision curves and traces, exact counts, optimizer fields, and the paper's explicit best-seed selection alongside mean, standard deviation, minimum, and maximum when aggregated. The initialization and state-interpretability plotters now emit their underlying arrays. [`benchmarks/benchmark_complex_kda.py`](../benchmarks/benchmark_complex_kda.py) records raw synchronized samples for absolute training and prefill timings, together with the software/hardware environment and the exact timed operation.

## Scope and remaining gaps

The archived audio result is a single-seed diagnostic on circular shifts of one fixed synthetic groove. It supports the reported values for that experiment but not a general audio-modeling claim.

The repository includes compact checked best-of-three state-tracking values, but the historical per-seed files were rounded to four decimal places and are not presented here as unrounded raw evidence. Re-running the released driver produces unrounded per-seed logs. Likewise, no raw H200/H100 throughput report is archived here, so percentages from an external manuscript should not be attributed to this package until a report from the released benchmark is added. The released FineWeb-Edu scripts pin the dataset revision and tokenizer snapshot, but the exact selected-shard manifest and the historical per-step language-model training logs are not present. The language-model evaluation outputs and the Nemotron-CC ladder results *are* present, under `lm_scaling/harvest/` — what is missing there is the runs behind them, not the numbers. Per-step validation curves exist only for the superseded torchtitan campaign and are not archived; the Megatron cells validate once, at their annealed endpoint, so a cell contributes a point and not a curve. [`manifests/claim_status.json`](manifests/claim_status.json) records these boundaries in machine-readable form.

Large future artifacts should be attached to a versioned release and listed with SHA-256 hashes rather than silently omitted or committed without provenance.
