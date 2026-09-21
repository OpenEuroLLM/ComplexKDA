# Optional CKDA kernel implementations

These backends support both unsigned KDA and CKDA (previously Signed-KDA). They do not change model parameters, gate ranges, imports, or checkpoint fields. The default `auto` route retains the existing backend-selection policy; optimized normalization and tiling require explicit selection.

Set `FLA_KDA_BACKEND` before running a process:

- `auto` (default): existing backend selection.
- `triton`: baseline Triton recurrence and normalization.
- `tilelang`: baseline normalization with the existing TileLang WY backward kernel.
- `triton_optimized`: paired normalization, fused normalization/sign backward epilogue, tuned Triton backward kernels, and parallel integer parity for the validated long-sequence shapes.
- `hybrid_optimized`: the same Triton normalization/epilogue/parity components with the existing TileLang WY backward kernel. This is not a fully TileLang operator.

The optimized routes are opt-in and restricted to NVIDIA Hopper, matching FP32/BF16 query/key/value dtypes, equal query/value head counts, and key dimension at most 128. Explicit routes currently reject context parallelism. Unsupported explicit selections raise an error instead of silently falling back. TileLang requires an installed package and usable CUDA compiler. Use `auto` for other platforms or context parallelism.

Explicit selection requires backend dispatch: leave `FLA_DISABLE_BACKEND_DISPATCH` unset or set it to `0` before importing FLA. The selected backward implementation is retained by the autograd context. The old experimental normalization-fusion flags are superseded by these explicit routes.

For historical Triton-only runs, pin `FLA_KDA_BACKEND=triton` instead of relying on optional-package discovery. The dispatch wrapper fix makes enable flags effective on PyTorch 2.10; `auto` can therefore select an installed optional backend that the previous wrapper inadvertently bypassed.

```bash
FLA_KDA_BACKEND=auto FLA_DISABLE_BACKEND_DISPATCH=0 FLA_FLASH_KDA=0 FLA_TILELANG=0 \
  pytest tests/ops/test_kda_implementations.py -q

FLA_KDA_BACKEND=auto FLA_DISABLE_BACKEND_DISPATCH=0 FLA_FLASH_KDA=0 FLA_TILELANG=0 \
  pytest tests/ops/test_kda.py tests/ops/test_complex_kda.py -q
```

The baseline recurrence kernels and original single-input signed normalization remain available. The shared GLA forward launcher separately excludes a known misaligned-address configuration on Hopper at chunk size 32; this safety fix does not lower numerical precision. Exact historical launch configurations remain in the preceding release revision.

The optimized Triton WY backward uses FP32 multiplication and accumulation for the state-product contribution to the gate gradient. In Triton 3.6, an untyped sum of BF16 products stays in BF16, making rounding depend on the tile and warp configuration. The baseline route retains its historical calculation. Gate-gradient tests use an independent FP64 recurrence: the broad BF16 sweep retains the existing 0.015 relative RMS allowance, while FP32 cases and targeted BF16 reduction regressions use 0.005. Other outputs and gradients are compared against the baseline at 0.005. The original KDA test files and their tolerances are unchanged. This precision correction does not establish that earlier throughput measurements apply to the new implementation.

Tests compare outputs, final states, and gradients, and exercise dense/ragged inputs, state layouts, explicit routing, and rejection. Passing tests must be established on the target GPU before using new backend measurements in a paper; adding an implementation does not establish a new throughput result. Preserve backend choice, revision, dependency versions, hardware, and precision environment in experiment manifests.

Validation on September 18, 2026 used one NVIDIA H100 with PyTorch `2.10.0a0+a36e1d39eb.nv26.1.42222806`, Triton `3.6.0`, and TileLang `0.1.14`. The implementation suite passed all 54 tests. The unchanged original KDA/CKDA suites passed 251 tests and skipped 110: 12 unavailable FlashKDA cases, 2 NPU-only cases, and 96 unsupported signed combinations (42 grouped-value attention, 18 private recurrent-entrypoint, and 36 fused-gate cases). The runs used the standard NaN-poisoning test configuration; these counts do not imply coverage of the skipped features.
