# `titan_ext` — fla layers under torchtitan

Registers a torchtitan train spec, **`fla_custom`**, that is titan-oellm's
`qwen3_custom` with every block's attention replaced by an `fla` linear-attention
layer (`kda`, `complex-kda`, `pda`).

## Why a swap, not a fork

Everything torchtitan provides — FSDP2 sharding, activation checkpointing,
`torch.compile`, the loss/metrics pipeline — lives in
`parallelize_qwen3_custom` and the surrounding train spec, and none of it needs
to know what the mixer is. The whole integration is therefore one attribute
assignment per block (`block.attention = FLAMixer(...)`).

Copying titan-oellm's 650-line model file to change one attribute would mean
maintaining a divergent copy forever, and — more importantly — would weaken the
thing the backend comparison rests on: **`lm/` and torchtitan must run the same
layer code**, so that a difference in the training curve means a bug and not a
re-implementation. Here the mixer maths, the Triton kernels and the gate
initialization all stay in `fla`.

## What the shim does

torchtitan calls the mixer as

```python
x = x + self.attention(self.attention_norm(x), rope_cache, masks, positions)
```

and initializes it via `init_weights(init_std)`. `fla` layers follow the
HuggingFace convention instead. `FLAMixer` bridges the two, and does nothing
else.

Two deliberate choices in the shim:

- **RoPE and attention masks are dropped.** These mixers are NoPE by design —
  position enters through the recurrence — and torchtitan's mask object is not
  `fla`'s `[B, T]` padding mask, so forwarding it would trip `fla`'s dimension
  assertion while carrying no information (titan feeds packed, equal-length
  sequences and the recurrence is causal by construction).
- **`init_weights` leaves the gate parameters alone** (`A_log`, `dt_bias`,
  `f_proj`, plus conv and norms). Their initialization is precisely what the
  study varies: `gate_init_style="spread"` is a specific distribution over the
  gate's sign, and letting torchtitan's truncated normal overwrite it would
  silently turn every titan run into the "shipped" arm. Only the plain
  projections get torchtitan's per-layer std.

`swap_in_fla_mixers` raises if it swaps zero blocks, because the alternative —
training a plain transformer under a run labelled `complex-kda` — is the one
failure here that would not announce itself.

## A checkpoint before the wall

A job that outlives its allocation resumes from its newest checkpoint, so the
interval alone loses up to one interval of training at every wall. `wall.py`
saves once more when the job is within a margin of `SLURM_JOB_END_TIME`
(180 s, or `TITAN_EXT_PRE_WALL_MARGIN_S`), through torchtitan's own save path,
and then lets training run on until Slurm stops it: the job still ends as a
TIMEOUT and the monitor restarts it as before. The ranks agree on the step
through one all-reduce every 20 steps. Off with `TITAN_EXT_PRE_WALL=0`, and on
its own wherever there is no Slurm end time or checkpointing is disabled.

## Requirements

`torchtitan` and `titan_oellm` must be importable; both ship in the titan
container (`titan_0.2.1_aarch64_*.sif`, `TorchTitanTraining_aarch64.sif`), not
in this repo. `titan-oellm` is at `github.com/OpenEuroLLM/titan-oellm`.

## Usage

```python
import lm_scaling.titan_ext as ext
ext.register()            # registers the "fla_custom" train spec
```

then in the job config set `model.name = "fla_custom"` and
`model.mixer = "complex-kda"`.

## Status

The shim is unit-tested without torchtitan (`tests/lm/test_titan_ext.py`):
construction of all three mixers, the torchtitan-shaped forward, gate-parameter
preservation across `init_weights`, and the zero-swap guard. `register()` itself
needs the container and is exercised by the backend comparison job, not by the
unit tests.
