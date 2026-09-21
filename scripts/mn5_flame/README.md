# CKDA language-model training on MareNostrum 5

This directory contains the reproducible setup for the 487M-parameter CKDA language-model experiments. The runs use FineWeb-Edu text, the Qwen3 tokenizer, sequence length 4096, four H100 GPUs per arm, and Muon with auxiliary AdamW. Dataset tokenization occurs online in Flame; the stored dataset remains Parquet text.

The paths and Slurm account in these scripts match project `ehpc390` on MareNostrum 5. Change the constants in the configuration generators and the variables near the top of each submission script when using another account or filesystem layout.

## Experiment configuration

Each full run processes 15B tokens in 152,588 optimizer steps:

- per-device batch: 6 sequences;
- data-parallel replicas: 4;
- gradient accumulation: 1;
- global batch: 24 sequences, or 98,304 tokens;
- warm-up: 2,000 steps, or approximately 0.197B tokens;
- cosine decay begins over the final 20% of training and reaches 10% of the peak learning rate;
- model snapshots: bfloat16 weights approximately every 1B tokens;
- recoverable TorchTitan checkpoints: every 10,173 steps, retaining the latest two.

Muon is applied to two-dimensional non-embedding and non-output-head parameters. Embeddings, the language-model head, vectors, and other remaining parameters use auxiliary AdamW. The selected peak learning rates are 0.005 for Muon and `3e-4` for auxiliary AdamW, with momentum 0.95, weight decay 0.01, and gradient clipping at norm 1.0.

The model has 28 layers, hidden size 768, 12 heads of dimension 64, and intermediate size 2816. Standard-beta variants have 487,209,808 parameters; beta-spread variants add a 12-element beta bias in each layer and have 487,210,144 parameters. Regenerate the exact per-arm inventory with `python scripts/mn5_flame/kda_qwen3_15b_ablation/export_manifest.py --output /tmp/config_manifest.json`.

## Data preparation

`../mn5_stage_fineweb_edu.py` downloads the pinned official `HuggingFaceFW/fineweb-edu` Parquet shards on a networked local machine and streams them through a Storage5 transfer node. Completed shards are size-checked, resumable partial files are retained, and `--verify-sha256` verifies each shard against its Hugging Face LFS digest.

```bash
python scripts/mn5_stage_fineweb_edu.py \
  --host frei178785@transfer1.bsc.es \
  --target /gpfs/scratch/ehpc390/fineweb-edu/sample-100BT-qwen3-budget30B \
  --limit 21 \
  --jobs 4 \
  --verify-sha256
```

The training configs expect the selected shards under `/gpfs/scratch/ehpc390/fineweb-edu/qwen3-15b-split/train/*.parquet`. This directory can contain symbolic links to a deterministic subset of the verified download. Estimate its Qwen3 token count from a compute node with:

```bash
python scripts/mn5_flame/estimate_fineweb_tokens.py \
  --data-files '/gpfs/scratch/ehpc390/fineweb-edu/qwen3-15b-split/train/*.parquet' \
  --tokenizer /gpfs/projects/ehpc390/.huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
```

The estimate is based on sampled documents because FineWeb-Edu's stored `token_count` uses a different tokenizer. The optimization budget, rather than exhaustion of every source document, fixes each run at exactly 15B Qwen3 tokens.

## Cluster layout

The submission scripts expect this isolated experiment tree:

```text
/gpfs/scratch/ehpc390/complex_kda_flame_ablation_15b/
├── configs/
├── flame/
├── flash-linear-attention/
├── launcher/launch_flame_muon.py
├── muon/
├── torchtitan/
├── training/
└── venv/
```

They execute inside `/gpfs/projects/ehpc390/work/Projects/titan-sci/titan_x86_64_0.2.1_flash.sif`. The environment in that image must provide a compatible CUDA PyTorch stack, while the experiment tree provides Flame, TorchTitan, Muon, and this FLA checkout through `PYTHONPATH`.

Generate all model and training configs locally:

```bash
python scripts/mn5_flame/kda_qwen3_15b_ablation/prepare_configs.py
python scripts/mn5_flame/kda_qwen3_lr_sweep_1b/prepare_configs.py
```

Copy the generated `configs/*` directories to the remote `configs/` directory and the generated TOML files to `training/`. Place the LR-sweep TOML files under `training/lr_sweep_1b/`. Copy `complex_kda_qwen3_smoke/launch_flame_muon.py` to `launcher/launch_flame_muon.py`, and copy the desired submission scripts to the experiment root.

## Validation and submission

Always run a compiled two-step preflight after changing the layer, model configuration, environment, or container. For example:

```bash
cd /gpfs/scratch/ehpc390/complex_kda_flame_ablation_15b
sbatch --export=ALL,PREFLIGHT_PHASES=preflight_drop_silu_lr005,\
PREFLIGHT_RUN_ROOT=/gpfs/scratch/ehpc390/flame_runs/kda_qwen3_15b/preflight_drop_silu_lr005 \
  submit_preflight.sh
```

Verify that the job reaches step 2, writes its checkpoint, exits with code zero, reports 98,304 global tokens per step, and uses the expected optimizer learning rates before submitting a full array.

The final reduced no-SiLU experiment contains the four standard-initialization range combinations:

```bash
sbatch submit_drop_silu_lr005_array.sh
```

The key-only experiment retains SiLU on queries and values and removes it only from keys. It compares standard KDA against CKDA with both ranges extended; both use standard initialization:

```bash
sbatch submit_key_no_silu_lr005_array.sh
```

`submit_array.sh` reproduces the original seven-arm SiLU pilot at Muon LR 0.02. The two additional fully extended single-spread pilot arms are in `submit_init_isolation_array.sh`. These pilots predate the LR search and should not be combined with the LR-0.005 runs without labeling the optimizer difference.

## Learning-rate sweep

The two one-billion-token arrays cover Muon peak learning rates 0.0025 through 0.04 while holding auxiliary AdamW at `3e-4`:

```bash
sbatch kda_qwen3_lr_sweep_1b/submit_array.sh
sbatch kda_qwen3_lr_sweep_1b/submit_lower_array.sh
```

At a matched one-billion-token horizon, the observed optimum was broad over 0.0035--0.007, with 0.005 giving the best point estimate. This is why the reduced no-SiLU and key-only experiments use 0.005.

## Monitoring and plotting

Slurm writes one `.err` metric log per arm. Copy those logs locally and generate the figures with:

```bash
python scripts/mn5_flame/kda_qwen3_lr_sweep_1b/plot_live_loss.py \
  --log-dir /path/to/lr-logs \
  --lower-job-id LOWER_ARRAY_JOB_ID \
  --job-id UPPER_ARRAY_JOB_ID \
  --output figures/kda_baseline_lr_sweep_live.png

python scripts/mn5_flame/kda_qwen3_15b_ablation/plot_live_loss.py \
  --log-dir /path/to/no-silu-logs \
  --combined-job-id ARRAY_JOB_ID \
  --condition-label 'no SiLU, Muon LR 0.005' \
  --expected-arms 4 \
  --loss-top 13 \
  --output figures/kda_4arm_no_silu_lr005_live_loss.png
```

The left panel shows token-matched training cross-entropy. The right panel subtracts the standard KDA loss at the same optimizer step, so negative values favor the compared arm. These are training-loss comparisons; use held-out validation and downstream evaluation before making generalization claims.

For allocation, storage, queue, and GPU-utilization status, run `scripts/mn5_dashboard` from the local machine.
