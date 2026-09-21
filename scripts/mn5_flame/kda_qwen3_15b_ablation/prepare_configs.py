# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import json
from pathlib import Path

REMOTE_PROJECT = Path("/gpfs/scratch/ehpc390/complex_kda_flame_ablation_15b")
REMOTE_RUNS = Path("/gpfs/scratch/ehpc390/flame_runs/kda_qwen3_15b")
REMOTE_DROP_SILU_RUNS = Path("/gpfs/projects/ehpc390/complex_kda_flame_runs/kda_qwen3_15b_drop_silu")
REMOTE_DROP_SILU_LR005_RUNS = Path("/gpfs/projects/ehpc390/complex_kda_flame_runs/kda_qwen3_15b_drop_silu_lr005")
REMOTE_KEY_NO_SILU_LR005_RUNS = Path("/gpfs/projects/ehpc390/complex_kda_flame_runs/kda_qwen3_15b_key_no_silu_lr005")
TOKENIZER = Path(
    "/gpfs/projects/ehpc390/.huggingface/hub/"
    "models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
DATA_FILES = "/gpfs/scratch/ehpc390/fineweb-edu/qwen3-15b-split/train/*.parquet"

BASE_VARIANTS = (
    ("a01_b01_standard", "sigmoid", "shipped", False, "standard"),
    ("a11_b01_standard", "signed_sigmoid2", "shipped", False, "standard"),
    ("a01_b02_standard", "sigmoid", "shipped", True, "standard"),
    ("a11_b02_standard", "signed_sigmoid2", "shipped", True, "standard"),
    ("a11_b01_gate_spread", "signed_sigmoid2", "spread", False, "standard"),
    ("a01_b02_beta_spread", "sigmoid", "shipped", True, "spread"),
    ("a11_b02_both_spread", "signed_sigmoid2", "spread", True, "spread"),
)

EXTRA_VARIANTS = (
    ("a11_b02_gate_spread", "signed_sigmoid2", "spread", True, "standard"),
    ("a11_b02_beta_spread", "signed_sigmoid2", "shipped", True, "spread"),
)

VARIANTS = BASE_VARIANTS + EXTRA_VARIANTS

KEY_NO_SILU_VARIANTS = (
    ("a01_b01_standard", "sigmoid", "shipped", False, "standard"),
    ("a11_b02_standard", "signed_sigmoid2", "shipped", True, "standard"),
)


def model_config(
    gate: str,
    gate_init_style: str,
    allow_neg_eigval: bool,
    beta_init_style: str,
    drop_silu: bool = False,
    drop_key_silu: bool = False,
) -> dict:
    return {
        "architectures": ["ComplexKDAForCausalLM"],
        "attn_mode": "chunk",
        "allow_neg_eigval": allow_neg_eigval,
        "beta_init_style": beta_init_style,
        "bos_token_id": 151643,
        "conv_size": 4,
        "drop_silu": drop_silu,
        "drop_key_silu": drop_key_silu,
        "eos_token_id": 151645,
        "expand_v": 1,
        "fuse_cross_entropy": True,
        "fuse_norm": True,
        "fuse_swiglu": True,
        "gate": gate,
        "gate_init_style": gate_init_style,
        "head_dim": 64,
        "hidden_act": "swish",
        "hidden_ratio": None,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 2816,
        "lower_bound": -5.0,
        "max_position_embeddings": 4096,
        "model_type": "complex_kda",
        "norm_eps": 1e-6,
        "num_heads": 12,
        "num_hidden_layers": 28,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_short_conv": True,
        "vocab_size": 151936,
    }


def training_config(
    name: str,
    steps: int,
    warmup_steps: int,
    checkpoint_interval: int,
    log_freq: int = 20,
    compile_model: bool = True,
    run_root: Path = REMOTE_RUNS,
    optimizer_lr: float = 2e-2,
) -> str:
    return f'''[model]
name = "fla"
config = "{REMOTE_PROJECT / 'configs' / name}"
tokenizer_path = "{TOKENIZER}"

[job]
description = "KDA gate/rate ablation: {name}"
dump_folder = "{run_root / name}"
print_args = true

[training]
batch_size = 6
seq_len = 4096
context_len = 4096
gradient_accumulation_steps = 1
steps = {steps}
max_norm = 1.0
skip_nan_inf = true
data_parallel_replicate_degree = 4
data_parallel_shard_degree = 1
tensor_parallel_degree = 1
compile = {str(compile_model).lower()}
dataset = "parquet"
dataset_split = "train"
data_files = "{DATA_FILES}"
streaming = true
num_workers = 1
pin_memory = false
persistent_workers = false
prefetch_factor = 2
seed = 42
varlen = false

[optimizer]
name = "MuonWithAuxAdam"
eps = 1e-10
lr = {optimizer_lr}
adamw_lr = 3e-4
momentum = 0.95
weight_decay = 0.01

[lr_scheduler]
warmup_steps = {warmup_steps}
decay_ratio = 0.2
decay_type = "cosine"
lr_min = 0.1

[checkpoint]
enable_checkpoint = true
folder = "checkpoint"
interval_type = "steps"
interval = {checkpoint_interval}
keep_latest_k = 2
model_weights_only = false
export_dtype = "float32"
async_mode = "disabled"

[profiling]
enable_profiling = false
save_traces_folder = "profile_trace"
profile_freq = 512

[metrics]
log_freq = {log_freq}
enable_wandb = false

[experimental]
context_parallel_degree = 1
pipeline_parallel_degree = 1

[float8]
enable_fsdp_float8_all_gather = false
precompute_float8_dynamic_scale_for_fsdp = false

[activation_checkpoint]
mode = "none"
'''


def main() -> None:
    output_root = Path(__file__).parent / "generated"
    output_root.mkdir(parents=True, exist_ok=True)
    for name, gate, gate_init_style, allow_neg_eigval, beta_init_style in VARIANTS:
        config_root = output_root / "configs" / name
        config_root.mkdir(parents=True, exist_ok=True)
        (config_root / "config.json").write_text(
            json.dumps(model_config(gate, gate_init_style, allow_neg_eigval, beta_init_style), indent=2) + "\n"
        )
        (output_root / f"{name}.toml").write_text(training_config(name, 152588, 2000, 10173))

    for name, gate, gate_init_style, allow_neg_eigval, beta_init_style in KEY_NO_SILU_VARIANTS:
        key_no_silu_name = f"key_no_silu_lr005_{name}"
        config_root = output_root / "configs" / key_no_silu_name
        config_root.mkdir(parents=True, exist_ok=True)
        (config_root / "config.json").write_text(
            json.dumps(
                model_config(
                    gate,
                    gate_init_style,
                    allow_neg_eigval,
                    beta_init_style,
                    drop_key_silu=True,
                ),
                indent=2,
            ) + "\n"
        )
        (output_root / f"{key_no_silu_name}.toml").write_text(
            training_config(
                key_no_silu_name,
                152588,
                2000,
                10173,
                run_root=REMOTE_KEY_NO_SILU_LR005_RUNS,
                optimizer_lr=5e-3,
            )
        )

    for name, gate, gate_init_style, allow_neg_eigval, beta_init_style in BASE_VARIANTS:
        drop_silu_name = f"drop_silu_{name}"
        config_root = output_root / "configs" / drop_silu_name
        config_root.mkdir(parents=True, exist_ok=True)
        (config_root / "config.json").write_text(
            json.dumps(
                model_config(gate, gate_init_style, allow_neg_eigval, beta_init_style, drop_silu=True),
                indent=2,
            ) + "\n"
        )
        (output_root / f"{drop_silu_name}.toml").write_text(
            training_config(drop_silu_name, 152588, 2000, 10173, run_root=REMOTE_DROP_SILU_RUNS)
        )

    for name, gate, gate_init_style, allow_neg_eigval, beta_init_style in BASE_VARIANTS[:4]:
        drop_silu_name = f"drop_silu_lr005_{name}"
        config_root = output_root / "configs" / drop_silu_name
        config_root.mkdir(parents=True, exist_ok=True)
        (config_root / "config.json").write_text(
            json.dumps(
                model_config(gate, gate_init_style, allow_neg_eigval, beta_init_style, drop_silu=True),
                indent=2,
            ) + "\n"
        )
        (output_root / f"{drop_silu_name}.toml").write_text(
            training_config(
                drop_silu_name,
                152588,
                2000,
                10173,
                run_root=REMOTE_DROP_SILU_LR005_RUNS,
                optimizer_lr=5e-3,
            )
        )

    preflight = "a11_b02_both_spread"
    (output_root / "preflight_phase1.toml").write_text(
        training_config(preflight, 10, 2, 10, log_freq=1, compile_model=False)
    )
    (output_root / "preflight_phase2.toml").write_text(
        training_config(preflight, 100, 2, 100, log_freq=1, compile_model=False)
    )
    (output_root / "preflight_compiled.toml").write_text(
        training_config(preflight, 100, 2, 100, log_freq=1)
    )
    (output_root / "preflight_compiled_verify.toml").write_text(
        training_config(preflight, 2, 1, 2, log_freq=1)
    )
    (output_root / "preflight_drop_silu.toml").write_text(
        training_config(
            "drop_silu_a11_b02_both_spread",
            2,
            1,
            2,
            log_freq=1,
            run_root=REMOTE_DROP_SILU_RUNS,
        )
    )
    (output_root / "preflight_drop_silu_lr005.toml").write_text(
        training_config(
            "drop_silu_lr005_a11_b02_standard",
            2,
            1,
            2,
            log_freq=1,
            run_root=REMOTE_DROP_SILU_LR005_RUNS,
            optimizer_lr=5e-3,
        )
    )
    (output_root / "preflight_key_no_silu_lr005.toml").write_text(
        training_config(
            "key_no_silu_lr005_a11_b02_standard",
            2,
            1,
            2,
            log_freq=1,
            run_root=REMOTE_KEY_NO_SILU_LR005_RUNS,
            optimizer_lr=5e-3,
        )
    )


if __name__ == "__main__":
    main()
