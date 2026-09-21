# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Export every FineWeb-Edu ablation model configuration with an exact parameter count."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
from prepare_configs import VARIANTS, model_config  # noqa: E402

from fla.models.complex_kda import ComplexKDAConfig, ComplexKDAForCausalLM  # noqa: E402


def count_parameters(configuration: dict) -> int:
    config = ComplexKDAConfig(**configuration)
    with torch.device("meta"):
        model = ComplexKDAForCausalLM(config)
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = {}
    for name, gate, gate_init_style, allow_neg_eigval, beta_init_style in VARIANTS:
        configuration = model_config(gate, gate_init_style, allow_neg_eigval, beta_init_style)
        variants[name] = {
            "model_config": configuration,
            "exact_parameter_count": count_parameters(configuration),
        }
    manifest = {
        "scope": "released FineWeb-Edu 15B recipe; this is not the earlier Nemotron-CC experiment",
        "training": {
            "steps": 152588,
            "sequence_length": 4096,
            "per_device_batch": 6,
            "data_parallel_replicas": 4,
            "global_tokens_per_step": 98304,
            "seed": 42,
            "optimizer": "MuonWithAuxAdam",
            "selected_muon_learning_rate": 0.005,
            "auxiliary_adamw_learning_rate": 0.0003,
            "momentum": 0.95,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "warmup_steps": 2000,
            "decay_ratio": 0.2,
            "minimum_learning_rate_ratio": 0.1,
        },
        "data": {
            "repository": "HuggingFaceFW/fineweb-edu",
            "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
            "tokenizer": "Qwen/Qwen3-0.6B",
            "tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "selected_shard_manifest_archived": False,
        },
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
