# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from pathlib import Path

MUON_LRS = (0.010, 0.014, 0.020, 0.028, 0.040)
LOWER_MUON_LRS = (0.0025, 0.0035, 0.0050, 0.0070, 0.0090)
STEPS = 10_173
REMOTE_RUN_ROOT = Path("/gpfs/projects/ehpc390/kda_lr_sweep_1b")


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> None:
    script_root = Path(__file__).resolve().parent
    base_path = script_root.parent / "kda_qwen3_15b_ablation" / "generated" / "a01_b01_standard.toml"
    base = base_path.read_text()
    output_root = script_root / "generated"
    output_root.mkdir(parents=True, exist_ok=True)

    for learning_rates, precision in ((MUON_LRS, 3), (LOWER_MUON_LRS, 4)):
        for learning_rate in learning_rates:
            formatted_rate = f"{learning_rate:.{precision}f}"
            name = f"muon_lr_{formatted_rate}".replace(".", "p")
            config = replace_once(base, 'description = "KDA gate/rate ablation: a01_b01_standard"',
                                  f'description = "KDA baseline 1B Muon LR search: {formatted_rate}"')
            config = replace_once(
                config,
                'dump_folder = "/gpfs/scratch/ehpc390/flame_runs/kda_qwen3_15b/a01_b01_standard"',
                f'dump_folder = "{REMOTE_RUN_ROOT / name}"',
            )
            config = replace_once(config, "steps = 152588", f"steps = {STEPS}")
            config = replace_once(config, "lr = 2e-2", f"lr = {formatted_rate}")
            config = replace_once(config, "decay_ratio = 0.2", "decay_ratio = 0.0")
            config = replace_once(config, "enable_checkpoint = true", "enable_checkpoint = false")
            (output_root / f"{name}.toml").write_text(config)


if __name__ == "__main__":
    main()
