# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Reproduce the finite-group state-tracking experiments and figure."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("s3", "s4", "a5")
VARIANT_ARGUMENTS = {
    "a01_b01": ("--gates", "sigmoid", "--no-neg-eigval", "--gate-init-style", "shipped"),
    "a01_b02": ("--gates", "sigmoid", "--gate-init-style", "shipped"),
    "a11_b01": ("--gates", "signed_sigmoid2", "--no-neg-eigval", "--gate-init-style", "spread"),
    "a11_b02": ("--gates", "signed_sigmoid2", "--gate-init-style", "spread"),
}


def parse_selection(value: str, choices: tuple[str, ...], name: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(selected) - set(choices))
    if not selected or invalid:
        raise argparse.ArgumentTypeError(f"{name} must be a comma-separated subset of {choices}; got {invalid or value!r}")
    return selected


def run(command: list[str], environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def common_command(
    python: str,
    task: str,
    result: Path,
    checkpoint_dir: Path,
    device: str,
    mode: str,
) -> list[str]:
    smoke = mode == "smoke"
    return [
        python,
        "group_word_problems/train_wordproblem.py",
        "--task",
        task,
        "--device",
        device,
        "--backend",
        "auto",
        "--seeds",
        "1" if smoke else "3",
        "--seed-start",
        "0",
        "--steps",
        "20" if smoke else "60000",
        "--batch",
        "8" if smoke else "1024",
        "--eval-batch",
        "64" if smoke else "2048",
        "--train-len",
        "8",
        "--len-dist",
        "curriculum",
        "--curriculum-lens",
        "4,6,8,16,32",
        "--eval-continuous",
        "--eval-len",
        "32" if smoke else "512",
        "--d-model",
        "48",
        "--heads",
        "12",
        "--head-dim",
        "16",
        "--drop-silu",
        "--beta-init-style",
        "standard",
        "--readout",
        "mlp",
        "--optimizer",
        "muon",
        "--lr",
        "0.005",
        "--muon-lr",
        "0.02",
        "--muon-scope",
        "hidden",
        "--weight-decay",
        "1e-12",
        "--lr-schedule",
        "onecycle",
        "--log-every",
        "10" if smoke else "250",
        "--ckpt-dir",
        str(checkpoint_dir),
        "--out",
        str(result),
    ]


def ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to append duplicate runs to {path}; choose a new --output-root")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--variants", default=",".join(VARIANT_ARGUMENTS))
    parser.add_argument("--skip-theory-a5", action="store_true")
    parser.add_argument("--include-deltaproduct2", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/state_tracking"))
    args = parser.parse_args()
    tasks = parse_selection(args.tasks, TASKS, "tasks")
    variants = parse_selection(args.variants, tuple(VARIANT_ARGUMENTS), "variants")
    output_root = args.output_root if args.output_root.is_absolute() else REPOSITORY_ROOT / args.output_root
    results_dir = output_root / "results"
    checkpoints_dir = output_root / "checkpoints"
    figures_dir = output_root / "figures"
    for directory in (results_dir, checkpoints_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT)
    environment.setdefault("TORCHDYNAMO_DISABLE", "1")
    environment.setdefault("MPLCONFIGDIR", str(output_root / ".matplotlib"))
    python = sys.executable
    run([
        python,
        "evidence/capture_environment.py",
        "--output",
        str(output_root / "environment.json"),
        "--requested-device",
        args.device,
    ], environment)
    manifest = {
        "mode": args.mode,
        "requested_device": args.device,
        "tasks": list(tasks),
        "variants": list(variants),
        "include_deltaproduct2": args.include_deltaproduct2,
        "include_theory_initialized_a5": "a5" in tasks and not args.skip_theory_a5,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for task in tasks:
        for variant in variants:
            result = results_dir / f"{task}_{variant}.jsonl"
            ensure_new(result)
            command = common_command(python, task, result, checkpoints_dir / f"{task}_{variant}", args.device, args.mode)
            command.extend(VARIANT_ARGUMENTS[variant])
            run(command, environment)
        if args.include_deltaproduct2:
            result = results_dir / f"{task}_deltaproduct2.jsonl"
            ensure_new(result)
            command = common_command(python, task, result, checkpoints_dir / f"{task}_deltaproduct2", args.device, args.mode)
            command.extend(("--layer", "deltaproduct", "--dp-num-householder", "2"))
            if task == "a5":
                command.extend(("--dp-no-conv-silu", "qv"))
            run(command, environment)

    if "a5" in tasks and not args.skip_theory_a5:
        result = results_dir / "a5_theory_init.jsonl"
        ensure_new(result)
        smoke = args.mode == "smoke"
        command = common_command(python, "a5", result, checkpoints_dir / "a5_theory_init", args.device, args.mode)
        replacements = {
            "--steps": "20" if smoke else "3000",
            "--batch": "8" if smoke else "256",
            "--d-model": "32",
            "--heads": "4",
            "--head-dim": "4",
            "--curriculum-lens": "4,6,7,8,10,12,16,32",
        }
        for flag, value in replacements.items():
            command[command.index(flag) + 1] = value
        command.extend((
            "--gates",
            "signed_sigmoid2",
            "--gate-init-style",
            "spread",
            "--beta-init-style",
            "spread",
            "--a5-theory-init",
            "--a5-theory-heads",
            "2",
        ))
        run(command, environment)

    result_files = sorted(results_dir.glob("*.jsonl"))
    run([
        python,
        "evidence/aggregate_results.py",
        "--kind",
        "state",
        "--inputs",
        *(str(path) for path in result_files),
        "--output",
        str(results_dir / "aggregates.json"),
    ], environment)
    run([
        python,
        "group_word_problems/plot_state_tracking.py",
        "--results-dir",
        str(results_dir),
        "--output",
        str(figures_dir / "state_tracking.pdf"),
    ], environment)
    print(f"\nReproduction complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
