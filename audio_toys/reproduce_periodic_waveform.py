# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Reproduce the periodic-waveform experiment and its paper figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], environment: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def training_command(
    python: str,
    variant: str,
    device: str,
    steps: int,
    batch: int,
    warmup: int,
    log_every: int,
    result_path: Path,
    checkpoint_dir: Path,
    audio_dir: Path,
    array_dir: Path,
    extra: tuple[str, ...] = (),
) -> list[str]:
    return [
        python,
        "audio_toys/train_groove_waveform.py",
        "--variant",
        variant,
        "--seeds",
        "1",
        "--steps",
        str(steps),
        "--batch",
        str(batch),
        "--frame-size",
        "64",
        "--source-sample-rate",
        "4096",
        "--d-model",
        "128",
        "--heads",
        "8",
        "--head-dim",
        "16",
        "--backend",
        "auto",
        "--device",
        device,
        "--lr",
        "0.003",
        "--weight-decay",
        "1e-12",
        "--optimizer",
        "muon",
        "--scheduler",
        "warmup_cosine",
        "--warmup-steps",
        str(warmup),
        "--min-lr-ratio",
        "0.1",
        "--muon-lr",
        "0.02",
        "--muon-momentum",
        "0.95",
        "--muon-ns-steps",
        "5",
        "--muon-scope",
        "core",
        "--prefix-steps",
        "8",
        "--curriculum-lengths",
        "12,16,24,40,72,136",
        "--curriculum-mode",
        "scheduled",
        "--curriculum-fraction",
        "0.4",
        "--eval-lengths",
        "40,136,264",
        "--log-every",
        str(log_every),
        "--init",
        "default",
        "--audio-steps",
        "264",
        "--audio-sample-rate",
        "8000",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--audio-dir",
        str(audio_dir),
        "--array-dir",
        str(array_dir),
        "--out",
        str(result_path),
        *extra,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--device", default="auto", help="PyTorch device: auto, cpu, mps, or cuda")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/periodic_waveform"))
    parser.add_argument("--include-controls", action="store_true", help="also run positional and DeltaProduct-2 controls")
    args = parser.parse_args()

    output_root = args.output_root if args.output_root.is_absolute() else REPOSITORY_ROOT / args.output_root
    result_path = output_root / "results" / "models.jsonl"
    dense_path = output_root / "results" / "dense_metrics.json"
    checkpoint_dir = output_root / "checkpoints"
    audio_dir = output_root / "audio"
    array_dir = output_root / "arrays"
    figure_dir = output_root / "figures"
    for directory in (result_path.parent, checkpoint_dir, audio_dir, array_dir, figure_dir):
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
    steps = 20 if args.mode == "smoke" else 2500
    batch = 2 if args.mode == "smoke" else 64
    warmup = 2 if args.mode == "smoke" else 250
    log_every = 10 if args.mode == "smoke" else 250
    manifest = {
        "mode": args.mode,
        "requested_device": args.device,
        "main_variants": ["a01_b01", "a01_b02", "a11_b01", "a11_b02", "gru", "transformer"],
        "include_controls": args.include_controls,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    train_command = training_command(
        python,
        "paper",
        args.device,
        steps,
        batch,
        warmup,
        log_every,
        result_path,
        checkpoint_dir,
        audio_dir,
        array_dir,
    )
    run(train_command, environment)
    run([
        python,
        "evidence/aggregate_results.py",
        "--kind",
        "audio",
        "--inputs",
        str(result_path),
        "--output",
        str(output_root / "results" / "aggregates.json"),
    ], environment)
    run([
        python,
        "audio_toys/plot_groove_waveform_comparison.py",
        "--results",
        str(result_path),
        "--output-json",
        str(dense_path),
        "--output",
        str(figure_dir / "waveform_mse.pdf"),
    ], environment)
    run([
        python,
        "audio_toys/plot_groove_waveform_paper_figures.py",
        "--results",
        str(result_path),
        "--metrics",
        str(dense_path),
        "--main-output",
        str(figure_dir / "groove_waveform_main.pdf"),
        "--appendix-output",
        str(figure_dir / "groove_waveform_appendix.pdf"),
    ], environment)
    run([
        python,
        "audio_toys/analyze_groove_state_spectrum.py",
        "--results",
        str(result_path),
        "--model",
        "a11_b02",
        "--output-json",
        str(output_root / "results" / "complex_kda_spectrum.json"),
        "--output-figure",
        str(figure_dir / "complex_kda_spectrum.png"),
    ], environment)
    if args.include_controls:
        control_specs = (
            ("transformer_rope", "transformer_rope", ()),
            ("transformer_rope_500k", "transformer_rope_500k", ()),
            ("transformer_selective_rope", "transformer_selective_rope", ()),
            ("transformer_no_pos", "transformer_no_pos", ()),
            ("deltaproduct2_default", "deltaproduct2", ("--dp-init", "default")),
            ("deltaproduct2_ungated", "deltaproduct2_ungated", ("--dp-init", "default")),
            ("deltaproduct2_reflection", "deltaproduct2", ("--dp-init", "reflection")),
        )
        control_results = {}
        for name, variant, extra in control_specs:
            control_result = output_root / "results" / f"{name}.jsonl"
            control_results[name] = control_result
            run(training_command(
                python,
                variant,
                args.device,
                steps,
                batch,
                warmup,
                log_every,
                control_result,
                output_root / "checkpoints" / name,
                output_root / "audio" / name,
                output_root / "arrays" / name,
                extra,
            ), environment)
        run([
            python,
            "audio_toys/plot_groove_waveform_transformer_positions.py",
            "--results",
            str(result_path),
            *(str(control_results[name]) for name in (
                "transformer_rope",
                "transformer_rope_500k",
                "transformer_selective_rope",
                "transformer_no_pos",
            )),
            "--output-json",
            str(output_root / "results" / "transformer_positions_dense.json"),
            "--audio-dir",
            str(output_root / "audio" / "transformer_positions"),
            "--output",
            str(figure_dir / "transformer_positions.pdf"),
        ], environment)
        run([
            python,
            "audio_toys/plot_groove_deltaproduct2.py",
            "--baseline-results",
            str(result_path),
            "--default-result",
            str(control_results["deltaproduct2_default"]),
            "--ungated-result",
            str(control_results["deltaproduct2_ungated"]),
            "--reflection-result",
            str(control_results["deltaproduct2_reflection"]),
            "--output-json",
            str(output_root / "results" / "deltaproduct2_dense.json"),
            "--output",
            str(figure_dir / "deltaproduct2.pdf"),
        ], environment)
        run([
            python,
            "audio_toys/analyze_deltaproduct2_transition.py",
            str(control_results["deltaproduct2_reflection"]),
            "--output",
            str(output_root / "results" / "deltaproduct2_spectrum.json"),
        ], environment)
    print(f"\nReproduction complete: {output_root}", flush=True)


if __name__ == "__main__":
    main()
