# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Export all-phase waveform predictions and checkpoint checksums from completed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audio_toys.plot_groove_waveform_comparison import MODEL_ORDER, load_model
from audio_toys.train_groove_waveform import make_waveform_pattern, predict_all_phases


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--length", type=int, default=264)
    args = parser.parse_args()

    rows = {row["model"]: row for row in map(json.loads, args.results.read_text().splitlines())}
    missing = set(MODEL_ORDER) - rows.keys()
    if missing:
        raise ValueError(f"missing models: {sorted(missing)}")
    reference = rows[MODEL_ORDER[0]]
    pattern = make_waveform_pattern(
        reference["frame_size"],
        reference["source_sample_rate"],
        bpm=124,
        seed=2026,
    )
    mean = pattern.mean(dim=0)
    std = pattern.std(dim=0).clamp_min(0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"length": args.length, "models": {}}
    for model_name in MODEL_ORDER:
        row = dict(rows[model_name])
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = args.checkpoint_root / checkpoint
        row["checkpoint"] = str(checkpoint)
        model = load_model(row)
        phases, inputs, targets, predictions = predict_all_phases(
            model,
            pattern,
            mean,
            std,
            args.length,
            row["prefix_steps"],
        )
        output = args.output_dir / f"{model_name}-seed{row['seed']}.npz"
        np.savez_compressed(
            output,
            phases=phases,
            normalized_inputs=inputs,
            targets=targets,
            predictions=predictions,
            pattern=pattern.numpy(),
            normalization_mean=mean.numpy(),
            normalization_std=std.numpy(),
        )
        manifest["models"][model_name] = {
            "seed": row["seed"],
            "array_file": output.name,
            "array_sha256": sha256(output),
            "checkpoint_file": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
