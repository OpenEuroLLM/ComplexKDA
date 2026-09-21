#!/usr/bin/env python3
"""Verify archived periodic-waveform arrays, hashes, and reported metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["model"], int(row["seed"]))] = row
    return rows


def verify(root: Path, metric_rtol: float) -> dict:
    manifest = json.loads((root / "arrays" / "manifest.json").read_text())
    rows = load_rows(root / "models.jsonl")
    checked = []

    for model, item in manifest["models"].items():
        array_path = root / "arrays" / item["array_file"]
        actual_hash = sha256(array_path)
        if actual_hash != item["array_sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {array_path}")

        key = (model, int(item["seed"]))
        if key not in rows:
            raise RuntimeError(f"No result row for {key}")
        row = rows[key]
        with np.load(array_path) as arrays:
            target = arrays["targets"].astype(np.float64)
            prediction = arrays["predictions"].astype(np.float64)
        if target.shape != prediction.shape:
            raise RuntimeError(f"Shape mismatch for {array_path}")

        metrics = {}
        for length in (40, 136, 264):
            prefix_steps = int(row["prefix_steps"])
            measured = float(
                np.mean((prediction[:, prefix_steps:length] - target[:, prefix_steps:length]) ** 2)
            )
            expected = float(row["metrics"][str(length)]["waveform_mse"])
            if not np.isclose(measured, expected, rtol=metric_rtol, atol=1e-12):
                raise RuntimeError(
                    f"MSE mismatch for {key} at length {length}: "
                    f"archive={measured}, row={expected}"
                )
            metrics[str(length)] = {"array": measured, "reported": expected}

        checked.append(
            {
                "model": model,
                "seed": int(item["seed"]),
                "path": str(array_path.relative_to(root)),
                "sha256": actual_hash,
                "shape": list(target.shape),
                "mse": metrics,
            }
        )

    return {
        "status": "pass",
        "metric_relative_tolerance": metric_rtol,
        "arrays_checked": len(checked),
        "checks": checked,
        "note": (
            "Small metric differences are expected because the archived rows used "
            "PyTorch float32 reductions while this verifier uses NumPy float64."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("evidence/results/periodic_waveform"),
    )
    parser.add_argument("--metric-rtol", type=float, default=2e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify(args.root, args.metric_rtol)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
