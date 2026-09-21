# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Write a compact software, hardware, and source-revision manifest."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def git(command: tuple[str, ...]) -> str | None:
    try:
        return subprocess.check_output(("git", *command), stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requested-device")
    args = parser.parse_args()
    report = {
        "source": {
            "git_revision": git(("rev-parse", "HEAD")),
            "git_branch": git(("rev-parse", "--abbrev-ref", "HEAD")),
            "git_dirty": bool(git(("status", "--porcelain"))),
        },
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "requested_device": args.requested_device,
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "packages": {
            name: package_version(name)
            for name in ("einops", "numpy", "scipy", "transformers", "triton", "scienceplots")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
