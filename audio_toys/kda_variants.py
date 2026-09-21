# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Shared KDA range-ablation definitions for the audio experiments."""

from __future__ import annotations

import argparse

VARIANTS = {
    "a01_b01": {
        "gate": "sigmoid",
        "allow_neg_eigval": False,
        "gate_init_style": "shipped",
        "beta_init_style": "standard",
    },
    "a01_b02": {
        "gate": "sigmoid",
        "allow_neg_eigval": True,
        "gate_init_style": "shipped",
        "beta_init_style": "spread",
    },
    "a11_b01": {
        "gate": "signed_sigmoid2",
        "allow_neg_eigval": False,
        "gate_init_style": "spread",
        "beta_init_style": "standard",
    },
    "a11_b02": {
        "gate": "signed_sigmoid2",
        "allow_neg_eigval": True,
        "gate_init_style": "spread",
        "beta_init_style": "spread",
    },
}


def parse_ints(value: str) -> list[int]:
    """Parse a strictly increasing comma-separated sequence of lengths."""
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 2 for item in values):
        raise argparse.ArgumentTypeError("lengths must be comma-separated integers >= 2")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("lengths must be strictly increasing")
    return values
