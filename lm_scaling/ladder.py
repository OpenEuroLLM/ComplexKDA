"""Single source of truth for the complex-KDA scaling ladder.

The ladder itself is **data** and lives in `ladder_spec.yaml` next to this
file. It is the geometry and the fitted laws only: the campaign that runs them
is Megatron's, and its config lives under `megatron_ext/config/`.
This module only loads it and exposes the names the config layer cites, so the geometry, the budgets and the fitted laws can be read and
edited without going through Python.

``SCALING``
    The transformer ladder of Ajroldi et al., "Deriving Scaling Laws for
    OpenEuroLLM Models" (arXiv:2608.28308), Table 1.  Tied embeddings,
    head_dim 64, FFN expansion 4 with GLU, **seq 4096**, GPT-NeoX-20B.  Their
    fitted optimal-hyperparameter laws (Table 2 of that paper) are only
    meaningful at the N of *this* ladder, which is why we adopt it verbatim
    for the baseline rather than inventing our own sizes.

``BUDGETS`` are the intermediate-decay token budgets of the same paper,
which coincide with the ones the oellm-autoexp multilingual sweep uses.

Not in the YAML on purpose: the *form* of the laws.  A coefficient is a
number to tune; a power law is a modelling choice, and burying it in a config
would let it be changed without anyone noticing the paper no longer applies.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

_SPEC = yaml.safe_load((Path(__file__).resolve().parent / "ladder_spec.yaml").read_text())

_SCALING_SPEC = _SPEC["scaling"]
_LAWS = _SCALING_SPEC["laws"]
_GRIDS = _SCALING_SPEC["grids"]

# --- arXiv:2608.28308 Table 1: dense transformer baseline ------------------
# (d_model, heads, layers, ffn, paper's reported total params incl. embeddings)
SCALING = [dict(r) for r in _SCALING_SPEC["rungs"]]

# Intermediate decay points, in tokens.  arXiv:2608.28308 Sec. 2.
BUDGETS = [float(b) for b in _SCALING_SPEC["budgets"]]

SEQ_LEN = int(_SCALING_SPEC["seq_len"])
VOCAB_SIZE = int(_SCALING_SPEC["vocab_size"])
HEAD_DIM = int(_SCALING_SPEC["head_dim"])

_LR_GRID = [float(x) for x in _GRIDS["lr"]]
_B_LOG2_MIN = int(_GRIDS["batch_size_log2"]["min"])
_B_LOG2_MAX = int(_GRIDS["batch_size_log2"]["max"])


def _power_law(spec: dict, N: float, D: float) -> float:
    return float(spec["coefficient"]) * N ** float(spec["n_exponent"]) * D ** float(spec["d_exponent"])


# --- fitted optimal hyperparameters, arXiv:2608.28308 Table 2 --------------
# N is total parameters INCLUDING (tied) embeddings; D is tokens.
def opt_batch_size(N: float, D: float) -> float:
    """Jointly optimal batch size in SEQUENCES (b, not tokens).

    In sequences is why SEQ_LEN matters to the ladder and not only to the
    step arithmetic: the same b* at half the context is half the tokens per
    step, which is off-optimum in a direction the law cannot express.
    """
    return _power_law(_LAWS["batch_size"], N, D)


def opt_lr(N: float, D: float) -> float:
    """Jointly optimal peak learning rate."""
    return _power_law(_LAWS["lr"], N, D)


def snap_batch_size(b: float) -> int:
    """Snap to the log2 grid the paper searched (2**4 .. 2**10)."""
    return int(2 ** min(_B_LOG2_MAX, max(_B_LOG2_MIN, round(math.log2(b)))))


def snap_lr(lr: float) -> float:
    """Snap to the LR grid the paper searched."""
    return min(_LR_GRID, key=lambda g: abs(math.log(g) - math.log(lr)))
