#!/bin/bash
# Build the venv that PLANS and SUBMITS the campaign, on a login node.
#
#   lm_scaling/make_login_venv.sh
#
# WHY THIS EXISTS AND THE OVERLAY DOES NOT COVER IT.
#
# `sbatch` is not in the container. It is a host binary that talks to the
# slurm controller over a socket and a set of configuration files that the
# image does not carry, so `apptainer exec ... submit.py --run` gets as far as
# writing every job file, registers them with the monitor, and then fails to
# submit a single one. The failure arrives after the part that looks like the
# work.
#
# So the split is by WHO NEEDS SLURM:
#
#   login venv (here)   plan.py, submit.py, watch.py, the probe drivers.
#                       They resolve configs, render sbatch scripts, call
#                       `sbatch`/`squeue`, and poll. No torch, no fla, no CUDA
#                       -- checked below, because the day one of them grows a
#                       torch import is the day this venv silently needs 3 GB.
#
#   container           the training itself. The rendered sbatch enters it via
#                       `slurm.launcher_cmd`, so a submitted job is already in
#                       the image with torchtitan, titan_oellm and the overlay.
#                       Also anything that builds a model to count parameters:
#                       param_match.py, the titan_ext tests.
#
# The venv lives in the repo at `.venv`, which `sync_to_cluster.sh` excludes,
# so a sync cannot overwrite or delete it. Same name and place as the local
# one, so the two clusters and the laptop all read the same way.
set -euo pipefail

REPO=${CKDA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV=${CKDA_VENV:-$REPO/.venv}

[ -x "$VENV/bin/python3" ] || python3 -m venv "$VENV"

"$VENV/bin/python3" -m pip install --no-cache-dir --quiet --upgrade \
  pip compoconf omegaconf "hydra-core>=1.3" networkx tomli-w pyyaml

# --force-reinstall --no-deps for the git packages, for the reason
# make_overlay.sh gives: they are pinned to a BRANCH, so the version string
# does not move between commits and `--upgrade` decides they are already
# satisfied. The install prints nothing, exits 0, and the venv keeps the old
# code.
"$VENV/bin/python3" -m pip install --no-cache-dir --quiet \
  --force-reinstall --no-deps \
  git+https://github.com/kpoeppel/hydra_staged_sweep \
  git+https://github.com/kpoeppel/slurm_gen \
  git+https://github.com/kpoeppel/monitor

# It has to RESOLVE A CAMPAIGN and find `sbatch`, not merely import. A venv
# that imports everything and cannot see the slurm binaries is the exact
# failure this script exists to remove, one layer down.
PYTHONPATH="$REPO:$REPO/lm_scaling" "$VENV/bin/python3" - <<'PY'
import shutil
import sys

from monitor.actions import LogEventConfig  # noqa: F401
from slurm_gen.schema import SlurmConfig

import plan as P

jobs = P.plans("fwedu_1p3B")
P.check(jobs)
checks = {
    "resolves a campaign": len(jobs) > 0,
    "srun_args": "srun_args" in SlurmConfig.__dataclass_fields__,
    "sbatch on PATH": bool(shutil.which("sbatch")),
    "squeue on PATH": bool(shutil.which("squeue")),
    # The whole point of a separate venv: if this turns True the split has
    # stopped being about slurm and the venv is quietly carrying a CUDA stack.
    "no torch pulled in": "torch" not in sys.modules,
}
for k, v in checks.items():
    print(f"  {k:<22} {v}")
if not all(checks.values()):
    raise SystemExit("login venv is not usable for submission")
print(f"  {'jobs resolved':<22} {len(jobs)}")
PY
echo "login venv ready: $VENV"
