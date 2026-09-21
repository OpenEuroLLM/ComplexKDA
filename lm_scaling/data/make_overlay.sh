#!/bin/bash
# Build the orchestration overlay: hydra_staged_sweep, slurm_gen, monitor and
# their dependencies, as one apptainer overlay image beside the container.
#
#   bash lm_scaling/data/make_overlay.sh                          # this cluster's titan image
#   CKDA_CONTAINER=jupiter_megatron bash lm_scaling/data/make_overlay.sh
#
# WHICH CONTAINER, and where the overlay lands, come from
# `config/container/<name>.yaml` -- the same file `container_run` reads and the
# same file a campaign mounts. They used to be spelled out again here, which is
# the "one number in two places" this project has paid for repeatedly: an image
# path that drifts between the builder and the job produces an overlay built
# against one image and mounted into another, and the symptom is an ImportError
# in a compute-node log.
#
# So a SECOND backend needs no second script. Add its container config and name
# it: the libraries installed below are what the orchestration layer needs and
# does not declare, which is as true of a Megatron stack as of this one.
#
# WHY NOT A VENV. A venv on a shared filesystem carries absolute paths and a
# python it does not own: it breaks when the image's interpreter moves, it has
# to be recreated per cluster, and a hundred jobs importing from it are a
# hundred readers of a directory tree nobody can freeze. An overlay is a
# single immutable file -- `--overlay img:ro` and the packages are on
# sys.path, with the same guarantees the image itself has.
#
# TRAINING NEEDS ONE THING FROM IT, on JUPITER: tilelang. torchtitan and
# titan_oellm are in the image and fla comes from the repo on PYTHONPATH, so
# the overlay is otherwise orchestration-only -- but the image ships Triton
# 3.6.0, and on Hopper fla REFUSES to run the gated chunk backward there:
#
#   RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces
#   incorrect results for gated chunk_bwd_dqkwg (see #640). Please upgrade
#   Triton to >= 3.7.1 or install tilelang
#
# That is a correctness guard, not a performance one -- the results would be
# WRONG -- and it blocks the gdn arm outright on GH200. It never fired on
# JUWELS because A100 is Ampere. The overlay is already mounted by training
# jobs (`container.overlays` is folded into slurm.launcher_cmd), so putting
# tilelang here is enough; `FLA_TILELANG` in config/slurm/jupiter.yaml turns
# it on.
set -eu

REPO=${CKDA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
CLUSTER=${CLUSTER:-$(hostname | grep -qi juwels && echo juwels || echo jupiter)}
CONTAINER=${CKDA_CONTAINER:-${CLUSTER}_titan}
CFG="$REPO/lm_scaling/config/container/$CONTAINER.yaml"
[ -f "$CFG" ] || { echo "no container config at $CFG" >&2; exit 1; }

# Image and overlay path, read out of that config. OmegaConf when it is
# installed, so `${oc.env:VAR,default}` resolves exactly as the campaign
# resolves it; PyYAML plus that one pattern otherwise. An interpolation this
# cannot resolve is an ERROR rather than a path with a literal `${...}` left in
# it, which apptainer would report as "no such file" naming something nobody
# wrote.
mapfile -t PARTS < <(python3 - "$CFG" <<'PY'
import os, re, sys

path = sys.argv[1]
try:
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
except ImportError:
    import yaml

    def env(m):
        var, _, default = m.group(1).partition(",")
        return os.environ.get(var, default)

    raw = re.sub(r"\$\{oc\.env:([^}]*)\}", env, open(path).read())
    cfg = yaml.safe_load(raw)

overlays = cfg.get("overlays") or []
if not overlays:
    sys.exit(f"{path} declares no overlays, so there is nothing to build into")
flat = repr([cfg["image"], overlays[0]])
left = re.search(r"\$\{[^}]*\}", flat)
if left:
    sys.exit(f"{path}: cannot resolve {left.group(0)} without hydra; "
             "install omegaconf on this node or simplify the container config")
print(cfg["image"])
print(overlays[0])
PY
) || exit 1

SIF=${SIF:-${PARTS[0]}}
OV=${OV:-${PARTS[1]}}
CACHE=$(dirname "$OV")
SIZE=${SIZE:-2048}

echo "container: $CONTAINER"
echo "  image:   $SIF"
echo "  overlay: $OV"

mkdir -p "$CACHE"
[ -f "$SIF" ] || { echo "no container at $SIF" >&2; exit 1; }
[ -f "$OV" ] || apptainer overlay create --size "$SIZE" "$OV"

apptainer exec --overlay "$OV" "$SIF" python3 -m pip install --no-cache-dir --quiet \
  --upgrade compoconf omegaconf "hydra-core>=1.3" networkx tomli-w pyyaml

# --force-reinstall --no-deps for the git packages. They are pinned to a
# BRANCH, not a version, so their version string does not move between
# commits and `--upgrade` alone decides they are already satisfied: the
# install prints nothing, exits 0, and the overlay keeps the old code. That
# silently cost a round of "why is the new API missing".
apptainer exec --overlay "$OV" "$SIF" python3 -m pip install --no-cache-dir --quiet \
  --force-reinstall --no-deps \
  git+https://github.com/kpoeppel/hydra_staged_sweep \
  git+https://github.com/kpoeppel/slurm_gen \
  git+https://github.com/kpoeppel/monitor

# tilelang, for the Hopper kernel path above, WITH its dependencies.
#
# `--no-deps` was the first attempt, on the theory that it would drag in a
# torch of its own and a second torch under an overlay is how a job comes to
# import one CUDA runtime and link another. It does not: pip resolves exactly
# four packages here -- apache-tvm-ffi, z3-solver, cloudpickle and
# torch_c_dlpack_ext -- and treats the image's torch as satisfying the rest.
# Without them tilelang installs happily and then dies on `import tvm`.
apptainer exec --overlay "$OV" "$SIF" python3 -m pip install --no-cache-dir --quiet \
  tilelang

# It has to IMPORT, not merely install: a wheel that unpacks and then fails on
# a missing symbol is the failure this script exists to catch early, on the
# login node, rather than on a compute node forty minutes into a queue.
apptainer exec --overlay "$OV" "$SIF" python3 -c \
  "import tilelang; print('tilelang', tilelang.__version__)"

# --- the JUWELS image's stale model code -------------------------------
# The x86 build carries the SAME torchtitan as JUPITER's aarch64 image (190
# files, md5 04a47be94e9a) but an older titan_oellm (36 files against 41), and
# the two halves disagree: torchtitan renamed the device-mesh dimension
# `dp_shard_cp` to `fsdp`, and qwen3_custom/infra/parallelize.py still asks
# for the old name. Every run dies at startup with
#
#   KeyError: Invalid mesh_dim_names ('dp_shard_cp',) specified.
#             Valid mesh_dim_names are ['world', 'loss_mesh']
#
# titan_oellm is pure Python, so the overlay carries JUPITER's copy and both
# clusters run byte-identical model code -- which is worth more than avoiding
# the patch: the alternative image (0.2.0) is internally consistent but ships
# a torchtitan 62 files different from JUPITER's, and then the two clusters
# would not be measuring the same thing.
#
# Extract it with, on JUPITER:
#   apptainer exec --bind /p <jupiter.sif> \
#     cp -a /opt/titan-oellm/titan_oellm $CACHE/titan_oellm_from_jupiter/
if [ "$CLUSTER" = "juwels" ] && [ -d "$CACHE/titan_oellm_from_jupiter/titan_oellm" ]; then
  apptainer exec --overlay "$OV" --bind /p "$SIF" bash -c \
    "rm -rf /opt/titan-oellm/titan_oellm && \
     cp -a $CACHE/titan_oellm_from_jupiter/titan_oellm /opt/titan-oellm/titan_oellm"
  echo "  patched titan_oellm from JUPITER's image"
fi

# Read-only is how every job will mount it, so that is how it is checked. The
# heredoc goes through stdin rather than a file in /tmp on purpose: python puts
# a script's own directory first on sys.path, and a shared /tmp on a login node
# is a place where another user's `inspect.py` can shadow the standard library.
# That is not hypothetical -- it happened here.
apptainer exec --overlay "$OV:ro" "$SIF" python3 - <<'PY'
from monitor.actions import LogEventConfig
from slurm_gen.schema import SlurmConfig
import compoconf, hydra_staged_sweep, monitor, slurm_gen  # noqa: F401

fields = LogEventConfig.__dataclass_fields__
checks = {
    "compoconf": compoconf.__version__,
    "progress events": "progress" in str(fields["pattern_type"].type),
    "inactivity knobs": "inactivity_polls" in fields,
    "srun_args": "srun_args" in SlurmConfig.__dataclass_fields__,
}
for k, v in checks.items():
    print(f"  {k:<18} {v}")
if not all(v for k, v in checks.items() if k != "compoconf"):
    raise SystemExit("overlay is stale: the install did not take")
PY
echo "overlay ready: $OV"
