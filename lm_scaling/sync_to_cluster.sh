#!/bin/bash
# Sync this repo to a cluster.  The LOCAL working tree is the source of truth and
# is pushed as-is, independently of git state -- uncommitted edits included.
# Nothing is ever pulled back, so anything produced on the cluster (checkpoints,
# logs, wandb) must live under one of the excluded paths below or it is at risk.
#
#   CLUSTER_HOST=jupiter CLUSTER_DEST='~/work/complex_kda_staging' \
#     ./lm_scaling/sync_to_cluster.sh
#
# Host and destination are parameters rather than constants, because the
# layout differs between machines. JUPITER has a thin wrapper that fills them
# in, and that is what to use day to day:
#
#   CLUSTER_HOST=jupiter CLUSTER_DEST='~/work/complex_kda' \
#     ./lm_scaling/sync_to_cluster.sh
#
# CLUSTER_HOST/CLUSTER_DEST reach any host with an ~/.ssh/config entry.
#
# The destinations differ by one path component, and getting it wrong does not
# fail -- rsync creates the wrong directory, reports success, and every later
# command runs against a tree one campaign behind.
#
# Requires a matching entry in ~/.ssh/config.
set -euo pipefail

REMOTE="${CLUSTER_HOST:-${JUPITER_HOST:-jupiter}}"
DEST="${CLUSTER_DEST:-${JUPITER_DEST:-~/work/complex_kda}}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/"

EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --delete)
      # Exact mirror.  Kept opt-in: run outputs on the cluster that are not
      # excluded below would be deleted.
      EXTRA+=(--delete)
      ;;
    --dry-run) EXTRA+=(--dry-run --itemize-changes) ;;
    *) EXTRA+=("$arg") ;;
  esac
done

# Syntax-gate the tree before shipping it. The local repo is the source of
# truth and is synced independently of commits, so a file that does not even
# parse reaches the cluster as fast as a good one -- and only announces itself
# ~40 s later, once every rank has spun up, as an IndentationError buried in a
# four-rank ChildFailedError dump. Five speed probes died that way. Compiling
# takes under a second and catches it here instead.
echo "Syntax-checking lm_scaling/ fla/ ..."
if ! python3 -m compileall -q -x '(\.venv|build|\.git)' "${SRC}lm_scaling" "${SRC}fla" >/dev/null; then
  echo "ABORT: a Python file does not compile; nothing synced." >&2
  python3 -m compileall -q -x '(\.venv|build|\.git)' "${SRC}lm" "${SRC}lm_scaling" "${SRC}fla" >&2
  exit 1
fi

# Same argument for the batch scripts. A shell syntax error does not surface
# as a shell error: slurm reports it as one dead array task per shard, from
# /var/spool/..., naming a line number in a file the user never wrote. A
# heredoc opened inside a brace block cost 15 tasks that way.
echo "Syntax-checking slurm scripts ..."
for f in "${SRC}lm_scaling/slurm/"*.sbatch "${SRC}lm_scaling/"*.sh; do
  [ -f "$f" ] || continue
  if ! bash -n "$f" 2>/dev/null; then
    echo "ABORT: $f is not valid shell; nothing synced." >&2
    bash -n "$f" >&2
    exit 1
  fi
done

rsync -rlptz --human-readable --progress \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '*.sif' \
  --exclude '*.egg-info/' \
  --exclude '.venv/' \
  `# autoexp builds its venv INSIDE the checkout, and this repo is`   \
  `# synced between architectures -- the local one is x86, JUPITER is`  \
  `# aarch64. Shipping it would overwrite the cluster's own venv with`  \
  `# binaries that cannot load there. make_megatron_stack.sh builds it`  \
  `# on the machine that will use it.`  \
  --exclude 'autoexp_venv/' \
  `# generated on the cluster -- never overwrite or delete from here` \
  `# output/ especially: submit.py writes each job's titan.toml, sbatch and` \
  `# resolved config INTO the run directory under cluster.runs_dir, so a sync` \
  `# can push a stale local copy over one a QUEUED job is about to read. That` \
  `# happened: a 20k-step loss-match job started against a 3000-step config` \
  `# left behind by an early planning run, and died on a token budget that no` \
  `# longer matched anything.` \
  --exclude 'output/' \
  --exclude 'results/' \
  --exclude 'wandb/' \
  --exclude 'monitor_state/' \
  `# the harvest is analysis data pulled OFF a cluster; shipping it back would` \
  `# put one cluster's extracts under another's tree and invite confusion` \
  --exclude 'lm_scaling/harvest/' \
  --exclude 'checkpoints/' \
  `# local paper build artefacts, no reason to ship them` \
  --exclude '*.pdf' \
  --exclude '*.aux' \
  --exclude '*.fls' \
  --exclude '*.fdb_latexmk' \
  --exclude '*.log' \
  --exclude '*.out' \
  --exclude '*.toc' \
  "${EXTRA[@]}" \
  "$SRC" "$REMOTE:$DEST/"

echo
echo "Synced $SRC -> $REMOTE:$DEST/"
