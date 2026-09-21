#!/bin/bash
# Clone and pin the Megatron backend's stack, exactly as `megatron_stack.lock`
# records it.
#
#   bash lm_scaling/make_megatron_stack.sh            # clone, check out, install
#   bash lm_scaling/make_megatron_stack.sh --verify   # check an existing tree, change nothing
#   bash lm_scaling/make_megatron_stack.sh --relock   # rewrite the lock from what is checked out
#
# The counterpart to `make_login_venv.sh`, for the second backend. That script
# builds the venv that plans and submits THIS repository's campaigns; this one
# materialises the external tree the Megatron ladder runs through, because the
# arms reach Megatron via oellm-autoexp's launcher rather than through an
# sbatch of our own.
#
# WHY A LOCKFILE AT ALL, and why checked rather than merely written, is argued
# in `megatron_stack.lock` itself. The short version: `hybrid_exp` is a branch
# and branches move, and the three orchestration libraries are pip-installed
# from a branch, so nothing in git pins them. A framework that shifts under a
# comparison built to hold the framework fixed is the one error this whole
# ladder exists to remove -- the two frameworks already disagree by 0.049 nats
# on identical configuration, which is larger than most of what is measured.
#
# WHAT THIS DOES NOT DO: build the container image. Megatron runs from an
# internally built image (fla + liger + mamba-ssm) that is not public, and no
# amount of git pinning reaches it. Everything else here is public and this
# script fetches all of it without a GitHub account.
set -euo pipefail

REPO=${CKDA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
LOCK=${CKDA_MEGATRON_LOCK:-$REPO/lm_scaling/megatron_stack.lock}
DEST=${OELLM_AUTOEXP:-$REPO/external/oellm-autoexp}
VENV=${AUTOEXP_VENV:-$DEST/autoexp_venv}

MODE=install
case "${1:-}" in
  --verify) MODE=verify ;;
  --relock) MODE=relock ;;
  "")       ;;
  *) echo "usage: $0 [--verify|--relock]" >&2; exit 2 ;;
esac

[ -f "$LOCK" ] || { echo "no lockfile at $LOCK" >&2; exit 1; }

# Every public repository in this stack is reachable over HTTPS, but three of
# autoexp's four submodules are DECLARED with `git@github.com:` URLs. Without
# this rewrite a clone fails on repositories that anyone can read, and the
# error names SSH rather than the real problem. Scoped to the commands here
# with `-c`, so nothing is written to the user's global git config.
GITC=(git -c "url.https://github.com/.insteadOf=git@github.com:")

# `<kind> <name> <url> <commit> [branch]`, comments and blank lines dropped.
entries() { grep -vE '^[[:space:]]*(#|$)' "$LOCK"; }
field() { entries | awk -v k="$1" -v n="$2" '$1==k && $2==n {print $'"$3"'; exit}'; }

AUTOEXP_NAME=$(entries | awk '$1=="repo" {print $2; exit}')
AUTOEXP_URL=$(entries  | awk '$1=="repo" {print $3; exit}')
AUTOEXP_SHA=$(entries  | awk '$1=="repo" {print $4; exit}')
AUTOEXP_BR=$(entries   | awk '$1=="repo" {print $5; exit}')
[ -n "$AUTOEXP_URL" ] && [ -n "$AUTOEXP_SHA" ] || {
  echo "$LOCK: no 'repo' entry to clone" >&2; exit 1; }

echo "lock:    $LOCK"
echo "dest:    $DEST"
echo "$AUTOEXP_NAME @ ${AUTOEXP_SHA:0:12}${AUTOEXP_BR:+ (via $AUTOEXP_BR)}"

# --- relock: record what is actually checked out ----------------------------
# Deliberate and separate from `install`. Refreshing a pin is a decision about
# which framework the ladder runs on, so it is never a side effect of setting
# the tree up -- otherwise the first `install` after an upstream push would
# quietly adopt it and the lockfile would describe nothing.
if [ "$MODE" = relock ]; then
  [ -d "$DEST/.git" ] || { echo "nothing checked out at $DEST" >&2; exit 1; }
  # Rewrites the DATA lines in place and leaves every comment exactly where it
  # is. The first version regenerated the file from a template and silently
  # dropped the per-section reasoning with it -- a lockfile whose "why" is
  # deleted by its own refresh command is worse than no refresh command.
  AUTOEXP_SHA_NOW=$(git -C "$DEST" rev-parse HEAD) \
  SUBS="$(git -C "$DEST" ls-tree HEAD submodules/ | awk '$2=="commit"{print $4" "$3}')" \
  LOCKFILE="$LOCK" DESTDIR="$DEST" python3 - <<'PY'
import os, re, subprocess, sys

lock = os.environ["LOCKFILE"]
subs = dict(l.split() for l in os.environ["SUBS"].splitlines() if l.strip())
lines = open(lock).read().splitlines(keepends=True)

def widths(kind):
    """Column widths this file already uses, so a relock is a value diff and
    not a whitespace diff."""
    for l in lines:
        p = l.split()
        if len(p) >= 4 and p[0] == kind:
            return l.index(p[1]), l.index(p[2]), l.index(p[3])
    return 11, 36, 92

seen, out, changed = set(), [], []
for l in lines:
    p = l.split()
    if len(p) < 4 or p[0] not in ("repo", "submodule", "pip") or l.lstrip().startswith("#"):
        out.append(l)
        continue
    kind, name, url, old = p[0], p[1], p[2], p[3]
    branch = p[4] if len(p) > 4 else ""
    if kind == "repo":
        new = os.environ["AUTOEXP_SHA_NOW"]
    elif kind == "submodule":
        new = subs.get(name, "")
        if not new:
            print(f"  WARNING: {name} is in the lock but not in the tree", file=sys.stderr)
            out.append(l); continue
        seen.add(name)
    else:
        new = subprocess.run(["git", "ls-remote", url, branch or "HEAD"],
                             capture_output=True, text=True).stdout.split("\t")[0].strip()
        if not new:
            print(f"  WARNING: could not resolve {name}; kept {old[:12]}", file=sys.stderr)
            out.append(l); continue
    if new != old:
        changed.append(f"{name}: {old[:12]} -> {new[:12]}")
    a, b, c = widths(kind)
    row = f"{kind:<{a}}{name:<{b - a}}{url:<{c - b}}{new}"
    out.append(row + (f"  {branch}" if branch else "") + "\n")

for name, sha in sorted(subs.items()):
    if name not in seen:
        print(f"  WARNING: {name} @ {sha[:12]} is in the tree but not in the lock; "
              "add it by hand, with a line saying what it is", file=sys.stderr)

open(lock, "w").writelines(out)
print("  " + ("\n  ".join(changed) if changed else "no pin moved"))
PY
  echo "relocked $LOCK"
  exit 0
fi

# --- our own commits in the external stack -----------------------------------
#
# Two of the pins in the lockfile are commits we made, not upstream's: one on
# Megatron adding `--ckda-arm`, one on autoexp carrying it through the config
# layer. Both sit on a branch called `feat/ckda-arm` in the respective
# OpenEuroLLM repository.
#
# They are pinned exactly like every other commit here, which is the point --
# an earlier version of this script shipped them as patch files applied after
# checkout, and a patch is a worse object than a commit: it has no author, no
# message, no parent, and it rots silently the moment the line it anchors to
# moves. A branch is the thing git is for.
#
# The cost is that a pin can name a commit the remote does not have yet, and
# git's own error for that is `fatal: reference is not a tree`, which says
# nothing about why. This turns it into a sentence.
unpushed_hint() {  # <dir> <commit> <branch> <name>
  local dir="$1" commit="$2" branch="$3" name="$4"
  git -C "$dir" cat-file -e "${commit}^{commit}" 2>/dev/null || return 0
  git -C "$dir" branch -r --contains "$commit" 2>/dev/null | grep -q . && return 0
  echo "  NOTE: $name is pinned at ${commit:0:12}, which exists here but is on" >&2
  echo "        no remote branch. It is ours, and pushing '$branch' to the" >&2
  echo "        OpenEuroLLM repository is what makes this stack installable" >&2
  echo "        from a fresh clone." >&2
}

# --- clone / fetch ----------------------------------------------------------
if [ "$MODE" = install ]; then
  if [ ! -d "$DEST/.git" ]; then
    mkdir -p "$(dirname "$DEST")"
    # --branch makes the fetch cheap; the COMMIT below is the pin. Cloning the
    # branch and stopping there is what a moving pin looks like.
    "${GITC[@]}" clone ${AUTOEXP_BR:+--branch "$AUTOEXP_BR"} "$AUTOEXP_URL" "$DEST"
  else
    "${GITC[@]}" -C "$DEST" fetch --all --tags
  fi
  "${GITC[@]}" -C "$DEST" checkout --quiet "$AUTOEXP_SHA"
  "${GITC[@]}" -C "$DEST" submodule update --init --recursive
fi

[ -d "$DEST/.git" ] || { echo "nothing checked out at $DEST" >&2; exit 1; }

# --- verify -----------------------------------------------------------------
fail=0
check() {  # name expected actual
  if [ "$2" = "$3" ]; then
    printf '  %-28s %s\n' "$1" "${3:0:12}"
  else
    printf '  %-28s %s  EXPECTED %s\n' "$1" "${3:0:12}" "${2:0:12}" >&2
    fail=1
  fi
}

echo "checking pins:"
check "$AUTOEXP_NAME" "$AUTOEXP_SHA" "$(git -C "$DEST" rev-parse HEAD)"

entries | awk '$1=="submodule"' | while read -r _ path _ want _; do
  got=$(git -C "$DEST" ls-tree HEAD "$path" | awk '{print $3}')
  if [ "$want" = "$got" ]; then
    printf '  %-28s %s\n' "$path" "${got:0:12}"
  else
    printf '  %-28s %s  EXPECTED %s\n' "$path" "${got:0:12}" "${want:0:12}" >&2
    echo "PIN-MISMATCH" >> "$DEST/.pin_mismatch"
  fi
done
# The loop above runs in a subshell (it is fed by a pipe), so its exit status
# cannot reach here. A sentinel file carries it instead -- a `fail=1` set in
# there would be discarded and every mismatch would report success.
if [ -f "$DEST/.pin_mismatch" ]; then fail=1; rm -f "$DEST/.pin_mismatch"; fi

# Our two commits: present in the tree is what matters for running, but a pin
# that no remote carries is a release that cannot be installed, so say so.
# A PIN DESCRIBES A COMMIT, NOT A TREE. Every check above compares SHAs, and
# they all pass against a checkout with uncommitted edits on top -- which is
# the state a stack is in exactly when someone is debugging it, and the state
# in which "matches the lock" is most misleading. Say so.
echo "checking for uncommitted edits:"
for d in "$DEST" "$DEST/submodules/Megatron-LM"; do
  n=$(git -C "$d" status --porcelain 2>/dev/null | grep -vc '^?? ' || true)
  if [ "${n:-0}" -gt 0 ]; then
    printf '  %-28s %s tracked file(s) modified\n' "$(basename "$d")" "$n" >&2
    echo "        the pin names a commit; this tree is not at it" >&2
    fail=1
  else
    printf '  %-28s clean\n' "$(basename "$d")"
  fi
done

echo "checking our own commits:"
unpushed_hint "$DEST" "$AUTOEXP_SHA" "$AUTOEXP_BR" "oellm-autoexp"
mg_want=$(entries | awk '$1=="submodule" && $2=="submodules/Megatron-LM" {print $4}')
mg_br=$(entries | awk '$1=="submodule" && $2=="submodules/Megatron-LM" {print $5}')
unpushed_hint "$DEST/submodules/Megatron-LM" "$mg_want" "$mg_br" "submodules/Megatron-LM"

if [ "$fail" -ne 0 ]; then
  echo "stack does not match $LOCK" >&2
  echo "  re-run without --verify to move the tree onto the lock," >&2
  echo "  or --relock to adopt what is checked out, deliberately." >&2
  exit 1
fi

[ "$MODE" = verify ] && { echo "stack matches the lock"; exit 0; }

# --- the venv ---------------------------------------------------------------
# Its own, not the login venv: autoexp is installed editable into it, and the
# three orchestration libraries are NOT in its pyproject.
[ -x "$VENV/bin/python3" ] || python3 -m venv "$VENV"
"$VENV/bin/python3" -m pip install --no-cache-dir --quiet --upgrade pip
"$VENV/bin/python3" -m pip install --no-cache-dir --quiet -e "$DEST"

# --force-reinstall --no-deps, for the reason make_login_venv.sh gives: pinned
# at a commit here, but pip compares the VERSION string, which does not move
# between commits of the same branch. Without it the install prints nothing,
# exits 0, and keeps whatever was there.
while read -r _ name url sha _; do
  echo "  pip: $name @ ${sha:0:12}"
  "$VENV/bin/python3" -m pip install --no-cache-dir --quiet \
    --force-reinstall --no-deps "git+$url@$sha"
done < <(entries | awk '$1=="pip"')

# Installed is not importable: a wheel that unpacks and then dies on a missing
# symbol is the failure this catches on a login node instead of forty minutes
# into a queue.
"$VENV/bin/python3" - <<'PY'
import importlib
for m in ("hydra_staged_sweep", "slurm_gen", "monitor"):
    importlib.import_module(m)
    print(f"  import {m}: ok")
PY

echo "megatron stack ready: $DEST"
echo "  venv: $VENV"
echo "  the container image is NOT built by this script; see lm_scaling/README.md"
