#!/bin/bash
# Regenerate every table and figure in `tex/` from the committed measurements.
#
#   lm_scaling/reproduce_tables.sh          # rewrite tex/ in place
#   lm_scaling/reproduce_tables.sh --check  # write to a scratch dir and diff
#
# NO CLUSTER, NO GPU, NO DOWNLOADS. Everything read here is in the repository:
# `harvest/megatron_ladder.tsv` (180 ladder cells), `harvest/ladder_downstream/`
# (180 eval JSONs), `harvest/downstream/` (the four 1.3B arms),
# `harvest/ruler/` (10 RULER files), `harvest/gate_spectrum/`, and the two
# geometry tables `ladder_matched.json` and `replication_1p3B.json`. Those are
# the expensive part of the paper; the scripts below are the cheap part.
#
# AND NO ORCHESTRATION LAYER. numpy, scipy, matplotlib, pyyaml and the one
# entry of `requirements-experiments.txt` (scienceplots, for the figure style)
# are the whole dependency list; `hydra_staged_sweep` and the rest of the login
# venv are for submitting the campaigns, not for reading them back.
#
# The FineWeb-Edu column of the architecture table used to be the exception: it
# resolved its recipe through the hydra planner, so this script failed with
# `ModuleNotFoundError` on any checkout that had not installed the submitter.
# That recipe now ships as `campaign_recipes.json`; see `arch_table.fixed_recipe`.
#
# --check is the one that matters for a fork: if it reports no differences,
# every number in `tex/` is exactly what the committed data produces, and each
# can be traced to a file here rather than taken on trust.
#
# NOT INSTANT, and the fitting is nearly all of it. Each fit variant does
# `--starts` random restarts for the point fit plus `--boot-starts` for every
# bootstrap replicate, so the floor is STARTS and not the replicate count: at
# the defaults that is 600 + 200x6 = 1,800 fits a variant, and the holdout
# refits every variant a second time on the reduced ladder.
#
#   (defaults, the paper's)                          tens of minutes
#   STARTS=40 BOOT_STARTS=2 THREE_REPS=20 REPS=20    a couple of minutes
#
# The fast setting exercises every code path and writes every file, and the
# well-determined numbers survive it -- the shared E comes out at 1.1852 either
# way. What it does move is the intervals, and the parameters the tables
# already flag as poorly determined: the shared beta of the pinned-beta
# variants goes from 0.3596 to 1.2476 when the restarts drop from 600 to 40,
# because 40 restarts do not find that optimum.
#
# So: the fast setting checks that the pipeline runs. The defaults check the
# numbers. Do not read a fast run's fit tables as results.
#
# EVERYTHING IN `tex/` IS GENERATED, so --check exempts nothing: a table this
# script does not write is a table nothing here can vouch for, and --check
# reports it rather than skipping it. Prose that is written by hand belongs in
# the paper, not in this directory.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
L=lm_scaling

OUT="$L/tex"
CHECK=0
SCRATCH=""
if [ "${1:-}" = "--check" ]; then
  CHECK=1
  SCRATCH="$(mktemp -d)"; OUT="$SCRATCH/tex"; mkdir -p "$OUT"
  # DELIBERATELY EMPTY. Seeding it with the committed tables was the first
  # version and it is exactly wrong: a generator that silently writes nothing
  # would leave the seeded copy in place and the diff would report a match. An
  # empty directory makes a file that was not written show up as missing, which
  # is what it is.
fi

REPS="${REPS:-400}"
THREE_REPS="${THREE_REPS:-200}"
STARTS="${STARTS:-600}"
BOOT_STARTS="${BOOT_STARTS:-6}"

run() { echo "  $*"; "$@" || { echo "FAILED: $*" >&2; exit 1; }; }
# Same, for a generator that writes its file and also prints a report.
quiet() { echo "  $*"; "$@" >/dev/null || { echo "FAILED: $*" >&2; exit 1; }; }
# Write through a temporary, because `> "$o"` truncates BEFORE the command
# runs: a generator that fails would leave an empty table behind, and in
# --write mode that table is the committed one.
cap() { local o="$1"; shift; echo "  $* > $(basename "$o")"
        "$@" > "$o.tmp" || { rm -f "$o.tmp"; echo "FAILED: $*" >&2; exit 1; }
        mv "$o.tmp" "$o"; }

# NOT RUN: `scaling_fit.py --latex`. Its in-sample tables (the per-arm fit,
# the joint fit, the per-cell losses) are not part of the paper -- the reported
# fits are held out, and the per-cell losses ship as scaling_data.tex with the
# reference column. It remains the library the holdout and the variants fit
# through.
# NOT RUN: `scaling_fit_variants.py`, the in-sample counterpart. It fits the
# same two variants on all 180 cells, and the paper does not report those --
# an in-sample residual says how well a form describes the ladder, not whether
# it predicts a cell it has not seen. Its tables are therefore not in `tex/`,
# and running it here would write four files the --check below would then have
# to call uncommitted. The holdout quotes it as a control, which is where those
# numbers appear.
echo "== the two fit variants, with the largest cell held out =="
# SKALING WRITES TABLES, CHINCHILLA DOES NOT. The paper reports the Skaling form
# only, so its three tables are the ones in `tex/`. Chinchilla is still fitted,
# because the third panel of the holdout figure is the comparison that justifies
# reporting Skaling -- held-out RMSE 0.0015 against 0.0258 -- and that panel
# reads `scaling_holdout_chinchilla.json`, which `--no-tables` still writes.
run $PY "$L/scaling_holdout.py" --form skaling --reps "$THREE_REPS" \
    --starts "$STARTS" --boot-starts "$BOOT_STARTS" --out "$OUT"
run $PY "$L/scaling_holdout.py" --form chinchilla --no-tables --reps "$THREE_REPS" \
    --starts "$STARTS" --boot-starts "$BOOT_STARTS" --out "$OUT"

echo "== the holdout figures =="
run $PY "$L/holdout_plot.py" --json "$OUT/scaling_holdout.json" \
    --other "$OUT/scaling_holdout_chinchilla.json" --out "$OUT"

echo "== scaling figures =="
run $PY "$L/scaling_plots.py" --out "$OUT"

echo "== the ladder's validation losses, against Ajroldi et al. =="
# `--latex PATH` writes the file and ALSO prints the 30-cell terminal table;
# only the file is wanted here.
quiet $PY "$L/megatron_ext/compare_table.py" --latex "$OUT/scaling_data.tex"

echo "== ladder downstream: 180 cells, paired against the loss table =="
for m in avg9 lmb wiki; do
  run $PY "$L/ladder_downstream.py" --latex "$OUT/downstream_$m.tex" --latex-metric "$m"
done

echo "== the 1.3B arms: downstream, both reporting conventions =="
cap "$OUT/fwedu_1p3B.tex" \
    $PY "$L/eval_table.py" --step 190976 --convention gdn2 --format paper
cap "$OUT/fwedu_1p3B_table_gdn.tex" \
    $PY "$L/eval_table.py" --step 190976 --convention gdn --format latex
cap "$OUT/fwedu_1p3B_table_gdn2.tex" \
    $PY "$L/eval_table.py" --step 190976 --convention gdn2 --format latex
# The same table with Based's recall suite appended, from harvest/recall.
cap "$OUT/fwedu_1p3B_table_gdn2_recall.tex" \
    $PY "$L/eval_table.py" --step 190976 --convention gdn2 --format latex --with-recall

echo "== the 1.3B arms: RULER needle retrieval =="
cap "$OUT/fwedu_1p3B_niah.tex" \
    $PY "$L/niah_table.py" --format latex --step 190976

echo "== the architecture table =="
cap "$OUT/arch_table.tex" $PY "$L/arch_table.py"

echo "== the gate spectrum: table, then six figures =="
cap "$OUT/gate_spectrum.tex" $PY "$L/gate_spectrum_plot.py" --latex
run $PY "$L/gate_spectrum_plot.py" --out "$OUT"

if [ "$CHECK" = 1 ]; then
  echo
  echo "== diff against the committed tex/ =="
  # Text only. The figures are matplotlib PDFs and PNGs, which differ byte for
  # byte between runs -- embedded timestamps and font subsetting -- while
  # drawing the same thing. Comparing them would report a difference every
  # time and mean nothing by it. The tables carry the numbers.
  bad=0
  # Walk the COMMITTED tables, not the regenerated ones, so a file that should
  # have been written and was not is reported rather than skipped.
  for f in "$L"/tex/*.tex; do
    b="$(basename "$f")"
    if [ ! -f "$OUT/$b" ]; then
      echo "  NOT WRITTEN  $b"; bad=1
    elif ! diff -q "$f" "$OUT/$b" >/dev/null 2>&1; then
      echo "  DIFFERS  $b"; diff "$f" "$OUT/$b" | head -8; bad=1
    fi
  done
  # And anything regenerated that is not committed: a new table nobody added.
  for f in "$OUT"/*.tex; do
    b="$(basename "$f")"
    [ -f "$L/tex/$b" ] || { echo "  UNCOMMITTED  $b"; bad=1; }
  done
  [ "$bad" = 0 ] && echo "  no differences: tex/ is exactly what harvest/ produces"
  rm -rf "$SCRATCH"
  exit "$bad"
fi
