"""Every finished Megatron cell of a campaign, as one row of the ladder TSV.

    lm_scaling/megatron_ext/harvest_megatron.py megatron_ladder megatron_ladder_beta \
        --out lm_scaling/harvest/megatron_ladder.tsv

Run ON the cluster. Emits the same columns as `lm_scaling/harvest_runs.py`
so the two ladders can be read by the same fits -- the whole point of
re-running in Megatron is to compare the shapes, and that needs one table
format, not two.

WHERE THE NUMBERS COME FROM. Each run directory holds `config-<jobid>.yaml`,
autoexp's fully resolved config, which carries the arm, the batch and the
budget in its `aux` block and the schedule in its `backend.megatron` block.
The loss comes from the log, from the line Megatron writes after its last
iteration:

    validation loss at iteration 45777 on validation set | lm loss value: 2.922429E+00

WHAT IT REFUSES TO DO, for the same reasons `harvest_runs.py` refuses them:

**Report a trunk segment as an endpoint.** A `stable<i>` job stops at constant
learning rate, above where the same chain lands after its cooldown. Skipped.

**Report a run that did not finish.** The final validation is the one at
`train_iters`; anything else is a periodic eval from a run still going or one
that died. Listed on stderr and left out.

**Mix schedules without saying so.** A cooldown branched off a shared trunk
measures 0.004-0.016 below a single-stage run of the same budget, and a fit
that cannot see which is which puts that straight into beta. `schedule` is a
column, decided by whether the job resumed from a checkpoint -- the config's
own `ckpt_step`, not a guess from the name.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUNS = "/e/scratch/e-sta-openeurollm/poeppel1/complex_kda_runs"
COLS = ("arm", "size", "budget_bt", "gbs", "lr", "steps", "tokens", "loss",
        "schedule")

# "validation loss at iteration N on validation set | lm loss value: X"
_VAL = re.compile(
    r"validation loss at iteration (\d+)[^|]*\| lm loss value:\s*([\d.E+-]+)")
_LR = re.compile(r"learning rate: ([\d.E+-]+)")
#: A WSD run ends at min_lr (1e-5). A cell whose last logged learning rate is
#: still near the peak never annealed: it is a stable-phase loss wearing an
#: endpoint's name, and it is higher than the real endpoint by the whole
#: cooldown. Measured: attn_588M_decay12B resumed from a stale trunk
#: checkpoint, never annealed, and reported 2.5702 where its qk-norm twin --
#: which should be 0.03 WORSE -- reported 2.4051.
#:
#: This is the one failure that survives every other check here. The job exits
#: 0, writes a validation line, and the number is a real loss of a real model.
LR_ANNEALED = 1e-4
#: The blocks we read, as PATHS from the document root. autoexp writes the
#: whole resolved config under a `config:` key, and there is a `backend.aux`
#: as well as the top-level `aux`, so a block cannot be found by its name
#: alone -- the first draft of this matched `backend: aux: {}` and returned
#: nothing, silently, for every run.
AUX = ("config", "aux")
MEGATRON = ("config", "backend", "megatron")
_KV = re.compile(r"^\s*([a-z_0-9]+):\s*'?([^'\n]*?)'?\s*$")

SIZES = ("47M", "124M", "302M", "588M", "983M", "1.7B")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _scalars(path: Path, block: tuple[str, ...]) -> dict:
    """The plain `key: value` lines directly under a dotted block path.

    A hand-written walk rather than a YAML load, because these files carry
    unresolved interpolations that a parser would either choke on or resolve
    against an environment that is not the one the run had.
    """
    lines = path.read_text().splitlines()
    # -2, so the first name is looked for at indent 0: the document root's
    # children are unindented, and starting at -1 looks for them at 1.
    lo, hi, depth = 0, len(lines), -2
    for name in block:
        for i in range(lo, hi):
            if _indent(lines[i]) == depth + 2 and lines[i].strip() == f"{name}:":
                lo, depth = i + 1, depth + 2
                hi = next((j for j in range(i + 1, hi)
                           if lines[j].strip() and _indent(lines[j]) <= depth), hi)
                break
        else:
            return {}
    want = depth + 2
    out = {}
    for line in lines[lo:hi]:
        if _indent(line) != want:
            continue
        m = _KV.match(line)
        if m:
            out.setdefault(m.group(1), m.group(2))
    return out


def rows(group: str, root: Path):
    base = root / group
    if not base.is_dir():
        print(f"no such campaign: {base}", file=sys.stderr)
        return
    for run in sorted(base.iterdir()):
        if not run.is_dir() or run.name == "ckpt":
            continue
        cfgs = sorted(run.glob("config-*.yaml"))
        logs = sorted(run.glob("slurm-*.log"))
        if not cfgs or not logs:
            print(f"skip {run.name}: no config or no log", file=sys.stderr)
            continue
        cfg = cfgs[-1]
        aux = _scalars(cfg, AUX)
        mg = _scalars(cfg, MEGATRON)
        tag = aux.get("additional_tag", "")
        if "_stable" in tag:
            continue

        # THE NEWEST LOG THAT ACTUALLY FINISHED, not simply the newest.
        #
        # A run directory accumulates a log per attempt, and an attempt that
        # was cancelled or died leaves a newer, shorter one. Taking logs[-1]
        # lets an aborted rerun SHADOW a valid earlier result -- which is what
        # happened when a redundant 588M resubmission was cancelled three
        # minutes in and four cells vanished from the table. The result was
        # never lost; it was just behind a newer file.
        want = int(mg.get("train_iters", 0) or 0)
        final = []
        for cand in reversed(logs):
            t = cand.read_text()
            f = [(int(i), float(v)) for i, v in _VAL.findall(t) if int(i) == want]
            if not f:
                continue
            # AND IT HAS TO HAVE ANNEALED, tested here rather than after the
            # loop. Deciding the log first and checking the learning rate
            # second lets a bad newest attempt block a good older one: four
            # 12BT cells vanished from the table when a resubmitted chain
            # produced a cooldown that validated at the right iteration but
            # held peak learning rate throughout, and the earlier good
            # attempts were never consulted.
            lr = _LR.findall(t)
            if lr and float(lr[-1]) > LR_ANNEALED:
                continue
            final = f
            break
        if not final:
            # The two ways to have no usable attempt read very differently to
            # whoever is looking at the table wondering where a cell went.
            newest = logs[-1].read_text()
            hits = _VAL.findall(newest)
            lrs = _LR.findall(newest)
            if any(int(i) == want for i, _v in hits) and lrs:
                print(f"skip {run.name}: validated at {want} but never "
                      f"annealed, last learning rate {float(lrs[-1]):.3E}, "
                      f"so that loss is a stable-phase loss", file=sys.stderr)
            else:
                last = f"last eval at {hits[-1][0]}" if hits else "no eval"
                print(f"skip {run.name}: never validated at {want} ({last})",
                      file=sys.stderr)
            continue
        step, loss = final[-1]

        size = next((s for s in SIZES if f"_{s}" in tag), "?")
        arm = tag.split(f"_{size}")[0]
        gbs = int(aux["gbs"])
        tokens = step * gbs * int(mg.get("seq_length", 4096))
        chained = bool(mg.get("ckpt_step", "").strip() not in ("", "null", "None"))
        # budget_bt is the NOMINAL budget as an integer, because that is what
        # `scaling_fit.load` parses with int() and what harvest_runs.py writes.
        # The realised count is 6.00008BT at a 6BT cell -- the budget rounded
        # up to a whole number of steps -- and it rides in `tokens`, which is
        # the column the fit actually uses for D.
        yield (arm, size, round(tokens / 1e9), gbs, mg.get("lr", "?"),
               step, tokens, f"{loss:.4f}", "chain" if chained else "single")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)

    out = ["\t".join(COLS)]
    n = 0
    for g in a.groups:
        for r in rows(g, Path(a.runs)):
            out.append("\t".join(str(x) for x in r))
            n += 1
    text = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).write_text(text)
        print(f"{n} cells -> {a.out}", file=sys.stderr)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
