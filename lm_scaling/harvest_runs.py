"""Every finished run of a campaign, as one row: settings, schedule, endpoint.

    lm_scaling/login_run lm_scaling/harvest_runs.py fwedu_1p3B \
        --out lm_scaling/harvest/fwedu_1p3B.tsv

This is the TORCHTITAN harvester, and `fwedu_1p3B` is the torchtitan campaign
in this release. The scaling ladder runs on Megatron and is harvested by
`megatron_ext/harvest_megatron.py` into `harvest/megatron_ladder.tsv`.

Runs ON the cluster, because the run directories are there and the logs are
large. Reads each run's own `config.yaml` rather than re-deriving anything from
the name: the name carries the budget and the batch, but not `actual_tokens`,
and the realised token count is the budget rounded to a whole number of steps.

WHAT IT REFUSES TO DO:

**Mix schedules without saying so.** The lower ladder's cells warm up, hold and
decay on their own. The medium rungs branch their cooldowns off a shared stable
trunk, which this study has measured at 0.004-0.016 nats BELOW a single-stage
run of the same budget. Both belong in the table; a fit that cannot see which is
which puts that systematic straight into the D direction and steepens `beta`.
So `schedule` is a column, and `branch_step > 0` is what decides it -- the run's
own record of where it resumed, not a guess from the campaign name.

**Report a stable trunk as an endpoint.** `stable*B` stages stop at constant
learning rate, above where the same run lands after its cooldown. They exist so
the decay branches have something to fork from. They are skipped.

**Report a run that has not reached its last step.** torchtitan validates on
`step % freq == 0` and, since titan_ext's `build_megatron_validator`, on the
final step as well. A run whose last validation is not at `training.steps` is
still going or died; it is listed on stderr and left out of the table.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RUNS = "/e/scratch/e-sta-openeurollm/poeppel1/complex_kda_runs"
# "[titan] ... validate step: N  loss:  X"
_VAL = re.compile(r"validate step:\s*(\d+)\s+loss:\s*([\d.]+)")
# config.yaml's aux block is one level of `key: 'value'` scalars.
_KV = re.compile(r"^  ([a-z_0-9]+): '?([^'\n]*)'?$")

COLS = ("arm", "size", "budget_bt", "gbs", "lr", "steps", "tokens", "loss",
        "schedule")
CURVE_COLS = ("arm", "size", "budget_bt", "gbs", "step", "tokens", "loss")


def aux_of(run: Path) -> dict:
    """The `aux:` block of the run's own resolved config."""
    out, inside = {}, False
    for line in (run / "config.yaml").read_text().splitlines():
        if line.startswith("aux:"):
            inside = True
            continue
        if inside:
            if line and not line.startswith("  "):
                break
            m = _KV.match(line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def trajectory(run: Path) -> dict[int, float]:
    """Every validation the run logged, keyed by step.

    A dict rather than a list because a job requeued at the wall re-validates
    the step it resumed from, and the same step must not appear twice on a
    curve. Last writer wins, which is the later of two identical evaluations.
    """
    out: dict[int, float] = {}
    for log in sorted(run.glob("slurm-*.log")) + [run / "current.log"]:
        if not log.exists():
            continue
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        for step, loss in _VAL.findall(text):
            out[int(step)] = float(loss)
    return out


def endpoint(run: Path) -> tuple[int, float] | None:
    """The last validation in any of the run's logs, and its step.

    Every log, because a job requeued at the wall writes a new `slurm-*.log`
    and the final step lands in the last one. Taking the max over steps rather
    than the last line of the newest file also survives the restart that
    re-validates an earlier step on resume.
    """
    t = trajectory(run)
    return max(t.items()) if t else None


def curves_for(campaign: Path):
    """Every validation of every run, as (step, tokens, loss).

    Includes the STABLE trunks, unlike the endpoint table: a chained run's
    trajectory is the trunk's plus its own cooldown, and the trunk is where
    most of it happened. Step 1 is kept here and dropped at plot time -- the
    untrained model sits near 10.85 and would own any axis it appears on.
    """
    for run in sorted(p for p in campaign.iterdir() if p.is_dir()):
        if not (run / "config.yaml").exists():
            continue
        a = aux_of(run)
        tps = int(a["tok_per_step"])
        for step, loss in sorted(trajectory(run).items()):
            yield dict(arm=a["model"], size=a["size"],
                       budget_bt=re.sub(r"[^0-9]", "", a.get("stage_tag", "")) or "0",
                       gbs=a["gbs"], step=step, tokens=step * tps,
                       loss=f"{loss:.4f}")


def rows_for(campaign: Path):
    for run in sorted(p for p in campaign.iterdir() if p.is_dir()):
        if not (run / "config.yaml").exists():
            continue
        a = aux_of(run)
        stage = a.get("stage_tag", "")
        if stage.startswith("stable"):
            continue
        end = endpoint(run)
        if end is None:
            print(f"no validation: {run.name}", file=sys.stderr)
            continue
        step, loss = end
        want = int(a["steps"])
        if step != want:
            print(f"unfinished ({step}/{want}): {run.name}", file=sys.stderr)
            continue
        tokens = int(a.get("actual_tokens") or want * int(a["tok_per_step"]))
        yield dict(
            arm=a["model"], size=a["size"],
            # The NOMINAL budget: `decay30B` and `wsd30B` are both 30.
            budget_bt=re.sub(r"[^0-9]", "", stage) or str(round(tokens / 1e9)),
            gbs=a["gbs"], lr=a["lr"], steps=want, tokens=tokens, loss=f"{loss:.4f}",
            schedule="chain" if int(a.get("branch_step") or 0) else "single",
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("campaigns", nargs="+")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--curves", action="store_true",
                    help="every validation of every run, not just the endpoints")
    a = ap.parse_args(argv)

    cols = CURVE_COLS if a.curves else COLS
    rows = []
    for name in a.campaigns:
        d = Path(a.runs) / name
        if not d.is_dir():
            print(f"no such campaign: {d}", file=sys.stderr)
            continue
        n = len(rows)
        rows.extend(curves_for(d) if a.curves else rows_for(d))
        print(f"{name}: {len(rows) - n} "
              f"{'points' if a.curves else 'endpoints'}", file=sys.stderr)

    rows.sort(key=lambda r: (r["arm"], r["size"], int(r["budget_bt"]),
                             int(r.get("step", 0))))
    text = "\t".join(cols) + "\n" + "".join(
        "\t".join(str(r[c]) for c in cols) + "\n" for r in rows)
    if a.out:
        a.out.write_text(text)
        print(f"wrote {len(rows)} rows to {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
