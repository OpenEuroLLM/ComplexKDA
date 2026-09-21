"""Every arm's Megatron parameter count, against the matched table.

    lm_scaling/megatron_ext/check_param_match.py megatron_arm_smoke

Run ON the cluster, after a smoke round. Reads the count Megatron itself
prints -- "number of parameters on (tensor, pipeline) model parallel rank
(0, 0): N" -- and compares it to `ladder_matched.json`, the table the
torchtitan flavors are generated from.

WHY THIS IS A CHECK AND NOT A CURIOSITY. The arms are parameter-matched
through their MLP width so that a difference between them is the mixer and not
the size. Every defect this ladder has had in its layers showed up here first,
and only here:

    hybrid attention built ungated       -3 x d_model^2   (442,368 at 47M)
    pre-mixer norm deleted by the spec   -n_layers x d_model (4,608 at 47M)

Both trained. Both fell. Neither raised anything. A count that is 0.01% off is
small enough to read as rounding, which is exactly why it needs a tool that
says the expected number out loud.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RUNS = "/e/scratch/e-sta-openeurollm/poeppel1/complex_kda_runs"
_REPO = Path(__file__).resolve().parents[2]
_N = re.compile(r"number of parameters on \(tensor, pipeline\) model parallel "
                r"rank \(0, 0\): (\d+)")
_LOSS = re.compile(r"iteration +(\d+)/ *\d+ .*lm loss: ([\d.E+]+)")
_VAL = re.compile(r"lm loss value: ([\d.E+]+)")
#: The geometry key an arm's parameter match is taken from. The two attention
#: arms share one, because the matched table has no `attn-qknorm` row.
GEOM = {"attn": "attn", "attn-qknorm": "attn"}
#: Parameters an arm carries that its table row does not, with the arithmetic.
#: Declared rather than tolerated: a blanket slack wide enough to cover this
#: would also cover a missing norm (n_layers x d_model = 4,608 at 47M), and
#: that is the defect this tool exists to catch.
def _extra(arm, table):
    if arm != "attn-qknorm":
        return 0
    # One RMSNorm over head_dim on each of q and k, per layer.
    return 2 * int(table["head_dim"]) * int(table["n_layers"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("group")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--rung", default="47M")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="parameters of slack allowed before a row fails")
    a = ap.parse_args(argv)

    table = json.load(open(_REPO / "lm_scaling" / "ladder_matched.json"))[a.rung]
    base = Path(a.runs) / a.group
    rows, bad = [], 0
    for run in sorted(base.iterdir()) if base.is_dir() else []:
        if not run.is_dir():
            continue
        logs = sorted(run.glob("slurm-*.log"))
        if not logs:
            continue
        text = logs[-1].read_text()
        arm = run.name.replace("ckda_", "").split(f"_{a.rung}")[0]
        key = GEOM.get(arm, arm)
        if key not in table["archs"]:
            print(f"  {arm}: no entry in the matched table", file=sys.stderr)
            continue
        want = int(table["archs"][key]["N"]) + _extra(arm, table)
        got = int(_N.findall(text)[-1]) if _N.search(text) else 0
        loss = _LOSS.findall(text)
        val = _VAL.findall(text)
        ok = got and abs(got - want) <= a.tolerance
        bad += not ok
        rows.append((ok, arm, want, got, loss[-1][1] if loss else "-",
                     val[-1] if val else "-"))

    print(f"{'':2s}{'arm':<30s}{'expected':>13s}{'megatron N':>13s}{'diff':>9s}"
          f"  {'last loss':<13s}{'validation':<13s}")
    for ok, arm, want, got, loss, val in rows:
        print(f"{'  ' if ok else '!!'}{arm:<30s}{want:>13,}{got:>13,}"
              f"{got - want:>9,}  {loss:<13s}{val:<13s}")
    print(f"\n{len(rows)} arms, {bad} off the matched table")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
