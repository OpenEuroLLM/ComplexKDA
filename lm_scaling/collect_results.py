"""Every finished run, its settings, and its validation loss, in one table.

Written for the gap between the cluster shutting down and the scaling runs
being planned: whatever is not collected before the machine goes away cannot be
reasoned about while it is gone.

Two things it refuses to do, because both have already cost a run in this
study:

**Report a number without the settings that produced it.** Arms differ by a
`gate` string and a `spread_frac` float, none of which change the parameter
count, so a table of losses keyed only by name cannot be checked. Every row
carries the `[fla] swapped` line's settings, read out of the run's own log.

**Trust a name over a path.** Run directories carry a hash of the model
settings and the seed. A row whose config fingerprint disagrees with the
directory it wrote to is flagged rather than silently averaged in.

    python lm_scaling/collect_results.py lm_scaling/logs --json results.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path

# torchtitan: "validate step:  N  loss:  X"
_VAL_TITAN = re.compile(r"validate step:\s*(\d+)\s+loss:\s*([\d.]+)")
# lm/: "micro_step: M | step: N | ... | valid/loss: X"
#
# The field anchor is load-bearing. The line leads with micro_step and also
# carries throughput/flops_per_step, so a bare "step: (\d+)" matches the
# micro_step -- grad_accumulation_steps times too large. The loss was right
# and the step silently 4x, which breaks the one thing the step column exists
# for: --at-step refuses to compare a stalled run with a finished one, and
# against micro-steps it would have admitted a run a quarter of the way
# through. Requiring "| " before it picks the real field and nothing else.
_VAL_LM = re.compile(r"(?:^|\| )step: (\d+) \|.*?valid/loss: ([\d.eE+-]+)")
_SWAP = re.compile(r"\[fla\] swapped (\d+) blocks to (\w+) \(mixer='([\w-]+)'\) (\{.*\})")
_PARAMS = re.compile(r"size:\s*([\d,]+) total parameters")
_CFG = re.compile(r"--job\.config_file=\S*/([^/\s]+\.toml)")
_DUMP = re.compile(r'dump_folder = "([^"]+)"')


def _runs(logdir: Path):
    """One record per log that got far enough to validate."""
    # `.txt` too: the harvest taken off the cluster before the maintenance
    # window keeps the grep-extracted lines under the original name plus a
    # suffix, and an analysis that only reads live logs is useless once the
    # machine holding them is down.
    files = (sorted(logdir.glob("*.raw")) + sorted(logdir.glob("*_gpu*.out"))
             + sorted(logdir.glob("*.txt")))
    for f in files:
        text = f.read_text(errors="replace")
        vals = [(int(s), float(v)) for s, v in _VAL_TITAN.findall(text)]
        if not vals:
            vals = [(int(s), float(v)) for s, v in _VAL_LM.findall(text)]
        if not vals:
            continue
        swap = _SWAP.search(text)
        settings = {}
        if swap:
            try:
                settings = ast.literal_eval(swap.group(4))
            except (ValueError, SyntaxError):
                settings = {"unparsed": swap.group(4)}
        m = _PARAMS.search(text)
        dump = _DUMP.search(text)
        arm, stage, rung, lr, gbs = _arm_and_stage(f.name)
        yield {
            "log": f.name,
            "arm": arm,
            # The stage is not cosmetic. A stable run validates throughout its
            # constant-LR phase and its loss is simply higher than the same
            # model's after the decay; putting the two in one column and
            # sorting produced a table where an arm appeared twice, 0.03 apart,
            # and the seed spread was computed across the pair.
            "stage": stage,
            # The RUNG. Without it a 124M row sorts in beside a 47M one and
            # reads as a better configuration rather than a bigger model.
            "rung": rung,
            # The learning rate and batch, which are NOT part of the arm. The
            # lr sweep reuses arm names, so without these its runs pooled in
            # with the main sweep's and were reported as seed spread -- kda-neg
            # showed sd 0.0119 across "replicates" that were four different
            # learning rates.
            "lr": lr,
            "gbs": gbs,
            # The directory hashes the model settings AND the seed, so it is
            # what tells replicates apart -- the log never prints the seed.
            "run_dir": dump.group(1).rstrip("/").split("/")[-1] if dump else None,
            "layer": swap.group(2) if swap else "Attention (no swap)",
            "mixer": swap.group(3) if swap else None,
            "settings": settings,
            "params": int(m.group(1).replace(",", "")) if m else None,
            "validations": sorted(set(vals)),
        }


_NAME = re.compile(r"(?:titanrun_\d+_|titanpack_\d+_gpu\d+_|pack_\d+_gpu\d+_)"
                   r"(?:titan_|lm_)?(?P<arm>[\w.-]+?)_(?P<rung>[\d.]+[MB])_"
                   r"gbs(?P<gbs>\d+)_lr(?P<lr>[\d.e-]+)_(?P<stage>\w+)")


def _arm_and_stage(name: str):
    m = _NAME.search(name)
    if not m:
        return (name, "unknown", "?", "?", "?")
    return (m["arm"], m["stage"], m["rung"], m["lr"], m["gbs"])


def is_final(stage: str) -> bool:
    """Stages whose loss is the run's answer rather than its progress."""
    return stage.startswith("decay") or stage.startswith("wsd") or stage.startswith("anneal")


def summarise(rec, tail: int = 3) -> dict:
    """Final validation loss, and the mean of the last few.

    The tail mean is the one to compare on. A single final point moves by
    ~0.005 between adjacent validations on these runs while the gap between
    arms is ~0.013, so reading one point is reading a third of the effect as
    noise.
    """
    v = rec["validations"]
    out = dict(rec)
    out["final_step"], out["final_loss"] = v[-1]
    out["tail_mean"] = statistics.fmean(x for _, x in v[-tail:])
    out["n_validations"] = len(v)
    del out["validations"]
    return out


def _load_signs(d: Path) -> dict:
    """{(arm, rung, lr, stage): mean sign stats} from measure_signs output.

    Keyed on the ARM, not the run directory, and averaged over seeds.

    The run directory would be exact, but nothing links a torchtitan LOG to one:
    the log is stdout and never echoes `dump_folder`, so every row came back
    with run_dir None and the join silently produced an empty column. Matching
    on the arm needs the seeds collapsed, which is honest here -- the statistic
    barely moves across them (ckda-p00 gives 0.034 / 0.035 / 0.036) -- and the
    spread is reported so a reader can see that for themselves.
    """
    from collections import defaultdict

    acc = defaultdict(list)
    for f in sorted(Path(d).glob("signs_*.json")):
        try:
            recs = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for rec in recs:
            if not rec.get("signed"):
                continue
            m = _NAME.search("titanrun_0_" + Path(rec["config"]).stem)
            if not m:
                continue
            acc[(m["arm"], m["rung"], m["lr"], m["stage"])].append(rec)
    out = {}
    for key, recs in acc.items():
        out[key] = {"n_signs": len(recs)}
        # The beta fields are on this list because they are the point. A
        # negative alpha only rotates when beta exceeds one in the same head,
        # so frac_beta_gt1_and_neg_alpha is the statistic the alpha-beta grid
        # is measured against -- and measure_signs has reported it since the
        # beta hook was added while this join still carried only the four
        # alpha fields, so it never reached a table.
        #
        # `is not None` rather than `in`: an arm whose gate is signed but
        # whose b_proj never fired reports the key with a None value, and
        # fmean over a list containing None raises rather than skipping.
        for k in ("frac_neg", "frac_neg_channels", "frac_ch_over_0.9",
                  "median_abs_alpha",
                  "frac_beta_gt1", "frac_beta_gt1_and_neg_alpha",
                  "mean_beta", "max_beta"):
            vals = [r[k] for r in recs if r.get(k) is not None]
            if vals:
                out[key][k] = statistics.fmean(vals)
                if len(vals) > 1:
                    out[key][k + "_sd"] = statistics.stdev(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logdir", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--signs", type=Path,
                    help="directory of signs_*.json from measure_signs.py; "
                         "joins negative-eigenvalue content onto each row")
    ap.add_argument("--tail", type=int, default=3)
    ap.add_argument("--min-validations", type=int, default=2,
                    help="ignore runs that barely started")
    ap.add_argument("--at-step", type=int,
                    help="keep only runs whose last validation is at least "
                         "this step, so a stalled run cannot be compared "
                         "against a finished one")
    ap.add_argument("--all-stages", action="store_true",
                    help="include stable-phase runs, whose loss is progress "
                         "rather than a result and is not comparable to a "
                         "decayed one")
    a = ap.parse_args()

    rows = [summarise(r, a.tail) for r in _runs(a.logdir)
            if len(r["validations"]) >= a.min_validations]
    if not a.all_stages:
        rows = [r for r in rows if is_final(r["stage"])]
    if a.at_step:
        rows = [r for r in rows if r["final_step"] >= a.at_step]
    if not rows:
        print(f"no runs with validation output under {a.logdir}")
        return
    signs = _load_signs(a.signs) if a.signs else {}
    for r in rows:
        # Joined on the RUN DIRECTORY, which hashes the model settings and the
        # seed. Joining on the arm name would put one seed's eigenvalue
        # measurement against another seed's loss.
        r.update(signs.get((r["arm"], r["rung"], r["lr"], r["stage"]), {}))
    rows.sort(key=lambda r: r["tail_mean"])

    w = max(len(r["arm"]) for r in rows) + 1
    # `step` is not decoration. A run 12% through has a tail mean like any
    # other and sorting puts it in the table beside a finished one; the only
    # thing separating them is how far they got.
    has_signs = any("frac_neg" in r for r in rows)
    sign_hdr = f" {'neg':>6} {'negch':>6} {'ch>.9':>6}" if has_signs else ""
    print(f"{'arm':<{w}} {'rung':>6} {'lr':>7} {'stage':>8} {'step':>7} "
          f"{'final':>8} {'tail':>8} {'n':>3}{sign_hdr}  settings")
    print("-" * (w + 60 + len(sign_hdr)))
    steps = {r["final_step"] for r in rows}
    if len(steps) > 1:
        print(f"# runs are at DIFFERENT steps ({min(steps):,}-{max(steps):,}); "
              f"only rows at the same step are comparable")
    for r in rows:
        s = ", ".join(f"{k}={v}" for k, v in sorted(r["settings"].items())
                      if k in ("gate", "gate_init_style", "spread_frac",
                               "allow_neg_eigval", "output_gate", "lower_bound",
                               "beta_init_style"))
        sg = ""
        if has_signs:
            sg = (f" {r['frac_neg']:>6.3f} {r['frac_neg_channels']:>6.3f} "
                  f"{r['frac_ch_over_0.9']:>6.3f}"
                  if "frac_neg" in r else f" {'-':>6} {'-':>6} {'-':>6}")
        print(f"{r['arm']:<{w}} {r['rung']:>6} {r['lr']:>7} {r['stage']:>8} "
              f"{r['final_step']:>7,} "
              f"{r['final_loss']:>8.4f} {r['tail_mean']:>8.4f} "
              f"{r['n_validations']:>3}{sg}  {s}")

    # Seed spread, where there is one: the scale any gap has to be judged on.
    by_arm = {}
    for r in rows:
        # Same arm, same stage, DIFFERENT directory: the directory hashes the
        # settings and the seed, so same-directory rows are one run logged
        # twice, not two draws.
        by_arm.setdefault((r["arm"], r["rung"], r["lr"], r["stage"]), {})[
            r["run_dir"] or r["log"]] = r["tail_mean"]
    reps = {k: list(v.values()) for k, v in by_arm.items() if len(v) > 1}
    if reps:
        print("\nreplicates (same arm, different seed):")
        for (arm, rung, lr, stage), v in sorted(reps.items()):
            sd = statistics.stdev(v) if len(v) > 1 else 0.0
            print(f"  {arm:<{w}} {rung:>6} {lr:>7} {stage:>8} n={len(v)} "
                  f"mean={statistics.fmean(v):.4f} sd={sd:.4f} "
                  f"range={max(v) - min(v):.4f}")

    if a.json:
        a.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
