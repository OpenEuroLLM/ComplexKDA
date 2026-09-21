"""What a running campaign is doing, in one screen.

    lm_scaling/login_run lm_scaling/watch.py fwedu_1p3B
    lm_scaling/login_run lm_scaling/watch.py fwedu_1p3B --loop 60
    lm_scaling/login_run lm_scaling/watch.py fwedu_1p3B --errors

The campaign's own config says which runs exist and how many steps each is
meant to take, so this reads the plan rather than globbing a directory and
guessing. That is the difference between "step 23040" and "step 23040 of
23040, done".

The ETA is computed from the rate the job is CURRENTLY achieving rather than
from the throughput table, because the interesting case is the one where they
disagree.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# torchtitan's metrics line, and the validation line that comes from the same
# call that writes `validation_metrics/loss` to tensorboard.
_STEP = re.compile(
    r"(?<!micro_)\bstep:\s*(\d+).*?loss:\s*([\d.]+).*?"
    r"memory:\s*([\d.]+)GiB.*?tps:\s*([\d,]+).*?mfu:\s*([\d.]+)%")
_VALID = re.compile(r"validate step:\s*(\d+)\s+loss:\s*([\d.]+)")
# `micro_step:` leads the metrics line and a bare `step:` matches THAT, which
# once made every collected step count 4x too large.


def squeue() -> dict[str, str]:
    """Job name -> SLURM state, when squeue is reachable."""
    try:
        out = subprocess.run(["squeue", "-u", subprocess.os.environ.get("USER", ""),
                              "-h", "-o", "%j|%T"], capture_output=True,
                             text=True, timeout=20).stdout
    except Exception:
        return {}
    states = {}
    for line in out.splitlines():
        if "|" in line:
            name, state = line.rsplit("|", 1)
            states[name.strip()] = state.strip()
    return states


def latest(run: Path) -> tuple[dict | None, tuple[int, float] | None]:
    """The last metrics line and the last validation loss in a run directory.

    Reads the newest log only. A requeued job writes a new one, and the last
    line of the previous log is by definition where it was killed.
    """
    logs = sorted(run.glob("slurm-*.log"))
    if not logs:
        return None, None
    text = logs[-1].read_text(errors="ignore")
    step = val = None
    for mo in _STEP.finditer(text):
        step = {"step": int(mo.group(1)), "loss": float(mo.group(2)),
                "mem": float(mo.group(3)),
                "tps": float(mo.group(4).replace(",", "")),
                "mfu": float(mo.group(5))}
    for mo in _VALID.finditer(text):
        val = (int(mo.group(1)), float(mo.group(2)))
    return step, val


def render(experiment: str, overrides: list[str], errors: bool) -> None:
    import plan as P

    jobs = P.plans(experiment, overrides)
    states = squeue()

    print(f"{'run':<52}{'state':>10}{'step':>16}{'%':>6}"
          f"{'loss':>8}{'valid':>8}{'mfu':>7}{'eta':>9}")
    running = done = 0
    for j in jobs:
        c, a = j.config, j.config.aux
        target = int(a["steps"])
        step, val = latest(Path(c.job.base_output_dir))
        state = states.get(c.job.name, "-")

        if step is None:
            print(f"{c.job.name:<52}{state:>10}{'not started':>16}")
            continue

        pct = 100.0 * step["step"] / target
        # tps is per GPU; a step consumes gbs x seq_len tokens across the world.
        world_tps = step["tps"] * int(a["world_gpus"])
        steps_s = world_tps / (int(a["gbs"]) * int(c.data.seq_len))
        eta_h = (target - step["step"]) / steps_s / 3600 if steps_s else 0.0

        if step["step"] >= target:
            done += 1
        elif state in ("RUNNING", "R"):
            running += 1

        print(f"{c.job.name:<52}{state:>10}"
              f"{step['step']:>9,}/{target:<6,}{pct:>5.0f}%"
              f"{step['loss']:>8.4f}"
              f"{(f'{val[1]:.4f}' if val else '-'):>8}"
              f"{step['mfu']:>6.1f}%{eta_h:>8.1f}h")

        if errors:
            for line in _problems(Path(c.job.base_output_dir)):
                print(f"    ! {line}")

    print(f"\n{len(jobs)} runs: {done} finished, {running} running")


def _problems(run: Path) -> list[str]:
    logs = sorted(run.glob("slurm-*.log"))
    if not logs:
        return []
    pat = re.compile(r"traceback|error:|out of memory|ChildFailedError|"
                     r"CANCELLED|TIME LIMIT", re.I)
    seen = []
    for line in logs[-1].read_text(errors="ignore").splitlines():
        if pat.search(line) and line.strip() not in seen:
            seen.append(line.strip()[:120])
    return seen[:3]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--loop", type=float, metavar="SECONDS",
                    help="refresh until interrupted")
    ap.add_argument("--errors", action="store_true",
                    help="show the failure lines in each run's log")
    args = ap.parse_args(argv)

    while True:
        if args.loop:
            print("\033[2J\033[H", end="")
        print(time.strftime("%Y-%m-%d %H:%M:%S"), f" {args.experiment}\n")
        render(args.experiment, args.override, args.errors)
        if not args.loop:
            return 0
        time.sleep(args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
