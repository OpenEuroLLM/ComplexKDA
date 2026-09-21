"""Find each cell's micro-batch ceiling IN TORCHTITAN, at powers of two.

    lm_scaling/login_run lm_scaling/titan_mbs_probe.py fwedu_1p3B --out DIR
    lm_scaling/login_run lm_scaling/titan_mbs_probe.py fwedu_1p3B --out DIR --collect

`login_run` and not `container_run`: this writes sbatch scripts and the venv
that renders them is the one that can reach slurm. The probes themselves enter
the container, the way every job does.

The ceilings in peak_mbs.yaml were measured with `lm/`'s TorchEngine, and they
do not transfer: at 124M with mbs 16 -- straight out of that table -- attn,
gdn, kda and kda-hybrid all ran out of memory under torchtitan. Different
model construction, different FSDP wrapping, different activation lifetimes.
A ceiling is a property of the stack, not of the architecture.

POWERS OF TWO ONLY. The paper's batch-size grid is 2^4..2^10, so every global
batch on this ladder is a power of two; dividing by 4 GPUs leaves a power of
two per GPU, and a power-of-two micro-batch then divides it exactly with a
power-of-two gradient accumulation. A ceiling of 12 or 20 could not be used
even if it fit.

Each candidate is a short run of the ladder's own job at that micro-batch --
so what is measured is what will run. The geometry is
`config/overlay/mbs.yaml`, which explains the choices; this file enumerates
the candidates, drives the plan and reads the logs back.

NOTHING HERE EDITS A CONFIG. The candidate reaches torchtitan as
`++aux.probe_mbs=<m>`, through the same resolution every other setting takes,
and the directory it lands in is named FROM that key. The version this
replaces deep-copied a resolved config and assigned `local_batch_size` into
it, which is the shape of failure that put `--ntasks=64` with no `--nodes`
into thirty-six rejected sbatch scripts.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import plan as P  # noqa: E402
import submit as S  # noqa: E402


def candidates(per_gpu: int, top: int | None = None) -> list[int]:
    """Powers of two from the largest that could be used, downward.

    Capped at the per-GPU share of the global batch: a micro-batch above it
    cannot be used however well it fits, because gradient accumulation would
    have to be fractional.
    """
    hi = per_gpu if top is None else min(per_gpu, top)
    out, m = [], 1
    while m <= hi:
        out.append(m)
        m *= 2
    return sorted(out, reverse=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=12,
                    help="enough to get past compilation and allocate the "
                         "steady-state activations")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--max-mbs", type=int, default=None, metavar="N",
                    help="do not probe above this micro-batch. The per-GPU "
                         "share of the global batch is the other cap and "
                         "always applies; this one bounds the cost at rungs "
                         "where everything above a small N is a certain OOM.")
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--section", default="gh200_seq4096_titan_g4",
                    help="peak_mbs.yaml section name to print under")
    ap.add_argument("--throughput", action="store_true",
                    help="what each micro-batch COST, from the same logs. "
                         "The ceiling is only half of what these runs "
                         "measured; the other half is that throughput is "
                         "roughly linear in the micro-batch below 4.")
    ap.add_argument("--yaml", action="store_true",
                    help="emit these rates as a throughput.yaml section")
    ap.add_argument("--unit", choices=("tflops", "tps"), default="tflops",
                    help="tflops for throughput.yaml, tps (tokens/s per GPU) "
                         "for token_rate.yaml -- the one that prices a run")
    ap.add_argument("--collect", action="store_true",
                    help="read the ceilings back out of --out instead of "
                         "writing probes")
    args = ap.parse_args(argv)

    if args.throughput:
        return report_throughput(Path(args.out))

    if args.yaml:
        return emit_yaml(Path(args.out), args.section, args.unit)

    if args.collect:
        found, problems = collect(Path(args.out))
        for line in problems:
            print(f"warning: {line}", file=sys.stderr)
        if not found:
            print(f"no completed probes under {args.out}", file=sys.stderr)
            return 1
        print(f"{args.section}:   # largest power of two that reached a step")
        for (arch, rung), mbs in sorted(found.items()):
            print(f"  {arch}|{rung}: {mbs}")
        return 0 if not problems else 2

    root = Path(args.out)

    # The per-GPU share of the global batch is the OTHER cap on a candidate: a
    # micro-batch above it cannot be used however well it fits, because
    # gradient accumulation would have to be fractional. It comes from the
    # campaign's own plan, and it has to be the LARGEST across budgets -- a
    # rung runs several, and taking the first silently stopped 47M at 8, which
    # is the 6BT cell's share, while its 12BT and 20BT cells ask for 16.
    tops: dict[tuple[str, str], int] = {}
    for j in P.plans(args.experiment, args.override):
        a = j.config.aux
        key = (a["model"], a["size"])
        tops[key] = max(tops.get(key, 0), int(a["per_gpu"]))

    written = 0
    for mbs in sorted({m for top in tops.values()
                       for m in candidates(top, args.max_mbs)}):
        jobs = P.plans(args.experiment, [
            "overlay=mbs",
            f"++aux.probe_mbs={mbs}",
            f"++aux.probe_steps={args.steps}",
            f"cluster.runs_dir={root}",
            *args.override,
        ])
        # One config per cell: the ceiling depends on the micro-batch, the
        # model and the context, and the budget changes none of them.
        jobs = P.per_cell(jobs)
        if args.only:
            jobs = [j for j in jobs
                    if any(o in j.config.job.name for o in args.only)]
        # Above a cell's own share this candidate is unusable even if it fits.
        jobs = [j for j in jobs
                if mbs <= tops[(j.config.aux["model"], j.config.aux["size"])]]
        P.check(jobs)
        for j in jobs:
            S.write_job(j.config)
            written += 1
    print(f"{written} probes for {len(tops)} cells under {root}")
    return 0


def collect(root: Path) -> tuple[dict[tuple[str, str], int], list[str]]:
    """Largest micro-batch that reached a training step, per cell.

    Reaching a STEP, not merely starting: torchtitan builds the model and the
    optimizer before the first forward, so a job that dies in the backward has
    still printed a good deal of encouraging log. The step line is the first
    evidence the whole loop fits.

    A ceiling is only believed when the next power of two ACTUALLY RAN OUT OF
    MEMORY. Any other way of not reaching a step -- a job still queued, one
    killed by the wall clock, one that died in setup -- looks exactly like a
    ceiling and is not one, and a ceiling that is too low costs gradient
    accumulation steps on every cell that reads it, forever, in silence. Those
    come back as complaints instead of numbers.
    """
    seen: dict[tuple[str, str], dict[int, str]] = {}
    speed: dict[tuple[str, str], dict[int, float]] = {}
    for d in sorted((root / "mbs").iterdir()) if (root / "mbs").is_dir() else []:
        if not d.is_dir():
            continue
        arch, rung, m = d.name.rsplit("_", 2)
        mbs = int(m.lstrip("m"))
        logs = sorted(d.glob("slurm-*.log"))
        text = logs[-1].read_text(errors="ignore") if logs else ""
        if re.search(r"\bstep:\s*\d+", text):
            verdict = "fits"
            got = read_metrics(logs[-1])
            if got:
                speed.setdefault((arch, rung), {})[mbs] = got[1]
        elif "out of memory" in text.lower() or "OutOfMemoryError" in text:
            verdict = "oom"
        else:
            verdict = "unresolved"
        seen.setdefault((arch, rung), {})[mbs] = verdict

    best: dict[tuple[str, str], int] = {}
    problems: list[str] = []
    for key, got in sorted(seen.items()):
        fits = [m for m, v in got.items() if v == "fits"]
        if not fits:
            problems.append(f"{key[0]}|{key[1]}: nothing fit ({_verdicts(got)})")
            continue
        ceiling = max(fits)

        # THE SMALLEST MICRO-BATCH THAT REACHES FULL THROUGHPUT, not simply
        # the largest that fits. Two different things go wrong above it.
        #
        # It can be catastrophically slower: near the top of the card the
        # allocator spends the step defragmenting and retrying instead of
        # training, and the run SURVIVES, so every "did it fit" test says yes.
        # gdn at 983M reaches a step at mbs 4, using 38.2 GiB of 39.5, at
        # 1.1 TFLOP/s against mbs 2's 167.5; kda at 1.7B manages 0.4 at mbs 2
        # against 128.2 at mbs 1. Taking the larger value would have cost 150x
        # and 320x with nothing in any log saying so.
        #
        # Or it can simply buy nothing: complex-kda at 1.7B on 8 GPUs runs at
        # 117.5 TFLOP/s at mbs 2 and 122.3 at mbs 1, using 34.8 GiB against
        # 19.9. Same speed, nearly twice the memory -- and memory held for no
        # throughput is pure risk in a run that lasts weeks, because
        # fragmentation grows and validation allocates on top.
        #
        # So: the smallest candidate within 3% of the best rate measured. 3%
        # rather than 0 because these are twelve-step probes and a percent or
        # two is noise.
        tf = speed.get(key, {})
        if len(tf) > 1:
            fastest = max(tf.values())
            chosen = min(m for m in tf if tf[m] >= fastest * 0.97)
            if chosen != ceiling:
                problems.append(
                    f"{key[0]}|{key[1]}: using m{chosen} at {tf[chosen]:.1f} "
                    f"TFLOP/s rather than m{ceiling} at "
                    f"{tf.get(ceiling, 0):.1f} -- a larger micro-batch that "
                    "does not buy throughput is memory held for nothing")
                best[key] = chosen
                continue

        above = ceiling * 2
        if above in got and got[above] != "oom":
            problems.append(
                f"{key[0]}|{key[1]}: m{ceiling} fits but m{above} is "
                f"{got[above]}, not an OOM -- ceiling not established")
            continue
        best[key] = ceiling
        if above not in got:
            problems.append(
                f"{key[0]}|{key[1]}: m{ceiling} is the largest PROBED, and "
                "nothing tested above it -- this is a floor, not a ceiling")
    return best, problems


def _verdicts(got: dict[int, str]) -> str:
    return ", ".join(f"m{m}={v}" for m, v in sorted(got.items()))


_METRICS = re.compile(
    r"step:\s*(\d+).*?memory:\s*([\d.]+)GiB\([\d.]+%\).*?"
    r"tps:\s*([\d,]+)\s+tflops:\s*([\d.]+)\s+mfu:\s*([\d.]+)%")
# groups: 1 step, 2 GiB, 3 tps, 4 TFLOP/s, 5 MFU%


def read_metrics(log: Path) -> tuple[float, float, float, float] | None:
    """(peak GiB, TFLOP/s, MFU%, tokens/s per GPU) from a probe log's last steps.

    The last steps rather than all of them: torchtitan's tps/tflops are an
    interval average, and the opening ones carry compilation and the first
    allocator growth.
    """
    import statistics

    seen: dict[int, tuple[float, float, float, float]] = {}
    for line in log.read_text(errors="ignore").splitlines():
        mo = _METRICS.search(line)
        if mo:
            seen[int(mo.group(1))] = (float(mo.group(2)), float(mo.group(4)),
                                      float(mo.group(5)),
                                      float(mo.group(3).replace(",", "")))
    if not seen:
        return None
    late = [v for step, v in seen.items() if step >= max(seen) - 2]
    return tuple(statistics.median(v[i] for v in late) for i in range(4))


def report_throughput(root: Path) -> int:
    """What each micro-batch bought, per cell.

    Worth reading before treating a ceiling as a memory fact and nothing
    else. Measured here on 4x A100-40GB at seq 4096: 302M complex-kda runs at
    15.2% MFU at mbs 1, 30.2% at 2 and 38.5% at 4 -- throughput is very nearly
    LINEAR in the micro-batch below 4, because the chunked linear-attention
    kernels are launch-bound and under-occupied there. Attention is far
    flatter (32.2% -> 46.5% over the same range) since its GEMMs already fill
    the card.

    So a ceiling is not only "will it fit". At the upper rungs it decides the
    MFU, and a cell that only fits mbs 1 is not slightly slower than one that
    fits 4 -- it is 2 to 3 times slower.
    """
    rows: dict[tuple[str, str], dict[int, tuple[float, float, float]]] = {}
    for d in sorted((root / "mbs").iterdir()) if (root / "mbs").is_dir() else []:
        if not d.is_dir():
            continue
        arch, rung, m = d.name.rsplit("_", 2)
        logs = sorted(d.glob("slurm-*.log"))
        got = read_metrics(logs[-1]) if logs else None
        if got:
            rows.setdefault((arch, rung), {})[int(m.lstrip("m"))] = got

    if not rows:
        print(f"no probe logs with metrics under {root}", file=sys.stderr)
        return 1

    widths = sorted({m for cell in rows.values() for m in cell})
    for rung in sorted({r for _, r in rows}, key=_rung_key):
        print(f"\n=== {rung} ===  TFLOP/s per GPU, MFU, peak memory")
        print(f"{'arch':14}" + "".join(f"{'mbs ' + str(m):>23}" for m in widths))
        for arch in sorted({a for a, r in rows if r == rung}):
            cells = []
            for m in widths:
                v = rows[(arch, rung)].get(m)
                cells.append(f"{v[1]:6.1f}TF {v[2]:5.1f}% {v[0]:5.1f}G" if v else "")
            print(f"{arch:14}" + "".join(f"{c:>23}" for c in cells))
    return 0


def emit_yaml(root: Path, section: str, unit: str = "tflops") -> int:
    """These rates as a `throughput.yaml` section.

    A command rather than a hand-copied table, because the numbers that price
    a campaign should arrive by a route that can be re-run.

    NON-MONOTONE POINTS ARE DROPPED. `throughput.measured_tflops` rounds a
    request DOWN to the nearest measured micro-batch, on the stated ground
    that "throughput rises with the micro-batch, so the lower neighbour
    under-promises". A point that is SLOWER than a smaller micro-batch breaks
    that: gdn|983M runs at 1.1 TFLOP/s at mbs 4 against 167.5 at mbs 2, so a
    lookup rounding down to 4 would price the cell 150x too slow and call it a
    measurement.

    Those points are real -- the card is thrashing, and that is worth knowing
    -- but they are not operating points, and this file's own convention is
    that a cell which cannot run is ABSENT rather than zero.
    """
    cells: dict[tuple[str, str], dict[int, float]] = {}
    for d in sorted((root / "mbs").iterdir()) if (root / "mbs").is_dir() else []:
        if not d.is_dir():
            continue
        arch, rung, m = d.name.rsplit("_", 2)
        logs = sorted(d.glob("slurm-*.log"))
        got = read_metrics(logs[-1]) if logs else None
        if got:
            rate = got[1] if unit == "tflops" else got[3]
            cells.setdefault((arch, rung), {})[int(m.lstrip("m"))] = rate

    if not cells:
        print(f"no probe logs with metrics under {root}", file=sys.stderr)
        return 1

    print(f"{section}:")
    dropped = []
    for (arch, rung), rates in sorted(cells.items()):
        keep, best = {}, 0.0
        for mbs in sorted(rates):
            if rates[mbs] < best:
                dropped.append(f"{arch}|{rung} mbs {mbs} at {rates[mbs]:.1f}")
                continue
            keep[mbs] = best = rates[mbs]
        fmt = "{:.1f}" if unit == "tflops" else "{:.0f}"
        body = ", ".join(f"{m}: {fmt.format(t)}" for m, t in sorted(keep.items()))
        print(f"  {arch}|{rung}: {{{body}}}")
    for line in dropped:
        print(f"# dropped (slower than a smaller micro-batch): {line}",
              file=sys.stderr)
    return 0


def _rung_key(tag: str) -> float:
    return float(tag[:-1]) * (1000 if tag.endswith("B") else 1)


if __name__ == "__main__":
    raise SystemExit(main())
