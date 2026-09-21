"""Write a campaign's job files, register them, and (optionally) watch them.

    python lm_scaling/submit.py fwedu_1p3B --out /tmp/fwedu  # write, do not submit
    python lm_scaling/submit.py fwedu_1p3B --write           # write in place
    python lm_scaling/submit.py fwedu_1p3B --write --run     # ... and monitor

Writing is separate from submitting on purpose. Everything a job needs -- the
backend config, the sbatch script, the resolved campaign config -- is a file
that can be read and diffed before anything reaches a queue, and `--out` puts
the whole campaign somewhere else to look at it.

`--out`, `--probe` and `--overlay` are all spellings of OVERRIDES. Nothing in
this file edits a config: every job is written exactly as compoconf resolved
it, which is what makes the archived `config.yaml` beside a run the config
that ran. See `plan.overlay_overrides`.

The three files per job:

    titan.toml   the backend's own config, dumped from `backend.titan`
    <name>.sbatch  rendered by slurm_gen from the template and `slurm`
    config.yaml  the resolved campaign config, archived beside its run

The third is not a convenience. This project has already had to work out, from
logs alone, which of two definitions of `kda` a finished run had used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import plan as P  # noqa: E402


def write_job(cfg) -> dict[str, Path]:
    """Write one job's three files, exactly as the resolved config names them.

    NOTHING IS EDITED HERE, and that is the whole point of the function.

    This used to take a `root` and repoint a job at it: `base_output_dir`,
    both log paths, the backend config file, the dump folder, four `slurm`
    paths, the launch command, the `--output` and `--error` directives, the
    checkpoint a chained cooldown resumes from, and three monitor conditions
    that name a SIBLING's directory. Thirteen assignments after resolution,
    every one of them a place where a path could be missed -- and three were,
    in turn: eighteen probes died on a missing TOML, the probe logs mixed with
    the campaign's, and an `--out` copy of a chained ladder waited forever on
    a checkpoint the relocated stable run was never going to write.

    All thirteen derive from `cluster.runs_dir`. Overriding that one field
    before the config resolves moves every one of them, including the
    siblings, because that is what an interpolation is for. `--out DIR` is
    now exactly `-o cluster.runs_dir=DIR`.
    """
    from compoconf import asdict
    from slurm_gen.generator import generate_script

    outdir = Path(cfg.job.base_output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    toml = outdir / "titan.toml"
    toml.write_text(P.to_toml(cfg.backend.titan))

    import yaml

    config = outdir / "config.yaml"
    config.write_text(yaml.safe_dump(asdict(cfg), sort_keys=True))

    slurm = cfg.slurm
    script = generate_script(
        slurm,
        job_name=cfg.job.name,
        script_path=outdir / f"{cfg.job.name}.sbatch",
        log_path=cfg.job.log_path,
        command=list(slurm.command),
    )
    return {"toml": toml, "config": config, "script": Path(script)}


def register(jobs, state_dir: Path) -> tuple[int, int]:
    """Put every job in the monitor's store -- WITHOUT resetting a submitted one.

    Returns (written, kept).

    A record that was already SUBMITTED is left exactly as it is. Its runtime
    is the only place the slurm job id lives, and `rehydrate()` re-attaches a
    job only if it finds `submitted` and that id. Overwriting it with a fresh
    runtime -- which this used to do, unconditionally -- turns a running job
    back into an unsubmitted one, and the next poll sbatches it a SECOND time
    into the same directory: two jobs appending to one log and racing each
    other's checkpoints. The natural recovery after a dropped login session,
    "run the same command again", was exactly the trigger.

    It also covers a campaign that already FINISHED: re-running `--run` must
    not retrain it.

    A record that was never submitted is rewritten. That is the state a failed
    submission leaves behind -- the container attempt on the ladder left one
    at `submitted=False, final_state=cancelled` -- and resetting it is what
    lets the next attempt go through.

    To force a submitted job back to fresh, delete its
    `<state_dir>/<job>.job.json` first; that is a decision about a real run
    and should be made by hand.
    """
    from monitor.loop import JobFileStore, JobRecordConfig, JobRuntimeConfig

    store = JobFileStore(state_dir)
    written = kept = 0
    for j in jobs:
        c = j.config
        existing = store.load(c.job.name, include_finished=True)
        if existing is not None and existing.runtime.submitted:
            kept += 1
            continue
        store.upsert(JobRecordConfig(
            job_id=c.job.name,
            definition=c.job.as_monitor_job(),
            runtime=JobRuntimeConfig(),
        ))
        written += 1
    return written, kept


def reconcile(state_dir: Path, queued: dict[str, list[str]] | None = None) -> list[str]:
    """What would go wrong if a monitor attached to this store right now.

    Returns a list of problems; empty means attaching is safe.

    The case that matters is a job slurm is running while its record says it
    was never submitted. That is what a monitor killed between `sbatch`
    returning and the record being written leaves behind, and the first poll
    of a re-attached monitor would submit it AGAIN -- a second job in the same
    directory, appending to the same log and racing the same checkpoints.
    Nothing downstream notices: both jobs train, and the loss curve is
    whichever wrote last.

    `queued` maps job name -> slurm ids, for testing; by default it is read
    from `squeue` for the current user.
    """
    import getpass
    import subprocess

    from monitor.loop import JobFileStore

    if queued is None:
        out = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%i %j"],
            capture_output=True, text=True, check=True).stdout
        queued = {}
        for line in out.splitlines():
            jid, _, name = line.partition(" ")
            queued.setdefault(name.strip(), []).append(jid.strip())

    problems = []
    for rec in JobFileStore(state_dir).load_all(include_finished=True):
        live = queued.get(rec.job_id, [])
        rid = str(rec.runtime.runtime_job_id or "")
        if len(live) > 1:
            problems.append(f"{rec.job_id}: {len(live)} slurm jobs share this "
                            f"name ({', '.join(live)})")
        if live and not rec.runtime.submitted:
            problems.append(
                f"{rec.job_id}: slurm is running it as {live[0]} but the record "
                "says it was never submitted -- attaching would submit it again")
        elif live and rid and rid not in live:
            problems.append(f"{rec.job_id}: the record names slurm job {rid} "
                            f"but squeue has {', '.join(live)}")
    return problems


def watch(state_dir: Path, poll_seconds: float) -> None:
    """Submit what is ready, then keep looking until nothing is left.

    `MonitorLoop` has no `run()`: it exposes `observe_once()` and the caller
    drives the cadence. This used to call `loop.run()` and would have died
    with AttributeError the moment anyone passed `--run` -- after writing 54
    job scripts and printing a GPU-hour total, which is the point at which it
    looks like it worked.

    `rehydrate()` first, and it is not optional. The clients only learn about
    a job through `submit()`, so a monitor started against an existing store
    -- after a crash, a logout, or a deliberate restart of the watcher --
    knows the job ids but has no client that will report on them. It would
    then treat running jobs as unsubmitted and start them a second time.
    """
    import time
    from collections import Counter

    from monitor.app import _import_registry
    from monitor.job_client_protocol import JobClientInterface
    from monitor.loop import JobFileStore, MonitorLoop

    _import_registry()
    from compoconf import parse_config

    client = parse_config(JobClientInterface.cfgtype,
                          {"class_name": "SlurmClient"}).instantiate(JobClientInterface)
    store = JobFileStore(state_dir)
    loop = MonitorLoop(store, slurm_client=client,
                       poll_interval_seconds=poll_seconds)
    loop.rehydrate()

    expected = len(store.load_all(include_finished=True))
    while True:
        loop.observe_once()
        # include_finished=True, and it is the whole difference between a
        # report and a lie. `load_all()` defaults to EXCLUDING finished jobs,
        # so a job that ran to completion drops out of the list -- and the
        # summary computed from that list said "all 0 jobs reached a terminal
        # state (0 COMPLETED)" about a campaign whose one job had just
        # submitted, trained 20 steps and exited COMPLETED in 1:48.
        records = store.load_all(include_finished=True)
        if len(records) != expected:
            # `load_all` swallows OSError, JSONDecodeError, ValueError and
            # KeyError per record and moves on, so a record it cannot parse
            # makes a job vanish from monitoring with no log line at all --
            # its own upsert docstring says so. Losing one silently is how a
            # campaign comes back a job short.
            raise SystemExit(
                f"the monitor store holds {len(records)} of {expected} jobs; "
                f"one or more records in {state_dir} cannot be parsed and "
                "would be dropped from monitoring without a message")
        pending = [r for r in records
                   if not r.runtime.submitted or not r.runtime.final_state]
        if not pending:
            states = Counter(r.runtime.final_state for r in records)
            print(f"all {len(records)} jobs reached a terminal state: "
                  + ", ".join(f"{n} {s}" for s, n in states.most_common()))
            return
        running = sum(1 for r in pending if r.runtime.submitted)
        print(f"{len(records) - len(pending)} done, {running} running, "
              f"{len(pending) - running} waiting", flush=True)
        time.sleep(poll_seconds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--out", metavar="DIR",
                    help="write the campaign under DIR instead of the "
                         "cluster's runs_dir. Sugar for "
                         "`-o cluster.runs_dir=DIR`, and it moves the whole "
                         "campaign -- checkpoints, logs, the branch a "
                         "cooldown resumes from -- because all of those "
                         "interpolate from that one field")
    ap.add_argument("--write", action="store_true",
                    help="write into each job's real output directory")
    ap.add_argument("--overlay", action="append", default=[],
                    help="a named bundle of overrides from config/overlay/ "
                         "-- `probe` for a short run of each job, `speed` for "
                         "one node at the campaign's micro-batch. Repeatable; "
                         "later ones win, and -o wins over all of them")
    ap.add_argument("--run", action="store_true",
                    help="register the jobs and run the monitor loop")
    ap.add_argument("--resume", action="store_true",
                    help="re-attach the monitor to a campaign that is already "
                         "submitted: writes NO job files and registers "
                         "nothing, after checking the store against squeue. "
                         "What to run after the monitor was stopped")
    ap.add_argument("--state-dir", default=None,
                    help="the monitor's job store (default: <campaign>/.monitor)")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--probe", type=int, metavar="STEPS",
                    help="shorthand for `--overlay probe "
                         "-o ++aux.probe_steps=STEPS`: each job as a short "
                         "throughput probe of itself, same config, fewer "
                         "steps, no checkpointing or validation")
    ap.add_argument("--per-cell", action="store_true",
                    help="keep one job per output directory -- the micro-batch "
                         "carrying most of that cell's GPU-hours. What the "
                         "cell-level overlays (`speed`, `mbs`) need, since "
                         "they name a job by arm and rung alone")
    ap.add_argument("--only", action="append", default=[],
                    help="substring a job name must contain")
    ap.add_argument("--exclude", action="append", default=[],
                    help="substring a job name must NOT contain. Applied "
                         "after --only, so the two compose: run one cell "
                         "first to see it work, then everything but it.")
    args = ap.parse_args(argv)

    if args.resume and (args.write or args.run or args.probe or args.overlay):
        ap.error("--resume only attaches; it does not combine with "
                 "--write/--run/--probe/--overlay")
    if not (args.out or args.write or args.resume):
        ap.error("say where the files go: --out DIR, or --write to write in place")

    # ONE list of overrides, in precedence order, and nothing after this point
    # touches a config. `--out` and `--probe` are spellings of overrides, not
    # a second way of configuring a run: the flags a user reaches for most
    # often were also the ones that used to edit resolved objects.
    overrides = [f"overlay={name}" for name in args.overlay]
    if args.probe:
        overrides += ["overlay=probe", f"++aux.probe_steps={args.probe}"]
    if args.out:
        overrides.append(f"cluster.runs_dir={args.out}")
    overrides += args.override

    jobs = P.plans(args.experiment, overrides)

    if args.resume:
        # NOTHING is written. `--write` here would rewrite titan.toml inside
        # the directories of RUNNING jobs, and a job that hits the wall clock
        # is resubmitted from that file -- so it would resume on whatever the
        # config says today, not what it started with. The archived
        # config.yaml would then describe a run that changed halfway.
        state = Path(args.state_dir) if args.state_dir else \
            Path(jobs[0].config.job.base_output_dir).parent / ".monitor"
        from monitor.loop import JobFileStore

        n = len(JobFileStore(state).load_all(include_finished=True))
        if not n:
            ap.error(f"nothing is registered in {state}; to start a campaign "
                     "use --write --run")
        problems = reconcile(state)
        if problems:
            raise SystemExit("not attaching; the store and slurm disagree:\n  "
                             + "\n  ".join(problems))
        print(f"attaching to {n} jobs in {state}")
        watch(state, args.poll)
        return 0

    if args.per_cell:
        # Weighted by the REAL campaign, planned again without the overlay:
        # under one, every job runs `probe_steps` and the weights would be
        # equal by construction.
        jobs = P.per_cell(jobs, P.campaign_weights(args.experiment, args.override))
    # Same validation the planner runs. Nothing is written for a campaign that
    # would not run.
    P.check(jobs)
    if args.only:
        jobs = [j for j in jobs
                if any(o in j.config.job.name for o in args.only)]
    if args.exclude:
        jobs = [j for j in jobs
                if not any(x in j.config.job.name for x in args.exclude)]
    if not jobs:
        ap.error("no job matches --only/--exclude; nothing to do")
    for j in jobs:
        paths = write_job(j.config)
        print(f"{j.config.job.name:<62} {paths['script']}")

    if args.probe or args.overlay:
        # Pricing the campaign here would be a lie: these run `--probe` steps,
        # not the budget the config names.
        # `backend.titan.training.steps` and NOT `aux.steps`: under an
        # overlay those differ on purpose -- `aux` goes on describing the
        # campaign cell a probe was cut from, and the TOML says what this job
        # runs. Reading `aux` here announced "1 jobs of 45824 steps" for a
        # 20-step probe, which is the one number a reader would check.
        steps = {int(j.config.backend.titan.training["steps"]) for j in jobs}
        shown = steps.pop() if len(steps) == 1 else f"{min(steps)}-{max(steps)}"
        print(f"\n{len(jobs)} jobs of {shown} steps each under "
              f"{Path(jobs[0].config.job.base_output_dir).parent}")
    else:
        total = sum(P.gpu_hours(j.config) for j in jobs)
        print(f"\n{len(jobs)} jobs, {total:,.1f} GPU-hours")

    if not args.run:
        print("\nnothing submitted. Add --run to register and monitor them.")
        return 0

    state = Path(args.state_dir) if args.state_dir else \
        Path(jobs[0].config.job.base_output_dir).parent / ".monitor"
    written, kept = register(jobs, state)
    print(f"registered {written} jobs in {state}"
          + (f"; kept {kept} already submitted as they are" if kept else ""))
    watch(state, args.poll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
