"""Expand a campaign config into jobs -- and price it -- without submitting.

    python lm_scaling/plan.py fwedu_1p3B
    python lm_scaling/plan.py fwedu_1p3B --toml out/   # write the torchtitan configs
    python lm_scaling/plan.py fwedu_1p3B -o ++aux.nodes=4

There is one place a run is defined (`config/experiments/*.yaml`), one place it
is expanded (`hydra_staged_sweep`), and one place it is rendered
(`slurm_gen`). This module only glues them and prints what came out. Nothing
here decides anything about an experiment: if a number appears below that is
not in the config, that is a bug in this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

CONF = _HERE / "config"


def overlay_overrides(name: str, seen: tuple[str, ...] = ()) -> list[str]:
    """`config/overlay/<name>.yaml`, flattened into `++key=value` overrides.

    An overlay is a named bundle of overrides -- "run this campaign short", "on
    one node", "at this micro-batch" -- and it is applied the way the command
    line is, because that is the only precedence in Hydra that does not depend
    on how deep an experiment's inheritance goes.

    Placing it as a config GROUP was the first attempt and it does not work
    here. A group wins only if it is LAST in the leaf's defaults list, and an
    experiment that inherits another inherits its defaults -- so a group
    declared in the parent is overwritten by everything the child says about
    the same key. Measured on the torchtitan ladder (whose configs this release
    does not ship -- the ladder moved to Megatron): the same overlay set `nodes`
    to 1 through the parent campaign and left it at 16 through the child,
    silently. An overlay that applies to one experiment and not another is
    worse than none.

    Flattening to overrides also keeps the property this whole refactor is
    for: the values land BEFORE resolution, so everything derived from them
    follows. `slurm.sbatch.nodes` is the case that matters -- `sbatch.ntasks`
    interpolates from it, and the version of this that assigned to a resolved
    config left `ntasks` at the campaign's 64 with no `--nodes` line at all.
    Thirty-six jobs were rejected by slurm at submit.

    `defaults:` inside an overlay names overlays it builds on, applied first.
    """
    import yaml

    if name in seen:
        raise SystemExit(f"overlay {name!r} extends itself: {' -> '.join(seen)}")
    path = CONF / "overlay" / f"{name}.yaml"
    if not path.exists():
        have = sorted(p.stem for p in (CONF / "overlay").glob("*.yaml"))
        raise SystemExit(f"no overlay {name!r} (have {have})")
    body = yaml.safe_load(path.read_text()) or {}

    out: list[str] = []
    for base in body.pop("defaults", []) or []:
        out += overlay_overrides(base, seen + (name,))
    out += [f"++{k}={_override_value(v)}" for k, v in _flatten(body)]
    return out


def _flatten(node, prefix: str = ""):
    """Nested mapping -> (dotted key, leaf) pairs.

    Leaves only: a nested mapping becomes several dotted overrides rather than
    one override of a whole subtree, so an overlay that sets
    `backend.titan.checkpoint.enable` leaves the rest of `[checkpoint]` alone.
    Overriding the subtree would replace it, and `interval` would go with it.
    """
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            yield from _flatten(value, dotted + ".")
        else:
            yield dotted, value


def _override_value(v) -> str:
    """A YAML leaf as Hydra's override grammar spells it.

    Quoting is not cosmetic. An overlay's values are mostly interpolations, and
    `${oc.muli:${aux.mbs},${backend.gpus_per_node}}` contains a comma, which
    the override parser reads as a LIST unless the value is quoted. Single
    quotes, because interpolations are written with double quotes inside
    `oc.eval` expressions far more often than the reverse.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    text = str(v)
    quote = '"' if "'" in text else "'"
    return f"{quote}{text}{quote}"


def expand(overrides: list[str] | None) -> list[str]:
    """User overrides with every `overlay=<name>` replaced by what it names.

    In place, so an override written after an overlay still wins over it --
    `--overlay probe -o ++aux.probe_steps=40` reads the way it looks.
    """
    out: list[str] = []
    for item in overrides or []:
        if item.startswith("overlay="):
            for name in item.split("=", 1)[1].split(","):
                out += overlay_overrides(name.strip())
        else:
            out.append(item)
    return out


def load(experiment: str, overrides: list[str] | None = None):
    """The root config, composed and validated."""
    return _load_expanded(experiment, expand(overrides))


def plans(experiment: str, overrides: list[str] | None = None):
    """Every job the config expands to, in dependency order."""
    from hydra_staged_sweep.config.schema import ConfigSetup
    from hydra_staged_sweep.dag_resolver import resolve_sweep_with_dag
    from hydra_staged_sweep.expander import expand_sweep
    from schema import LadderRoot

    # Expanded ONCE and handed to both. The sweep re-composes the tree per
    # point through `ConfigSetup`, so an overlay that reached `load` and not
    # `ConfigSetup` would apply to the root config and to none of the jobs --
    # visible only as a campaign that prices itself as a probe and submits as
    # a campaign.
    overrides = expand(overrides)
    root = _load_expanded(experiment, overrides)
    points = expand_sweep(root.sweep)
    setup = ConfigSetup(pwd=str(_HERE.parent), config_dir=str(CONF),
                        config_name=f"experiments/{experiment}",
                        overrides=list(overrides))
    return resolve_sweep_with_dag(root, points, setup, config_class=LadderRoot)


def _load_expanded(experiment: str, overrides: list[str]):
    """`load`, for overrides that have already been expanded."""
    import resolvers

    resolvers.install()
    from hydra_staged_sweep.config.loader import load_hydra_config
    from schema import LadderRoot

    return load_hydra_config(f"experiments/{experiment}", CONF,
                             overrides=list(overrides), config_class=LadderRoot)


def per_cell(jobs, weight=None):
    """One job per output directory: the micro-batch that carries most of that
    cell's GPU-hours.

    A SELECTION, not an edit -- it drops jobs and changes none of them.

    Cell-level overlays (`speed`, `mbs`) name a job by its arm and rung alone,
    because a rung runs several budgets at the same geometry and timing all of
    them measures one number several times. That makes their names collide on
    purpose, and something has to choose which of the colliding configs to
    keep.

    Not simply "the first one found". A rung runs several chains at different
    global batches, so its micro-batch is not unique: on the lower ladder 47M
    runs at mbs 8 for the 6BT cell and mbs 16 for the other two, and those are
    different arithmetic intensities. Measuring the minority regime answers a
    question the campaign does not ask.

    `weight(cfg) -> float` prices a config in the campaign it belongs to --
    the caller passes one built from an UN-OVERLAID plan, because under a
    probe overlay every job is twelve steps and the weights would be equal by
    construction.
    """
    from collections import defaultdict

    cost: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    seen: dict[str, dict[int, object]] = defaultdict(dict)
    order: list[str] = []
    for j in jobs:
        key, mbs = j.config.job.base_output_dir, int(j.config.aux["mbs"])
        if key not in seen:
            order.append(key)
        cost[key][mbs] += (weight(j.config) if weight else 0.0)
        seen[key].setdefault(mbs, j)

    out = []
    for key in order:
        by_mbs = cost[key]
        pick = max(sorted(by_mbs), key=lambda m: by_mbs[m]) if any(by_mbs.values()) \
            else min(by_mbs)
        out.append(seen[key][pick])
    return out


def campaign_weights(experiment: str, overrides: list[str] | None = None):
    """`weight` for `per_cell`: what each (arm, rung, micro-batch) costs in the
    real campaign.

    A second plan pass, without the overlay. It is the only honest source --
    the overlaid jobs all run `probe_steps` and would weigh the same.
    """
    hours: dict[tuple[str, str, int], float] = {}
    for j in plans(experiment, overrides):
        a = j.config.aux
        key = (a["model"], a["size"], int(a["mbs"]))
        hours[key] = hours.get(key, 0.0) + gpu_hours(j.config)

    def weight(cfg) -> float:
        a = cfg.aux
        return hours.get((a["model"], a["size"], int(a["mbs"])), 0.0)

    return weight


def check(jobs) -> None:
    """Invariants that must hold for ANY campaign, checked before pricing.

    These are not tests of this file -- they are the failure modes the first
    campaign actually hit, each of which produced a plausible-looking job that
    died (or worse, did not) hours later. A validation pass here catches them
    at plan time, on the login node, for free.

    TWO KINDS, and a probe is held to one of them.

      * Will this job RUN -- a batch torchtitan refuses at startup, an output
        directory the compute nodes cannot see, a ceiling that OOMs five
        minutes in. Checked for everything.
      * Is this job the EXPERIMENT IT CLAIMS TO BE -- a WSD run with no decay,
        a warmup that eats the run, a budget under one grid step, a final step
        that is never validated. Checked for campaigns only.

    A probe fails every check in the second group by construction: twelve
    steps, no cooldown, no validation, and in the ceiling probe's case a
    micro-batch chosen precisely because nobody knows yet whether it fits.
    Holding it to them would mean either a `--probe` that always errors or a
    `check` that a caller can switch off -- and the second is how the first
    campaign shipped with three copies of the micro-batch ceilings.

    So the config says which it is: `aux.probe` is set by
    config/overlay/probe.yaml and by nothing else.
    """
    import resolvers

    problems: list[str] = []
    seen: dict[str, str] = {}

    for j in jobs:
        c, a = j.config, j.config.aux
        name = c.job.name

        # 1. Two runs in one directory. The second silently resumes the first.
        if c.job.base_output_dir in seen:
            problems.append(f"{name} shares an output directory with "
                            f"{seen[c.job.base_output_dir]}")
        seen[c.job.base_output_dir] = name

        # Set by config/overlay/probe.yaml. A probe is a short run of a job,
        # not a claim about an experiment.
        probe = bool(int(a.get("probe", 0) or 0))

        # 2. A micro-batch above the measured ceiling: an OOM five minutes in,
        #    after the queue wait.
        #
        #    Not for a probe: the ceiling probe's whole job is to find out
        #    whether the next power of two fits, and the cell it is measuring
        #    has no entry in the table to compare against -- `peak_mbs` RAISES
        #    for it, which is right everywhere else and circular here.
        mbs, gas = int(a["mbs"]), int(a["grad_accum"])
        world = int(a["world_gpus"])
        if not probe:
            ceiling = resolvers.peak_mbs(a["peak_section"], a["size"], a["model"])
            if mbs > ceiling:
                problems.append(f"{name}: micro-batch {mbs} over the measured "
                                f"ceiling {ceiling}")
            if mbs * gas * world != int(a["gbs"]):
                problems.append(f"{name}: {mbs} x {gas} x {world} GPUs != gbs "
                                f"{a['gbs']}")
        if int(a["gbs"]) % world:
            problems.append(
                f"{name}: global batch {a['gbs']} is not divisible by the "
                f"{world} GPUs it runs on ({c.slurm.sbatch.nodes} nodes x "
                f"{c.backend.gpus_per_node}); torchtitan refuses this at "
                "startup rather than silently reshaping it")

        # 3. Warmup that eats the run. The source paper excludes such (b, D)
        #    cells outright, so a cell like that is not on the ladder it
        #    claims to be on.
        warm = int(c.backend.titan.lr_scheduler["warm_steps"])
        if not probe and warm > 0.3 * int(a["run_steps"]):
            problems.append(f"{name}: warmup {warm} is over 30% of the "
                            f"{a['run_steps']} steps it runs")

        # 4. A validation set that is not the same size for every arm.
        #    `steps` is eval_sequences // (mbs x dp) and truncates, and the
        #    micro-batch differs per arm -- so an eval_sequences that does not
        #    divide evenly has attn scoring one amount of text and pda
        #    another, and torchtitan divides the summed loss by the step
        #    count. Two arms then report validation losses over different
        #    corpora, which is the comparison the whole study is.
        #
        #    Per WORLD, not per node: every rank runs `validation.steps`
        #    batches. Checked per node, a 32-node arm passed while scoring 32x
        #    the sequences a one-node arm did -- every job of both upper
        #    ladders scored 2x to 32x `eval_sequences`, and on text that
        #    differed with the width.
        per_eval = int(a["mbs"]) * world
        if not probe and c.data.eval_sequences % per_eval:
            problems.append(
                f"{name}: eval_sequences {c.data.eval_sequences} is not "
                f"divisible by {a['mbs']} x {world} ranks; this arm would "
                f"score {c.data.eval_sequences // per_eval * per_eval}"
                " sequences while another scores a different number")
        elif not probe and c.backend.titan.validation.get("enable", True):
            scored = int(c.backend.titan.validation["steps"]) * per_eval
            if scored != c.data.eval_sequences:
                problems.append(
                    f"{name}: validation scores {scored} sequences "
                    f"({c.backend.titan.validation['steps']} steps x "
                    f"{a['mbs']} x {world} ranks), not the "
                    f"{c.data.eval_sequences} eval_sequences names")

        # 5. A final step that is never validated.
        #    torchtitan evaluates inside the training loop only --
        #    `step == 1 or step % freq == 0`, nothing after the loop ends --
        #    so unless freq divides steps, the last step is never scored. The
        #    final validation loss of a cooldown is this campaign's datapoint,
        #    so that is the whole result missing, silently. Every one of the
        #    48 jobs had it before the step counts went on a grid.
        freq = int(c.backend.titan.validation["freq"])
        steps, run = int(a["steps"]), int(a["run_steps"])
        if c.backend.titan.validation.get("enable", True):
            if steps % freq:
                problems.append(
                    f"{name}: validation freq {freq} does not divide "
                    f"{steps} steps; the last scored step would be "
                    f"{steps // freq * freq} and the final loss never measured")
            elif run // freq < int(a.get("min_evals", 1)):
                problems.append(
                    f"{name}: only {run // freq} evaluations in the "
                    f"{run} steps it runs")

        # 6. A budget the step grid cannot express. Each budget's endpoint is
        #    rounded onto the grid so a validation frequency that divides it
        #    exists; a budget under one grid step would round to nothing.
        if not probe and int(a["steps"]) < int(a["grid"]):
            problems.append(
                f"{name}: {a['steps']} steps is under one {a['grid']}-step "
                "grid unit, so the endpoint and the eval frequency collapse")

        # 6b. A run longer than its corpus. The reader wraps to the first
        #     window when it runs out -- it is built `infinite` -- so this
        #     neither fails nor warns: it trains a second epoch and reports a
        #     single-epoch budget. Checked where the data group records the
        #     corpus size; 0 there means "not recorded yet".
        corpus = int(c.data.train_tokens or 0)
        trained = int(a.get("actual_tokens", 0) or 0)
        if not probe and corpus and trained > corpus:
            problems.append(
                f"{name}: trains {trained:,} tokens on a {corpus:,}-token "
                "corpus, so the reader wraps around and repeats it")

        # 7. A cooldown that is the whole run, or none of it. WSD is warmup,
        #    constant, decay -- a run that is all decay is a different
        #    schedule, and one with no decay is not WSD at all.
        #
        #    A run whose MAIN phase decays is the other schedule, cosine, as
        #    the published 1.3B/100BT recipes train (fwedu_1p3B). It anneals
        #    without a cooldown, and that is the schedule it claims to be
        #    rather than a WSD run that lost its decay.
        cool = int(a["cooldown_steps"])
        sched = c.backend.titan.lr_scheduler
        main_decays = (sched.get("main_decay_type", "const") != "const"
                       and float(sched.get("main_decay_ratio", 1.0)) < 1.0)
        # A STABLE stage has no cooldown by definition -- it is the shared
        # constant-LR phase that the cooldowns branch off. Everything else
        # that is all decay, or none of it, is a different schedule from the
        # one it claims to be.
        if not probe and c.stage != "stable" and not (
                0 < cool < int(a["steps"]) or (cool == 0 and main_decays)):
            problems.append(
                f"{name}: cooldown is {cool} of {a['steps']} steps")
        if c.stage == "stable" and (cool or main_decays):
            problems.append(
                f"{name}: a stable stage must not decay, but cooldown is {cool}"
                + (" and the main phase decays" if main_decays else ""))

        # 8. A batch that is not a power of two anywhere down the chain.
        #    The paper's batch-size grid is 2^4..2^10, so `gbs` is a power of
        #    two by construction; four GPUs divide it to another one, and the
        #    micro-batch has to be a third so gradient accumulation stays
        #    integral. The ceilings the `lm/` sweep reports -- 5, 10, 21 --
        #    are not usable as micro-batches even where they are true, and a
        #    ceiling like 5 quietly drops the run to mbs 4 while looking as
        #    though 5 was measured and honoured. Torchtitan-measured ceilings
        #    are powers of two on purpose; this is what says so.
        for label, value in (("gbs", int(a["gbs"])), ("mbs", mbs),
                             ("grad_accum", gas)):
            if value < 1 or value & (value - 1):
                problems.append(
                    f"{name}: {label} is {value}, not a power of two")

        # 8b. A regime measured on a different GPU.
        #
        #     Pinning the WORLD SIZE is legitimate -- the upper ladder runs at
        #     8 to 64 GPUs and is priced off the 4-GPU table, which is a floor
        #     because FSDP only frees memory as the world grows. Pinning the
        #     GPU never is: an A100 ceiling on a GH200 is wrong in both
        #     directions and by an arm-dependent amount.
        #
        #     The torchtitan ladder's upper campaigns carried
        #     `a100_seq4096_titan_g4` after the campaign moved to JUPITER, so
        #     both priced GH200 runs off A100 measurements and reported a
        #     JUPITER cluster while doing it. Nothing else would have noticed.
        if not str(a["regime"]).startswith(f"{c.cluster.gpu}_"):
            problems.append(
                f"{name}: runs on {c.cluster.gpu} but is priced from "
                f"{a['regime']}, which was measured on a different GPU")

        # 9. An output directory the compute nodes cannot see.
        #
        #    /p is mounted on JUPITER's LOGIN node and not on its compute
        #    nodes. A campaign pointed there writes nothing and reports
        #    nothing: slurm cannot create the output file, apptainer has
        #    nothing to bind, and the job still exits 0. Ninety-eight probes
        #    did exactly that -- 55 seconds each, no log, no error, perfect
        #    exit codes, and the only symptom was a `--collect` that found no
        #    completed probes.
        #
        #    The container's bind list is the machine's own statement of what
        #    a job can reach, so a runs_dir outside it is one the job cannot
        #    write. Cheap to check and impossible to notice otherwise.
        roots = [b.split(":")[0] for b in (c.container.bind or [])]
        out = c.job.base_output_dir
        if roots and not any(out == r or out.startswith(r.rstrip("/") + "/")
                             for r in roots):
            problems.append(
                f"{name}: writes to {out}, which no container bind covers "
                f"({sorted(set(roots))}). On this cluster that directory is "
                "unreachable from a compute node, and the job would exit 0 "
                "having written nothing")

        # 10. A checkpoint the wall clock never reaches.
        #
        #    A job longer than its time limit is requeued, and resumes from
        #    its last checkpoint. If the checkpoint PERIOD is longer than the
        #    window, there is never a last checkpoint: the job is killed, comes
        #    back at step 0, and repeats forever while looking perfectly busy
        #    in squeue.
        #
        #    The ladder's default period is `full_steps // 4`, which is fine
        #    for a cell that finishes inside six hours and fatal for one that
        #    does not. At 1.7B/300BT a quarter of the run is 35,776 steps
        #    against the ~12,700 a 6-hour lowcont window covers.
        from throughput import token_rate

        rate = None if probe else \
            token_rate(a["model"], a["size"], mbs, a.get("regime"))
        limit_s = _seconds(getattr(c.slurm.sbatch, "time", None))
        if rate and limit_s:
            per_step = int(a["gbs"]) * int(c.data.seq_len) / (rate * world)
            if per_step * int(a["run_steps"]) > limit_s:  # needs a requeue
                ckpt = int(a.get("ckpt_interval") or 0)
                if not ckpt:
                    problems.append(
                        f"{name}: runs {per_step * int(a['run_steps']) / 3600:.1f}h "
                        f"against a {limit_s / 3600:.1f}h limit and saves no "
                        "checkpoint, so a requeue restarts it from nothing")
                elif per_step * ckpt > limit_s:
                    problems.append(
                        f"{name}: checkpoints every {ckpt} steps "
                        f"({per_step * ckpt / 3600:.1f}h) but the wall limit is "
                        f"{limit_s / 3600:.1f}h, so it is killed before it ever "
                        "writes one and every requeue starts from step 0")

    if problems:
        raise SystemExit("this campaign will not run:\n  "
                         + "\n  ".join(problems))


def _seconds(t) -> float | None:
    """A SLURM time limit in seconds. `D-HH:MM:SS`, `HH:MM:SS` or `MM:SS`."""
    if not t:
        return None
    text, days = str(t), 0
    if "-" in text:
        d, _, text = text.partition("-")
        days = int(d)
    parts = [int(x) for x in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def to_toml(titan) -> str:
    """`backend.titan` as a torchtitan TOML.

    A dump, not a translation: the config's sections ARE torchtitan's
    sections. A translation step is where a field goes missing, and a field
    torchtitan does not receive is one it silently defaults.
    """
    import tomli_w
    from compoconf import asdict

    d = {k: v for k, v in asdict(titan).items()
         if k not in ("class_name", "extra") and v}
    d.update(asdict(titan).get("extra", {}) or {})
    return tomli_w.dumps(d)


def gpu_hours(cfg) -> float:
    """What this job will cost, from the measured throughput table.

    Priced on `run_steps`, not `steps`: a cooldown resumes at its branch and
    executes only the tail. Pricing it on `steps` would triple the ladder's
    quoted cost and hide the whole reason the stages exist.
    """
    from throughput import gpu_hours as gh

    a = cfg.aux
    return gh(a["model"], a["size"], int(a["run_steps"]), int(a["gbs"]),
              int(cfg.data.seq_len), gpus=int(a["world_gpus"]),
              gpu=cfg.cluster.gpu, mbs=int(a["mbs"]),
              prefer=a["regime"]) or 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("-o", "--override", action="append", default=[])
    ap.add_argument("--toml", metavar="DIR",
                    help="write each job's backend config to DIR")
    args = ap.parse_args(argv)

    from throughput import measured_at

    jobs = plans(args.experiment, args.override)
    check(jobs)
    total = 0.0
    extrapolated = 0.0
    for job in jobs:
        c = job.config
        a = c.aux
        cost = gpu_hours(c)
        total += cost
        src = measured_at(a["model"], a["size"], int(c.data.seq_len),
                          c.cluster.gpu, prefer=a["regime"])
        want = a["regime"]
        if src != want:
            extrapolated += cost
        print(f"{c.job.name:<62} steps={int(a['steps']):>7} "
              f"mbs={a['mbs']:>3} ga={a['grad_accum']:>2} "
              f"cool={int(a['cooldown_steps']):>6} {cost:8.1f} GPU-h"
              f"{'' if src == want else '  ~' + src}")
        if args.toml:
            out = Path(args.toml)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{c.job.name}.toml").write_text(to_toml(c.backend.titan))
    print(f"\n{len(jobs)} jobs, {total:,.1f} GPU-hours")
    if extrapolated:
        print(f"{extrapolated / total:.0%} of that is extrapolated from a "
              "measurement on another GPU or at another context -- see `~` "
              "above. The A100/GH200 gap is 1.6x on attention and 2.3x on "
              "complex-kda, so those rows are wrong by an arm-dependent "
              "amount, not a constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
