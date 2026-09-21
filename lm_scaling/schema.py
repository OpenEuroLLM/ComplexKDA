"""Config schema for the scaling ladder, on compoconf + hydra_staged_sweep.

One config describes a whole campaign. `hydra_staged_sweep` expands its sweep
groups into points, resolves the stage DAG, and `slurm_gen` renders the sbatch
from the same object. Nothing is rebuilt in a second place.

That is the point rather than the aesthetic. The previous pipeline kept three
ladder builders, and every failure of the first campaign was one piece of
state living in more than one of them:

  * the micro-batch ceilings, in config/size/*.yaml, experiment.py and
    titan_ladder.py -- they disagreed, and kda, complex-kda and pda all OOM'd
    at 302M five minutes in;
  * the arm definitions, in ARCHS, experiment.py and ladder_matched.json --
    they disagreed, and the arm named `kda` was built with
    allow_neg_eigval=True, so it was really kda-neg;
  * the stable run's output directory, built once by compose and again by
    emit -- they disagreed the moment exp_name entered the path, and all 36
    annealings died on FileNotFoundError.

Here the ladder geometry comes from `ladder.yaml`, the arm definitions and
matched widths from `ladder_matched.json`, and both reach the config through
OmegaConf resolvers (see `resolvers.py`) rather than being copied. An
annealing names its parent as `${sibling.stable.job.base_output_dir}`, so the
branch path cannot drift from the directory the stable run actually wrote.

Machine specifics -- the container image, its bind mounts, the environment
variables JUPITER needs -- are CONFIG, not template. They are the things most
likely to need overriding per cluster and per debugging session, and burying
them in an sbatch template makes them invisible and unoverridable.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field
from typing import Any

from compoconf import ConfigInterface
from hydra_staged_sweep.config.schema import StagedSweepRoot, SweepConfig
from monitor.submission import SlurmJobConfig
from slurm_gen.schema import SlurmConfig


@dataclass(kw_only=True)
class ContainerConfig(ConfigInterface):
    """How to enter the container, as data.

    Fields only -- the `apptainer exec ...` line is assembled in the YAML
    (`slurm.launcher_cmd`) out of these, using `oc.join` and `oc.maptmpl`.
    Building it here in Python instead would put the launcher in two places
    the moment one cluster needs a different flag, which is the shape of every
    failure this port exists to remove.

    `bind` and `env` are lists/dicts so a cluster quirk can be overridden from
    the command line. Several of ours were learned expensively and are
    documented at their point of use in `config/container/*.yaml`: an empty
    LD_LIBRARY_PATH replaces the image's own and hides libcuda, which surfaces
    as "Found no NVIDIA driver on your system"; TRITON_CACHE_DIR left on the
    quota-limited project filesystem surfaces as "OSError: [Errno 122] Disk
    quota exceeded" thrown from inside kernel compilation.
    """

    class_name: str = "Container"
    image: str | None = None
    runtime: str = "apptainer"
    args: list[str] = field(default_factory=lambda: ["--nv", "--writable-tmpfs"])
    bind: list[str] = field(default_factory=list)
    # Extra libraries, as an apptainer overlay rather than a venv on the
    # shared filesystem. A venv carries absolute paths and a python it does
    # not own, so it breaks when the image's interpreter moves and it cannot
    # be mounted read-only by a hundred concurrent jobs with any confidence.
    # An overlay is one immutable file: `--overlay img:ro` and the packages
    # are simply on sys.path.
    overlays: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    python: str = "python3"


@dataclass(kw_only=True)
class JobConfig(SlurmJobConfig):
    """Identity, where the run's output lands, and what the monitor watches.

    Inherited from `monitor.submission.SlurmJobConfig`: `name`, `log_path`,
    `log_path_current`, `log_events`, `state_events`, `start_condition`,
    `cancel_condition`, `array_len`, and `slurm`. Extending the library type
    rather than redeclaring it means a job description IS what the monitor
    registers -- no translation step, and no second place for a log path to
    drift out of agreement with the one SLURM writes to.

    `name` is an interpolation template in the YAML, so a run is named by what
    distinguishes it -- model, size, tokens, batch, learning rate, stage --
    and `base_output_dir` derives FROM `name`. One definition. The previous
    pipeline built the name in `compose.Run.name` and the directory again in
    `emit`, which is how the annealings came to look somewhere the stable run
    had never written.
    """

    class_name: str = "Job"
    base_output_dir: str = field(default=MISSING)
    # Where the resolved config is archived next to the run it produced. A run
    # whose config cannot be read back is a number without a provenance.
    config_path: str = ""
    config_path_current: str = ""

    def as_monitor_job(self) -> SlurmJobConfig:
        """This job as the thing the monitor registers.

        A projection, not a translation: every field comes straight across and
        only the ones the monitor has no use for -- where the output goes,
        where the config was archived -- are dropped. It exists because
        `class_name` has to say "SlurmJob" for compoconf to reload the record
        from disk, and ours says "Job".
        """
        keep = {f for f in SlurmJobConfig.__dataclass_fields__ if f != "class_name"}
        return SlurmJobConfig(**{f: getattr(self, f) for f in keep})


@dataclass(kw_only=True)
class TitanConfig(ConfigInterface):
    """torchtitan, nested the way its own TOML is.

    Mirroring the target format means the emitter is a dump rather than a
    translation: `[model]`, `[training]`, `[optimizer]` here are the same
    sections torchtitan reads. A field that torchtitan does not know is a
    field it silently ignores, so the shapes have to match rather than be
    approximately right.
    """

    class_name: str = "Titan"
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    optimizer: dict[str, Any] = field(default_factory=dict)
    lr_scheduler: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    parallelism: dict[str, Any] = field(default_factory=dict)
    activation_checkpoint: dict[str, Any] = field(default_factory=dict)
    compile: dict[str, Any] = field(default_factory=dict)
    job: dict[str, Any] = field(default_factory=dict)
    # torchtitan keeps the seed under [debug], not [training]; putting it in
    # [training] is rejected outright by its config merger.
    debug: dict[str, Any] = field(default_factory=dict)
    # [experimental] custom_import is how `titan_ext` gets loaded -- the train
    # spec `fla_custom` and every ladder flavor are registered from there.
    experimental: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    parameter_logging: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class DataConfig(ConfigInterface):
    """The corpus, its tokenizer and the context it is read at.

    `seq_len` lives here rather than on the backend because every backend has
    to agree on it, and a per-backend copy is how torchtitan quietly stayed at
    2048 while `lm/` moved to 4096 -- same config file, two different models.

    `eval_sequences` has no default on purpose. torchtitan divides the summed
    validation loss by the STEP COUNT, so the number of windows consumed has
    to be exact and identical across arms -- one arm scoring a different
    amount of text produces a number that looks like a validation loss and is
    not comparable to the others. It is a decision about how much of the
    held-out split to score, so it is written down rather than inferred.
    """

    class_name: str = "Data"
    name: str = field(default=MISSING)
    prefix: str = field(default=MISSING)
    format: str = "megatron_indexed"
    tokenizer: str = field(default=MISSING)
    vocab_size: int = field(default=MISSING)
    seq_len: int = field(default=MISSING)
    tie_embeddings: bool = True
    valid_prefix: str = ""
    eval_sequences: int = field(default=MISSING)
    # Token counts read off the .idx headers, so a moved, truncated or swapped
    # corpus is caught before a campaign runs on it rather than after. 0 means
    # "not recorded"; `verify_corpus.py` fills them in.
    train_tokens: int = 0
    valid_tokens: int = 0


@dataclass(kw_only=True)
class ClusterConfig(ConfigInterface):
    """Which machine, and which GPU is in it.

    `gpu` is not decoration: throughput.yaml is keyed by it because the gap
    between an A100-40GB and a GH200 is 1.6x on attention and 2.3x on
    complex-kda at the same context. Pricing a JUWELS campaign off the JUPITER
    table would be wrong by an amount that depends on the arm -- which is the
    one kind of error that survives a sanity check.
    """

    class_name: str = "Cluster"
    name: str = field(default=MISSING)
    gpu: str = field(default=MISSING)
    # Where runs land on THIS machine. It belongs here and not in the
    # experiment: a campaign that carried its own output root wrote JUPITER's
    # /e path into a JUWELS job script, where it does not exist, and the job
    # would have failed on its first checkpoint rather than at submit.
    runs_dir: str = field(default=MISSING)


@dataclass(kw_only=True)
class BackendConfig(ConfigInterface):
    """Which trainer, and its native config.

    `lm` is kept alongside `titan` only for the backend-agreement check: the
    first campaign's numbers came off `lm/`, and moving to torchtitan without
    showing the two agree would make everything before the switch
    incomparable to everything after it, silently.
    """

    class_name: str = "Backend"
    name: str = "titan"
    titan: TitanConfig = field(default_factory=TitanConfig)
    lm: dict[str, Any] = field(default_factory=dict)
    # How this backend is launched, as data, so the sbatch command line is
    # composed once in `experiments/base.yaml` for every backend instead of
    # once per backend in a template nobody reads.
    launcher: str = "torchrun"
    # Flags for the launcher, as a mapping so a cluster group can add one
    # without restating the list -- `node_rank` and `local_addr` are the two
    # that only matter once a job spans more than one node.
    torchrun_args: dict[str, Any] = field(default_factory=dict)
    entrypoint: str = ""
    config_flag: str = ""
    config_file: str = ""
    gpus_per_node: int = 4


@dataclass(kw_only=True)
class LadderRoot(StagedSweepRoot):
    """The whole campaign.

    Inherited from StagedSweepRoot: `sweep`, `stage`, `index`, `sibling`.
    `sibling` is what makes an annealing able to name its stable run without
    anyone reconstructing a path.
    """

    job: JobConfig = field(default_factory=MISSING)
    slurm: SlurmConfig = field(default_factory=MISSING)
    data: DataConfig = field(default_factory=MISSING)
    cluster: ClusterConfig = field(default_factory=MISSING)
    backend: BackendConfig = field(default_factory=BackendConfig)
    container: ContainerConfig = field(default_factory=ContainerConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    # Scratch space, deliberately untyped. `aux` holds the intermediates a
    # config derives on its way to filling in the REAL configs -- the
    # citations it looked up, the batch it computed, the step count it
    # divided out. Those are working notes, not an interface, and typing them
    # would mean every new intermediate needs a schema change.
    #
    # Validation belongs on the typed sections `aux` feeds:
    # backend.titan.training.local_batch_size is the number torchtitan
    # actually reads, and that is where a wrong type or an unknown key has to
    # be rejected. A typo in `aux` surfaces as an unresolved interpolation at
    # the point of use, which is equally loud and needs no schema.
    #
    # `dict[str, str]` and not `Any`: scratch space should not be validated,
    # and strings say so. compoconf coerces the resolved numbers on the way
    # in and the arithmetic resolvers take `str | int`, so the chain is
    # unaffected. (`dict[str, object]` would REJECT an int outright, which
    # would make the scratch space unusable for what it exists for.)
    aux: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
