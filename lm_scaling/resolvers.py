"""Citations: resolvers that read a table and nothing else.

Each one answers "what does <file> say about X" and stops there. The
arithmetic that combines them belongs in the YAML, using the composable
primitives `hydra_staged_sweep` already registers -- `oc.divi`, `oc.cdivi`,
`oc.muli`, `oc.eval`, `oc.if`. So a config reads:

    peak:   ${peak_mbs:${aux.peak_section},${size},${model}}
    want:   ${oc.divi:${aux.gbs},${gpus}}
    mbs:    ${oc.eval:'max(d for d in range(1, ${aux.want} + 1)
                           if ${aux.want} % d == 0 and d <= ${aux.peak})'}

rather than calling one `${micro_batch:...}` that hides the cap, the
divisibility and the failure mode inside Python. Every step is visible in the
file you are reading, and any of them can be overridden from the command line.

Two rules these follow:

**A citation never computes.** The moment a resolver takes five arguments and
returns a decision, the config stops describing the experiment and starts
delegating it -- which is the situation the port exists to leave.

**A citation never defaults.** Missing entries raise. An unmeasured
micro-batch ceiling that quietly returns something plausible is exactly what
OOM'd three 302M runs five minutes in; a silent default would reintroduce it
behind a nicer interface. Type-level validation is compoconf's job and
happens at load; these raise at resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

_cache: dict[str, object] = {}


def _geometries() -> dict:
    """Every geometry a campaign can name, as one table.

    Read through `geometries` rather than from `ladder_matched.json` directly:
    torchtitan's flavors come from the same loader, so any geometry a config
    can cite also has a flavor, and the other way round.
    """
    import geometries

    return geometries.table()


def _load_yaml(name: str) -> dict:
    import yaml

    if name not in _cache:
        _cache[name] = yaml.safe_load((_HERE / name).read_text())
    return _cache[name]  # type: ignore[return-value]


# --- citations into the geometry table -------------------------------------
def matched(rung: str, arch: str, field: str):
    """One field of the parameter-matched table: `d_ffn`, `N`, `kwargs`, ..."""
    table = _geometries()
    if rung not in table:
        raise KeyError(f"the geometry table has no rung {rung!r} "
                       f"(has {sorted(table)})")
    archs = table[rung]["archs"]
    if arch not in archs:
        raise KeyError(
            f"the geometry table has no arch {arch!r} at {rung}; regenerate "
            "the file that defines the rung rather than defaulting")
    entry = archs[arch]
    if field == "kwargs":
        return OmegaConf.create(entry.get("kwargs", {}))
    if field not in entry:
        raise KeyError(f"{arch}@{rung} has no {field!r} (has {sorted(entry)})")
    return entry[field]


def flavor(rung: str, arch: str, drop_silu=False) -> str:
    """The torchtitan flavor a run names: `<arch>-<rung>`, or that with
    `-nosilu` when the campaign drops SiLU from q/k/v and the arm's layer can.

    gdn and attn are upstream layers and keep their SiLU. For them this returns
    the plain name, so the flavor never claims a change that did not happen.
    """
    import geometries

    matched(rung, arch, "d_ffn")        # the arm exists, or a KeyError naming it
    base = f"{arch}-{rung}"
    if str(drop_silu).strip().lower() not in ("1", "true", "yes"):
        return base
    layer = _geometries()[rung]["archs"][arch].get("mixer")
    return base + geometries.NOSILU if layer in geometries.DROPS_SILU else base


def rung(rung: str, field: str):
    """One RUNG-level field of the matched table: `d_model`, `n_layers`,
    `n_heads`, `head_dim`, `tie_word_embeddings`.

    Separate from `matched` because these are properties of the rung, shared
    by every arm at it -- the arms differ only in `d_ffn`, which is what makes
    them parameter-matched.
    """
    table = _geometries()
    if rung not in table:
        raise KeyError(f"the geometry table has no rung {rung!r} "
                       f"(has {sorted(table)})")
    entry = table[rung]
    if field not in entry:
        raise KeyError(
            f"rung {rung} has no {field!r} (has {sorted(k for k in entry if k != 'archs')})")
    return entry[field]


def arch_kwarg(rung: str, arch: str, key: str):
    """One layer setting for an arm, from the matched table's `kwargs`.

    The canonical definition. The first campaign's `lm/` path took these from
    its own hardcoded block instead, so the arm named `kda` was built with
    allow_neg_eigval=True and was really kda-neg.
    """
    kwargs = _geometries()[rung]["archs"][arch].get("kwargs", {})
    if key not in kwargs:
        raise KeyError(
            f"{arch}@{rung} does not set {key!r} (sets {sorted(kwargs)}). "
            "Add it to param_match.ARCHS if the arm needs it.")
    return kwargs[key]


# --- citations into peak_mbs.yaml ------------------------------------------
def peak_mbs(section: str, rung: str, arch: str) -> int:
    """Largest per-GPU micro-batch measured to fit, for this stack on this GPU
    at this context.

    `section` is `<gpu>_seq<context>_<stack>`, composed in the config from
    `${cluster.gpu}`, `${data.seq_len}` and `${backend.name}` rather than
    passed as three more arguments -- the citation reads one table and the
    config says which.

    No interpolation along any of the three: attention holds its ceiling from
    2048 to 4096 while the linear arms halve, a GH200 holds three times an
    A100-40GB, and the stack moves it again -- the `lm/` sweep puts 124M at 16
    on an A100 where torchtitan OOMs four of the six arms. A guess is wrong in
    several directions at once.
    """
    spec = _load_yaml("peak_mbs.yaml")
    section_name = str(section)
    section = spec.get(section_name)
    if section is None:
        raise KeyError(
            f"peak_mbs.yaml has no {section_name!r} section (has "
            f"{sorted(spec)}). Sections are <gpu>_seq<context>_<stack>; "
            "measure the missing one with lm_scaling/titan_mbs_probe.py "
            "rather than borrowing another stack's, which is how four arms "
            "OOM'd at 124M.")
    key = f"{arch}|{rung}"
    if key not in section:
        raise KeyError(
            f"no measured micro-batch ceiling for {key} in {section_name}. "
            "Measure it (titan_mbs_probe.py); do not interpolate.")
    return int(section[key])


# --- naming ----------------------------------------------------------------
def budget_tag(tokens: float) -> str:
    """Filesystem-safe name for a token count.

    Enough decimals to keep small budgets apart: `{D:.0f}B` collapses 0.05B
    and 0.1B onto "0B", which would make two annealings share one config file
    and one checkpoint name.
    """
    b = float(tokens) / 1e9
    text = f"{b:.3f}".rstrip("0").rstrip(".") if b < 1 else f"{b:.1f}".rstrip("0").rstrip(".")
    return text.replace(".", "p") + "B"


_RESOLVERS = {
    "matched": matched,
    "flavor": flavor,
    "rung": rung,
    "arch_kwarg": arch_kwarg,
    "peak_mbs": peak_mbs,
    "budget_tag": budget_tag,
}


def install(replace: bool = True) -> None:
    """Register the citations, plus the library's composition primitives.

    `hydra_staged_sweep` supplies `oc.divi`, `oc.cdivi`, `oc.muli`, `oc.eval`,
    `oc.if` and friends -- the arithmetic a config needs to combine citations
    itself, so none of it has to be wrapped in a resolver here.
    """
    from hydra_staged_sweep.config.resolvers import register_default_resolvers

    register_default_resolvers(force=replace)
    for name, fn in _RESOLVERS.items():
        OmegaConf.register_new_resolver(name, fn, replace=replace)
