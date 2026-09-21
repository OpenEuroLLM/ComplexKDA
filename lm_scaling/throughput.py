"""Measured throughput, loaded from `throughput.yaml`.

The numbers are data and live in the YAML; this is only the lookup. It exists
because `pack_launch.py` groups runs by expected wall clock, and getting that
from a measurement rather than from step counts is what keeps a node's four
jobs finishing together.

It used to live in `launch.py`, which was one of two Python launchers
duplicating what `compose.py` already does from YAML. Deleting those took the
table with it, so it moved here rather than being copied a fourth time --
there were already three separate copies of the micro-batch ceilings, and one
of them silently OOM'd a campaign.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_SPEC = yaml.safe_load((_HERE / "throughput.yaml").read_text())
# Tokens per second per GPU, measured. Separate file because it is a
# different UNIT, and one file holding two units is how a number gets read
# with the wrong one.
_RATES = yaml.safe_load((_HERE / "token_rate.yaml").read_text())


def _table(section: str) -> dict:
    """`{(arch, rung): {mbs: tflops}}` for one section.

    Micro-batch keys are coerced to int: YAML will hand back `8` or `"8"`
    depending on how the file was written, and a string key silently makes
    every `m <= mbs` comparison raise -- or worse, sorts lexically.
    """
    return {tuple(k.split("|")): {int(m): float(t) for m, t in v.items()}
            for k, v in (_SPEC.get(section) or {}).items()}


def _sections(gpu: str) -> list[tuple[str, dict]]:
    """This GPU's tables, most-preferred first, then the other GPU's.

    Falling back across GPUs is allowed and LABELLED (see `measured_at`),
    because an estimate from the wrong machine beats no estimate -- but it has
    to say so. The A100/GH200 gap is 1.6x on attention and 2.3x on
    complex-kda, so an unlabelled fallback is wrong by an amount that depends
    on which arm you ask about.
    """
    names = [n for n in _SPEC if n.startswith(f"{gpu}_")]
    names += [n for n in _SPEC if not n.startswith(f"{gpu}_")]
    return [(n, _table(n)) for n in names]


# Stand-ins for a geometry or an arm with no measurement of its own: read the
# nearest one and say so. Empty here -- every arm this release runs was probed
# at its own geometry (peak_mbs.yaml, token_rate.yaml) -- and kept as the
# place a proxy goes, because the alternative is a lookup that silently
# returns a number measured on something else.
PROXY_RUNG: dict[str, str] = {}
PROXY_ARCH: dict[str, str] = {}


def _ordered(gpu: str, seq_len: int,
             prefer: str | None = None) -> list[tuple[str, dict]]:
    """Sections to try, best first: the caller's exact regime if it named one,
    then this GPU at this context, then this GPU's other contexts, then
    everything else.

    `prefer` is the whole regime -- `<gpu>_seq<context>_<stack>_g<world>` --
    and it outranks everything because it is the only name that pins the
    stack and the world size. Two sections measured under different trainers
    can share a `<gpu>_seq<context>` prefix and be 1.08-1.54x apart; without
    this the alphabet would decide which prices the campaign.

    The GPU has to outrank the alphabet for the same reason. Sorting names
    alone put `a100_seq4096` ahead of `gh200_seq4096_derived`, so a GH200 cost
    silently came off the A100 table -- a 2.3x error, labelled but in the
    wrong direction from what the label implied.
    """
    want = f"{gpu}_seq{int(seq_len)}"
    return sorted(_sections(gpu),
                  key=lambda nt: (nt[0] != prefer,
                                  nt[0] != want,
                                  not nt[0].startswith(f"{gpu}_"),
                                  "_derived" in nt[0], nt[0]))


def measured_tflops(arch: str, rung_tag: str, mbs: int, seq_len: int = 4096,
                    gpu: str = "gh200", prefer: str | None = None):
    """Throughput at the nearest measured micro-batch at or below `mbs`.

    Rounds DOWN rather than to the nearest: throughput rises with the
    micro-batch, so the lower neighbour under-promises. A pessimistic estimate
    costs a longer time limit; an optimistic one costs a job killed at 90%.

    Prefers a measurement at the requested context and falls back to another,
    because that is better than nothing -- but contexts are kept in separate
    sections so the fallback is a choice rather than an accident.
    """
    for _, table in _ordered(gpu, seq_len, prefer):
        for a in (arch, PROXY_ARCH.get(arch, arch)):
            for r in (rung_tag, PROXY_RUNG.get(rung_tag, rung_tag)):
                entry = table.get((a, r))
                if entry:
                    below = [m for m in entry if m <= mbs]
                    return entry[max(below)] if below else entry[min(entry)]
    return None


def measured_at(arch: str, rung_tag: str, seq_len: int,
                gpu: str = "gh200", prefer: str | None = None) -> str:
    """Which sweep a price for this cell actually comes from.

    Worth asking out loud. Only one cell of the 4096 sweep has been measured,
    so almost every estimate is a 2048 number wearing a 4096 label -- and the
    two are not close for attention, whose cost per token roughly doubles
    between the contexts. A price that does not say where it came from is a
    price that gets quoted as a measurement.
    """
    want = prefer or f"{gpu}_seq{int(seq_len)}"
    for section, table in _ordered(gpu, seq_len, prefer):
        for a in (arch, PROXY_ARCH.get(arch, arch)):
            for r in (rung_tag, PROXY_RUNG.get(rung_tag, rung_tag)):
                if table.get((a, r)):
                    return section if section == want else f"{section}->{want}"
    return "none"


def token_rate(arch: str, rung_tag: str, mbs: int,
               prefer: str | None = None) -> float | None:
    """Measured tokens per second per GPU, or None if this cell has none.

    Exact by construction: it is torchtitan's own `tps:` field, so using it
    needs no FLOPs-per-token convention. Rounds the micro-batch DOWN like
    `measured_tflops`, and for the same reason -- the lower neighbour
    under-promises, and a pessimistic estimate costs a longer time limit
    where an optimistic one costs a job killed at 90%.
    """
    # EXACT REGIME ONLY -- no fallback of any kind.
    #
    # `measured_tflops` may fall back across GPUs and contexts because
    # `measured_at` LABELS the result, and an estimate from the wrong machine
    # beats no estimate as long as it says so. Nothing labels this one: it
    # returns a bare number that goes straight into a wall-clock figure. A
    # first draft searched every section and priced a GH200 run off the A100
    # table -- a 2.3x error on complex-kda, silently, in the direction that
    # makes a job look done before it is.
    #
    # An unmeasured regime returns None, and `gpu_hours` drops to the TFLOP
    # path, which does label what it used.
    section = (_RATES.get(prefer) or {}) if prefer else {}
    entry = section.get(f"{arch}|{rung_tag}") or section.get(
        f"{PROXY_ARCH.get(arch, arch)}|{PROXY_RUNG.get(rung_tag, rung_tag)}")
    if not entry:
        return None
    entry = {int(m): float(t) for m, t in entry.items()}
    below = [m for m in entry if m <= mbs]
    return entry[max(below)] if below else entry[min(entry)]


def gpu_hours(arch: str, rung_tag: str, total_steps: int, gbs: int,
              seq_len: int, gpus: int = 4, gpu: str = "gh200",
              mbs: int | None = None, prefer: str | None = None):
    """Estimate from measured throughput; None when we have no measurement.

    Passes `seq_len` through to the lookup so a 4096 run is priced on the 4096
    measurements. It used to be priced on the 2048 ones whatever the context,
    which understated attention by ~40%.
    """
    # The micro-batch the job will RUN at, when the caller knows it. Deriving
    # it as gbs // gpus is the per-GPU share, which is what the job would use
    # if memory were free -- at 302M/20BT that is 32 where the cell actually
    # runs at 4, and the rate at 4 is not the rate at 32.
    at_mbs = mbs or max(1, gbs // gpus)
    tokens = total_steps * gbs * seq_len

    # A MEASURED TOKEN RATE FIRST, because it needs no conversion.
    #
    # Going through TFLOP/s requires a FLOPs-per-token convention, and it has
    # to be the one the rate was measured under. torchtitan's is
    #   6*(nparams - nparams_embedding) + 6*n_layers*n_heads*head_dims*seq_len
    # and the arithmetic below drops that second term, so the round trip is
    # wrong by their ratio -- 3.0x at 47M/4096, all of it optimistic. The
    # token rate has no denominator to disagree about.
    rate = token_rate(arch, rung_tag, at_mbs, prefer)
    if rate:
        return tokens / rate / 3600

    # NO FALLBACK THROUGH throughput.yaml, deliberately. Reconstructing a rate
    # from TFLOP/s needs a FLOPs-per-token convention, and getting it wrong is
    # a silent 3x -- see above. Every cell this release runs was probed at its
    # own geometry, so an unpriced cell means an unmeasured one, and the
    # honest answer to that is None: the caller reports "no estimate" instead
    # of quoting an arithmetic guess as a measurement.
    return None
