"""Every geometry a campaign can name, read as one table.

Two kinds of entry live here and they must not be mixed:

    ladder_matched.json    the six rungs of the scaling ladder, tied, NeoX
                           vocabulary, context 4096 -- one ladder, whose
                           fitted hyperparameter laws apply across it
    replication_1p3B.json  a FIXED POINT: the 1.3B geometry other groups
                           publish 100BT results at, untied, 32,000 vocabulary.
                           An experiment of its own, and the ladder's laws do
                           not apply to it

Both are generated -- `ladder_matched.py` and `replication_1p3B.py` -- and both
match every arm to the dense transformer's parameter count at that geometry.

Two consumers have to agree on this set: `titan_ext.ladder_flavors`, which
registers a torchtitan flavor per (arm, geometry), and the citations in
`resolvers.py`, which read widths and head geometry out of the same entries.
Each used to keep its own list, which held only while no config named a
geometry the other could not see.

Standard library only: `resolvers` imports this on the login node, whose venv
must not grow a torch dependency (make_login_venv.sh checks).
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent

LADDER = "ladder_matched.json"
FIXED_POINTS = ("replication_1p3B.json",)

# What every entry carries.
_SHARED = ("d_model", "n_layers", "n_heads", "head_dim", "tie_word_embeddings")
# The fields that make a fixed point a different experiment from any other.
# Passed through when present, never defaulted: a reader that needs one and
# does not find it should fail where it looks.
_OPTIONAL = ("vocab_size", "seq_len", "target_N")

# "This arm with SiLU dropped from q/k/v", as a flavor suffix, and the layers
# that can honour it. A campaign asks for it with `aux.drop_silu`
# (resolvers.flavor), and titan_ext.ladder_flavors registers the variant for
# every entry whose layer is listed here. The names live here because both of
# those read this module and neither may import the other: resolvers runs on
# the login node, titan_ext needs torch.
NOSILU = "-nosilu"
DROPS_SILU = ("complex-kda",)

_table: dict | None = None


def table() -> dict:
    """`{geometry tag: entry}` for the ladder's rungs and every fixed point.

    A tag defined twice RAISES. Two files each defining "1.3B" would make the
    widths a run gets depend on which was read last.
    """
    global _table
    if _table is None:
        out: dict = json.loads((_HERE / LADDER).read_text())
        for name in FIXED_POINTS:
            fixed = json.loads((_HERE / name).read_text())
            tag = fixed["tag"]
            if tag in out:
                raise ValueError(
                    f"{name} defines geometry {tag!r}, which {LADDER} or an "
                    "earlier fixed point already defines")
            entry = {k: fixed[k] for k in _SHARED}
            entry.update({k: fixed[k] for k in _OPTIONAL if k in fixed})
            entry["archs"] = fixed["archs"]
            out[tag] = entry
        _table = out
    return _table
