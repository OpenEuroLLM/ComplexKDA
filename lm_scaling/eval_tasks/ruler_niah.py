"""lm-eval's own RULER builders, re-exported so the configs beside this file
can name them.

`!function` resolves its module from the directory the YAML lives in, so a
config here cannot write `niah_utils.niah_single_1` -- that module is inside
lm-eval's `tasks/ruler` package, not this one. Re-exporting is the entire
content of this file.

NOTHING about the benchmark is redefined here: the haystacks, the needles, the
500 samples per cell, the prompt template and the substring scoring are
upstream's, which is the only way our rows and Gated DeltaNet-2's Table 3 are
the same measurement. What the configs beside this file change is the METRIC
LIST -- see `niah_single_1.yaml`.
"""

from lm_eval.tasks.ruler.common_utils import (  # noqa: F401
    aggregate_metrics,
    process_results,
)
from lm_eval.tasks.ruler.niah_utils import (  # noqa: F401
    niah_multikey_1,
    niah_single_1,
    niah_single_2,
    niah_single_3,
)
