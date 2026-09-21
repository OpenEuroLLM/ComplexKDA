"""Notes on resolving sweeps inside the test suite.

`hydra_staged_sweep` hands independent dependency chains to a `fork` pool, and
forking from a process that already has threads -- which this one does, since
importing torch starts several -- is the case its own DeprecationWarning
names: "This process is multi-threaded, use of fork() may lead to deadlocks in
the child."

It happened once here, after three tests were added that each resolved the
228-job upper ladder: seventeen forked workers idle for eleven minutes, no
output and no failure. A suite that deadlocks is worse than one that fails,
because the hang is indistinguishable from a slow machine.

The fix is NOT to pin the pool to one worker. Measured on the upper ladder,
that is 75s a resolution against 11s -- the whole suite would spend minutes
resolving configs it already resolved.

The fix is to resolve each campaign ONCE. Every plan in these tests comes from
a module-scoped fixture, so a campaign is expanded a single time however many
properties are asserted about it. If a hang comes back, look first for a test
calling `P.plans` directly instead of taking a fixture.

    HYDRA_STAGED_SWEEP_WORKERS=1 pytest tests/lm/   # to rule the pool out
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "lm_scaling") not in sys.path:
    sys.path.insert(0, str(_ROOT / "lm_scaling"))


def reachable(experiment, path, overrides=()):
    """An override that ADDS `path` to the container's bind list.

    `plan.check` refuses a `runs_dir` no bind covers, because on JUPITER /p is
    mounted on the login node and not on the compute nodes: 98 probes pointed
    there ran 55 seconds each and exited 0 having written nothing. A test that
    relocates a campaign into `tmp_path` therefore has to say the container can
    reach it.

    It has to EXTEND the list rather than replace it. `++container.bind=[/tmp]`
    drops the two host-driver binds, and the rendered script then has no
    `libcuda.so.1` -- which is the "Found no NVIDIA driver on your system"
    failure, arriving here as an assertion instead of at 3am.
    """
    import plan as P

    binds = list(P.load(experiment, list(overrides)).container.bind) + [str(path)]
    return "++container.bind=[" + ",".join(binds) + "]"
