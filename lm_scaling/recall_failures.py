"""Split a recall-suite score into the two things it could be measuring.

    lm_scaling/recall_failures.py --results <dir with *_diag.samples.json>
    lm_scaling/recall_failures.py --results DIR --task fda --show 5

THE QUESTION IS THE ONE THE RULER CELLS FAILED. `contains_score` gives 0 both
to a model that fetched the wrong value and to a model that produced no value
at all, and on the S-NIAH tasks these arms turned out to be doing the second --
which retired a reading of the whole RULER table (see
`config/experiments/fwedu_1p3B_results.md`, "What those single-needle cells
actually measure"). Any difference in `recall_table.py` has to survive the same
question before it can be read as recall.

IT DOES. At n = 200 a task the answer rates are 95-100% everywhere except the
two recurrent arms on FDA (~83%), and `correct | answered` tracks the raw score
throughout. The decisive cell is SQuAD in the hybrid stack: both arms answer
200 of 200 and the bounded gate is right 32.5% against the signed gate's 22.0%,
so nothing about compliance can explain the gap.

WHY THE CRITERION DIFFERS FROM `niah_failures.py`. There an answer has a
machine-checkable SHAPE -- a 7-digit number, a UUID -- so a wrong answer can be
recognised as an answer. Here the targets are arbitrary prose ("Clearance of a
new device", "Paris, France"), and no pattern separates a wrong value from
prose that is not a value at all. So the test is inverted: a generation counts
as an ANSWER unless it is empty or it simply continues the prompt. That is a
weaker instrument -- it cannot catch a fluent non-answer that does not echo --
and it is calibrated to catch the failure mode that was actually observed on
these arms, which is the degenerate restatement loop.

Needs `--log-samples` output from eval_downstream.py, which eval_recall.sbatch
writes under a `_diag` TAG. Those dumps are diagnostic: `recall_table.py`
refuses to merge them and nothing they produce belongs in the table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ["squad_completion", "swde", "fda"]

# How much of a generation has to reappear in the prompt before it is an echo
# rather than a value. Long enough that a value which repeats the key -- "year:
# 1983" against a prompt ending "year:" -- is not miscounted as an echo, and
# short enough to catch a loop that restates the tail and then drifts.
ECHO_PREFIX = 40


def _text(row: dict) -> str:
    """The generated string, out of lm-eval's nested resps."""
    g = row.get("generated")
    while isinstance(g, list) and g:
        g = g[0]
    return str(g or "").strip()


def analyse(rows: list[dict]) -> dict:
    """(scored, answered, empty, echo, correct | answered) over one task."""
    n = len(rows)
    hit = ans = empty = echo = 0
    for r in rows:
        gen, tail = _text(r), str(r.get("prompt_tail") or "")
        # `hit` is the sample dump's own containment flag, which
        # eval_downstream computes case-insensitively to agree with
        # `contains_score`. Recomputing it here would be a second definition of
        # the metric and a chance for the two to drift.
        if r.get("hit"):
            hit += 1
        if not gen:
            empty += 1
        elif len(gen) >= ECHO_PREFIX and gen[:ECHO_PREFIX] in tail:
            echo += 1
        else:
            ans += 1
    return {"n": n, "scored": 100 * hit / max(n, 1),
            "answered": 100 * ans / max(n, 1),
            "empty": 100 * empty / max(n, 1), "echo": 100 * echo / max(n, 1),
            "correct_given_answered": 100 * hit / max(ans, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True,
                    help="a directory of *.samples.json from --log-samples")
    ap.add_argument("--task", default=None, help="restrict to one task")
    ap.add_argument("--show", type=int, default=0,
                    help="print this many failing generations per (arm, task)")
    a = ap.parse_args(argv)

    dumps = sorted(a.results.glob("*.samples.json"))
    if not dumps:
        raise SystemExit(
            f"no *.samples.json under {a.results} -- this reads a --log-samples "
            f"dump, which eval_recall.sbatch writes with LOG_SAMPLES=1")

    print(f"{'arm':30s} {'task':17s} {'n':>4s} {'scored':>7s} {'answered':>9s} "
          f"{'empty':>6s} {'echo':>6s} {'corr|ans':>9s}")
    for path in dumps:
        arm = path.name.split("_step")[0]
        data = json.loads(path.read_text())
        for task in (TASKS if a.task is None else [a.task]):
            rows = data.get(task)
            if not rows:
                continue
            s = analyse(rows)
            print(f"{arm:30s} {task:17s} {s['n']:4d} {s['scored']:6.1f}% "
                  f"{s['answered']:8.1f}% {s['empty']:5.1f}% {s['echo']:5.1f}% "
                  f"{s['correct_given_answered']:8.1f}%")
            for row in [r for r in rows if not r.get("hit")][:a.show]:
                print(f"      target {str(row.get('target'))[:60]!r}")
                print(f"      got    {_text(row)[:110]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
