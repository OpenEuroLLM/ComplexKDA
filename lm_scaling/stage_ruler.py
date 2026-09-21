"""Put everything RULER's needle tasks need where a compute node can reach it.

    lm_scaling/container_run lm_scaling/stage_ruler.py        # ON A LOGIN NODE

RUN THIS ON A LOGIN NODE. Everything below either downloads or verifies a
download, and the compute nodes have no route out -- which is the failure this
script exists to move forward in time. Unstaged, an eval job dies after the
export and before any generation, with a message about reaching the Hub.

THREE THINGS, none of which is in the image or the shared cache today:

1. `wonderwords`, imported at the top of lm-eval's `prepare_niah` and so
   required by every NIAH task whether or not its haystack uses generated
   words. Installed into CKDA_PYLIBS beside the harness, --no-deps, the way
   the rest of that tree was built.

2. nltk's `punkt_tab`. `prepare_niah.download_nltk_resources()` runs AT IMPORT
   and calls `nltk.download` when `nltk.data.find` misses, so an unstaged
   compute node does not fail at the download -- it fails inside an import,
   which reads as a broken harness rather than a missing corpus. NLTK_DATA has
   to be exported by the job too; eval_ruler.sbatch does it.

3. `baber/paul_graham_essays`, the haystack for S-NIAH-2, S-NIAH-3 and
   MK-NIAH-1. Note that lm-eval ALSO ships `tasks/ruler/essays.py`, which
   scrapes those essays over HTTP with httpx and BeautifulSoup. That path is
   not the one `get_haystack` takes (prepare_niah.py:335 loads the dataset),
   so none of httpx, html2text or bs4 is needed here. Staging them would be
   staging for a code path that never runs.

Then it BUILDS one cell of each task offline, which is the only check that
matters: the three steps above are each individually plausible and still leave
a job that cannot construct a sample.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PYLIBS = os.environ.get("CKDA_PYLIBS", "/e/project1/e-sta-openeurollm/poeppel1/pylibs")

# THE EVAL CACHE, and it is not the training cache.
#
# `/e/project1/jureap59/poeppel1/cache` -- what this and every other eval script
# used to pin -- is now a DANGLING SYMLINK; its target under /e/scratch was
# purged. Both failures it produces read as network faults rather than as a
# missing directory, and 109 eval jobs died on it before that was understood.
# The live eval cache is below, and the training cache
# (/e/fscratch/jureap59/poeppel1/cache, CKDA_HF_HOME) is a different place that
# carries the tokenizer and not one task dataset.
#
# BOTH HALVES ARE PINNED, because HF_HOME controls only one of them. A
# `load_dataset` writes the hub download to `$HF_HOME/hub` and the built arrow
# to HF_DATASETS_CACHE, the container sets HF_DATASETS_CACHE explicitly, and an
# explicit setting wins over HF_HOME. Setting only HF_HOME therefore moves half
# the cache and leaves the other half wherever the container points -- which
# cost a second failed pass, and which made an earlier RULER staging run agree
# with its job only because both inherited the same container.
EVAL_HF = os.environ.get("CKDA_EVAL_HF_HOME",
                         "/e/project1/e-sta-openeurollm/poeppel1/cache/hf")
HF_HOME = EVAL_HF
HF_DATASETS_CACHE = EVAL_HF
NLTK_DATA = os.environ.get("NLTK_DATA", os.path.join(HF_HOME, "nltk_data"))
ESSAYS = "baber/paul_graham_essays"
TOKENIZER = os.environ.get(
    "CKDA_TOKENIZER",
    "/e/scratch/e-sta-openeurollm/poeppel1/datasets/fineweb-edu-100BT-llama2/tokenizer")


def pylibs_on_path() -> None:
    """The harness tree, on sys.path the way an eval job puts it there.

    NOT just PYLIBS itself. nltk lives one level down, in the `--target`
    directory `$PYLIBS/lm_eval` that holds the dependency tree, so appending
    only the root leaves `import nltk` failing inside the container -- which
    reads as "the image has no nltk" rather than "this script did not set the
    path the job sets". APPENDED, never prepended: those directories carry
    their own copies of torch's dependencies.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_downstream import PYLIB_DIRS

    for d in PYLIB_DIRS:
        p = os.path.join(PYLIBS, d) if d else PYLIBS
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


def stage_wonderwords() -> None:
    try:
        import wonderwords  # noqa: F401
        print(f"  wonderwords: already in {PYLIBS}")
        return
    except ImportError:
        pass
    print(f"  wonderwords: installing into {PYLIBS}")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps",
                    "--target", PYLIBS, "wonderwords"], check=True)


def stage_nltk() -> None:
    import nltk

    os.makedirs(NLTK_DATA, exist_ok=True)
    if NLTK_DATA not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA)
    try:
        nltk.data.find("tokenizers/punkt_tab")
        print(f"  punkt_tab: already under {NLTK_DATA}")
        return
    except LookupError:
        pass
    print(f"  punkt_tab: downloading into {NLTK_DATA}")
    nltk.download("punkt_tab", download_dir=NLTK_DATA)
    nltk.data.find("tokenizers/punkt_tab")


def stage_essays() -> None:
    import datasets

    # Online on purpose: this is the one process allowed to fetch. The eval job
    # sets HF_DATASETS_OFFLINE=1 and reads what this leaves behind.
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    ds = datasets.load_dataset(ESSAYS, split="train")
    # The RESOLVED path, not the variable we set. datasets computes its cache
    # at import from whichever of HF_DATASETS_CACHE / HF_HOME it finds first,
    # and reporting the intention rather than the result is how a staging run
    # says "cached under $WORK" while the arrow tables sit on another
    # filesystem entirely.
    print(f"  {ESSAYS}: {len(ds)} essays, built under "
          f"{datasets.config.HF_DATASETS_CACHE}")


TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1")


def verify(tokenizer: str) -> None:
    """Build the tasks THE WAY THE JOB DOES, with the caches closed.

    OFFLINE FROM HERE. A verification allowed to download verifies nothing: it
    would pass on the login node and the job would still die on the node.

    Through the TaskManager and the include path, not by calling the builders
    directly, because three separate things have to line up and only one of
    them is the data:

      * our config in `lm_scaling/eval_tasks` has to WIN over lm-eval's own
        config of the same name -- it is a same-name override, so if the
        include path is not honoured the job silently runs the stock grid;
      * `!function ruler_niah.*` has to resolve from that directory;
      * the metric list has to contain every length asked for. A returned key
        with no metric entry is dropped at aggregation, which is a cell that
        renders blank after the generation has already been paid for.

    Each of those fails quietly at a point hours after the job starts.
    """
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["NLTK_DATA"] = NLTK_DATA

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from eval_downstream import ensure_lm_eval

    ensure_lm_eval()
    from lm_eval.tasks import TaskManager, get_task_dict

    lengths = [1024, 2048]
    tm = TaskManager(include_path=str(here / "eval_tasks"),
                     metadata={"max_seq_lengths": lengths, "tokenizer": tokenizer})
    for name, task in get_task_dict(list(TASKS), tm).items():
        metrics = [m["metric"] for m in task.config.metric_list]
        missing = [str(n) for n in lengths if str(n) not in metrics]
        assert not missing, (
            f"{name}: no metric entry for {missing}. The stock config's grid "
            f"starts at 4096, so this is our override NOT being picked up -- "
            f"check the include path and the `task:` names.")

        docs = list(task.test_docs())
        built = sorted({d["max_length"] for d in docs})
        assert built == lengths, (
            f"{name}: built {built}, asked for {lengths}. The metadata did not "
            f"reach the builder -- see the TaskManager note in "
            f"eval_downstream.main.")

        doc = docs[0]
        # The needle has to be IN the haystack. prepare_niah asserts it too; a
        # task whose samples are unanswerable scores 0.0 for every arm and
        # reads as a finding about the models.
        assert doc["outputs"][0] in doc["input"], name
        # And a perfect answer has to score 1.0 under the metric key for its own
        # length: that is the scoring path end to end, minus the model.
        scored = task.process_results(doc, [doc["outputs"][0]])
        assert set(scored) <= set(metrics), (
            f"{name}: returned {sorted(set(scored) - set(metrics))}, which no "
            f"metric entry names; those cells would be dropped at aggregation")
        assert scored[str(doc["max_length"])] == 1.0, (name, scored)
        print(f"  {name}: {len(docs)} docs at {built}, needle present, "
              f"perfect answer scores 1.0")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help="the campaign's tokenizer -- RULER counts tokens with "
                         "it to fill each cell, so a sample built with another "
                         "one is a different length than its label says")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--hf-home", default=HF_HOME,
                    help="hub cache; the ambient value is overridden on purpose")
    ap.add_argument("--hf-datasets-cache", default=HF_DATASETS_CACHE,
                    help="built-dataset cache, which HF_HOME does NOT control "
                         "-- see the constants' note")
    a = ap.parse_args(argv)

    os.environ["HF_HOME"] = a.hf_home
    os.environ["HF_DATASETS_CACHE"] = a.hf_datasets_cache
    globals()["HF_HOME"] = a.hf_home
    # NLTK_DATA hangs off the hub cache, so it follows --hf-home rather than
    # keeping the value the module computed from the default.
    globals()["NLTK_DATA"] = (os.environ.get("NLTK_DATA")
                              or os.path.join(a.hf_home, "nltk_data"))
    print(f"staging RULER with CKDA_PYLIBS={PYLIBS}")
    print(f"  hub cache      HF_HOME          = {a.hf_home}")
    print(f"  dataset cache  HF_DATASETS_CACHE= {a.hf_datasets_cache}")
    print(f"  nltk           NLTK_DATA        = {NLTK_DATA}")
    print("  eval_ruler.sbatch pins the same three; if they drift, the job "
          "re-downloads on a node with no route out")
    pylibs_on_path()
    if not a.verify_only:
        stage_wonderwords()
        stage_nltk()
        stage_essays()
    print("verifying, offline:")
    verify(a.tokenizer)
    print("staged. eval_ruler.sbatch can run on a compute node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
