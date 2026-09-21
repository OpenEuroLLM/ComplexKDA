"""Put the three recall-intensive datasets where a compute node can reach them.

    lm_scaling/container_run lm_scaling/stage_recall.py       # ON A LOGIN NODE

RUN THIS ON A LOGIN NODE, for the same reason `stage_ruler.py` says so: every
download below happens here or not at all, and a compute node that has to fetch
dies after the model is loaded with a message about reaching the Hub.

THE THREE TASKS are Based's recall-intensive suite (arXiv:2402.18668), which
lm-eval ships as `squad_completion`, `swde` and `fda`. They are GENERATION
tasks scored by substring containment -- the model is given a document and an
answer prefix and has to emit the value -- which is what makes them the
in-context recall counterpart to the common-sense table: nothing in them can be
answered from parameters.

  squad_completion  hazyresearch/based-squad      SQuAD, rewritten as completion
  swde              hazyresearch/based-swde-v2    relation extraction from HTML
  fda               hazyresearch/based-fda        key-value extraction from 510(k)

All three are PARQUET repositories, so unlike social_iqa/boolq/winogrande they
need no in-repo task override: `datasets` 4.x loads them directly and the stock
configs work once the bytes are in the cache. That is checked here rather than
assumed -- a script-based repo would fail on the node, offline, as a
ConnectionError.

Then it BUILDS all three offline and prints what they will cost: the document
count, the generation budget each task asks for, and the PROMPT LENGTH
distribution against the 4,096-token context the arms trained at. The last one
is the number that decides whether the job measures recall or measures
truncation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PYLIBS = os.environ.get("CKDA_PYLIBS", "/e/project1/e-sta-openeurollm/poeppel1/pylibs")

# THE EVAL CACHE, BOTH HALVES -- the same two values eval_recall.sbatch pins.
# `/e/project1/jureap59/poeppel1/cache`, which the older eval scripts still
# name, is a DANGLING SYMLINK; see stage_ruler.py's note and the 109 jobs it
# cost. HF_HOME owns only the hub download: `datasets` builds its arrow tables
# under HF_DATASETS_CACHE, the container sets that explicitly, and an explicit
# setting wins -- so staging with one of them moves half the cache and the job
# re-downloads the other half on a node with no route out.
EVAL_HF = os.environ.get("CKDA_EVAL_HF_HOME",
                         "/e/project1/e-sta-openeurollm/poeppel1/cache/hf")

TOKENIZER = os.environ.get(
    "CKDA_TOKENIZER",
    "/e/scratch/e-sta-openeurollm/poeppel1/datasets/fineweb-edu-100BT-llama2/tokenizer")

# (task name, repo, config name). The repo ids are lm-eval's own -- they live in
# `lm_eval/tasks/<name>/task.py` as DATASET_PATH, and `swde` points at the *-v2
# repository, not at `based-swde`. Staging the wrong one leaves a cache that
# looks full and a job that still cannot find its dataset.
DATASETS = [("squad_completion", "hazyresearch/based-squad", "default"),
            ("swde", "hazyresearch/based-swde-v2", "default"),
            ("fda", "hazyresearch/based-fda", "default")]

TASKS = [name for name, _, _ in DATASETS]


def pylibs_on_path() -> None:
    """The harness tree, on sys.path the way an eval job puts it there.

    APPENDED, never prepended: those directories carry their own copies of
    torch's dependencies, and putting them first is how
    `operator torchvision::nms does not exist` appeared here.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_downstream import PYLIB_DIRS

    for d in PYLIB_DIRS:
        p = os.path.join(PYLIBS, d) if d else PYLIBS
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)


def stage() -> None:
    """Fetch the three repositories. Online on purpose -- this is the one
    process allowed out; the job sets HF_*_OFFLINE=1 and reads what is left.

    THE FDA REPOSITORY HAS A TYPO'D FILE. Beside
    `data/validation-00000-of-00001.parquet` sits
    `data/validaion-00000-of-00001.parquet`, and how `datasets` treats it
    decides how many documents the task has: absorbed into the validation split
    it would double every FDA row and halve the accuracy, silently. The split
    sizes are printed below and `verify()` checks the count against the task's
    own, so a change in that resolution is visible rather than absorbed.
    """
    import datasets

    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    for name, repo, config in DATASETS:
        ds = datasets.load_dataset(repo, config)
        sizes = {split: len(rows) for split, rows in ds.items()}
        print(f"  {name:17s} {repo:30s} {sizes}")
    # The RESOLVED path, not the variable we set: `datasets` computes its cache
    # at import from whichever of HF_DATASETS_CACHE / HF_HOME it sees first, and
    # reporting the intention rather than the result is how a staging run claims
    # to have filled a cache the job does not read.
    print(f"  built under {datasets.config.HF_DATASETS_CACHE}")


def verify(tokenizer_dir: str, context: int) -> None:
    """Build the tasks THE WAY THE JOB DOES, with the caches closed.

    OFFLINE FROM HERE. A verification allowed to download verifies nothing: it
    passes on the login node and the job still dies on the compute node.

    Four things are checked, and each of them fails quietly hours into a job:

    * the dataset builds with no Hub access and has the documents the published
      task has;
    * `doc_to_target` is IN the prompt's document -- these are extraction tasks,
      so an answer that is not in the text is unanswerable and would score 0 for
      every arm, reading as a finding about the models;
    * a perfect answer scores 1.0 through the task's own `process_results`,
      which is the scoring path end to end minus the model;
    * how many prompts EXCEED the scoring context. `generate_until` keeps the
      END of an over-long prompt, so a truncated document loses its head --
      and for SWDE, where the answer sits near the top of an HTML page, that
      turns a recall measurement into a truncation measurement. The number is
      printed per task so the job's `--max-length` is a decision and not a
      default.
    """
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from eval_downstream import ensure_lm_eval, load_tokenizer

    ensure_lm_eval()
    from lm_eval.tasks import TaskManager, get_task_dict

    tok = load_tokenizer(tokenizer_dir)
    tm = TaskManager(include_path=str(here / "eval_tasks"))
    for name, task in get_task_dict(TASKS, tm).items():
        docs = list(task.validation_docs())
        doc = docs[0]

        # WHAT THE TASK WILL ASK FOR, read off the task rather than off its
        # README. These three set `until` and `max_gen_toks` inside
        # `construct_requests`, on top of whatever ConfigurableTask defaulted
        # generation_kwargs to -- so the budget that actually reaches the
        # decoder is only visible by building a request.
        (_, gen_kw), = [inst.arguments for inst in
                        task.construct_requests(doc, task.doc_to_text(doc))]
        target = task.doc_to_target(doc)
        assert target and target in task.doc_to_text(doc), (
            f"{name}: the target {target!r} is not in its own document. These "
            f"are extraction tasks; an unanswerable prompt scores 0 for every "
            f"arm and reads as a model result.")
        scored = task.process_results(doc, [target])
        assert scored.get("contains") == 1.0, (name, scored)

        lens = sorted(len(tok(task.doc_to_text(d))["input_ids"]) for d in docs)
        budget = context - int(gen_kw.get("max_gen_toks", 48))
        over = sum(1 for n in lens if n > budget)
        pct = lens[int(0.99 * (len(lens) - 1))]
        print(f"  {name:17s} {len(docs):5d} docs | until={gen_kw.get('until')!r} "
              f"max_gen_toks={gen_kw.get('max_gen_toks')} | prompt tokens "
              f"median {lens[len(lens) // 2]}, p99 {pct}, max {lens[-1]} | "
              f"{over} over the {budget}-token budget")
        if over:
            print(f"    NOTE: {100 * over / len(lens):.1f}% of {name} prompts "
                  f"lose their FRONT at --max-length {context}. lm-eval keeps "
                  f"the end of an over-long prompt, so those documents are "
                  f"measured with their head cut off.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help="the campaign's tokenizer -- the prompt-length report "
                         "is only about this job if it counts tokens with the "
                         "tokenizer the arms were trained with")
    ap.add_argument("--context", type=int, default=4096,
                    help="the scoring context the job will use, for the "
                         "truncation report. 4,096 is these arms' own "
                         "max_position_embeddings and what eval_recall.sbatch "
                         "scores at")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--hf-home", default=EVAL_HF,
                    help="hub cache; the ambient value is overridden on purpose")
    ap.add_argument("--hf-datasets-cache", default=EVAL_HF,
                    help="built-dataset cache, which HF_HOME does NOT control "
                         "-- see the constants' note")
    a = ap.parse_args(argv)

    os.environ["HF_HOME"] = a.hf_home
    os.environ["HF_DATASETS_CACHE"] = a.hf_datasets_cache
    print(f"staging the recall suite with CKDA_PYLIBS={PYLIBS}")
    print(f"  hub cache      HF_HOME           = {a.hf_home}")
    print(f"  dataset cache  HF_DATASETS_CACHE = {a.hf_datasets_cache}")
    print("  eval_recall.sbatch pins the same two; if they drift, the job "
          "re-downloads on a node with no route out")
    pylibs_on_path()
    if not a.verify_only:
        stage()
    print("verifying, offline:")
    verify(a.tokenizer, a.context)
    print("staged. eval_recall.sbatch can run on a compute node.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
