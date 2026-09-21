"""Downstream evaluation of an exported arm, on the GDN 1.3B table's tasks.

    lm_scaling/container_run lm_scaling/eval_downstream.py \
        --model <hf export dir> --out results.json

THE TASKS are the ones the paper this campaign is compared against reports
(Gated DeltaNet, arXiv:2412.06464, Table 3): wikitext and LAMBADA perplexity,
then accuracy on LAMBADA, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, SIQA and
BoolQ. SIQA is `social_iqa` in lm-eval-harness; `siqa` is not a registered
task name and asking for it fails the run rather than skipping the task.

WHY NOT `HFLM`, and so why not evals/harness.py. lm-eval's HuggingFace wrapper
imports `transformers.AutoModelForVision2Seq`, which transformers 5 removed;
the container ships 5.3.0, so every released lm_eval (0.4.5 and 0.4.9.1 both
checked) fails at import of that wrapper. Downgrading transformers inside the
image's path shadows the NVIDIA torch build the container's torchvision was
compiled against -- that is how `operator torchvision::nms does not exist`
appeared here. The task and metric machinery imports fine; only the model
wrapper is broken. So this subclasses the harness's own `LM` interface and
drives the fla model directly.

HOW THE HARNESS IS FOUND. lm_eval is not in the image: it is a --no-deps
checkout with its own dependency tree beside it, under `CKDA_PYLIBS`. Those
directories are APPENDED to sys.path, never prepended -- they hold a second
copy of torch's dependencies, and putting them first is how
`operator torchvision::nms does not exist` appeared here.

That has a second benefit. `AutoModelForCausalLM` cannot be used on these
exports at all: fla/models/new_ckda registers the same `model_type`
("complex_kda") as fla/models/complex_kda and wins the registry, so an Auto load
silently builds a layer whose config has no `conv_silu` -- evaluating with
SiLU ON a checkpoint trained with it off, at identical parameter counts and
with no error. `export_hf.load_exported()` constructs the classes explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Where the lm-eval-harness checkout and its dependencies live when they are
# not in the image. `lm_eval_049` (0.4.9.1) comes before `lm_eval`, which is the
# --target directory holding the dependency tree AND an older 0.4.5 package: on
# JUPITER both are present and the order is what decides which harness runs.
PYLIBS = os.environ.get(
    "CKDA_PYLIBS", "/e/project1/e-sta-openeurollm/poeppel1/pylibs")
PYLIB_DIRS = ("lm_eval_049", "lm_eval", "")


def ensure_lm_eval() -> None:
    """Make `import lm_eval` work in this container. Two obstacles, both outside
    this repo.

    THE PATH. The harness is not installed in the image, so `PYLIBS` is put on
    sys.path -- APPENDED, because those directories carry their own copies of
    torch's dependencies and shadowing the image's build is what produced
    `operator torchvision::nms does not exist`.

    THE MISSING CLASS. `lm_eval.models.hf_vlms` reads
    `transformers.AutoModelForVision2Seq` in a CLASS BODY, and transformers 5
    renamed that class to AutoModelForImageTextToText. The read happens while
    `lm_eval.models` imports, so it takes down `simple_evaluate` -- a
    vision-language wrapper this evaluation never touches. Assigning the
    attribute is not enough: transformers exposes its classes through a
    `_LazyModule.__getattr__`, so the name has to be answered there. The wrapper
    below delegates everything else to the original and resolves exactly the
    Auto* names transformers no longer has, which today is that one.
    """
    import transformers

    for d in PYLIB_DIRS:
        p = os.path.join(PYLIBS, d) if d else PYLIBS
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)

    lazy = type(transformers)
    if getattr(lazy, "_ckda_shimmed", False):
        return
    original = lazy.__getattr__

    def __getattr__(self, name):
        try:
            return original(self, name)
        except AttributeError:
            if self.__name__ != "transformers" or not name.startswith("AutoModel"):
                raise
            # The rename, and a generic fallback so a future removal costs a
            # wrong class in an unused wrapper rather than the whole run.
            return original(self, "AutoModelForImageTextToText"
                            if "Vision2Seq" in name else "AutoModel")

    lazy.__getattr__ = __getattr__
    lazy._ckda_shimmed = True


# The tasks, in the published tables' order. wikitext and lambada carry the
# perplexities; the rest are accuracies.
#
# openbookqa is not in Gated DeltaNet's Table 3 but IS in Gated DeltaNet-2's
# Table 2 (arXiv:2605.22791), which also averages over it. Scoring it costs
# 2,000 of ~80,000 requests, and not scoring it would make the newer -- and
# closer, it has a KDA row -- comparison impossible. Which columns and which
# average a table shows is eval_table.py's business, not this list's.
TASKS = ["wikitext", "lambada_openai", "piqa", "hellaswag", "winogrande",
         "arc_easy", "arc_challenge", "openbookqa", "social_iqa", "boolq"]

# Based's recall-intensive suite (Arora et al., arXiv:2402.18668), which
# lm-eval ships as these three names. NOT in `TASKS`, and that is the point:
# they GENERATE rather than score, they belong to a different published table,
# and folding them into the list above would put them inside the GDN average,
# which is defined over the common-sense tasks alone.
#
# What they measure is in-context recall -- a document and an answer prefix, and
# the answer is in the prompt or nowhere -- which is the axis a fixed-size
# recurrent state is expected to lose on. `eval_recall.sbatch` runs them and
# `recall_table.py` renders them.
RECALL_TASKS = ["squad_completion", "swde", "fda"]


def greedy_decode(model, rows, max_gen, pad_id, eos_id=None, device="cuda"):
    """Greedy-decode a batch of prompts of DIFFERENT lengths. Returns, per row,
    the generated token ids -- no prompt, stopped at `eos_id`.

    PADDING IS ON THE LEFT, and carries an attention mask. A recurrent model
    cannot ignore a pad token the way attention can: its state is a running
    product, so a pad consumed anywhere is part of the answer. fla's layer
    unpads before the kernel and repads after (`unpad_hidden_states`, with
    cu_seqlens), which is what makes a padded row equal to the same row run
    alone. `tests/lm/test_generate_padding.py` asserts that equality, because a
    contaminated state produces fluent wrong text rather than an error -- it
    would read as a model that cannot retrieve.

    LEFT rather than right so the last real token of every row sits at position
    -1, where `logits_to_keep=1` reads it. The alternative is materialising a
    [rows, 8192, 32000] logit tensor -- 4 TB -- to use 32,000 of its numbers.

    The DECODE steps pass no mask at all: every row contributes exactly one
    real token per step, which is precisely what an absent mask means. Rows
    that have already emitted `eos_id` keep being fed and their output
    discarded; stopping them properly would mean editing a batched recurrent
    state, and the saving is a few forwards.
    """
    import torch

    width = max(len(ids) for ids in rows)
    x = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for r, ids in enumerate(rows):
        x[r, width - len(ids):] = torch.tensor(ids, dtype=torch.long, device=device)
        mask[r, width - len(ids):] = 1

    with torch.no_grad():
        out = model(x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    cache, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)

    gen = [[] for _ in rows]
    alive = [True] * len(rows)
    for step in range(max_gen):
        for r in range(len(rows)):
            if not alive[r]:
                continue
            tok_id = int(nxt[r])
            if eos_id is not None and tok_id == eos_id:
                alive[r] = False
            else:
                gen[r].append(tok_id)
        if not any(alive) or step == max_gen - 1:
            break
        with torch.no_grad():
            out = model(nxt[:, None], past_key_values=cache, use_cache=True,
                        logits_to_keep=1)
        cache, nxt = out.past_key_values, out.logits[:, -1].argmax(-1)
    return gen


def load_tokenizer(name: str):
    """The tokenizer, offline, whether `name` is a directory or a Hub id.

    WHY THE FALLBACK. On a compute node with `HF_HUB_OFFLINE=1`, transformers 5
    resolves a Hub id by first fetching that repo's **config.json** -- it reads
    the model config to pick the tokenizer class -- and the cached
    EleutherAI/gpt-neox-20b snapshot holds only tokenizer files. There is no
    config.json to find and no Hub to ask, so `AutoTokenizer.from_pretrained`
    dies with "We couldn't connect to 'https://huggingface.co'", which reads as
    a network problem and is a cache-shape problem.

    Pointed at the snapshot DIRECTORY instead it never needs the config:
    `tokenizer_config.json` names `tokenizer_class` and that is enough.

    The id is tried first, so nothing changes wherever this already works; the
    fallback only runs after that has failed. (`$WORK/cache` used to hold a full
    snapshot and no longer holds anything, which is how a path that worked for
    the torchtitan ladder's sweep stopped working for this one.)
    """
    import glob
    import os

    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except OSError as exc:
        if os.path.isdir(name) or "/" not in name:
            raise
        home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
        repo = "models--" + name.replace("/", "--")
        cands = sorted(glob.glob(os.path.join(
            home, "hub", repo, "snapshots", "*", "tokenizer_config.json")))
        if not cands:
            raise SystemExit(
                f"cannot load tokenizer {name!r} offline: {exc}\n"
                f"  no snapshot with a tokenizer_config.json under "
                f"{os.path.join(home, 'hub', repo, 'snapshots')}") from exc
        snap = os.path.dirname(cands[-1])
        print(f"  tokenizer {name} -> {snap} (Hub id unresolvable offline)",
              flush=True)
        return AutoTokenizer.from_pretrained(snap)


def ablate_attention(model) -> int:
    """Silence every ATTENTION sublayer of a hybrid, leaving the rest intact.
    Returns how many were silenced; 0 means the export was not a hybrid.

    WHAT THIS ANSWERS. Both hybrids have the same six attention layers in the
    same places, and one of them retrieves perfectly while the other does not.
    That is either a difference in what the attention learned, or a difference
    in whether the model ROUTES retrieval through it at all -- and the two
    predict opposite things here. Take attention away: an arm that was relying
    on it collapses, an arm that was answering from its recurrent state barely
    notices.

    ZEROED, not removed. The sublayer keeps its place in the residual stream
    and contributes nothing, so every other layer sees the shapes and the
    normalisation it trained with. Deleting the block instead would renumber
    the layers and change the residual depth, which is a different model rather
    than the same model without attention.

    This is a DIAGNOSTIC. Nothing it produces belongs in the published table:
    an ablated arm is not the model that trained.
    """
    import torch

    from fla.layers.attn import Attention

    silenced = 0
    for block in model.model.layers:
        inner = getattr(block, "attn", None)
        # A hybrid block's `.attn` is `Attention`; a recurrent block's is the
        # KDA mixer, which is the thing being measured and must be left alone.
        if not isinstance(inner, Attention):
            continue

        def _silent(hidden_states, *args, past_key_values=None, **kwargs):
            return torch.zeros_like(hidden_states), None, past_key_values

        inner.forward = _silent
        silenced += 1
    return silenced


def window_attention(model, window: int) -> int:
    """Make a hybrid's attention LOCAL, keeping it otherwise intact.

    The gentler half of the pair with `ablate_attention`, and the one that can
    actually answer the routing question. Zeroing six of twenty-four layers
    removes the capability AND breaks the model -- both hybrids went to 0.0,
    which is what "attention carried retrieval" and "the ablation destroyed the
    network" both predict, so it discriminates nothing.

    A window keeps every layer computing, keeps the residual stream's
    statistics roughly where training left them, and removes only the LONG-RANGE
    half of attention. A RULER needle sits hundreds to thousands of tokens
    before the query, so a window of a few hundred puts it out of reach while
    leaving local attention to do whatever local work it was doing. Retrieval
    that survives that came from the recurrent state; retrieval that does not
    was being carried by attention.

    `window_size` is read at forward time and passed to flash-attn (and to the
    cache), so setting it here is the same switch the config would have set.

    DIAGNOSTIC. A windowed arm is not the model that trained.
    """
    from fla.layers.attn import Attention

    changed = 0
    for block in model.model.layers:
        inner = getattr(block, "attn", None)
        if not isinstance(inner, Attention):
            continue
        inner.window_size = window
        changed += 1
    return changed


def context_prefix_ids(tok, policy: str) -> list[int]:
    """The token ids to put in front of every scored context.

    WHY THIS IS A KNOB AND NOT A DEFAULT OF THE TOKENIZER. lm-eval's own HFLM
    tokenizes a causal model's context with `add_special_tokens=False` unless
    you ask otherwise -- it OVERRIDES whatever `add_bos_token` a tokenizer
    config carries (the one exception is a hardcoded one for Gemma). This
    class is a plain `LM` rather than an HFLM, because an export cannot be
    loaded through `AutoModelForCausalLM`, and in writing its own tokenisation
    it called `self.tok(context)` with specials ON. The fwedu tokenizer copy
    sets `add_bos_token: True`, so every context was prefixed with `<s>` -- a
    token `prepare_fineweb_edu.py` deliberately never writes
    (`add_special_tokens=False`, EOS appended by the writer), and which occurs
    5 times in 200M tokens of the corpus.

    The NeoX ladder was unaffected: GPTNeoX sets no `add_bos_token`, so its
    contexts were already bare.

    "eos" is not the old behaviour restored under another name -- it is the
    separator these corpora actually use, and the token that precedes every
    document's first token in training.
    """
    if policy == "none":
        return []
    if policy == "bos":
        tid = tok.bos_token_id
    elif policy == "eos":
        tid = tok.eos_token_id
    else:
        raise ValueError(f"unknown context prefix policy {policy!r}")
    if tid is None:
        raise SystemExit(
            f"--context-prefix {policy} but this tokenizer has no {policy}_token_id")
    return [int(tid)]


def build_lm(model_dir: str, tokenizer_dir: str | None, batch_size: int,
             max_length: int, device: str, allow_extrapolation: bool = False,
             ablate_attn: bool = False, attn_window: int = 0,
             prompt_suffix: str = "", context_prefix: str = "none"):
    """An lm-eval `LM` over one exported arm."""
    ensure_lm_eval()
    import torch
    from export_hf import load_exported
    from lm_eval.api.instance import Instance  # noqa: F401  (typing only)
    from lm_eval.api.model import LM

    model, cfg = load_exported(model_dir)
    model = model.to(device)
    if ablate_attn:
        n = ablate_attention(model)
        if not n:
            raise SystemExit(
                "--ablate-attn on an export with no attention layers: this is a "
                "recurrent arm, and the flag would silently measure nothing.")
        print(f"  ABLATED {n} attention sublayers -- DIAGNOSTIC ONLY, this is "
              f"not the model that trained", flush=True)
    if attn_window:
        n = window_attention(model, attn_window)
        if not n:
            raise SystemExit(
                "--attn-window on an export with no attention layers: this is a "
                "recurrent arm, and the flag would silently measure nothing.")
        print(f"  WINDOWED {n} attention sublayers to {attn_window} tokens -- "
              f"DIAGNOSTIC ONLY, this is not the model that trained", flush=True)
    tok = load_tokenizer(tokenizer_dir or model_dir)
    prefix_ids = context_prefix_ids(tok, context_prefix)
    print(f"  context prefix: {context_prefix}"
          + (f" -> {prefix_ids} ({tok.convert_ids_to_tokens(prefix_ids)})"
             if prefix_ids else " (bare context, lm-eval's own default)"),
          flush=True)

    # How many TOKENS go through one forward, which is what bounds memory: the
    # scoring pass takes a float32 log-softmax over the vocabulary, so a batch
    # costs tokens x 32,000 x 4 bytes -- 1.05 GB at this cap. Rows per batch
    # follow from the bucket's length, so short requests (most of PIQA and
    # HellaSwag) run 40 at a time and long ones fewer.
    TOKEN_CAP = 8192

    # The same quantity for GENERATION, where the bound is a different one. A
    # RULER prompt is 8,000 tokens against a HellaSwag continuation's 80, but
    # the prefill keeps only ONE position's logits (`logits_to_keep=1`), so
    # what has to fit is the activation rather than the vocabulary: four rows
    # of 8,192 here, thirty-two rows of 1,024.
    GEN_TOKEN_CAP = 32768

    class ExportedLM(LM):
        def __init__(self):
            super().__init__()
            self.model, self.tok = model, tok
            self.prefix_ids = list(prefix_ids)
            self.batch_size = batch_size
            # The MODEL's context unless asked otherwise. lm-eval's own HFLM
            # scores at max_position_embeddings, and wikitext is scored in
            # NON-OVERLAPPING windows: a shorter window gives every token less
            # context than the model trained with, and inflates the perplexity
            # that a published row is compared against. 4096 here.
            #
            # An EXPLICIT --max-length ABOVE that is honoured rather than
            # clamped. RULER's 8K cells are a deliberate question -- what a
            # fixed-size state does past the length it trained at -- and
            # clamping would quietly answer a different one while keeping the
            # 8K label. A recurrent arm has no positional table to run off the
            # end of, so the only thing that changes is the regime.
            #
            # A HYBRID needs --allow-extrapolation to go past it, because there
            # the question is genuinely open rather than merely interesting.
            # Its attention is NoPE (`attn.use_rope: false`), so there is no
            # position table to run off either -- that is the argument for
            # expecting it to extend -- but the layers still never saw a
            # sequence this long, and a full-attention hybrid at 8K is not the
            # 2K sliding-window design the published hybrid rows use, which
            # extends by construction. The flag exists so that number is asked
            # for on purpose and never arrives as a default.
            if max_length and max_length > cfg.max_position_embeddings:
                if getattr(cfg, "attn", None) and not allow_extrapolation:
                    raise SystemExit(
                        f"--max-length {max_length} exceeds this export's "
                        f"{cfg.max_position_embeddings}-token context and the "
                        f"export is a HYBRID (config.attn is set). Its "
                        f"attention layers never saw a sequence this long. "
                        f"Pass --allow-extrapolation to ask for it anyway -- "
                        f"NoPE attention has no position table to run off, so "
                        f"this is a real question, but the answer is about "
                        f"length generalisation and not about the gate.")
                if getattr(cfg, "attn", None):
                    print(f"  NOTE: HYBRID export evaluated at {max_length} tokens, "
                          f"past the {cfg.max_position_embeddings} its attention "
                          f"layers trained at. NoPE, so no position table is "
                          f"exceeded; the result is a length-generalisation "
                          f"measurement.", flush=True)
                print(f"  NOTE: context {max_length} > the {cfg.max_position_embeddings} "
                      f"tokens this model trained at -- extrapolation, not the "
                      f"trained regime", flush=True)
                self.max_length = max_length
            else:
                self.max_length = (min(max_length, cfg.max_position_embeddings)
                                   if max_length else cfg.max_position_embeddings)

        def _ids(self, context: str, continuation: str):
            """(token ids, how many of them are the continuation)."""
            ctx = (self.prefix_ids + self.tok(context, add_special_tokens=False)["input_ids"]
                   if context else list(self.prefix_ids) or [self.tok.eos_token_id or 0])
            cont = self.tok(continuation, add_special_tokens=False)["input_ids"]
            ids = (ctx + cont)[-self.max_length:]
            return ids, min(len(cont), len(ids) - 1)

        def _forward(self, rows):
            """Score a batch of EQUAL-LENGTH id lists.

            Equal length on purpose: no padding, so no attention mask, so the
            result is the batch-1 result rather than something that depends on
            what a request was batched with. Bucketing by exact length costs a
            sort and buys the GPU back -- at batch 1 a GH200 sits at 13%
            utilization while Python assembles the next single sequence.
            """
            x = torch.tensor([ids for ids, _ in rows], device=device)
            with torch.no_grad():
                logits = self.model(x).logits[:, :-1].float()
            target = x[:, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            picked = logprobs.gather(-1, target[..., None])[..., 0]
            greedy = logprobs.argmax(-1) == target
            out = []
            for i, (_, n_cont) in enumerate(rows):
                out.append((float(picked[i, -n_cont:].sum()),
                            bool(greedy[i, -n_cont:].all())))
            return out

        def loglikelihood(self, requests, **kw):
            import time

            prepared = [self._ids(*r.args) for r in requests]
            # Bucket by length, longest first: the big buckets dominate the run
            # and finishing them early makes the progress line informative.
            buckets = {}
            for i, (ids, n_cont) in enumerate(prepared):
                buckets.setdefault(len(ids), []).append(i)

            results = [None] * len(prepared)
            done, t0, total = 0, time.time(), len(prepared)
            for length in sorted(buckets, reverse=True):
                idxs = buckets[length]
                rows_per_batch = max(1, min(self.batch_size, TOKEN_CAP // max(1, length)))
                for i in range(0, len(idxs), rows_per_batch):
                    chunk = idxs[i:i + rows_per_batch]
                    for j, res in zip(chunk, self._forward([prepared[j] for j in chunk])):
                        results[j] = res
                    done += len(chunk)
                    if done % 5000 < rows_per_batch:
                        rate = done / max(1e-9, time.time() - t0)
                        print(f"    scored {done:>6d}/{total} at {rate:6.1f} req/s",
                              flush=True)
            print(f"    scored {total}/{total} in {time.time() - t0:.0f}s", flush=True)
            return results

        def loglikelihood_rolling(self, requests, **kw):
            out = []
            for r in requests:
                (text,) = r.args
                ids = self.prefix_ids + self.tok(text, add_special_tokens=False)["input_ids"]
                # Non-overlapping windows, which is what the harness's own
                # rolling scorer does at stride = max_length.
                windows = [ids[i:i + self.max_length]
                           for i in range(0, len(ids), self.max_length)]
                windows = [w for w in windows if len(w) >= 2]
                total = 0.0
                for w in windows:
                    x = torch.tensor([w], device=device)
                    with torch.no_grad():
                        logits = self.model(x).logits[0, :-1].float()
                    lp = torch.log_softmax(logits, dim=-1)
                    tgt = x[0, 1:]
                    total += float(lp[range(len(tgt)), tgt].sum())
                out.append(total)
            return out

        def _generate(self, chunk, max_gen, until):
            """Greedy-decode one batch of prompts. Returns their continuations."""
            pad_id = self.tok.pad_token_id
            if pad_id is None:
                pad_id = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0
            gen = greedy_decode(self.model, [ids for _, ids in chunk], max_gen,
                                pad_id=pad_id, eos_id=self.tok.eos_token_id,
                                device=device)
            texts = []
            for g in gen:
                text = self.tok.decode(g)
                # Every RULER task passes `until: []` and is scored by
                # substring match, so nothing is cut there. Honoured anyway:
                # a task that asks for a stop sequence and silently does not
                # get one is scored against a different string than it asked
                # for.
                for stop in until:
                    if stop and stop in text:
                        text = text.split(stop)[0]
                texts.append(text)
            return texts

        def generate_until(self, requests, **kw):
            """Greedy generation, which is what the RULER needle tasks need.

            WHY THIS LOOP EXISTS rather than lm-eval's. Its generator lives in
            HFLM, and HFLM loads through `AutoModelForCausalLM` -- the one path
            that silently builds fla/models/new_ckda instead of the layer that
            trained, see the module docstring. So the decode is here.
            """
            import time

            prepared, truncated, worst = [], 0, 0
            for i, r in enumerate(requests):
                ctx, gen_kw = r.args
                # DIAGNOSTIC ONLY. RULER's template ends at its answer prefix
                # ("...mentioned in the provided text is") with nothing after
                # it, exactly as NVIDIA's own template does -- so every
                # generation spends its first token closing the primer, and on
                # these arms 200 of 200 emitted ":" there. Appending changes
                # what the benchmark asks and breaks comparability with every
                # published row; it exists to measure how much of an answer
                # rate is the model and how much is the last two characters.
                if prompt_suffix:
                    ctx = ctx + prompt_suffix
                # GREEDY, and only greedy. Every RULER task sets
                # `do_sample: false`; a task that asked for sampling and was
                # decoded greedily anyway would report a number under its own
                # name that this code did not produce.
                if gen_kw.get("do_sample"):
                    raise NotImplementedError(
                        "this decoder is greedy; a task asking for do_sample "
                        "needs a sampler written for it, not silently ignored")
                until = gen_kw.get("until") or []
                if isinstance(until, str):
                    until = [until]
                max_gen = int(gen_kw.get("max_gen_toks", 128))
                ids = self.prefix_ids + self.tok(ctx, add_special_tokens=False)["input_ids"]
                # lm-eval's own convention: keep the END of an over-long
                # context and leave room for the generation. RULER sizes its
                # prompts to the cell's budget MINUS the answer, and lm-eval
                # then appends the task's `gen_prefix` -- about fifteen tokens
                # that the budget did not account for -- so the longest prompts
                # in a cell can cross it by a little. Dropping the first tokens
                # of a haystack is harmless; dropping enough of one to lose the
                # needle is a retrieval failure that is really a flag, so the
                # overshoot is reported rather than absorbed.
                budget = self.max_length - max_gen
                if len(ids) > budget:
                    truncated += 1
                    worst = max(worst, len(ids) - budget)
                    ids = ids[-budget:]
                prepared.append((i, ids, max_gen, tuple(until)))
            if truncated:
                print(f"    NOTE: {truncated}/{len(prepared)} prompts exceeded "
                      f"the {self.max_length}-token context and lost up to "
                      f"{worst} tokens from the FRONT (the needle sits in the "
                      f"middle of a RULER haystack; a large number here means "
                      f"--max-length is too small for the cell)", flush=True)

            # One group per (length, stop sequence) pair -- they decode for
            # different numbers of steps -- and longest first inside a group,
            # so the row count per batch is set by the longest row in it.
            groups = {}
            for i, ids, max_gen, until in prepared:
                groups.setdefault((max_gen, until), []).append((i, ids))

            results = [None] * len(prepared)
            done, t0, total = 0, time.time(), len(prepared)
            for (max_gen, until), recs in sorted(groups.items()):
                recs.sort(key=lambda rec: len(rec[1]), reverse=True)
                pos = 0
                while pos < len(recs):
                    longest = len(recs[pos][1])
                    n = max(1, min(self.batch_size, GEN_TOKEN_CAP // max(1, longest)))
                    chunk = recs[pos:pos + n]
                    pos += n
                    for (i, _), text in zip(chunk, self._generate(chunk, max_gen, list(until))):
                        results[i] = text
                    done += len(chunk)
                    rate = done / max(1e-9, time.time() - t0)
                    print(f"    generated {done:>5d}/{total} at {rate:5.2f} req/s "
                          f"(prompt {longest} tokens, {len(chunk)} rows)", flush=True)
            print(f"    generated {total}/{total} in {time.time() - t0:.0f}s", flush=True)
            return results

    lm = ExportedLM()
    # PROVENANCE. The scoring context decides the rolling perplexities, and a
    # run whose context nobody recorded cannot be compared with one that used
    # another -- which is how half of a wikitext gap turns out to be a flag.
    print(f"  scoring context {lm.max_length} tokens, batch up to "
          f"{lm.batch_size} rows", flush=True)
    return lm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="an export_hf.py output directory")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--batch-size", type=int, default=32,
                    help="rows per forward, capped so a batch stays "
                         "under 8,192 tokens; requests are bucketed by "
                         "exact length, so batching changes no number")
    ap.add_argument("--max-length", type=int, default=0,
                    help="scoring context in tokens; 0 means the model's own "
                         "max_position_embeddings (4096 for these arms), which "
                         "is what lm-eval's HFLM uses and what the published "
                         "rows were measured at")
    ap.add_argument("--limit", type=int, default=None,
                    help="documents per task; for smoke runs only")
    ap.add_argument("--log-samples", action="store_true",
                    help="write each document's prompt, generation and target "
                         "beside --out as *.samples.json. THE ONLY WAY to tell "
                         "a model that cannot retrieve from a harness that is "
                         "mis-wired: both report a low accuracy and nothing "
                         "else. Large -- use with --limit.")
    ap.add_argument("--include-path", default=str(_HERE / "eval_tasks"),
                    help="task configs that override the stock ones. THREE TASKS "
                         "NEED THIS: social_iqa, boolq and winogrande name the "
                         "legacy canonical datasets (social_i_qa, super_glue, "
                         "winogrande), which resolve to script-based repos that "
                         "datasets 4.4.2 refuses outright. The overrides repoint "
                         "them at parquet mirrors with the same rows and columns "
                         "and change nothing else. Pass '' to use the stock set.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--context-prefix", default="none", choices=("none", "bos", "eos"),
                    help="token to put in front of every scored context. "
                         "'none' is lm-eval's own default for causal models and "
                         "matches how these corpora were tokenised (no BOS, EOS "
                         "appended per document). 'bos' reproduces the runs made "
                         "before this flag existed, which prefixed `<s>` -- a "
                         "token the fwedu corpus never contains. 'eos' prefixes "
                         "the separator the documents actually carry. NOTE: "
                         "this only bites on the Llama-2-tokenised fwedu arms. "
                         "GPT-NeoX has bos_token_id == eos_token_id == 0 (both "
                         "`<|endoftext|>`), so on the ladder all three settings "
                         "differ by at most that one token and 'bos' vs 'eos' "
                         "is no comparison at all -- do not read a NeoX "
                         "bos-vs-eos difference as a result.")
    ap.add_argument("--ablate-attn", action="store_true",
                    help="zero every attention sublayer of a HYBRID export, to "
                         "see whether its retrieval was coming from attention "
                         "or from the recurrent state. Diagnostic only -- an "
                         "ablated arm is not the model that trained, and its "
                         "numbers belong in no table.")
    ap.add_argument("--prompt-suffix", default="",
                    help="append this to every generation context. DIAGNOSTIC: "
                         "it changes the prompt away from RULER's own template "
                         "and away from every published row. Use it to ask "
                         "whether an answer rate is a property of the model or "
                         "of how the primer terminates.")
    ap.add_argument("--attn-window", type=int, default=0,
                    help="restrict a HYBRID's attention to this many tokens, "
                         "instead of removing it. Answers what --ablate-attn "
                         "could not: zeroing the layers breaks the model, so "
                         "0.0 says nothing, while a window leaves it working "
                         "and removes only the long-range reach a needle needs. "
                         "Diagnostic only.")
    ap.add_argument("--allow-extrapolation", action="store_true",
                    help="evaluate a HYBRID export past the context its "
                         "attention layers trained at. Their attention is NoPE, "
                         "so nothing runs off a position table -- but the "
                         "result measures length generalisation, and the "
                         "published hybrid rows use 2K sliding-window "
                         "attention, which extends by construction and is not "
                         "the same design. Recurrent arms need no flag.")
    ap.add_argument("--ruler-lengths", default="",
                    help="comma-separated context lengths for the RULER needle "
                         "tasks, e.g. 1024,2048,4096,8192. RULER BUILDS ITS OWN "
                         "SAMPLES to fill each length, so this decides what is "
                         "generated and not just how it is read; every length "
                         "named here must have a metric entry in the task "
                         "config (see lm_scaling/eval_tasks/niah_single_1.yaml). "
                         "Gated DeltaNet-2's Table 3 uses 1K/2K/4K/8K for "
                         "S-NIAH-1 and -2 and 1K/2K/4K for S-NIAH-3 and "
                         "MK-NIAH-1, which is two invocations, not one.")
    a = ap.parse_args(argv)

    ensure_lm_eval()
    import lm_eval

    lm = build_lm(a.model, a.tokenizer, a.batch_size, a.max_length, a.device,
                  a.allow_extrapolation, a.ablate_attn, a.attn_window,
                  a.prompt_suffix, a.context_prefix)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    kw = {}
    # RULER's samples are SYNTHESISED at task-build time, against a tokenizer,
    # to fill a target length. So both settings reach the task through the
    # TaskManager's metadata rather than through the model.
    #
    # AND THEY HAVE TO GO THROUGH *THIS* TaskManager. `simple_evaluate` takes a
    # `metadata=` of its own, but look at evaluator.py: it is used to build a
    # TaskManager only `if task_manager is None`. We always pass one, for the
    # include path, so a metadata= handed to simple_evaluate is accepted,
    # ignored, and the tasks build at the default 4,096 -- one length, silently,
    # under whatever label the job was given.
    meta = {}
    if a.ruler_lengths:
        meta["max_seq_lengths"] = [int(x) for x in a.ruler_lengths.split(",") if x.strip()]
        # A LIMIT SLICES THE FIRST LENGTH ONLY. RULER builds one dataset per
        # task by concatenating its lengths in order -- 500 docs at 4K, then
        # 500 at 8K -- and lm-eval's `--limit N` takes the first N documents of
        # that. With two lengths and --limit 200 the run reports a 4K cell,
        # never reaches 8K, and says nothing about it: a job that looks like it
        # measured what it was asked for and did not. Cost a diagnostic run.
        if a.limit and len(meta["max_seq_lengths"]) > 1:
            raise SystemExit(
                f"--limit {a.limit} with {len(meta['max_seq_lengths'])} "
                f"--ruler-lengths would evaluate only the first "
                f"({meta['max_seq_lengths'][0]}), because the limit slices a "
                f"dataset that concatenates the lengths in order. Run one "
                f"length per invocation when limiting.")
        meta["tokenizer"] = a.tokenizer or a.model
        longest = max(meta["max_seq_lengths"])
        if lm.max_length < longest:
            raise SystemExit(
                f"--ruler-lengths asks for {longest}-token samples and the "
                f"scoring context is {lm.max_length}. The prompts would be "
                f"truncated at the front, where the needle often is, and the "
                f"cell would report a retrieval failure that is this flag. "
                f"Pass --max-length {longest}.")
    elif any(t.startswith(("niah", "ruler")) for t in tasks):
        raise SystemExit(
            "the RULER tasks need --ruler-lengths; without it they build at "
            "lm-eval's default 4096 alone and the shorter cells come back empty")
    if a.include_path:
        from lm_eval.tasks import TaskManager
        kw["task_manager"] = TaskManager(include_path=a.include_path,
                                         metadata=meta or None)
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=a.limit,
                                  bootstrap_iters=0, log_samples=a.log_samples,
                                  **kw)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res["results"], indent=2, default=str) + "\n")
    # WHAT THIS RUN WAS, beside the numbers rather than inside them: eval_table
    # reads the results file by its exact shape, and two runs that differ only
    # in how the context was tokenised are otherwise indistinguishable on disk.
    a.out.with_suffix(".meta.json").write_text(json.dumps({
        "model": a.model, "tokenizer": a.tokenizer, "tasks": tasks,
        "context_prefix": a.context_prefix,
        "context_prefix_ids": lm.prefix_ids,
        "max_length": lm.max_length, "limit": a.limit,
    }, indent=2, default=str) + "\n")
    if a.log_samples:
        # The PROMPT TAIL rather than the prompt: a RULER context is 8,000
        # tokens of haystack and the part that decides the answer is the
        # question at the end. The needle itself is in `doc`, so a sample is
        # still enough to check by hand that the thing asked for was present.
        def _row(r):
            doc = r.get("doc") or {}
            # WHERE THE NEEDLE SAT, as a fraction of the haystack. This is the
            # covariate that tells a retrieval failure from a formatting one:
            # retrieval degrades with distance from the query, so its failures
            # concentrate at low depth (early in the context). A model that
            # simply does not answer in the expected form fails uniformly, at
            # every depth alike.
            #
            # TWO DOCUMENT SHAPES REACH HERE. RULER's carries `input`/`outputs`;
            # the recall suite (squad_completion, swde, fda) carries
            # `text`/`value`. Reading only RULER's names does not fail on a
            # recall sample -- it produces `hit: false` on every row of a task
            # that scored 60%, which is a dump that looks like a finding.
            needle = ((doc.get("outputs") or [""])[0]
                      if "outputs" in doc else doc.get("value") or "")
            text = doc.get("input") or doc.get("text") or ""
            at = text.find(needle) if needle else -1
            depth = round(at / len(text), 3) if at >= 0 and text else None
            got = r.get("resps")
            flat = got[0] if isinstance(got, list) and got else got
            if isinstance(flat, list) and flat:
                flat = flat[0]
            # CASE-INSENSITIVE, because both scorers are: RULER's
            # `string_match_all` compares `r.lower() in pred.lower()` and the
            # recall suite's `contains_score` searches with re.IGNORECASE. A
            # case-sensitive `hit` here would disagree with the number in the
            # results file, which is worse than having no `hit` at all.
            return {"doc_id": r.get("doc_id"),
                    "length": doc.get("max_length"),
                    "depth": depth,
                    "hit": bool(needle) and needle.lower() in str(flat).lower(),
                    "prompt_tail": str(r.get("arguments", [["", ""]])[0][0])[-400:],
                    "target": r.get("target"),
                    "generated": r.get("resps"),
                    # RULER names a metric after the context length, so its
                    # keys are digits; the recall suite's is `contains`. Taking
                    # only the digits leaves every recall sample with an empty
                    # `score`, which reads as a document the harness failed to
                    # grade rather than one it graded under another name.
                    "score": {k: v for k, v in r.items()
                              if k.isdigit() or k == "contains"}}

        dump = {}
        for task, rows in (res.get("samples") or {}).items():
            dump[task] = [_row(r) for r in rows]
        samples_path = a.out.with_suffix(".samples.json")
        samples_path.write_text(json.dumps(dump, indent=2, default=str) + "\n")
        print(f"wrote {samples_path}")
    for task, vals in sorted(res["results"].items()):
        keep = {k: v for k, v in vals.items()
                if isinstance(v, float) and not k.endswith("_stderr,none")}
        print(f"  {task:16s} " + "  ".join(f"{k.split(',')[0]}={v:.4f}" for k, v in keep.items()))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
