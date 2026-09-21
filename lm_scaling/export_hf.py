"""Turn a torchtitan DCP checkpoint into a directory `fla` can load.

    lm_scaling/container_run lm_scaling/export_hf.py \
        --checkpoint <runs>/<job>/checkpoint/step-190976 \
        --arm kda-sig-lowrank --geometry 1.3B --out <dir>

WHY THIS EXISTS. The campaigns train through torchtitan, whose checkpoint is a
sharded DCP directory of `__N_0.distcp` files under torchtitan's own parameter
names. Every downstream tool -- lm-eval-harness included -- wants a
transformers directory: one `config.json`, one weight file, HF's names. Nothing
in the repo bridged the two: `lm/services/prepare_hf_export.py` uploads models
`lm/` trained, and `to_hf.py` wraps `lm/`'s own class.

THE NAMES, and they are the whole job. torchtitan builds the model as
qwen3_custom with each block's attention replaced by an fla layer (titan_ext),
so the mixer's own parameters arrive verbatim under `.attention.inner.`, and
everything around them is torchtitan's:

    tok_embeddings.weight              -> model.embeddings.weight
    output.weight                      -> lm_head.weight
    norm.weight                        -> model.norm.weight
    layers.N.attention_norm.weight     -> model.layers.N.attn_norm.weight
    layers.N.ffn_norm.weight           -> model.layers.N.mlp_norm.weight
    layers.N.attention.inner.<rest>    -> model.layers.N.attn.<rest>
    layers.N.feed_forward.w1.weight    -> model.layers.N.mlp.gate_proj.weight
    layers.N.feed_forward.w3.weight    -> model.layers.N.mlp.up_proj.weight
    layers.N.feed_forward.w2.weight    -> model.layers.N.mlp.down_proj.weight

w1 is the GATE and w3 the UP projection, not the other way round: torchtitan's
FeedForward computes `w2(silu(w1(x)) * w3(x))`, and fla's GatedMLP computes
`down_proj(swish(gate_proj(x)) * up_proj(x))`. Swapping them builds a model
whose parameter count is right and whose outputs are not.

The config is generated from the same tables the run was planned from, so an
exported model cannot disagree with the arm it came from: geometry and d_ffn
from geometries.table(), layer settings from that entry's `kwargs`.

NEVER LOAD THIS WITH `AutoModelForCausalLM`. Two fla models declare
`model_type = "complex_kda"`: ours (fla/models/complex_kda) and
fla/models/new_ckda, and the registry keeps whichever registered last --
new_ckda, as the container currently imports fla. An Auto load therefore
builds a DIFFERENT layer, whose config has no `conv_silu` at all, so a
checkpoint trained with SiLU dropped evaluates with SiLU ON. It does not
error: the parameter counts agree to the digit, because the two layers differ
in what they compute rather than in what they hold. `load_exported()` below
constructs the classes explicitly for that reason, and anything that reads
these directories must do the same.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


def hf_config(arm: str, geometry: str, seq_len: int | None = None) -> dict:
    """The fla config for this arm at this geometry, from the campaign's table."""
    import geometries

    table = geometries.table()
    if geometry not in table:
        raise SystemExit(f"no geometry {geometry!r} (have {sorted(table)})")
    entry = table[geometry]
    if arm not in entry["archs"]:
        raise SystemExit(f"no arm {arm!r} at {geometry} "
                         f"(have {sorted(entry['archs'])})")
    a = entry["archs"][arm]
    if a.get("mixer") != "complex-kda":
        raise SystemExit(
            f"{arm} is mixer {a.get('mixer')!r}; this exporter writes the "
            f"complex_kda config. Add the mapping before exporting it.")
    kw = dict(a.get("kwargs") or {})
    cfg = {
        "architectures": ["ComplexKDAForCausalLM"],
        "model_type": "complex_kda",
        "attn_mode": "chunk",
        "hidden_size": entry["d_model"],
        "num_hidden_layers": entry["n_layers"],
        "num_heads": entry["n_heads"],
        "head_dim": entry["head_dim"],
        "intermediate_size": a["d_ffn"],
        "hidden_ratio": None,
        "hidden_act": "swish",
        "vocab_size": entry.get("vocab_size", 50304),
        "max_position_embeddings": seq_len or entry.get("seq_len", 4096),
        "tie_word_embeddings": entry["tie_word_embeddings"],
        "norm_eps": 1e-6,
        "conv_size": 4,
        "use_short_conv": True,
        "fuse_norm": True,
        "fuse_swiglu": True,
        "fuse_cross_entropy": True,
        "initializer_range": 0.02,
    }
    # The arm's own layer settings -- gate, init, output gate, SiLU, beta -- so
    # the exported model is the arm and not the config class's defaults.
    cfg.update(kw)
    # A HYBRID's attention layers, as fla's own hybrid mechanism expresses them.
    #
    # WHICH layers comes from titan_ext.hybrid_layers -- imported rather than
    # recomputed, because `attn_every` counts from 1 (layer i is attention when
    # (i + 1) % attn_every == 0, so [3, 7, 11, 15, 19, 23] at 24 layers and every
    # 4) and a second copy of that convention is how two different models end up
    # under one name.
    #
    # WHAT they are comes from the same table entry the run was planned from:
    # gated (Qwen3-Next's sigmoid before o_proj) and NoPE. `num_heads` is
    # d_model // head_dim, which is what titan_ext computes for `mixer == "attn"`
    # -- 16 heads of 128 at 1.3B, not the 32 of 64 a published Transformer++ row
    # would have. qkv_bias and qk_norm are off, as in the run.
    if hybrid := a.get("hybrid"):
        from titan_ext import hybrid_layers

        n_heads = entry["d_model"] // entry["head_dim"]
        cfg["attn"] = {
            "layers": hybrid_layers(entry["n_layers"], int(hybrid["attn_every"])),
            "num_heads": n_heads,
            "num_kv_heads": n_heads,
            "qkv_bias": False,
            "qk_norm": False,
            "output_gate": bool(hybrid.get("attn_output_gate", False)),
            "use_rope": bool(hybrid.get("attn_use_rope", True)),
            "rope_theta": float(cfg.pop("rope_theta", 10000.0)) if "rope_theta" in cfg else 10000.0,
            "window_size": None,
        }

    # `drop_silu` is a campaign setting rather than an arm one: it reaches the
    # run through the flavor (`<arch>-<rung>-nosilu`), so it cannot be read off
    # the table. The caller passes it; see --drop-silu.
    return cfg


def rename(key: str) -> str | None:
    """torchtitan's name -> fla's, or None for things a model does not carry."""
    if key.startswith(("optimizer.", "dataloader.", "lr_scheduler.", "train_state")):
        return None
    if key in ("tok_embeddings.weight",):
        return "model.embeddings.weight"
    if key in ("output.weight",):
        return "lm_head.weight"
    if key in ("norm.weight",):
        return "model.norm.weight"
    if key.startswith("layers."):
        n, rest = key.split(".", 2)[1], key.split(".", 2)[2]
        if rest.startswith("attention.inner."):
            return f"model.layers.{n}.attn.{rest[len('attention.inner.'):]}"
        if rest == "attention_norm.weight":
            return f"model.layers.{n}.attn_norm.weight"
        if rest == "ffn_norm.weight":
            return f"model.layers.{n}.mlp_norm.weight"
        if rest.startswith("feed_forward."):
            w = {"w1.weight": "gate_proj.weight",
                 "w3.weight": "up_proj.weight",
                 "w2.weight": "down_proj.weight"}.get(rest[len("feed_forward."):])
            if w:
                return f"model.layers.{n}.mlp.{w}"
        # A hybrid's attention layers are fla's own Attention, also under .inner.
        if rest.startswith("attention."):
            return f"model.layers.{n}.attn.{rest[len('attention.'):]}"
    return None


def load_dcp(checkpoint: Path) -> dict:
    """Every model tensor in a sharded DCP directory, on CPU, unsharded."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    reader = FileSystemReader(str(checkpoint))
    md = reader.read_metadata()
    want = {k: v for k, v in md.state_dict_metadata.items()
            if isinstance(v, TensorStorageMetadata) and rename(k)}
    if not want:
        raise SystemExit(f"{checkpoint} holds no model tensors this exporter knows")
    sd = {k: torch.empty(tuple(v.size), dtype=v.properties.dtype) for k, v in want.items()}
    _load_state_dict(sd, storage_reader=reader, planner=DefaultLoadPlanner(),
                     no_dist=True)
    return sd


def load_exported(path, dtype="bfloat16"):
    """The exported directory, as the model that trained it.

    Explicitly, never through Auto -- see the module docstring: `complex_kda`
    resolves to fla/models/new_ckda in this container, which is a different
    layer with the same parameter shapes.
    """
    import json as _json

    import torch
    from safetensors.torch import load_file

    from fla.models.complex_kda import ComplexKDAForCausalLM
    from fla.models.complex_kda.configuration_complex_kda import ComplexKDAConfig

    path = Path(path)
    raw = _json.loads((path / "config.json").read_text())
    # TIE THE HEAD OURSELVES. fla declares `_tied_weights_keys` as a LIST, which
    # is right for transformers 4.x; transformers 5.3 reads it as a mapping and
    # `post_init` dies with "'list' object has no attribute 'keys'" for ANY tied
    # config. Every ladder rung ties its embeddings (fwedu does not), so this is
    # the difference between evaluating the ladder and not.
    #
    # Building untied and assigning afterwards is contained in this loader, where
    # upstream fla stays as upstream ships it -- the training path uses fla's
    # LAYERS and never constructs this model class.
    tied = bool(raw.get("tie_word_embeddings", False))
    cfg = ComplexKDAConfig(**{k: (False if k == "tie_word_embeddings" else v)
                             for k, v in raw.items()
                             if k not in ("architectures", "model_type")})
    model = ComplexKDAForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(
        load_file(str(path / "model.safetensors")), strict=False)
    # TIED EMBEDDINGS. When the geometry ties them -- every ladder rung does,
    # fwedu does not -- the exporter writes no `lm_head.weight`, because there is
    # no separate tensor to write. The model still LISTS that key, so a strict
    # comparison reports it missing and rejects a correct export. Re-tie and drop
    # it from the complaint; anything else missing is still a real mismatch.
    if tied:
        # One tensor, two names -- and `strict` complains about the name the
        # export could not carry, which is why it is filtered rather than fixed
        # by writing a duplicate into the file.
        model.lm_head.weight = model.model.embeddings.weight
        missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise SystemExit(
            f"{path}: {len(missing)} missing and {len(unexpected)} unexpected "
            f"tensors -- the export does not match this layer. "
            f"missing[:3]={missing[:3]} unexpected[:3]={unexpected[:3]}")
    return model.to(getattr(torch, dtype)).eval(), cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--geometry", required=True, help="a tag in geometries.table(), e.g. 1.3B")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tokenizer", default=None,
                    help="directory to copy tokenizer files from")
    ap.add_argument("--drop-silu", action="store_true",
                    help="the run used a -nosilu flavor")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    a = ap.parse_args(argv)

    import torch

    cfg = hf_config(a.arm, a.geometry)
    cfg["drop_silu"] = bool(a.drop_silu)
    sd = load_dcp(a.checkpoint)
    out = {}
    for k, v in sd.items():
        new = rename(k)
        if new:
            out[new] = v.to(getattr(torch, a.dtype))
    if cfg["tie_word_embeddings"]:
        out.pop("lm_head.weight", None)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in out.items()}, str(a.out / "model.safetensors"),
              metadata={"format": "pt"})
    if a.tokenizer:
        import shutil
        for f in Path(a.tokenizer).iterdir():
            if f.is_file() and f.name.startswith(("tokenizer", "special_tokens")):
                shutil.copy2(f, a.out / f.name)
    n = sum(v.numel() for v in out.values())
    print(f"wrote {a.out}: {len(out)} tensors, {n:,} parameters "
          f"({'tied' if cfg['tie_word_embeddings'] else 'untied'}), dtype {a.dtype}")
    print(f"  arm {a.arm} at {a.geometry}: d_ffn {cfg['intermediate_size']}, "
          f"gate {cfg.get('gate')}, init {cfg.get('gate_init_style')}, "
          f"output_gate {cfg.get('output_gate')}, drop_silu {cfg['drop_silu']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
