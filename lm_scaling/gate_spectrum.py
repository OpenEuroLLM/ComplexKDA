"""The spectrum of the state transition, checkpoint by checkpoint.

`measure_signs.py` answers two neighbouring questions -- is alpha negative, and
is a head CAPABLE of rotating (beta > 1 and some alpha < 0 at the same token) --
and both are proxies. This measures the thing itself: of the transitions a
trained model actually applies, what share carry negative real eigenvalues, what
share carry complex ones, and how far each rotates.

    python lm_scaling/gate_spectrum.py --config run.toml --all-steps --json out.json

THE MATRIX. At one (token, value head) the recurrence applies

    M = (I - beta k k^T) Diag(alpha),    ||k|| = 1

over `head_k_dim` dimensions. Which side the reflection sits on does not matter
-- AB and BA share a characteristic polynomial -- so this is the spectrum of the
step, not of a convention.

WHAT EACH FACTOR CAN DO ALONE, which is why the split is worth measuring:

  * alpha < 0 alone            negative REAL eigenvalues, no rotation
  * beta > 1 alone             one negative real eigenvalue, along k
  * both together              COMPLEX pairs, i.e. rotation

The 2x2 case is exact and is this module's test: beta = 2, alpha = (1, -1) and
k = (-sin(phi/2), cos(phi/2)) give eigenvalues e^{+-i phi}, a planar rotation by
phi (appendix_wfa_min.tex, lem:flip). `tests/lm/test_gate_spectrum.py` checks it
to 1e-12 rather than trusting the derivation.

WHAT THE UNSIGNED ARMS MUST REPORT. `kda-sig-*` runs `allow_neg_eigval: False`,
so beta is in (0, 1) and alpha > 0: M is a product of two positive definite
matrices and every eigenvalue is real and positive. Those arms are not a
baseline to compare against, they are a CONTROL -- anything but zero here is a
bug in this file, and the sweep keeps them for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Thresholds on |Im lambda| / spectral radius. Reported at three rather than
# one because "complex" is a numerical judgement at the bottom: a real repeated
# eigenvalue of a nonsymmetric matrix picks up an imaginary part of order
# sqrt(eps) from the QR iteration itself, and a threshold below that counts
# arithmetic noise as rotation. If the three disagree, the answer is the
# threshold and not the model.
IM_TOLS = (1e-8, 1e-6, 1e-4)


def transitions(alpha, beta, k):
    """The batch of transition matrices, one per (token, head).

    alpha (N, d), beta (N,), k (N, d) -- k need not arrive normalized, and is
    normalized here, because the kernel takes an un-normalized k and does its
    own l2norm (`use_qk_l2norm_in_kernel=True`). Reading the raw k as the
    reflection direction would scale beta by ||k||^2 and report a reflection
    strength the model never applies.
    """
    import torch

    kh = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    eye = torch.eye(kh.shape[-1], dtype=kh.dtype, device=kh.device)
    refl = eye - beta[:, None, None] * kh[:, :, None] * kh[:, None, :]
    return refl * alpha[:, None, :]          # (I - b k k^T) Diag(alpha)


def eigenvalues(alpha, beta, k, chunk: int = 512):
    """Eigenvalues of every transition, in float64.

    float64 is not caution for its own sake: the classification below asks
    whether an imaginary part is zero, and in float32 a 128x128 nonsymmetric
    eigensolve carries ~1e-4 relative error, which is the same size as the
    smallest rotations worth reporting.
    """
    import torch

    out = []
    for i in range(0, alpha.shape[0], chunk):
        m = transitions(alpha[i:i + chunk].double(),
                        beta[i:i + chunk].double(),
                        k[i:i + chunk].double())
        out.append(torch.linalg.eigvals(m.cpu()))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.complex128)


def classify(eigs, tol: float = 1e-6) -> dict:
    """What the spectrum says, per eigenvalue and per transition.

    `eigs` is (N, d) complex. Complex eigenvalues of a real matrix come in
    conjugate pairs, so the per-eigenvalue share counts both members -- a
    transition with one rotating plane out of 64 reads 2/128, not 1/128.
    """
    import torch

    if eigs.numel() == 0:
        return {"n_transitions": 0}
    radius = eigs.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    is_cplx = eigs.imag.abs() > tol * radius
    real_part = eigs.real
    is_neg = (~is_cplx) & (real_part < 0)
    ang = eigs.angle().abs()                      # rotation per step, radians

    def q(t, ps=(0.5, 0.9, 0.99)):
        return ([float(torch.quantile(t.double(), p)) for p in ps]
                if t.numel() else [None] * len(ps))

    any_c = float(is_cplx.any(dim=-1).float().mean())
    any_n = float(is_neg.any(dim=-1).float().mean())
    per = float(is_cplx.float().sum(-1).mean())
    return {
        "n_transitions": int(eigs.shape[0]),
        "head_dim": int(eigs.shape[1]),
        "im_tol": tol,
        # PER TRANSITION -- one (token, value head), i.e. one application of M.
        # Named "transition" and not "step": in a document about training
        # trajectories "step" reads as a TRAINING step, which is the one thing
        # it does not mean here.
        "frac_transition_any_complex": any_c,
        "frac_transition_any_real_neg": any_n,
        "mean_complex_per_transition": per,
        # The old spellings, so the 88 checkpoints already on disk and anything
        # reading them keep working. Same values; do not add a third name.
        "frac_step_any_complex": any_c,
        "frac_step_any_real_neg": any_n,
        "mean_complex_per_step": per,
        # Per eigenvalue. DETERMINED, not measured: M is a rank-one update of a
        # diagonal, so at most ONE conjugate pair exists per transition and this
        # is frac_transition_any_complex * 2/head_dim exactly. Kept because it
        # is free; reported nowhere, because at a ceiling of 2/128 it reads as
        # "vanishingly rare" when it means "a quarter of what is possible".
        "frac_eig_complex": float(is_cplx.float().mean()),
        "frac_eig_real_neg": float(is_neg.float().mean()),
        # The rotation itself: an angle near 0 is a complex pair that barely
        # turns, which is not the same claim as "the recurrence rotates".
        "angle_q50_q90_q99": q(ang[is_cplx]),
        "max_angle": float(ang[is_cplx].max()) if is_cplx.any() else None,
        # Moduli: a rotation that decays to nothing in two steps carries no
        # phase information, however complex it is.
        "modulus_complex_q50_q90_q99": q(eigs.abs()[is_cplx]),
        "modulus_all_q50_q90_q99": q(eigs.abs().reshape(-1)),
        "spectral_radius_q50_q90_q99": q(eigs.abs().amax(dim=-1)),
    }


def _tol_sweep(eigs) -> dict:
    """The same share at three thresholds, so the reader can see it is stable."""

    if eigs.numel() == 0:
        return {}
    radius = eigs.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return {f"frac_eig_complex@{t:g}":
            float((eigs.imag.abs() > t * radius).float().mean())
            for t in IM_TOLS}


def _sample(t, idx):
    """Flatten (B, T, H, ...) to (B*T*H, ...) and take the sampled rows."""
    return t.reshape(-1, *t.shape[3:])[idx] if t.dim() > 3 else \
        t.reshape(-1)[idx]


def capture(model, batch_iter, ctx, batches: int, per_call: int, seed: int = 0):
    """Run the model and keep a random sample of (alpha, beta, k) PER LAYER.

    Patched at the KERNEL CALL rather than at `compute_gate`, because the three
    tensors that define the transition only meet there: compute_gate sees alpha
    alone, and a hook on `b_proj` sees beta's logits with no way to tell which
    head's alpha they belong to. The layer hands the kernel exactly the
    (k, beta, sign, g) the recurrence uses, which is the point.

    LAYER INDEX is the call's position within one forward pass, and the counter
    resets per batch. Not `calls % n_layers`, which needs a layer count and is
    wrong for a hybrid the moment one is miscounted; and not a module hook,
    which cannot see the kernel's arguments. Within a sequential stack the i-th
    kernel call of a forward IS the i-th recurrent layer.

    Per layer because alpha and beta are ACTIVATIONS, not parameters: they
    depend on the token, so the answer is an average over a dataset, and
    averaging across layers on top of that hides the thing worth seeing -- one
    layer rotating hard and twenty not is a different model from all of them
    rotating a little, and the pooled number cannot tell them apart.
    """
    import torch

    import fla.layers.complex_kda_layer as ckda

    g = torch.Generator().manual_seed(seed)
    kept = {"alpha": {}, "beta": {}, "k": {}, "pos": {}, "unsigned": 0,
            "calls": 0, "in_batch": 0, "seq_len": 0}

    def record(k, beta, sign, glog, use_beta_sigmoid, allow_neg):
        b = beta.detach().float()
        if use_beta_sigmoid:
            # The kernel's own form. Mirrored here rather than read off the
            # layer: `beta_activation: plateau` arms pre-activate and pass the
            # value through, and applying sigmoid twice would report a beta
            # the model never used.
            b = torch.sigmoid(b) * (2.0 if allow_neg else 1.0)
        a = glog.detach().float().exp()
        if sign is not None:
            a = a * sign.detach().float()
        else:
            kept["unsigned"] += 1
        nb, nt, nh = a.shape[0], a.shape[1], a.shape[2]
        n = nb * nt * nh
        take = min(per_call, n)
        idx = torch.randperm(n, generator=g)[:take]
        li = kept["in_batch"]
        kept["alpha"].setdefault(li, []).append(_sample(a.cpu(), idx))
        kept["beta"].setdefault(li, []).append(_sample(b.cpu(), idx))
        kept["k"].setdefault(li, []).append(_sample(k.detach().float().cpu(), idx))
        # WHERE IN THE SEQUENCE each sample came from. The flattening is
        # (b, t, h) -> ((b*T + t)*H + h), so the position is (idx // H) % T.
        # Kept because the sample is uniform over positions and the recurrence
        # is not: at t = 0 the state is empty, there is nothing to forget, and
        # a gate near 1 there says something different from a gate near 1 at
        # t = 4000. Without this the two are averaged together silently.
        kept["pos"].setdefault(li, []).append((idx // nh) % nt)
        kept["seq_len"] = nt
        kept["in_batch"] += 1
        kept["calls"] += 1

    originals = {}
    for name in ("chunk_kda", "fused_recurrent_kda"):
        fn = getattr(ckda, name, None)
        if fn is None:
            continue
        originals[name] = fn

        def wrap(orig):
            def inner(*args, **kw):
                if kw.get("k") is not None and kw.get("beta") is not None:
                    record(kw["k"], kw["beta"], kw.get("sign"), kw["g"],
                           kw.get("use_beta_sigmoid_in_kernel", True),
                           kw.get("allow_neg_eigval", False))
                return orig(*args, **kw)
            return inner

        setattr(ckda, name, wrap(fn))
    try:
        with torch.no_grad(), ctx:
            for xs in batch_iter(batches):
                kept["in_batch"] = 0
                model(xs)
    finally:
        for name, fn in originals.items():
            setattr(ckda, name, fn)

    if not kept["calls"]:
        raise SystemExit("the kernel was never called with keyword arguments; "
                         "this model does not go through chunk_kda")
    layers = sorted(kept["alpha"])
    return ([torch.cat(kept["alpha"][i]) for i in layers],
            [torch.cat(kept["beta"][i]) for i in layers],
            [torch.cat(kept["k"][i]) for i in layers],
            [torch.cat(kept["pos"][i]) for i in layers],
            kept["unsigned"] == kept["calls"], kept["seq_len"])


def bias_preference(model) -> list[dict]:
    """What each layer's BIAS alone asks for, with no input at all.

    alpha and beta are activations, so an occupancy is an average over data --
    but the parameters underneath it say which way the layer LEANS, and they
    are free to read. At init the two leans are opposite and extreme:
    `dt_bias` sits at 3.0-7.6 with `A_log = 0`, so u = z + dt_bias and a
    channel only goes negative if f_proj(x) < -5 or so; while `b_proj` has no
    bias under `beta_init_style: standard`, so beta = 2*sigmoid(0) = 1 exactly,
    with no preference either way.

    So this answers "has the model moved the bias toward alpha < 0, or toward
    beta = 2" separately from "how often does the data take it there".
    """
    import torch

    from fla.layers.complex_kda_layer import compute_gate

    out = []
    inners = [m for _, m in model.named_modules() if hasattr(m, "b_proj")]
    for i, m in enumerate(inners):
        # .cpu() and not just .detach(): torch.quantile requires q on the
        # input's device, and the whole row is scalars for a JSON file anyway.
        dt = m.dt_bias.detach().float().cpu() if getattr(m, "dt_bias", None) is not None else None
        row = {"layer": i}
        if dt is not None:
            q = torch.quantile(dt.double(), torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
            row["dt_bias_min_med_max"] = [float(x) for x in q]
            row["frac_dt_bias_neg"] = float((dt < 0).float().mean())
        if getattr(m, "A_log", None) is not None:
            a = m.A_log.detach().float()
            row["A_log_mean"] = float(a.mean())
        # The gate at zero input: the sign and magnitude the bias alone gives.
        z = torch.zeros(1, 1, *m.dt_bias.shape, device=m.dt_bias.device) \
            if getattr(m, "dt_bias", None) is not None else None
        if z is not None and getattr(m, "gate", None):
            try:
                # Detached: these are live parameters, and float() on a tensor
                # that still wants a grad warns on every one of 24 layers.
                al = getattr(m, "A_log", None)
                sign, logabs = compute_gate(m.gate, z,
                                            None if al is None else al.detach(),
                                            m.dt_bias.detach(), m.lower_bound)
                if sign is not None:
                    row["frac_alpha_neg_at_zero_input"] = float((sign < 0).float().mean())
                row["median_abs_alpha_at_zero_input"] = float(logabs.exp().median())
            except Exception as e:                  # a gate this file has not met
                row["gate_error"] = f"{type(e).__name__}: {e}"
        # beta's bias. No bias parameter is itself the answer: beta sits at the
        # centre of its range, which for allow_neg_eigval is exactly 1.
        scale = 2.0 if getattr(m, "allow_neg_eigval", False) else 1.0
        bb = getattr(m.b_proj, "bias", None)
        if bb is None:
            row["beta_at_zero_input"] = 0.5 * scale
            row["b_proj_has_bias"] = False
        else:
            b = torch.sigmoid(bb.detach().float()) * scale
            row["b_proj_has_bias"] = True
            row["beta_at_zero_input_mean"] = float(b.mean())
            row["frac_beta_bias_over_1"] = float((b > 1).float().mean())
        out.append(row)
    return out


# Histogram bins for the two gates, fixed so every layer and every checkpoint
# is directly comparable and a figure can stack them without rebinning. alpha
# is in [-1, 1] by construction (|alpha| = eps + (1-eps)|tanh|), beta in [0, 2]
# under allow_neg_eigval and [0, 1] without.
ALPHA_BINS, BETA_BINS = 81, 81
ALPHA_RANGE, BETA_RANGE = (-1.0, 1.0), (0.0, 2.0)


def _hist(t, bins: int, rng) -> list[int]:
    """Counts only. The edges are implied by the constants above and writing
    them per layer per checkpoint would be 90% of the file."""
    import torch

    h = torch.histc(t.float().flatten().clamp(*rng), bins=bins, min=rng[0], max=rng[1])
    return [int(x) for x in h]


def _quantiles(t, ps=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> list[float]:
    import torch

    f = t.float().flatten()
    if f.numel() > 2_000_000:                 # torch.quantile's input limit
        f = f[torch.linspace(0, f.numel() - 1, 2_000_000).long()]
    return [float(x) for x in torch.quantile(f, torch.tensor(ps))]


def _range(alpha):
    """max_i alpha_{t,h,i} - min_i alpha_{t,h,i}, one value per transition."""
    return alpha.amax(dim=-1) - alpha.amin(dim=-1)


def _factors(alpha, beta) -> dict:
    """The two ingredients, beside the spectrum they produce.

    Kept so a share can be attributed: complex needs alpha < 0 AND beta > 1, so
    when the complex share moves it matters which side moved.

    The HISTOGRAMS are here because a share is a summary of a shape, and the
    shape is what moves: "alpha < 0 is 5.9%" is compatible with a small hard
    negative mode and with a broad distribution whose tail crosses zero, and
    those are different models. 81 bins per gate per layer is ~1 KB, against a
    40 KB checkpoint summary.
    """
    return {"frac_alpha_neg": float((alpha < 0).float().mean()),
            "frac_beta_gt1": float((beta > 1).float().mean()),
            "hist_alpha": _hist(alpha, ALPHA_BINS, ALPHA_RANGE),
            "hist_beta": _hist(beta, BETA_BINS, BETA_RANGE),
            "alpha_q": _quantiles(alpha),
            "beta_q": _quantiles(beta),
            # max_i alpha - min_i alpha WITHIN one (token, head): how much the
            # gate differentiates across its channels. A head whose alphas are
            # all equal applies uniform decay however large that decay is; the
            # range is what says whether the gate is selective. It also bounds
            # the spectrum -- a range of 0 forces every eigenvalue real.
            "alpha_range_q": _quantiles(_range(alpha)),
            "mean_alpha_range": float(_range(alpha).mean()),
            "hist_alpha_range": _hist(_range(alpha), ALPHA_BINS, (0.0, 2.0)),
            "mean_beta": float(beta.mean()),
            "frac_beta_over_1p5": float((beta > 1.5).float().mean()),
            "median_abs_alpha": float(alpha.abs().median())}


def measure(config_path: str, step: int | None = None, batches: int = 8,
            per_call: int = 128, seq_len: int = 4096, tol: float = 1e-6,
            dump_eigs: Path | None = None, dump_max: int = 40000) -> dict:
    """`dump_eigs` writes the raw spectrum to an .npz beside the summary.

    The summary says 0.41% of eigenvalues are complex; a picture of the complex
    plane says what that MEANS -- a cloud hugging the positive real axis for the
    unsigned arm, and for the signed one a spread down the negative reals with a
    ring of rotating pairs off the axis. No summary statistic carries that, and
    the eigenvalues are already computed, so keeping a sample is free.
    """
    import tomllib
    import torch
    from measure_signs import _load_titan

    model, ckpt, batch_iter, ctx = _load_titan(config_path, seq_len, step=step)
    alphas, betas, ks, poss, unsigned, seq = capture(model, batch_iter, ctx,
                                                     batches, per_call)
    # `local_batch_size` sequences per batch, not one. Recorded wrongly as
    # batches * seq_len until 2026-09-14, which understated it by the batch
    # size -- 4x for fwedu, 16x for the ladder. Informational only; nothing
    # computes from it, and the JSON already written carries the old value.
    vcfg = tomllib.loads(Path(config_path).read_text()).get("validation", {})
    bs = int(vcfg.get("local_batch_size", 8))
    out = {"config": config_path, "checkpoint": ckpt,
           "step": int(Path(ckpt).name.split("-")[1]),
           "unsigned_gate": bool(unsigned),
           "n_recurrent_layers": len(alphas),
           # The windows are taken from the HEAD of the validation split, in
           # order, with no shuffle: the same text at every checkpoint, which is
           # what makes the trajectory paired rather than resampled.
           "validation_path": vcfg.get("dataset_path"),
           "tokens_seen": batches * bs * seq_len,
           "sequences_seen": batches * bs,
           "transitions_per_layer": batches * per_call,
           "per_layer": [],
           # What the parameters lean toward, with no input at all.
           "bias": bias_preference(model)}
    dumped = []
    for i, (a, b, k) in enumerate(zip(alphas, betas, ks)):
        eigs = eigenvalues(a, b, k)
        row = {"layer": i}
        row.update(classify(eigs, tol))
        row.update(_factors(a, b))
        out["per_layer"].append(row)
        if dump_eigs is not None:
            dumped.append((i, eigs))

    # Pooled, over every layer's sample together. Second, not first: it is the
    # headline number and it is also the one that can hide a single rotating
    # layer inside twenty that do not.
    alpha, beta = torch.cat(alphas), torch.cat(betas)
    eigs = eigenvalues(alpha, beta, torch.cat(ks))
    out.update(classify(eigs, tol))
    out.update(_tol_sweep(eigs))
    out.update(_factors(alpha, beta))
    # BY POSITION IN THE SEQUENCE. The sample is uniform over positions and the
    # recurrence is not: at t = 0 the state is empty and a gate near 1 means
    # "nothing to forget", while at t = 4000 it means "carry what is there".
    # Averaging the two is how a measurement of the model becomes a measurement
    # of where the tokens were. Buckets are geometric because whatever happens
    # early happens fast.
    pos = torch.cat(poss)
    radius = eigs.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    rot = (eigs.imag.abs() > tol * radius).any(dim=-1)
    edges = [0, 1, 8, 64, 512, 2048, max(seq, 2049)]
    out["seq_len_sampled"] = int(seq)
    out["by_position"] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pos >= lo) & (pos < hi)
        if not bool(m.any()):
            continue
        out["by_position"].append({
            "pos_lo": lo, "pos_hi": hi, "n": int(m.sum()),
            "frac_alpha_neg": float((alpha[m] < 0).float().mean()),
            "frac_beta_gt1": float((beta[m] > 1).float().mean()),
            "median_abs_alpha": float(alpha[m].abs().median()),
            "mean_alpha_range": float(_range(alpha[m]).mean()),
            "frac_transition_any_complex": float(rot[m].float().mean()),
        })

    # The spread across layers, so "pooled" can be read against how uneven it is.
    for key in ("frac_eig_complex", "frac_eig_real_neg", "frac_alpha_neg",
                "frac_beta_gt1"):
        vals = [r[key] for r in out["per_layer"] if r.get(key) is not None]
        out[f"{key}_layer_min_max"] = [min(vals), max(vals)] if vals else None
        out[f"{key}_argmax_layer"] = (max(range(len(vals)), key=vals.__getitem__)
                                      if vals else None)

    if dump_eigs is not None and dumped:
        import numpy as np

        # Thinned to `dump_max` TOTAL, evenly across layers, so a 24-layer
        # 128-dim sample does not write 400k complex numbers per checkpoint.
        per = max(1, dump_max // len(dumped))
        lay, re, im = [], [], []
        for i, e in dumped:
            flat = e.reshape(-1)
            if flat.numel() > per:
                sel = torch.linspace(0, flat.numel() - 1, per).long()
                flat = flat[sel]
            lay.append(np.full(flat.numel(), i, dtype=np.int16))
            re.append(flat.real.numpy().astype(np.float32))
            im.append(flat.imag.numpy().astype(np.float32))
        f = Path(dump_eigs)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            f.with_name(f"{f.stem}_step{out['step']}.npz"),
            layer=np.concatenate(lay), re=np.concatenate(re),
            im=np.concatenate(im), step=out["step"],
            unsigned=out["unsigned_gate"])
        out["eigs_dumped"] = str(f.with_name(f"{f.stem}_step{out['step']}.npz"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, nargs="+")
    ap.add_argument("--step", type=int, action="append",
                    help="checkpoint step; repeatable. Default: the last one")
    ap.add_argument("--all-steps", action="store_true",
                    help="every checkpoint under the run, oldest first")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--per-call", type=int, default=64,
                    help="(token, head) transitions sampled per layer per batch")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--dump-eigs", type=Path,
                    help="also write the raw spectrum to PATH_step<N>.npz")
    a = ap.parse_args(argv)

    from measure_signs import checkpoint_steps

    out = []
    for c in a.config:
        steps = (checkpoint_steps(c) if a.all_steps else (a.step or [None]))
        for s in steps:
            try:
                r = measure(c, s, a.batches, a.per_call, a.seq_len, a.tol,
                            dump_eigs=a.dump_eigs)
            except Exception as e:                 # one bad step must not end
                r = {"config": c, "step": s, "error": f"{type(e).__name__}: {e}"}
            out.append(r)
            if "error" in r:
                print(f"{Path(c).stem:<44} step {str(s):>7}  {r['error']}")
            else:
                lo, hi = r["frac_eig_complex_layer_min_max"]
                print(f"{Path(c).stem:<44} step {r['step']:>7}  "
                      f"cplx={100 * r['frac_eig_complex']:5.2f}%  "
                      f"neg={100 * r['frac_eig_real_neg']:5.2f}%  "
                      f"rot={100 * r['frac_transition_any_complex']:5.1f}%  "
                      f"a<0={100 * r['frac_alpha_neg']:4.1f}%  "
                      f"b>1={100 * r['frac_beta_gt1']:4.1f}%  "
                      f"| cplx by layer {100 * lo:.2f}-{100 * hi:.2f}% "
                      f"(max L{r['frac_eig_complex_argmax_layer']} "
                      f"of {r['n_recurrent_layers']})")
            if a.json:                              # written as it goes, so a
                a.json.write_text(json.dumps(out, indent=2) + "\n")
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
