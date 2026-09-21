"""Figures for the gate spectrum over training.

    lm_scaling/gate_spectrum_plot.py --out lm_scaling/tex

reading `lm_scaling/harvest/gate_spectrum/` by default -- the measurements
behind the paper's figures, committed. `--results DIR` plots another run;
`gate_spectrum.sbatch` is what produces one.

THREE FIGURES, in the order they earn their space.

1. THE COMPLEX PLANE (`gate_spectrum_plane`). The one that makes the claim
   rather than summarising it: a log-density map of where the transition's
   eigenvalues actually sit, unit circle drawn, unsigned arm beside signed. The
   unsigned arm is a cloud on the positive real axis and nothing else -- it
   cannot be otherwise, `M` is a product of two positive definite matrices --
   and the signed arm spreads down the negative reals with a low-density haze
   off the axis where the rotations live. A density map and not a scatter,
   because 40,000 points on a 2-inch panel is a blob.

2. THE TRAJECTORY (`gate_spectrum_trajectory`). Three stacked panels on one
   token axis, which is the only way to see that the three quantities move on
   different schedules:
     (a) dt_bias on a log axis with the pure-weight-decay curve over it, so
         "decay explains the parameter" is visible rather than asserted
     (b) the two factors: alpha < 0 saturates at ~6% a third of the way in,
         beta > 1 climbs the whole way
     (c) the outcome, which follows beta and not alpha
   The unsigned arms are drawn as the flat zero they must be: a control on the
   page, not a claim in a caption.

3. DEPTH AGAINST TIME (`gate_spectrum_depth`). Layers down, tokens across,
   colour is the share of transitions that rotate. Says in one glance that this
   is a bottom-of-the-stack phenomenon and that it stays there. An appendix
   figure -- it answers a question the other two raise.

WHAT IS DELIBERATELY NOT PLOTTED: the rotation ANGLE against the share. The
angles are near pi and the moduli are small, so "rotates by 118 degrees" and
"is forgotten within two steps" are the same population, and a panel showing
only the first invites the wrong reading. Both go in the caption as numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import plot_style  # noqa: E402  (needs the bootstrap above)

TOK_PER_STEP = 128 * 4096

# The campaign's own schedule, for the weight-decay overlay. AdamW's decoupled
# decay multiplies a parameter by (1 - lr_t * wd) each step, so with no gradient
# at all the bias follows dt_bias(0) * exp(-wd * sum lr_t).
PEAK_LR, WARMUP, WD, TOTAL = 4e-4, 1907, 0.1, 190976
DT_BIAS_INIT = 5.29        # 2*atanh(exp(-0.01)), the median of logU(1e-3, 1e-1)

DISPLAY = {"ckda-shipped-lowrank": "CKDA",
           "kda-sig-lowrank": "KDA, bounded gate",
           "ckda-shipped-hybrid-lowrank": "CKDA + attn 3:1",
           "kda-sig-hybrid-lowrank": "KDA + attn 3:1"}
COLOR = {"ckda-shipped-lowrank": "#1f77b4", "kda-sig-lowrank": "#d62728",
         "ckda-shipped-hybrid-lowrank": "#2ca02c",
         "kda-sig-hybrid-lowrank": "#ff7f0e"}


def arm_of(config_path: str) -> str | None:
    m = re.search(r"fwedu_1p3B_(.+?)_1\.3B", config_path)
    return m.group(1) if m else None


def load(results: Path) -> dict[str, list[dict]]:
    """{arm: [checkpoint, ...]} from every spectrum_*.json under `results`."""
    out: dict[str, list[dict]] = {}
    for f in sorted(results.glob("spectrum_*.json")):
        for r in json.loads(f.read_text()):
            if "error" in r or not (arm := arm_of(r.get("config", ""))):
                continue
            out.setdefault(arm, []).append(r)
    for arm in out:                      # one row per step, newest file wins
        by_step = {r["step"]: r for r in out[arm]}
        out[arm] = [by_step[s] for s in sorted(by_step)]
    return out


def wd_only(steps):
    """dt_bias under weight decay alone, no gradient."""
    cum, s = 0.0, set(steps)
    floor, seen = 0.1 * PEAK_LR, {}
    for t in range(1, TOTAL + 1):
        if t <= WARMUP:
            cum += PEAK_LR * t / WARMUP
        else:
            p = (t - WARMUP) / (TOTAL - WARMUP)
            cum += floor + 0.5 * (PEAK_LR - floor) * (1 + math.cos(math.pi * p))
        if t in s:
            seen[t] = DT_BIAS_INIT * math.exp(-WD * cum)
    return [seen.get(t, float("nan")) for t in steps]


def rotating(r) -> float:
    """Share of (token, head) transitions carrying a complex pair.

    Accepts either spelling: the JSON already on disk from the first sweep says
    `frac_step_any_complex`, which was renamed because "step" reads as a
    training step in a document about training.
    """
    return r.get("frac_transition_any_complex", r.get("frac_step_any_complex"))


def dt_median(r) -> float:
    m = [b["dt_bias_min_med_max"][1] for b in r["bias"] if "dt_bias_min_med_max" in b]
    return float(np.median(m)) if m else float("nan")


def fig_trajectory(data, out: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(7.0, 8.2), sharex=True)
    for arm, rows in sorted(data.items()):
        x = [r["step"] * TOK_PER_STEP / 1e9 for r in rows]
        c, lbl = COLOR.get(arm, "grey"), DISPLAY.get(arm, arm)
        signed = not rows[0]["unsigned_gate"]
        ls = "-" if signed else "--"
        ax[0].plot(x, [abs(dt_median(r)) for r in rows], ls, color=c, label=lbl)
        ax[1].plot(x, [100 * r["frac_alpha_neg"] for r in rows], ls, color=c)
        ax[1].plot(x, [100 * r["frac_beta_gt1"] for r in rows], ls, color=c,
                   alpha=0.45, lw=1.2, marker="^", ms=3, markevery=3)
        # The share of (token, head) transitions that rotate, and ONLY that.
        # The share of eigenvalues is not a second measurement: a rank-one
        # update of a diagonal admits at most one conjugate pair, so it is this
        # number times 2/head_dim exactly (checked to 1e-7 over 44 checkpoints).
        # It reads 60x smaller, which invites "rotation is vanishingly rare"
        # when the truth is "a quarter of the maximum the parameterisation
        # allows".
        ax[2].plot(x, [100 * rotating(r) for r in rows], ls, color=c)
        if signed and arm == "ckda-shipped-lowrank":
            ax[0].plot(x, wd_only([r["step"] for r in rows]), ":", color="k",
                       lw=1.6, label="weight decay alone, no gradient")

    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$|$dt_bias$|$  (median over layers)")
    ax[0].axhline(DT_BIAS_INIT, color="grey", lw=0.8, alpha=0.5)
    ax[0].text(0.6, DT_BIAS_INIT * 1.1, "init 5.29", fontsize=8, color="grey")
    ax[0].legend(fontsize=8, loc="lower left")
    ax[0].set_title("The gate bias collapses; what it controls does not follow",
                    fontsize=10)

    ax[1].set_ylabel("occupancy (%)")
    ax[1].text(0.02, 0.93, r"bold: $\alpha<0$      faint with $\triangle$: $\beta>1$",
               transform=ax[1].transAxes, fontsize=8, va="top")
    ax[2].set_ylabel("(token, head) transitions that rotate (%)")
    ax[2].text(0.02, 0.93, "share carrying a complex conjugate pair "
               "(at most one is possible)", transform=ax[2].transAxes,
               fontsize=8, va="top")
    ax[2].set_xlabel("training tokens (B)")
    for a in ax:
        a.grid(alpha=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gate_spectrum_trajectory.{ext}", dpi=160,
                    bbox_inches="tight")
    return "gate_spectrum_trajectory"


def fig_depth(data, out: Path):
    import matplotlib.pyplot as plt

    arms = [a for a in ("ckda-shipped-lowrank", "ckda-shipped-hybrid-lowrank")
            if a in data]
    if not arms:
        return None
    fig, axes = plt.subplots(1, len(arms), figsize=(5.6 * len(arms), 4.4),
                             squeeze=False)
    for ax, arm in zip(axes[0], arms):
        rows = data[arm]
        m = np.array([[100 * rotating(l) for l in r["per_layer"]]
                      for r in rows]).T
        x = [r["step"] * TOK_PER_STEP / 1e9 for r in rows]
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="magma",
                       extent=[min(x), max(x), -0.5, m.shape[0] - 0.5])
        ax.set_title(DISPLAY.get(arm, arm), fontsize=10)
        ax.set_xlabel("training tokens (B)")
        ax.set_ylabel("layer")
        fig.colorbar(im, ax=ax, label="(token, head) transitions that rotate (%)")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gate_spectrum_depth.{ext}", dpi=160,
                    bbox_inches="tight")
    return "gate_spectrum_depth"


ALPHA_RANGE, BETA_RANGE = (-1.0, 1.0), (0.0, 2.0)


def _edges(n, rng):
    return np.linspace(rng[0], rng[1], n + 1)


def _stack(rows, key, rng, target_per_bin: int = 20):
    """(bins, len(rows)) column-normalised density, or None if not captured.

    Column-normalised because the columns are compared with each other and a
    checkpoint with more samples would otherwise read as a denser one.

    REBINNED to the sample. alpha has head_dim values per transition and beta
    has one, so at 256 transitions per layer an 81-bin beta histogram holds
    three counts a bin and renders as static -- a picture of the sample size,
    not of the distribution. Adjacent bins are summed until a bin holds
    `target_per_bin` on average; alpha is left alone because it already does.
    """
    # EVERY row, not just the first: a results directory legitimately mixes
    # sweeps, and a field added later is present in some checkpoints and not
    # others. Checking rows[0] alone raised a KeyError halfway through the
    # stack, which is a confusing way to learn that.
    if not rows or any(key not in r for r in rows):
        return None, None
    m = np.array([r[key] for r in rows], dtype=float).T
    nbins = m.shape[0]
    per_col = m.sum(axis=0).mean()
    group = max(1, int(np.ceil(nbins * target_per_bin / max(per_col, 1))))
    while group > 1 and nbins % group:          # keep the edges exact
        group -= 1
    if group > 1:
        m = m.reshape(nbins // group, group, m.shape[1]).sum(axis=1)
    s = m.sum(axis=0, keepdims=True)
    return (np.divide(m, s, out=np.zeros_like(m), where=s > 0),
            _edges(m.shape[0], rng))


def fig_gates(data, out: Path):
    """Where alpha and beta actually sit -- as distributions, over time and depth.

    A share is a summary of a shape. "alpha < 0 is 5.9%" is equally consistent
    with a hard negative mode and with a broad distribution whose tail crosses
    zero, and those are different models; only the shape distinguishes them.

    Left column is the distribution against TRAINING, pooled over layers. Right
    column is against DEPTH at the final checkpoint. Log density, each column
    normalised on its own so the comparison is of shape and not of sample size.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    arms = [a for a in DISPLAY if a in data and not data[a][0]["unsigned_gate"]]
    if not arms or "hist_alpha" not in data[arms[0]][0]:
        return None
    arm = arms[0]
    rows = data[arm]
    panels = [("hist_alpha", ALPHA_RANGE, r"$\alpha$"),
              ("hist_beta", BETA_RANGE, r"$\beta$")]
    if all("hist_alpha_range" in r for r in rows):
        # max_i alpha - min_i alpha within a head. A head at 0 applies uniform
        # decay and cannot rotate at all -- (I - beta k k^T)cI is symmetric --
        # so this is the gate's selectivity and a bound on its spectrum at once.
        panels.append(("hist_alpha_range", (0.0, 2.0),
                       r"$\max_i\alpha_i-\min_i\alpha_i$"))
    fig, ax = plt.subplots(len(panels), 2, figsize=(11.5, 3.8 * len(panels)))
    for i, (key, rng, name) in enumerate(panels):
        dens, ed = _stack(rows, key, rng)
        if dens is None:
            continue
        x = [r["step"] * TOK_PER_STEP / 1e9 for r in rows]
        im = ax[i][0].imshow(dens, aspect="auto", origin="lower", cmap="magma",
                             norm=LogNorm(vmin=max(dens[dens > 0].min(), 1e-6),
                                          vmax=dens.max()),
                             extent=[min(x), max(x), ed[0], ed[-1]])
        ax[i][0].set_xlabel("training tokens (B)")
        ax[i][0].set_ylabel(name)
        ax[i][0].set_title(f"{name} over training, all layers pooled", fontsize=10)
        fig.colorbar(im, ax=ax[i][0], label="density")

        last = rows[-1]["per_layer"]
        dl, ed = _stack(last, key, rng)
        if dl is None:
            continue
        im = ax[i][1].imshow(dl, aspect="auto", origin="lower", cmap="magma",
                             norm=LogNorm(vmin=max(dl[dl > 0].min(), 1e-6),
                                          vmax=dl.max()),
                             extent=[-0.5, len(last) - 0.5, ed[0], ed[-1]])
        ax[i][1].set_xlabel("layer")
        ax[i][1].set_ylabel(name)
        ax[i][1].set_title(f"{name} by layer, final checkpoint", fontsize=10)
        fig.colorbar(im, ax=ax[i][1], label="density")
        line = {0: 0.0, 1: 1.0}.get(i)          # no boundary for the range
        if line is not None:
            for a in (ax[i][0], ax[i][1]):
                a.axhline(line, color="w", lw=0.8, ls="--", alpha=0.7)
    fig.suptitle(f"{DISPLAY.get(arm, arm)}: the gates themselves. Dashed line is "
                 r"the boundary that matters ($\alpha=0$, $\beta=1$).", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gate_spectrum_gates.{ext}", dpi=160, bbox_inches="tight")
    return "gate_spectrum_gates"


def fig_position(data, out: Path):
    """Does the answer depend on WHERE in the sequence the token sat?

    The sample is uniform over a 4,096-token window and the recurrence is not:
    at t = 0 there is no state to forget, so a gate near 1 means something
    different there than at t = 4000. If the curves here are flat, the pooled
    numbers are a property of the model; if they slope, the pooled numbers are
    partly a property of where the tokens were, and the honest statistic is the
    one measured late in the window.
    """
    import matplotlib.pyplot as plt

    arms = [a for a in DISPLAY if a in data and not data[a][0]["unsigned_gate"]]
    arms = [a for a in arms if data[a] and "by_position" in data[a][0]]
    if not arms:
        return None
    keys = [("frac_alpha_neg", r"$\alpha<0$  (%)", 100),
            ("frac_beta_gt1", r"$\beta>1$  (%)", 100),
            ("frac_transition_any_complex", "transitions rotating (%)", 100),
            ("median_abs_alpha", r"median $|\alpha|$", 1),
            ("mean_alpha_range", r"mean $\max_i\alpha_i-\min_i\alpha_i$", 1)]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.6),
                             squeeze=False)
    arm = arms[0]
    rows = data[arm]
    picks = [rows[0], rows[-1]] if len(rows) > 1 else rows
    for ax, (key, label, mul) in zip(axes[0], keys):
        for r, style in zip(picks, ("--o", "-o")):
            bp = r["by_position"]
            # Plot at the bucket's geometric midpoint, +1 so t = 0 is drawable.
            x = [max(1, (b["pos_lo"] * max(b["pos_hi"] - 1, 1)) ** 0.5) for b in bp]
            ax.plot(x, [mul * b[key] for b in bp], style, ms=4,
                    label=f"{r['step'] * TOK_PER_STEP / 1e9:.0f}B tokens")
        ax.set_xscale("log")
        ax.set_xlabel("position in sequence")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{DISPLAY.get(arm, arm)}: the gates against position in the "
                 f"{rows[-1].get('seq_len_sampled', '?')}-token window", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gate_spectrum_position.{ext}", dpi=160,
                    bbox_inches="tight")
    return "gate_spectrum_position"


def fig_by_layer(data, out: Path):
    """One line per layer, coloured by depth: both gates, over training.

    The heatmaps show the shape; this shows every layer's trajectory at once,
    which is what says whether the stack moves together or layer 0 does
    something the rest do not.

    LAYER INDEX COUNTS RECURRENT LAYERS ONLY. In a hybrid, every fourth block is
    attention and never reaches the kernel, so "layer 4" here is the fifth KDA
    layer and sits at block 5. Comparing depth across the pure and hybrid panels
    means comparing fractions of the stack, not block numbers.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    arms = [a for a in DISPLAY if a in data and not data[a][0]["unsigned_gate"]]
    if not arms:
        return None
    fig, axes = plt.subplots(2, len(arms), figsize=(6.0 * len(arms), 7.2),
                             squeeze=False, sharex=True)
    for j, arm in enumerate(arms):
        rows = data[arm]
        n = len(rows[0]["per_layer"])
        cmap, norm = plt.get_cmap("viridis"), Normalize(0, n - 1)
        x = [r["step"] * TOK_PER_STEP / 1e9 for r in rows]
        for li in range(n):
            c = cmap(norm(li))
            axes[0][j].plot(x, [100 * r["per_layer"][li]["frac_alpha_neg"] for r in rows],
                            color=c, lw=1.1)
            axes[1][j].plot(x, [100 * r["per_layer"][li]["frac_beta_gt1"] for r in rows],
                            color=c, lw=1.1)
        axes[0][j].set_title(f"{DISPLAY.get(arm, arm)}  ({n} recurrent layers)",
                             fontsize=10)
        axes[0][j].set_ylabel(r"$\alpha<0$  (%)")
        axes[1][j].set_ylabel(r"$\beta>1$  (%)")
        axes[1][j].set_xlabel("training tokens (B)")
        for a in (axes[0][j], axes[1][j]):
            a.grid(alpha=0.25)
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=a, label="layer")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gate_spectrum_by_layer.{ext}", dpi=160,
                    bbox_inches="tight")
    return "gate_spectrum_by_layer"


def fig_plane(results: Path, out: Path, step: int | None = None):
    """Where the eigenvalues are: the plane, and the real line under it.

    TWO ROWS, because one view cannot carry this. Almost every eigenvalue is
    real, so the plane alone is a bright line with a faint speck of rotation
    beside it -- honest about how rare rotation is, useless for seeing the
    negative tail. The marginal histogram of Re(lambda) underneath, on a log
    count, is where "the signed arm reaches negative eigenvalues and the
    unsigned one cannot" is actually legible.

    The inset zooms the rotating cloud, which lives inside |lambda| < 0.45 --
    at the scale of the unit circle it is otherwise a few pixels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    files = sorted(results.glob("eigs_*_step*.npz"))
    if not files:
        return None
    want = step or max(int(re.search(r"_step(\d+)\.npz", f.name).group(1))
                       for f in files)
    picked = [f for f in files if f.name.endswith(f"_step{want}.npz")]
    if not picked:
        return None
    picked.sort(key=lambda f: "ckda" in f.name)   # control on the left

    n = len(picked)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 7.4), squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    ylims = []
    for j, f in enumerate(picked):
        d = np.load(f)
        re_, im_ = d["re"], d["im"]
        arm = re.sub(r"^eigs_|_step\d+\.npz$", "", f.name)
        cplx = np.abs(im_) > 1e-6
        ax = axes[0][j]
        lim = 1.12
        h = ax.hist2d(re_, im_, bins=200, range=[[-lim, lim], [-lim, lim]],
                      norm=LogNorm(), cmap="viridis")
        th = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(th), np.sin(th), color="0.45", lw=0.9, ls="--")
        ax.axhline(0, color="0.75", lw=0.5)
        ax.axvline(0, color="0.75", lw=0.5)
        ax.set_title(f"{DISPLAY.get(arm, arm)}\n"
                     f"complex {100 * cplx.mean():.2f}%   negative real "
                     f"{100 * float(np.mean(~cplx & (re_ < 0))):.2f}%", fontsize=10)
        ax.set_xlabel(r"Re $\lambda$")
        ax.set_ylabel(r"Im $\lambda$")
        ax.set_aspect("equal")
        fig.colorbar(h[3], ax=ax, label="eigenvalues per bin", fraction=0.046)
        if cplx.sum() > 20:          # the rotating cloud, at its own scale
            # Lower right: the only quadrant with no data in it, and clear of
            # the panel title, which the inset's own title collided with.
            ins = ax.inset_axes([0.63, 0.04, 0.34, 0.34])
            z = 0.45
            ins.hist2d(re_[cplx], im_[cplx], bins=60, range=[[-z, z], [-z, z]],
                       norm=LogNorm(), cmap="magma")
            ins.set_xticks([])
            ins.set_yticks([])
            ins.set_title(r"complex only, $|\lambda|<0.45$", fontsize=7)

        axl = axes[1][j]
        axl.hist(re_[~cplx], bins=180, range=(-1.05, 1.05), color="#1f77b4",
                 log=True, label="real")
        if cplx.any():
            axl.hist(re_[cplx], bins=180, range=(-1.05, 1.05), color="#d62728",
                     log=True, alpha=0.8, label="in a complex pair")
        axl.axvline(0, color="0.4", lw=0.8)
        axl.set_xlabel(r"Re $\lambda$")
        axl.set_ylabel("count")
        axl.legend(fontsize=8)
        axl.grid(alpha=0.2)
        lo, hi = axl.get_ylim()
        ylims.append((lo, hi))
    # One y range across the bottom row: the panels are read against each other,
    # and two different log floors make the control's wall at Re = 0 look like a
    # difference in counts rather than in support.
    if ylims:
        lo, hi = min(l for l, _ in ylims), max(h for _, h in ylims)
        for a in axes[1]:
            a.set_ylim(lo, hi)
    fig.suptitle(f"Transition spectrum at step {want:,} "
                 f"({want * TOK_PER_STEP / 1e9:.0f}B tokens)", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"gate_spectrum_plane.{ext}", dpi=160,
                    bbox_inches="tight")
    return "gate_spectrum_plane"


def markdown(data):
    """The tables for the results record, printed rather than transcribed.

    Same source as the figures, so a number in the record and a point on a
    curve cannot disagree.
    """
    arms = [a for a in DISPLAY if a in data]
    steps = [r["step"] for r in data[arms[0]]]
    wd = dict(zip(steps, wd_only(steps)))

    print("### dt_bias over training (median across layers of each layer's median)")
    print()
    print("| tokens (B) | step | " + " | ".join(DISPLAY[a] for a in arms)
          + " | wd alone |")
    print("|---" * (len(arms) + 3) + "|")
    for i, s in enumerate(steps):
        cells = [f"{dt_median(data[a][i]):.3f}" for a in arms]
        print(f"| {s * TOK_PER_STEP / 1e9:.1f} | {s:,} | " + " | ".join(cells)
              + f" | {wd[s]:.3f} |")
    print()
    print("### Occupancy and spectrum, signed arms")
    print()
    print("| tokens (B) | step | " + " | ".join(
        f"{DISPLAY[a]}: a<0 / b>1 / rotating" for a in arms
        if not data[a][0]["unsigned_gate"]) + " |")
    sg = [a for a in arms if not data[a][0]["unsigned_gate"]]
    print("|---" * (len(sg) + 2) + "|")
    for i, s in enumerate(steps):
        cells = []
        for a in sg:
            r = data[a][i]
            cells.append(f"{100 * r['frac_alpha_neg']:.2f}% / "
                         f"{100 * r['frac_beta_gt1']:.1f}% / "
                         f"{100 * rotating(r):.1f}%")
        print(f"| {s * TOK_PER_STEP / 1e9:.1f} | {s:,} | " + " | ".join(cells) + " |")
    print()
    print("### Final checkpoint, every arm")
    print()
    # No bare pipes in a markdown header: `median |a|` closes two cells.
    print("| arm | layers | a<0 | b>1 | mean beta | median abs alpha | "
          "rotating (token, head) | angle q50 | modulus q50 |")
    print("|---" * 9 + "|")
    for a in arms:
        r = data[a][-1]
        ang = r["angle_q50_q90_q99"][0]
        mod = r["modulus_complex_q50_q90_q99"][0]
        print(f"| {DISPLAY[a]} | {r['n_recurrent_layers']} | "
              f"{100 * r['frac_alpha_neg']:.2f}% | {100 * r['frac_beta_gt1']:.1f}% | "
              f"{r['mean_beta']:.3f} | {r['median_abs_alpha']:.4f} | "
              f"{100 * rotating(r):.1f}% | "
              f"{'--' if ang is None else f'{ang:.2f} rad'} | "
              f"{'--' if mod is None else f'{mod:.3f}'} |")


def latex(data):
    """The paper's table and figures, from the same rows the record uses."""
    arms = [a for a in DISPLAY if a in data]
    print("% Generated by lm_scaling/gate_spectrum_plot.py -- regenerate, do not edit:")
    print("%   lm_scaling/gate_spectrum_plot.py --latex "
          "> lm_scaling/tex/gate_spectrum.tex")
    print("%   lm_scaling/gate_spectrum_plot.py --out lm_scaling/tex")
    print("% NEEDS: booktabs, graphicx. Figures are written beside this file.")
    print(r"\begin{table}[t]")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \caption{Transition spectrum at the end of training, 1.3B/100BT on "
          r"FineWeb-Edu. The unsigned arms are a \emph{control}: with "
          r"$\beta<1$ and $\alpha>0$ the transition "
          r"$M=(I-\beta kk^{\top})\mathrm{Diag}(\alpha)$ is similar to a "
          r"symmetric positive definite matrix, so no eigenvalue can be negative "
          r"or complex, and none is, at any of their 22 checkpoints. "
          r"\emph{Rotating} is the share of transitions carrying at least one "
          r"conjugate pair; the share of individual eigenvalues in one is "
          r"$\sim$60$\times$ smaller and is given beside it. Rotations are wide "
          r"but shallow: median $|\arg\lambda|$ near $2$\,rad with median "
          r"$|\lambda|$ below $0.1$, i.e. forgotten within a step or two.}")
    print(r"  \label{tab:gate-spectrum}")
    print(r"  \begin{tabular}{lrrrrrr}")
    print(r"    \toprule")
    print(r"    & & \multicolumn{2}{c}{occupancy} & rotating "
          r"& \multicolumn{2}{c}{when it rotates} \\")
    print(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-5} \cmidrule(lr){6-7}")
    print(r"    Model & layers & $\alpha<0$ & $\beta>1$ & (token, head) "
          r"& $|\arg\lambda|$ & $|\lambda|$ \\")
    print(r"    \midrule")
    for a in arms:
        r = data[a][-1]
        ang, mod = r["angle_q50_q90_q99"][0], r["modulus_complex_q50_q90_q99"][0]
        print(f"    {DISPLAY[a]} & {r['n_recurrent_layers']} & "
              f"{100 * r['frac_alpha_neg']:.2f}\\% & {100 * r['frac_beta_gt1']:.1f}\\% & "
              f"{100 * rotating(r):.1f}\\% & "
              + (r"-- & -- \\" if ang is None else f"{ang:.2f} & {mod:.3f} \\\\"))
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")
    for name, cap in (
        ("gate_spectrum_plane",
         r"Eigenvalues of the state transition at 100B tokens. Top: log density "
         r"in the complex plane, unit circle dashed, with the rotating "
         r"population inset at its own scale ($|\lambda|<0.45$). Bottom: the "
         r"marginal in $\mathrm{Re}\,\lambda$, shared axes, which is where the "
         r"result is legible -- the bounded-gate arm stops dead at $0$, as it "
         r"must, while the signed arm runs to $-1$. A rank-one update of a "
         r"diagonal admits at most \emph{one} conjugate pair per head, so the "
         r"rotating population is thin by construction, and shallow in fact "
         r"(median $|\lambda|=0.08$). The negative reals are neither."),
        ("gate_spectrum_trajectory",
         r"Over training. (a) the gate bias \texttt{dt\_bias} against what "
         r"decoupled weight decay alone predicts, with no gradient at all: the "
         r"first checkpoint matches to $1.00$. (b) the two factors: "
         r"$\alpha<0$ saturates a third of the way in while $\beta>1$ climbs "
         r"throughout. (c) the outcome, which follows $\beta$."),
        ("gate_spectrum_depth",
         r"Share of transitions carrying a conjugate pair, by layer and token "
         r"budget. Rotation is a bottom-of-the-stack phenomenon and stays one."),
        ("gate_spectrum_gates",
         r"The gates themselves, as distributions rather than shares: log "
         r"density of $\alpha$ (top) and $\beta$ (bottom) against training "
         r"tokens with layers pooled (left) and against layer at the final "
         r"checkpoint (right). Each column is normalised on its own, so the "
         r"comparison is of shape and not of sample size, and bins are merged "
         r"where the sample is thin ($\beta$ has one value per transition "
         r"against $\alpha$'s \texttt{head\_dim}). Dashed: the boundary that "
         r"decides the spectrum, $\alpha=0$ and $\beta=1$."),
        ("gate_spectrum_by_layer",
         r"Every layer's trajectory at once, coloured by depth. $\alpha<0$ is "
         r"cleanly stratified and flattens by $\sim$40B tokens in every layer. "
         r"$\beta>1$ climbs throughout --- except in layer 0, which opens at "
         r"96\%, falls to 43\% by 35B, and climbs back to 83\%. Layer index "
         r"counts recurrent layers only: in a hybrid every fourth block is "
         r"attention and never reaches the kernel.")):
        print(r"\begin{figure}[t]")
        print(r"  \centering")
        print(f"  \\includegraphics[width=\\linewidth]{{{name}}}")
        print(f"  \\caption{{{cap}}}")
        print(f"  \\label{{fig:{name.replace('_', '-')}}}")
        print(r"\end{figure}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Defaults to the committed measurements, like the other generators here,
    # so the figures regenerate from a fresh clone with no cluster and no
    # arguments. Point it at a `gate_spectrum.sbatch` output directory to plot
    # a new run instead.
    ap.add_argument("--results", type=Path, default=_HERE / "harvest" / "gate_spectrum")
    ap.add_argument("--out", type=Path, default=_HERE / "tex")
    ap.add_argument("--step", type=int, help="which step for the complex plane")
    ap.add_argument("--markdown", action="store_true",
                    help="print the results record's tables instead of plotting")
    ap.add_argument("--latex", action="store_true",
                    help="print the paper's table and figure environments")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    data = load(a.results)
    if not data:
        raise SystemExit(f"no spectrum_*.json with a recognizable arm under {a.results}")
    if a.markdown:
        markdown(data)
        return 0
    if a.latex:
        latex(data)
        return 0
    # ONE style for every figure in the package, set once before the first
    # figure is built. These figures land in the same document as the scaling
    # ladder's; two typefaces in one paper reads as an accident.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_style.apply(plt)

    made = [fig_trajectory(data, a.out), fig_depth(data, a.out),
            fig_gates(data, a.out), fig_by_layer(data, a.out),
            fig_position(data, a.out),
            fig_plane(a.results, a.out, a.step)]
    for name in made:
        print(f"wrote {a.out / name}.pdf/.png" if name else
              "skipped a figure: no input for it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
