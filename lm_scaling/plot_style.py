"""The one plot style every figure in this package uses.

    import plot_style
    plot_style.apply()            # after matplotlib.use("Agg")

ONE COPY, because the alternative is two. `scaling_plots.py` and
`gate_spectrum_plot.py` write figures that land in the same document, and a
style block pasted into both drifts the moment one of them is edited -- the
reader then sees two typefaces in one paper and cannot tell which is
deliberate.

WHY THE TIMES FAMILY IS RESOLVED RATHER THAN NAMED. Setting

    rcParams["font.family"] = "Times New Roman"

on a machine without that font does NOT fall back to a serif. matplotlib logs
`findfont: Font family ['Times New Roman'] not found. Falling back to DejaVu
Sans.` -- at DEBUG level, with no warning -- and renders the whole figure in a
SANS face. That is the exact opposite of the intent and it is silent. Checked
on this box, which has no Times New Roman: every label came out DejaVu Sans.

So the name is resolved against the installed fonts, preferring Times New
Roman and falling back through its metric-compatible clones (Nimbus Roman is
URW's Times, Liberation Serif is metric-compatible with Times New Roman), and
only then to a generic serif. `font.family` takes a list and would degrade on
its own; `mathtext.rm` and friends take ONE name each and would not, which is
why the resolution happens here instead.

IF YOU HAVE JUST INSTALLED THE FONT and the figures still come out in the
fallback, matplotlib is reading a CACHED font list: delete `fontlist-*.json`
under `$MPLCONFIGDIR` (or `~/.cache/matplotlib`) and rerun. Installing
msttcorefonts does not invalidate that cache.
"""
from __future__ import annotations

#: In preference order. The first one actually installed wins, so a machine
#: with real Times New Roman renders with it and this box renders with the
#: clone rather than silently with a sans face.
TIMES = ("Times New Roman", "Nimbus Roman", "Liberation Serif", "Times",
         "FreeSerif", "DejaVu Serif")


def times_family() -> tuple[str, list[str]]:
    """(best installed member of `TIMES`, the installed ones in preference order).

    Only INSTALLED names go in the list. A `font.family` list is a fallback
    CHAIN, and matplotlib resolves every entry in it -- so leaving the missing
    names in prints `findfont: Font family 'Times New Roman' not found.` to
    stdout on every run even though the chain resolves fine. These scripts'
    stdout carries the numbers, and a warning that means nothing trains a
    reader to skip warnings that do.
    """
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    found = [n for n in TIMES if n in have]
    if not found:
        return "DejaVu Serif", ["DejaVu Serif", "serif"]
    return found[0], [*found, "serif"]


def apply(plt=None, size: float = 11.5):
    """Install the style. `size` is the base font size everything scales from."""
    if plt is None:
        import matplotlib.pyplot as plt  # noqa: F811
    import scienceplots  # noqa: F401  (registers the styles by importing)

    plt.style.use(["science", "no-latex", "light"])
    plt.rcParams["figure.constrained_layout.use"] = True
    fam, chain = times_family()
    plt.rcParams["font.family"] = chain
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = fam
    plt.rcParams["mathtext.it"] = f"{fam}:italic"
    plt.rcParams["mathtext.bf"] = f"{fam}:bold"
    # `mathtext.fontset = "custom"` resolves EVERY class, not just the three
    # above, and the defaults for the rest are generic families this box does
    # not have -- `mathtext.cal` is "cursive", which logs a findfont fallback
    # on every figure. `tt` stays monospace: it is the typewriter class.
    plt.rcParams["mathtext.cal"] = f"{fam}:italic"
    plt.rcParams["mathtext.sf"] = fam
    plt.rcParams.update({
        "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "font.size": size,
    })
    return fam
