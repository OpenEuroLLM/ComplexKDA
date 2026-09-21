"""The table's metric conventions, pinned.

The harness reports several numbers per task and the published table names one.
HellaSwag and ARC-c are length-normalized there (`acc_norm`) and everything else
is plain `acc`; taking `acc` for HellaSwag moves it 2-4 points, which is larger
than any difference this campaign measures. The average is over the eight
accuracies and excludes the perplexities.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO), str(_REPO / "lm_scaling")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_downstream import TASKS  # noqa: E402
from eval_table import ACC, PPL, row  # noqa: E402


def test_no_column_names_a_task_nobody_scores():
    # One direction only: a convention may LEAVE OUT a task (Gated DeltaNet's
    # table has no OpenBookQA), but a column naming a task the suite never runs
    # would print an empty cell and quietly drop out of the average.
    assert {t for _, t, _ in PPL + ACC} <= set(TASKS)


def test_the_normalized_metrics_are_the_published_ones():
    metric = {task: m for _, task, m in ACC}
    assert metric["hellaswag"] == "acc_norm,none"
    assert metric["arc_challenge"] == "acc_norm,none"
    for task in ("piqa", "winogrande", "arc_easy", "social_iqa", "boolq",
                 "lambada_openai"):
        assert metric[task] == "acc,none"


def test_the_average_is_over_the_eight_accuracies():
    res = {task: {m: 0.5} for _, task, m in ACC}
    res["wikitext"] = {"word_perplexity,none": 20.0}
    res["lambada_openai"]["perplexity,none"] = 10.0
    r = row(res)
    assert len(ACC) == 8
    assert r["avg"] == 0.5
    assert r["wiki.ppl"] == 20.0 and r["lmb.ppl"] == 10.0


def test_a_missing_task_gets_no_average():
    # Seven of eight is not the average column, and reporting it as one would
    # make an arm look better or worse than its peers for a reason no reader can
    # see.
    res = {task: {m: 0.5} for _, task, m in ACC if task != "boolq"}
    assert row(res)["avg"] is None


def test_the_published_rows_carry_every_column():
    """A transcribed table is only useful if it is complete and well-formed.

    The rows are compared against ours column by column; a missing or misspelled
    key would print a KeyError at the end of a campaign, and a number typed into
    the wrong column would print nothing at all and be believed.
    """
    import json

    from eval_table import ACC, PPL, PUBLISHED

    pub = json.loads(PUBLISHED.read_text())
    cols = [c for c, _, _ in PPL + ACC] + ["avg"]
    assert "arXiv:2412.06464" in pub["source"]
    assert pub["rows"], "no published rows"
    for name, pub_row in pub["rows"].items():
        assert pub_row.get("group") in ("recurrent", "hybrid"), f"{name}: bad group"
        assert set(pub_row) - {"group"} == set(cols), f"{name}: {set(cols) ^ set(pub_row)}"
        # Perplexities are perplexities and accuracies are percentages: a value
        # in the wrong column is the transcription error that does not announce
        # itself.
        assert 10 < pub_row["wiki.ppl"] < 40 and 10 < pub_row["lmb.ppl"] < 40
        accs = [pub_row[c] for c, _, _ in ACC]
        assert all(30 < v < 80 for v in accs), name
        # The paper's own averages, to a tenth. Every row agrees to 0.005
        # except Gated DeltaNet-H1, whose eight accuracies mean 56.46 against a
        # printed 56.40 -- checked against two independent fetches of the paper,
        # so it is their row and not a transcription slip.
        assert abs(sum(accs) / len(accs) - pub_row["avg"]) < 0.1, f"{name}: avg disagrees"


def test_the_two_conventions_differ_where_the_papers_differ():
    """Each table follows its own paper's rules, and they are not the same.

    Gated DeltaNet-2 adds OpenBookQA and reports ARC-c unnormalized. Reading our
    acc_norm into its column would flatter us by ~3 points -- the same Gated
    DeltaNet model is printed 38.39 in the older table and 35.15 in the newer.
    """
    from eval_table import ACC, ACC_GDN2, CONVENTIONS

    assert set(CONVENTIONS) == {"gdn", "gdn2"}
    gdn_acc, gdn_pub = CONVENTIONS["gdn"]
    gdn2_acc, gdn2_pub = CONVENTIONS["gdn2"]
    assert gdn_acc is ACC and gdn2_acc is ACC_GDN2
    assert len(ACC) == 8 and len(ACC_GDN2) == 9
    assert {t: m for _, t, m in ACC}["arc_challenge"] == "acc_norm,none"
    assert {t: m for _, t, m in ACC_GDN2}["arc_challenge"] == "acc,none"
    assert "openbookqa" not in {t for _, t, _ in ACC}
    assert "openbookqa" in {t for _, t, _ in ACC_GDN2}
    assert gdn_pub.exists() and gdn2_pub.exists()


def test_the_gdn2_rows_carry_every_column_and_their_average():
    import json

    from eval_table import ACC_GDN2, CONVENTIONS, PPL

    _, path = CONVENTIONS["gdn2"]
    pub = json.loads(path.read_text())
    cols = [c for c, _, _ in PPL + ACC_GDN2] + ["avg"]
    assert "arXiv:2605.22791" in pub["source"]
    assert "KDA" in pub["rows"], "the row our baseline is compared with"
    for name, r in pub["rows"].items():
        assert r.get("group") in ("recurrent", "hybrid"), f"{name}: bad group"
        assert set(r) - {"group"} == set(cols), f"{name}: {set(cols) ^ set(r)}"
        accs = [r[c] for c, _, _ in ACC_GDN2]
        assert abs(sum(accs) / len(accs) - r["avg"]) < 0.05, f"{name}: avg disagrees"


def test_every_task_scored_appears_in_one_convention_or_the_other():
    from eval_downstream import TASKS
    from eval_table import ACC, ACC_GDN2, PPL

    covered = {t for _, t, _ in PPL + ACC} | {t for _, t, _ in ACC_GDN2}
    assert covered == set(TASKS)


def test_latex_and_markdown_render_without_dropping_a_value():
    """The renderers are what a paper pastes, so they get exercised, not trusted.

    The bug this catches: bolding by `.format` on a doubled-brace template drops
    the number and emits an empty \\textbf{}. A table of blanks in the one column
    a model wins is worse than no table.
    """
    import io
    import json
    from contextlib import redirect_stdout

    from eval_table import CONVENTIONS, main

    results = Path("/nonexistent")  # no arms; the published rows still render
    for convention in sorted(CONVENTIONS):
        acc, published = CONVENTIONS[convention]
        pub = json.loads(published.read_text())
        for fmt in ("text", "markdown", "latex"):
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--results", str(results), "--step", "190976",
                      "--convention", convention, "--format", fmt])
            out = buf.getvalue()
            assert r"\textbf{}" not in out and "****" not in out
            # Every published row reaches the output with its average.
            for name, r in pub["rows"].items():
                if fmt == "text":
                    assert name[:20] in out, (convention, fmt, name)
                else:
                    assert f"{r['avg']:.2f}" in out, (convention, fmt, name)
        # LaTeX structure the document needs to compile.
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["--results", str(results), "--step", "190976",
                  "--convention", convention, "--format", "latex"])
        tex = buf.getvalue()
        assert tex.count(r"\begin{table*}") == 1 and tex.count(r"\end{table*}") == 1
        assert tex.count(r"\begin{tabular}") == 1 and tex.count(r"\end{tabular}") == 1
        assert tex.count(r"\toprule") == 1 and tex.count(r"\bottomrule") == 1
        # One cell per column on every body row.
        ncols = len([c for c, _, _ in PPL + acc]) + 1
        for line in tex.splitlines():
            if line.strip().endswith(r"\\") and "multicolumn" not in line:
                assert line.count("&") == ncols, line


def _fake_results(tmp_path, step=190976):
    """Four arms' worth of harness output, so the paper file renders in full.

    The real JSONs live on the cluster; a structural test must not need them,
    and what is being checked here is the LaTeX, not the numbers.
    """
    import json

    from eval_table import ACC_GDN2, ARMS, PPL

    for i, arm in enumerate(ARMS):
        res: dict = {}
        for _, task, metric in PPL + ACC_GDN2:
            res.setdefault(task, {})[metric] = 15.0 + i if "perplexity" in metric \
                else 0.50 + i / 100
        (tmp_path / f"{arm}_step{step}.json").write_text(json.dumps(res))
    return tmp_path


def _render(results, convention="gdn2", fmt="paper", step=190976):
    import io
    from contextlib import redirect_stdout

    from eval_table import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--results", str(results), "--step", str(step),
              "--convention", convention, "--format", fmt])
    return buf.getvalue()


def test_the_paper_file_is_two_well_formed_tables():
    """`--format paper` is what the paper \\input's, so its structure is pinned.

    Two tables, not one: the comparison against the published rows, and the
    pairing that is what this campaign actually varied. A body row short one
    cell is a LaTeX error at submission time, and a stray `$` is a document that
    compiles into gibberish -- neither shows up in a diff of numbers.
    """
    import re
    import tempfile

    from eval_table import ACC_GDN2, PAIRS, PPL

    with tempfile.TemporaryDirectory() as d:
        tex = _render(_fake_results(Path(d)))

    assert len(re.findall(r"\\begin\{table\*\}", tex)) == 1
    assert len(re.findall(r"\\end\{table\*\}", tex)) == 1
    assert len(re.findall(r"\\begin\{table\}", tex)) == 1
    assert len(re.findall(r"\\end\{table\}", tex)) == 1
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 2
    assert tex.count(r"\toprule") == tex.count(r"\bottomrule") == 2
    assert r"\label{tab:fwedu-1p3b-gdn2}" in tex
    assert r"\label{tab:fwedu-1p3b-pairing}" in tex
    assert r"\textbf{}" not in tex

    body = tex.split(r"\begin{table}")  # [0] the comparison, [1] the pairing
    wide = len([c for c, _, _ in PPL + ACC_GDN2]) + 1
    for chunk, ncols in ((body[0], wide), (body[1], 6)):
        for line in chunk.splitlines():
            if line.strip().endswith(r"\\") and "multicolumn" not in line:
                assert line.count("&") == ncols, line
            assert line.count("$") % 2 == 0, f"unbalanced math: {line}"

    # One row per pair, labelled by its stack rather than by an arm's filename.
    assert len(PAIRS) == 2
    assert "Pure (KDA) &" in tex and "Hybrid (KDA + attn 3:1) &" in tex


def test_the_paper_file_states_what_may_not_be_claimed():
    """The caption carries the replication check, not just the win.

    Table 1 has the signed arm ahead on downstream accuracy. Eighteen paired
    ladder cells say otherwise, and held-out loss says otherwise here. A table
    that travels without that is a claim this study cannot support.
    """
    import tempfile

    from eval_table import LADDER_AVG9

    with tempfile.TemporaryDirectory() as d:
        tex = _render(_fake_results(Path(d)))

    assert f"{abs(LADDER_AVG9['mean']):.2f} points" in tex
    assert f"{LADDER_AVG9['negative']} of {LADDER_AVG9['cells']}" in tex
    assert "matches the baseline, not that it beats it" in tex
    assert "disagree in sign" in tex


def test_the_replication_figure_agrees_with_the_results_document():
    """LADDER_AVG9 is the companion ladder's pooled pairing, and it ends up in
    the caption of a paper table -- which makes it the one number here that a
    reader carries away without a way to check it.

    So the caption's source and the document that explains it must state the
    same figures. If the ladder is ever re-scored, one of the two moves and
    this fails, rather than letting a stale figure ship in a caption.
    """
    import re

    from eval_table import LADDER_AVG9

    record = (_REPO / "lm_scaling" / "config" / "experiments"
              / "fwedu_1p3B_results.md").read_text()
    # The document states it as a signed mean; the constant carries the sign
    # separately, so the magnitudes are compared and the sign asserted below.
    m = re.search(r"reads \*\*[−-]([0-9.]+) points, se ([0-9.]+), "
                  r"negative in (\d+) of (\d+)\*\*", record)
    assert m, "the results document no longer states the pooled avg9 difference"
    assert abs(float(m.group(1)) - abs(LADDER_AVG9["mean"])) < 5e-4
    assert abs(float(m.group(2)) - LADDER_AVG9["se"]) < 5e-4
    assert (int(m.group(3)), int(m.group(4))) == \
        (LADDER_AVG9["negative"], LADDER_AVG9["cells"])
    assert LADDER_AVG9["mean"] < 0, (
        "the caption says the ladder TRAILS; a positive mean would make the "
        "generated text say the opposite of the number")


def test_every_column_is_won_inside_every_block():
    """A blocked table is a comparison within its blocks, so each block bolds
    its own best.

    Pooling both blocks -- the old behaviour, still available as
    `--bold global` -- gave the recurrent half 3 of 12 marked columns and our
    own baseline row none at all, because a hybrid beats a pure recurrent model
    on nearly everything. A reader sees a table where a whole section won
    nothing, which is not what the numbers say.
    """
    import tempfile

    from eval_table import ACC_GDN2, PPL

    cols = [c for c, _, _ in PPL + ACC_GDN2] + ["avg"]
    with tempfile.TemporaryDirectory() as d:
        tex = _render(_fake_results(Path(d)), fmt="latex")

    body = tex.split(r"\midrule")[1:]          # one chunk per block
    assert len(body) == 2, "expected a recurrent and a hybrid block"
    for blk in body:
        marked = [0] * len(cols)
        for line in blk.splitlines():
            s = line.strip()
            if not s.endswith(r"\\") or "multicolumn" in s:
                continue
            for i, c in enumerate(s.replace(r"\\", "").split("&")[1:]):
                marked[i] += r"\textbf" in c
        assert all(m >= 1 for m in marked), \
            dict(zip(cols, marked))


def test_cells_that_print_the_same_number_are_bolded_together():
    """Bolding compares what is printed, not what is stored.

    Our rows come from the harness at full precision and the published rows were
    transcribed at two decimals. 0.727965 and 0.7280 both print 72.80; bolding
    only the larger one puts two identical numbers in a column with a mark on
    one of them, which a reader reads as a broken table and not as the
    seven-thousandth of a point that separates them.
    """
    from eval_table import cell

    mark = r"\textbf{{{}}}"
    assert cell("piqa", 0.727965, 0.7280, mark) == r"\textbf{72.80}"
    assert cell("wiki.ppl", 15.7749, 15.7702, mark) == r"\textbf{15.77}"
    # A difference that IS visible still marks only the winner.
    assert cell("piqa", 0.7270, 0.7280, mark) == "72.70"
    assert cell("wiki.ppl", 15.78, 15.77, mark) == "15.78"


def test_the_bold_mode_is_stated_in_the_caption():
    """Whoever reads the table has to know what a bold means, and the two modes
    mean different things."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        results = _fake_results(Path(d))
        assert "within each block in bold" in _render(results, fmt="latex")
        assert "across both blocks in bold" in \
            _render_argv(results, "--bold", "global")


def _render_argv(results, *extra, convention="gdn2", fmt="latex", step=190976):
    import io
    from contextlib import redirect_stdout

    from eval_table import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--results", str(results), "--step", str(step),
              "--convention", convention, "--format", fmt, *extra])
    return buf.getvalue()
