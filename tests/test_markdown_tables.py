"""Markdown table integrity for generated documentation and reports (Week 3 review).

GitHub-flavored Markdown splits table cells on every unescaped ``|``, including inside code
spans, so a formula such as ``mean(|y - y_hat|)`` silently breaks a table row.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from arcora.config import get_project_root
from arcora.models.modeling_strategy import METRIC_SPECS

_PIPE = re.compile(r"(?<!\\)\|")


def broken_table_rows(text: str) -> list[int]:
    """Return 1-based line numbers of table rows whose cell count differs from the header."""
    broken: list[int] = []
    header_cells: int | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = len(_PIPE.findall(stripped))
            if header_cells is None:
                header_cells = cells
            elif cells != header_cells:
                broken.append(number)
        else:
            header_cells = None
    return broken


def test_detector_flags_pipe_inside_code_span() -> None:
    table = "| a | b |\n| :--- | :--- |\n| `mean(|y|)` | ok |\n"
    assert broken_table_rows(table) == [3]
    assert broken_table_rows("| a | b |\n| --- | --- |\n| 1 | 2 |\n") == []


def test_metric_formulas_contain_no_table_pipes() -> None:
    for spec in METRIC_SPECS:
        assert "|" not in spec.formula, spec.name
        assert "|" not in spec.undefined_handling, spec.name


_ROOT = get_project_root()
_MARKDOWN_FILES = sorted(
    [*(_ROOT / "docs").glob("*.md"), *(_ROOT / "reports").rglob("*.md"), _ROOT / "README.md"]
)


@pytest.mark.parametrize("path", _MARKDOWN_FILES, ids=lambda p: p.relative_to(_ROOT).as_posix())
def test_repository_markdown_tables_are_well_formed(path: Path) -> None:
    assert broken_table_rows(path.read_text(encoding="utf-8")) == []


# --- Reporting integrity: canonical baseline-comparison names and table headings -------------

LEGACY_COMPARISON_NAMES: tuple[str, ...] = (
    "improvement_vs_baseline",
    "delta_mae",
    "mae_reduction",
    "Δ MAE vs",
    "Δ log loss vs",
    "MAE reduction",
    "Log-loss reduction",
    "Absolute Δ",
    "Δ loss",
)
_REPORTS = _ROOT / "reports"
_GENERATED_REPORTS = sorted(
    p
    for p in [*_REPORTS.glob("*.md"), *(_REPORTS / "models").glob("*.md")]
    if "week-1" not in p.name
)


def legacy_names_in(text: str) -> list[str]:
    """Return legacy baseline-comparison names that appear in ``text``."""
    return [name for name in LEGACY_COMPARISON_NAMES if name in text]


def images_directly_above_tables(text: str) -> list[int]:
    """Line numbers of images followed (after blank lines) by a table with no heading between.

    Markdown renderers that fail to load an image show its alt text, which then reads like a
    heading for the unrelated table below it.
    """
    lines = text.splitlines()
    flagged = []
    for i, line in enumerate(lines):
        if not line.strip().startswith("!["):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines) and lines[j].strip().startswith("|"):
            flagged.append(i + 1)
    return flagged


def test_legacy_detector() -> None:
    assert legacy_names_in("| delta_mae |") == ["delta_mae"]
    assert legacy_names_in("| `delta_vs_baseline` | `relative_loss_reduction_pct` |") == []
    assert images_directly_above_tables("![x](a.png)\n\n| a |\n| --- |\n") == [1]
    assert images_directly_above_tables("![x](a.png)\n\n### T\n\n| a |\n") == []


@pytest.mark.parametrize("path", _GENERATED_REPORTS, ids=lambda p: p.relative_to(_ROOT).as_posix())
def test_generated_reports_use_canonical_comparison_names(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert legacy_names_in(text) == []
    assert images_directly_above_tables(text) == []


def test_week3_report_error_summary_heading() -> None:
    path = _REPORTS / "week-3-modeling-report.md"
    if not path.is_file():
        pytest.skip("Week 3 report not generated")
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = lines.index("### Regression Error Summary (Validation)")
    table_header = next(line for line in lines[idx + 1 :] if line.startswith("|"))
    assert "Bias" in table_header and "Within ±5" in table_header
    chart = lines.index("### Relative loss reduction vs best baseline (chart)")
    assert lines[chart + 2].startswith("![relative_loss_reduction_pct")


def test_comparison_csv_and_json_use_canonical_columns() -> None:
    csv_path, json_path = _REPORTS / "model_comparison.csv", _REPORTS / "model_comparison.json"
    if not csv_path.is_file() or not json_path.is_file():
        pytest.skip("Comparison outputs not generated")
    columns = set(pd.read_csv(csv_path, nrows=1).columns)
    assert {"delta_vs_baseline", "relative_loss_reduction_pct", "best_baseline_score"} <= columns
    assert not legacy_names_in(",".join(columns))
    ranking = json.loads(json_path.read_text(encoding="utf-8"))["ranking"]
    for row in ranking:
        assert not legacy_names_in(",".join(row))
        assert row["delta_vs_baseline"] == pytest.approx(
            row["primary_score"] - row["best_baseline_score"]
        )
