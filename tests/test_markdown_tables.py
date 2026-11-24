"""Markdown table integrity for generated documentation and reports (Week 3 review).

GitHub-flavored Markdown splits table cells on every unescaped ``|``, including inside code
spans, so a formula such as ``mean(|y - y_hat|)`` silently breaks a table row.
"""

from __future__ import annotations

import re
from pathlib import Path

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
