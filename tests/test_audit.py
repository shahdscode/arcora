"""Tests for NBA dataset schema, relationship, and quality audit (Issue #3)."""

from __future__ import annotations

from pathlib import Path

from arcora.config import ArcoraConfig
from arcora.data.audit import run_dataset_audit, write_audit_artifacts


def test_run_dataset_audit_quantifies_all_files_and_issues(tmp_path: Path) -> None:
    """Verify audit inspects all 5 tables, relationships, duplicates, and missing values."""
    cfg = ArcoraConfig.from_env()
    report = run_dataset_audit(cfg.raw_data_dir)

    assert len(report.tables) == 5
    table_map = {t.filename: t for t in report.tables}

    assert table_map["games.csv"].row_count == 26651
    assert table_map["games.csv"].primary_key_duplicates == 29
    assert table_map["games_details.csv"].row_count == 668628
    assert table_map["games_details.csv"].primary_key_duplicates == 289
    assert table_map["players.csv"].row_count == 7228
    assert table_map["players.csv"].primary_key_duplicates == 0
    assert table_map["ranking.csv"].row_count == 210342
    assert table_map["ranking.csv"].primary_key_duplicates == 29
    assert table_map["teams.csv"].row_count == 30
    assert table_map["teams.csv"].primary_key_duplicates == 0

    rel_map = {r.name: r for r in report.relationships}
    assert rel_map["games_home_team_to_teams"].unmatched_row_count == 0
    assert rel_map["games_visitor_team_to_teams"].unmatched_row_count == 0
    assert rel_map["details_game_to_games"].unmatched_row_count == 0
    assert rel_map["games_to_details_coverage"].unmatched_row_count == 99
    assert rel_map["details_player_to_players_csv"].unmatched_distinct_keys == 918

    assert len(report.anomalies) >= 8
    assert len(report.limitations) >= 4

    artifacts = write_audit_artifacts(
        report=report,
        docs_dir=tmp_path / "docs",
        reports_dir=tmp_path / "reports",
    )
    assert artifacts["schema_audit_markdown"].is_file()
    assert artifacts["audit_json"].is_file()

    md_content = artifacts["schema_audit_markdown"].read_text(encoding="utf-8")
    assert "DQ-001" in md_content
    assert "Foreign Key Verification Summary" in md_content
