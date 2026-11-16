"""Tests for historical NBA dataset acquisition and documentation (Issue #2)."""

from __future__ import annotations

from pathlib import Path

from arcora.config import ArcoraConfig
from arcora.data.acquisition import (
    EXPECTED_RAW_FILES,
    EXPECTED_SHA256,
    inspect_raw_dataset,
    verify_raw_checksums,
    verify_raw_files_present,
    write_acquisition_artifacts,
)


def test_raw_files_present_and_unmodified() -> None:
    """Verify all 5 raw CSV files exist in data/raw/ and match original SHA-256 hashes."""
    cfg = ArcoraConfig.from_env()
    present, missing = verify_raw_files_present(cfg.raw_data_dir)
    assert present, f"Missing raw files: {missing}"

    checksums_ok, mismatches = verify_raw_checksums(cfg.raw_data_dir)
    assert checksums_ok, f"Raw files modified or corrupted: {mismatches}"


def test_inspect_raw_dataset_coverage_and_artifacts(tmp_path: Path) -> None:
    """Verify dataset inspection metrics, season coverage, and generated documentation."""
    cfg = ArcoraConfig.from_env()
    report = inspect_raw_dataset(cfg.raw_data_dir, retrieved_at_utc="2026-11-16T15:20:00Z")

    assert len(report.files) == len(EXPECTED_RAW_FILES)
    for file_meta in report.files:
        assert file_meta.sha256 == EXPECTED_SHA256[file_meta.filename]
        assert file_meta.row_count > 0
        assert file_meta.column_count > 0

    assert report.season_count == 20
    assert report.seasons_available == list(range(2003, 2023))
    assert report.overall_min_game_date == "2003-10-05"
    assert report.overall_max_game_date == "2022-12-22"
    assert report.total_games_rows == 26651
    assert report.unique_teams_in_teams_csv == 30

    docs_out = tmp_path / "docs"
    raw_out = tmp_path / "raw"
    artifacts = write_acquisition_artifacts(report, raw_dir=raw_out, docs_dir=docs_out)

    assert artifacts["manifest"].is_file()
    assert artifacts["dataset_source"].is_file()
    assert artifacts["coverage_summary"].is_file()

    source_text = artifacts["dataset_source"].read_text(encoding="utf-8")
    assert "https://www.kaggle.com/datasets/nathanlauga/nba-games" in source_text
    assert "CC0: Public Domain" in source_text

    cov_text = artifacts["coverage_summary"].read_text(encoding="utf-8")
    assert "2003-04" in cov_text
    assert "2022-23" in cov_text
