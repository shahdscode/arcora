"""Tests for the reproducible NBA data ingestion pipeline (Issue #5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arcora.config import ArcoraConfig
from arcora.data.ingestion import (
    classify_game_type,
    format_season_label,
    parse_minutes_value,
    run_ingestion_pipeline,
)
from arcora.models.schema import CANONICAL_ENTITIES


def test_parse_minutes_value_edge_cases() -> None:
    """Verify MIN string parser handles MM:SS, integers, decimals, nulls, and negatives."""
    assert parse_minutes_value("36:30") == (36.5, False)
    assert parse_minutes_value("12.000000:00") == (12.0, False)
    assert parse_minutes_value("23") == (23.0, False)
    assert parse_minutes_value(None) == (0.0, False)
    assert parse_minutes_value("") == (0.0, False)
    assert parse_minutes_value("-9") == (0.0, True)
    assert parse_minutes_value("invalid") == (0.0, True)


def test_season_and_game_type_helpers() -> None:
    """Verify season label formatting and game_type classification."""
    assert format_season_label(2003) == "2003-04"
    assert format_season_label(2022) == "2022-23"
    assert classify_game_type(12200001) == "Preseason"
    assert classify_game_type(22200477) == "Regular Season"
    assert classify_game_type(42100406) == "Playoffs"
    assert classify_game_type(52100101) == "Play-In"


def test_ingestion_pipeline_end_to_end_and_idempotency(tmp_path: Path) -> None:
    """Verify ingestion produces canonical Parquet files and re-running does not duplicate rows."""
    cfg = ArcoraConfig.from_env()
    proc_dir = tmp_path / "processed"
    logs_dir = tmp_path / "logs"
    reports_dir = tmp_path / "reports"

    res1 = run_ingestion_pipeline(
        raw_dir=cfg.raw_data_dir,
        processed_dir=proc_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
    )

    assert res1.entities["teams"].rows_written == 30
    assert res1.entities["games"].rows_written == 26523
    assert res1.entities["games"].duplicates_removed == 29
    assert res1.entities["player_game_stats"].rows_written == 668338
    assert res1.entities["player_game_stats"].duplicates_removed == 289
    assert res1.entities["players"].rows_written == 2687

    # Re-run against the same processed directory to verify idempotency
    res2 = run_ingestion_pipeline(
        raw_dir=cfg.raw_data_dir,
        processed_dir=proc_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
    )

    for entity_name, spec in CANONICAL_ENTITIES.items():
        parquet_path = proc_dir / spec.storage_file
        assert parquet_path.is_file()
        df = pd.read_parquet(parquet_path)
        assert list(df.columns) == spec.column_names
        assert len(df) == res1.entities[entity_name].rows_written
        assert len(df) == res2.entities[entity_name].rows_written
        assert df.duplicated(subset=list(spec.primary_key)).sum() == 0

    assert (logs_dir / "ingestion.log").is_file()
    assert (reports_dir / "ingestion_manifest.json").is_file()


def test_ingestion_raises_on_missing_raw_files(tmp_path: Path) -> None:
    """Verify pipeline logs and raises FileNotFoundError when raw files are absent."""
    with pytest.raises(FileNotFoundError):
        run_ingestion_pipeline(
            raw_dir=tmp_path / "empty_raw",
            processed_dir=tmp_path / "processed",
            logs_dir=tmp_path / "logs",
        )
