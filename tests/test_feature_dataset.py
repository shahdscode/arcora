"""End-to-end tests for the final ML feature dataset and documentation (Issue #15)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arcora.config import ArcoraConfig
from arcora.features.leakage import split_features_and_targets
from arcora.features.pipeline import FEATURE_PARQUET_FILENAME, run_feature_pipeline
from arcora.features.strategy import (
    FEATURE_COLUMN_NAMES,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)


def test_final_feature_dataset_pipeline_and_reproducibility(tmp_path: Path) -> None:
    """Verify full feature dataset generation, schema, leakage checks, and reproducibility."""
    cfg = ArcoraConfig.from_env()
    feat_dir = tmp_path / "features"
    docs_dir = tmp_path / "docs"
    rep_dir = tmp_path / "reports"
    logs_dir = tmp_path / "logs"

    df1, summary1 = run_feature_pipeline(
        processed_dir=cfg.processed_data_dir,
        features_dir=feat_dir,
        docs_dir=docs_dir,
        reports_dir=rep_dir,
        logs_dir=logs_dir,
    )

    # 24,848 eligible non-preseason games (23,244 Regular Season + 1,591 Playoffs + 13 Play-In)
    assert summary1.total_games == 24848
    assert len(df1) == 24848
    assert summary1.duplicate_game_ids == 0
    assert summary1.total_metadata_columns == len(METADATA_COLUMNS)
    assert summary1.total_feature_columns == len(FEATURE_COLUMN_NAMES)
    assert summary1.total_target_columns == len(TARGET_COLUMNS)
    assert summary1.unexpected_missing_feature_count == 0
    assert summary1.leakage_checks_passed is True
    assert summary1.stage_distribution == {
        "Regular Season": 23244,
        "Playoffs": 1591,
        "Play-In": 13,
    }

    meta_df, x_df, y_df = split_features_and_targets(df1)
    assert list(meta_df.columns) == list(METADATA_COLUMNS)
    assert list(x_df.columns) == list(FEATURE_COLUMN_NAMES)
    assert list(y_df.columns) == list(TARGET_COLUMNS)

    # Verify generated files exist
    parquet_path = feat_dir / FEATURE_PARQUET_FILENAME
    assert parquet_path.is_file()
    assert (docs_dir / "feature_engineering_strategy.md").is_file()
    assert (docs_dir / "feature_dictionary.md").is_file()
    assert (rep_dir / "feature_quality_report.md").is_file()
    assert (rep_dir / "feature_summary.json").is_file()
    assert (logs_dir / "feature_engineering.log").is_file()

    # Re-run to verify idempotency and exact reproducibility
    df2 = pd.read_parquet(parquet_path)
    pd.testing.assert_frame_equal(df1, df2)


def test_season_2022_breakdown_and_preseason_reconciliation() -> None:
    """Verify exact 2022-23 season game_type breakdown and 1,675 preseason exclusion."""
    cfg = ArcoraConfig.from_env()
    games_df = pd.read_parquet(cfg.processed_data_dir / "games.parquet")
    features_df = pd.read_parquet(cfg.features_data_dir / FEATURE_PARQUET_FILENAME)

    # Full dataset reconciliation: 26,523 - 1,675 Preseason = 24,848 eligible games
    assert len(games_df) == 26523
    assert int(games_df["game_id"].nunique()) == 26523
    full_counts = games_df["game_type"].value_counts().to_dict()
    assert full_counts == {
        "Regular Season": 23244,
        "Preseason": 1675,
        "Playoffs": 1591,
        "Play-In": 13,
    }

    assert len(features_df) == 24848
    assert int(features_df["game_id"].nunique()) == 24848
    assert len(games_df) - len(features_df) == 1675
    assert int((features_df["game_type"] == "Preseason").sum()) == 0

    # Season 2022 (2022-23 partial season through 2022-12-22) reconciliation:
    # 542 canonical games = 477 Regular Season + 65 Preseason
    s2022_games = games_df[games_df["season"] == 2022]
    assert len(s2022_games) == 542
    assert int(s2022_games["game_id"].nunique()) == 542
    s2022_counts = s2022_games["game_type"].value_counts().to_dict()
    assert s2022_counts == {
        "Regular Season": 477,
        "Preseason": 65,
    }

    # In nba_game_features.parquet, all 65 Preseason games are excluded, leaving 477 games
    s2022_features = features_df[features_df["season"] == 2022]
    assert len(s2022_features) == 477
    assert int(s2022_features["game_id"].nunique()) == 477
    assert (s2022_features["game_type"] == "Regular Season").all()
    assert s2022_features["is_partial_season"].all()

