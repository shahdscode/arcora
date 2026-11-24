"""Tests for model targets, input-feature selection, and metric definitions (Issue #18)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arcora.evaluation.metrics import classification_metrics, regression_metrics
from arcora.features.strategy import (
    FEATURE_COLUMN_NAMES,
    FORBIDDEN_POSTGAME_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)
from arcora.models.modeling_strategy import (
    APPROVED_MISSING_INDICATORS,
    CLASSIFICATION_METRICS,
    EXCLUDED_INPUT_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGETS,
    REGRESSION_METRICS,
    assert_feature_allowlist,
    derive_targets,
    feature_list_fingerprint,
    render_modeling_strategy_markdown,
    select_model_features,
    task_for_target,
    validate_feature_schema,
)
from arcora.models.splits import build_split_manifest, split_dataset


def _synthetic_feature_dataset(n_per_season: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    game_id = 1
    for season in range(2003, 2023):
        for i in range(n_per_season):
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": pd.Timestamp(f"{season}-11-01") + pd.Timedelta(days=i),
                    "season": season,
                    "season_label": f"{season}-{(season + 1) % 100:02d}",
                    "game_type": "Regular Season",
                    "is_partial_season": season == 2022,
                    "home_team_id": 10,
                    "away_team_id": 20,
                }
            )
            game_id += 1
    df = pd.DataFrame(rows)
    feats = pd.DataFrame(
        rng.normal(size=(len(df), len(FEATURE_COLUMN_NAMES))), columns=list(FEATURE_COLUMN_NAMES)
    )
    for col in APPROVED_MISSING_INDICATORS:
        feats[col] = rng.integers(0, 2, len(df)).astype(bool)
    df = pd.concat([df, feats], axis=1)
    df["home_pts"] = rng.integers(90, 130, len(df))
    df["away_pts"] = rng.integers(90, 130, len(df))
    df["point_margin"] = df["home_pts"] - df["away_pts"]
    df["total_pts"] = df["home_pts"] + df["away_pts"]
    df["home_win"] = df["home_pts"] > df["away_pts"]
    return df


def test_model_targets_and_tasks() -> None:
    assert MODEL_TARGETS == ("home_pts", "away_pts", "total_pts", "point_margin", "home_win")
    assert set(MODEL_TARGETS) == set(TARGET_COLUMNS)
    for t in ("home_pts", "away_pts", "total_pts", "point_margin"):
        assert task_for_target(t) == "regression"
    assert task_for_target("home_win") == "classification"
    with pytest.raises(KeyError):
        task_for_target("game_id")


def test_derive_targets_from_final_scores() -> None:
    df = pd.DataFrame({"home_pts": [110, 95, 100], "away_pts": [100, 101, 99]})
    targets = derive_targets(df)
    assert list(targets.columns) == list(MODEL_TARGETS)
    assert targets["total_pts"].tolist() == [210, 196, 199]
    assert targets["point_margin"].tolist() == [10, -6, 1]
    assert targets["home_win"].tolist() == [1, 0, 1]


def test_derive_targets_detects_inconsistent_stored_targets() -> None:
    df = pd.DataFrame({"home_pts": [110], "away_pts": [100], "total_pts": [200]})
    with pytest.raises(ValueError, match="total_pts"):
        derive_targets(df)


def test_feature_allowlist_excludes_metadata_targets_and_postgame() -> None:
    assert len(MODEL_FEATURE_COLUMNS) == 242
    assert not set(MODEL_FEATURE_COLUMNS) & set(METADATA_COLUMNS)
    assert not set(MODEL_FEATURE_COLUMNS) & set(TARGET_COLUMNS)
    assert not set(MODEL_FEATURE_COLUMNS) & set(FORBIDDEN_POSTGAME_COLUMNS)
    assert not set(MODEL_FEATURE_COLUMNS) & EXCLUDED_INPUT_COLUMNS
    assert {"game_id", "home_team_id", "away_team_id"} <= EXCLUDED_INPUT_COLUMNS
    assert not any(c.endswith("_id") for c in MODEL_FEATURE_COLUMNS)


def test_approved_missing_indicators_are_kept() -> None:
    assert len(APPROVED_MISSING_INDICATORS) == 8
    assert set(APPROVED_MISSING_INDICATORS) <= set(MODEL_FEATURE_COLUMNS)
    assert "home_hist_insufficient_history" in APPROVED_MISSING_INDICATORS
    assert "away_is_season_opener" in APPROVED_MISSING_INDICATORS


def test_select_model_features_returns_ordered_float_matrix() -> None:
    df = _synthetic_feature_dataset()
    x = select_model_features(df)
    assert tuple(x.columns) == MODEL_FEATURE_COLUMNS
    assert (x.dtypes == "float64").all()
    assert set(x[APPROVED_MISSING_INDICATORS[0]].unique()) <= {0.0, 1.0}
    for col in (*METADATA_COLUMNS, *TARGET_COLUMNS):
        assert col not in x.columns


def test_select_model_features_preserves_missing_values() -> None:
    df = _synthetic_feature_dataset()
    df.loc[0, "home_hist_pts_avg"] = np.nan
    x = select_model_features(df)
    assert bool(x["home_hist_pts_avg"].isna().iloc[0])


def test_assert_feature_allowlist_rejects_leakage_and_reordering() -> None:
    with pytest.raises(ValueError, match="Excluded"):
        assert_feature_allowlist((*MODEL_FEATURE_COLUMNS, "home_pts"))
    with pytest.raises(ValueError, match="Excluded"):
        assert_feature_allowlist((*MODEL_FEATURE_COLUMNS, "game_id"))
    with pytest.raises(ValueError, match="Excluded"):
        assert_feature_allowlist((*MODEL_FEATURE_COLUMNS, "home_fg_pct"))
    with pytest.raises(ValueError, match="Unapproved"):
        assert_feature_allowlist((*MODEL_FEATURE_COLUMNS, "mystery_feature"))
    with pytest.raises(ValueError, match="canonical order"):
        assert_feature_allowlist(tuple(reversed(MODEL_FEATURE_COLUMNS)))
    assert assert_feature_allowlist(MODEL_FEATURE_COLUMNS) == MODEL_FEATURE_COLUMNS


def test_validate_feature_schema() -> None:
    df = _synthetic_feature_dataset()
    validate_feature_schema(df)
    with pytest.raises(ValueError, match="missing feature columns"):
        validate_feature_schema(df.drop(columns=["home_hist_pts_avg"]))
    with pytest.raises(ValueError, match="forbidden post-game"):
        validate_feature_schema(df.assign(home_fg_pct=0.5))
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_feature_schema(df.assign(mystery_column=1.0))


def test_feature_list_fingerprint_is_order_sensitive() -> None:
    assert feature_list_fingerprint(["a", "b"]) != feature_list_fingerprint(["b", "a"])
    assert feature_list_fingerprint(MODEL_FEATURE_COLUMNS) == feature_list_fingerprint(
        list(MODEL_FEATURE_COLUMNS)
    )


def test_metric_definitions_cover_required_metrics() -> None:
    assert [m.name for m in REGRESSION_METRICS] == ["mae", "rmse", "r2"]
    assert [m.name for m in CLASSIFICATION_METRICS] == [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "roc_auc",
        "log_loss",
        "brier",
    ]
    assert set(regression_metrics([1, 2], [1, 2])) == {"mae", "rmse", "r2"}
    assert set(classification_metrics([0, 1], [0.2, 0.8])) == {
        m.name for m in CLASSIFICATION_METRICS
    }


def test_render_modeling_strategy_documentation(tmp_path: Path) -> None:
    df = _synthetic_feature_dataset()
    manifest = build_split_manifest(split_dataset(df), "abc", "synthetic.parquet")
    text = render_modeling_strategy_markdown(manifest)
    for heading in (
        "## 2. Chronological Splits",
        "## 3. Prediction Targets",
        "## 4. Model Input Rules",
        "## 5. Evaluation Metrics",
        "## 6. Reproducibility Requirements",
    ):
        assert heading in text
    assert "SPLIT-005" in text
    assert "Brier score" in text
