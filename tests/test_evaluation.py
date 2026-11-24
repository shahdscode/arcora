"""Tests for the unified evaluation, comparison, and error-analysis framework (Issue #23)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arcora.evaluation.comparison import (
    ComparabilityError,
    best_models,
    comparison_table,
    paired_bootstrap,
    per_game_loss,
    rank_models,
    validate_comparability,
)
from arcora.evaluation.error_analysis import (
    attach_metadata,
    calibration_table,
    classification_error_analysis,
    confusion_counts,
    guard_locked_split,
    regression_error_analysis,
)
from arcora.evaluation.metrics import (
    classification_metrics,
    compute_metrics,
    regression_metrics,
    undefined_metrics,
)
from arcora.evaluation.report import generate_evaluation_report
from arcora.models.base import ModelResult
from arcora.models.data import ModelingData
from arcora.models.experiment import run_model_family
from arcora.models.splits import HOLDOUT_SPLIT, TEST_SPLIT, VALIDATION_SPLIT

# --- metrics -------------------------------------------------------------------------------


def test_regression_metrics_match_definitions() -> None:
    m = regression_metrics([10, 20, 30], [12, 18, 33])
    assert m["mae"] == pytest.approx((2 + 2 + 3) / 3)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert m["r2"] == pytest.approx(1 - 17 / 200)


def test_regression_r2_undefined_for_constant_target() -> None:
    m = regression_metrics([5, 5, 5], [4, 5, 6])
    assert m["r2"] is None
    assert undefined_metrics(m) == ["r2"]
    assert regression_metrics([], [])["mae"] is None


def test_classification_metrics_match_definitions() -> None:
    y = [1, 1, 0, 0, 1]
    p = [0.9, 0.4, 0.6, 0.1, 0.8]
    m = classification_metrics(y, p)
    # labels: 1,0,1,0,1 -> TP=2, FN=1, FP=1, TN=1
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert m["brier"] == pytest.approx(np.mean((np.array(p) - np.array(y)) ** 2))
    assert m["roc_auc"] == pytest.approx(5 / 6)
    expected_ll = -np.mean([np.log(0.9), np.log(0.4), np.log(0.4), np.log(0.9), np.log(0.8)])
    assert m["log_loss"] == pytest.approx(expected_ll)


def test_classification_metrics_single_class_are_null_not_zero() -> None:
    m = classification_metrics([1, 1, 1], [0.7, 0.8, 0.4])
    assert m["roc_auc"] is None
    assert m["balanced_accuracy"] is None
    assert m["log_loss"] is not None and np.isfinite(m["log_loss"])
    assert set(undefined_metrics(m)) == {"roc_auc", "balanced_accuracy"}


def test_precision_and_recall_undefined_cases() -> None:
    no_positive_predictions = classification_metrics([0, 1], [0.1, 0.2])
    assert no_positive_predictions["precision"] is None
    no_positive_labels = classification_metrics([0, 0], [0.1, 0.9])
    assert no_positive_labels["recall"] is None


def test_degenerate_probabilities_are_clipped() -> None:
    m = classification_metrics([0, 1], [1.0, 0.0])
    assert np.isfinite(m["log_loss"])
    assert m["log_loss"] > 30


def test_metric_input_validation() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        regression_metrics([1, 2], [1])
    with pytest.raises(ValueError, match="NaN"):
        regression_metrics([1, np.nan], [1, 2])
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError, match="0/1"):
        classification_metrics([0, 2], [0.5, 0.5])
    with pytest.raises(ValueError, match="Unknown task"):
        compute_metrics("ranking", [1], [1])


# --- comparison ----------------------------------------------------------------------------


def _result(model: str, family: str, mae: float, **rep: str) -> ModelResult:
    reproducibility = {
        "dataset_fingerprint": "d",
        "split_game_id_sha256": "s",
        "feature_fingerprint": "f",
        **rep,
    }
    return ModelResult(
        model_name=model,
        model_family=family,
        target="point_margin",
        task="regression",
        split=VALIDATION_SPLIT,
        n_games=100,
        metrics={"mae": mae, "rmse": mae * 1.3, "r2": 0.1},
        undefined_metrics=[],
        params={},
        training_metadata={},
        reproducibility=reproducibility,
        evaluated_at="2026-01-01T00:00:00+00:00",
    )


def test_validate_comparability_rejects_mixed_evaluation_data() -> None:
    ok = [_result("a", "baseline", 11.0), _result("b", "linear", 10.5)]
    validate_comparability(ok)
    with pytest.raises(ComparabilityError, match="split"):
        validate_comparability([*ok, _result("c", "linear", 10.0, split_game_id_sha256="other")])
    with pytest.raises(ComparabilityError, match="features"):
        validate_comparability([*ok, _result("c", "linear", 10.0, feature_fingerprint="x")])
    mixed_n = dataclasses.replace(_result("d", "linear", 10.0), n_games=99)
    with pytest.raises(ComparabilityError, match="n_games"):
        validate_comparability([*ok, mixed_n])


def test_rank_models_and_improvement_vs_baseline() -> None:
    table = comparison_table(
        [
            _result("mean_baseline", "baseline", 12.0),
            _result("ref", "baseline", 11.0),
            _result("ridge", "linear", 10.0),
            _result("rf", "random_forest", 10.5),
        ]
    )
    ranked = rank_models(table, VALIDATION_SPLIT)
    assert ranked["model_name"].tolist() == ["ridge", "rf", "ref", "mean_baseline"]
    assert ranked["rank"].tolist() == [1, 2, 3, 4]
    ridge = ranked[ranked["model_name"] == "ridge"].iloc[0]
    assert ridge["best_baseline_score"] == 11.0
    assert ridge["improvement_vs_baseline"] == pytest.approx(1.0)
    assert ridge["improvement_vs_baseline_pct"] == pytest.approx(100 / 11)
    assert best_models(ranked)["model_name"].tolist() == ["ridge"]


def _paired_predictions() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    y = rng.normal(0, 10, 400)
    rows = []
    for model, noise in (("good__point_margin", 2.0), ("bad__point_margin", 8.0)):
        pred = y + rng.normal(0, noise, len(y))
        rows.append(
            pd.DataFrame(
                {
                    "model_id": model,
                    "target": "point_margin",
                    "task": "regression",
                    "split": VALIDATION_SPLIT,
                    "game_id": np.arange(len(y)),
                    "y_true": y,
                    "y_pred": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_paired_bootstrap_detects_meaningful_improvement() -> None:
    preds = _paired_predictions()
    result = paired_bootstrap(
        preds, "good__point_margin", "bad__point_margin", "point_margin", VALIDATION_SPLIT, 500
    )
    assert result.mean_difference < 0
    assert result.ci_high < 0
    assert result.significant
    same = paired_bootstrap(
        preds, "good__point_margin", "good__point_margin", "point_margin", VALIDATION_SPLIT, 200
    )
    assert same.mean_difference == 0 and not same.significant


def test_paired_bootstrap_requires_identical_games() -> None:
    preds = _paired_predictions()
    preds = preds[~((preds["model_id"] == "bad__point_margin") & (preds["game_id"] == 0))]
    with pytest.raises(ComparabilityError):
        paired_bootstrap(
            preds, "good__point_margin", "bad__point_margin", "point_margin", VALIDATION_SPLIT
        )


def test_per_game_loss_means_match_primary_metrics() -> None:
    y, p = np.array([1, 0, 1]), np.array([0.8, 0.3, 0.6])
    assert per_game_loss("classification", y, p).mean() == pytest.approx(
        classification_metrics(y, p)["log_loss"]
    )
    assert per_game_loss("regression", [1, 2], [2, 4]).mean() == pytest.approx(1.5)


# --- error analysis ------------------------------------------------------------------------


def test_locked_splits_are_guarded() -> None:
    for split in (TEST_SPLIT, HOLDOUT_SPLIT):
        with pytest.raises(PermissionError):
            guard_locked_split(split)
    guard_locked_split(VALIDATION_SPLIT)


def test_confusion_counts_sum_to_total() -> None:
    counts = confusion_counts([1, 0, 1, 0], [0.9, 0.7, 0.2, 0.1])
    assert counts == {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 1,
        "false_negative": 1,
    }


def test_calibration_table_bins_probabilities() -> None:
    frame = pd.DataFrame({"y_true": [0, 1, 1, 1], "y_pred": [0.05, 0.15, 0.95, 0.92]})
    table = calibration_table(frame)
    assert table["n_games"].sum() == 4
    assert table.iloc[-1]["observed_rate"] == 1.0


@pytest.fixture(scope="module")
def evaluated_reports(
    synthetic_modeling_data: ModelingData, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    root = tmp_path_factory.mktemp("eval")
    for family in ("baseline", "linear"):
        run_model_family(
            family, synthetic_modeling_data, root / "reports" / "models", root / "models"
        )
    return root


def test_error_analysis_breakdowns(
    synthetic_modeling_data: ModelingData, evaluated_reports: Path
) -> None:
    preds = pd.read_parquet(evaluated_reports / "models" / "linear" / "predictions.parquet")
    frame = attach_metadata(preds, synthetic_modeling_data, {10: "AAA"})
    reg = regression_error_analysis(frame, "ridge_regression__point_margin", VALIDATION_SPLIT)
    assert reg["n_games"] == synthetic_modeling_data.get(VALIDATION_SPLIT).n_games
    assert set(reg["by_season"]["season_label"]) == {"2019-20", "2020-21"}
    assert set(reg["by_game_type"]["game_type"]) == {"Regular Season", "Playoffs"}
    assert "AAA" in set(reg["by_home_team"]["home_team"])
    assert reg["by_home_team"]["n_games"].sum() == reg["n_games"]
    assert len(reg["largest_errors"]) == 10
    assert reg["largest_errors"]["abs_error"].is_monotonic_decreasing

    clf = classification_error_analysis(frame, "logistic_regression__home_win", VALIDATION_SPLIT)
    assert sum(clf["confusion"].values()) == clf["n_games"]
    assert set(clf["by_season"]["season_label"]) == {"2019-20", "2020-21"}
    with pytest.raises(PermissionError):
        regression_error_analysis(frame, "ridge_regression__point_margin", TEST_SPLIT)


def test_evaluation_report_is_generated_reproducibly(
    synthetic_modeling_data: ModelingData, evaluated_reports: Path
) -> None:
    out_a = evaluated_reports / "out_a"
    out_b = evaluated_reports / "out_b"
    written = {}
    for out in (out_a, out_b):
        written = generate_evaluation_report(
            synthetic_modeling_data,
            evaluated_reports / "reports" / "models",
            evaluated_reports / "models",
            out,
            n_bootstrap=200,
        )
    for key in ("comparison_csv", "comparison_json", "evaluation_report"):
        assert written[key].is_file()
    assert sum(1 for k in written if k.startswith("figure:")) == 5
    csv_a = (out_a / "model_comparison.csv").read_text()
    assert csv_a == (out_b / "model_comparison.csv").read_text()

    payload = json.loads((out_a / "model_comparison.json").read_text())
    assert payload["evaluation_split"] == VALIDATION_SPLIT
    assert {b["target"] for b in payload["best_models"]} == {
        "home_pts",
        "away_pts",
        "total_pts",
        "point_margin",
        "home_win",
    }
    text = (out_a / "model_evaluation_report.md").read_text()
    assert "## Best model per target" in text
    assert "Paired bootstrap" in text
    assert "Error analysis — classification" in text
