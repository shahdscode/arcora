"""Tests for linear, regularized, and logistic regression models (Issue #20)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcora.models.artifacts import load_model_artifact, save_model_artifact
from arcora.models.base import TrainedModel
from arcora.models.data import ModelingData
from arcora.models.experiment import run_model_family
from arcora.models.linear_models import (
    build_preprocessing_pipeline,
    linear_report_sections,
    standardized_coefficients,
    train_linear_models,
)
from arcora.models.modeling_strategy import MODEL_FEATURE_COLUMNS, REGRESSION_TARGETS
from arcora.models.splits import TRAIN_SPLIT, VALIDATION_SPLIT


@pytest.fixture(scope="module")
def linear_models(synthetic_modeling_data: ModelingData) -> list[TrainedModel]:
    return train_linear_models(synthetic_modeling_data)


def test_all_linear_models_train(linear_models: list[TrainedModel]) -> None:
    ids = {m.model_id for m in linear_models}
    for target in REGRESSION_TARGETS:
        for name in ("linear_regression", "ridge_regression", "lasso_regression"):
            assert f"{name}__{target}" in ids
    assert "logistic_regression__home_win" in ids
    assert len(linear_models) == 13
    for m in linear_models:
        assert m.feature_names == MODEL_FEATURE_COLUMNS
        assert m.model_family == "linear"


def test_preprocessing_is_fitted_on_training_data_only(
    linear_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    train_x = synthetic_modeling_data.get(TRAIN_SPLIT).X
    model = next(m for m in linear_models if m.model_id == "ridge_regression__point_margin")
    imputer = model.estimator.named_steps["imputer"]
    scaler = model.estimator.named_steps["scaler"]
    assert np.allclose(imputer.statistics_, np.nanmedian(train_x.to_numpy(), axis=0))
    imputed = train_x.fillna(dict(zip(train_x.columns, imputer.statistics_, strict=True)))
    assert np.allclose(scaler.mean_, imputed.mean().to_numpy())
    assert imputer.n_features_in_ == len(MODEL_FEATURE_COLUMNS)


def test_preprocessing_keeps_feature_width_with_empty_columns(
    synthetic_modeling_data: ModelingData,
) -> None:
    x = synthetic_modeling_data.get(TRAIN_SPLIT).X.copy()
    x["home_hist_pts_avg"] = np.nan
    transformed = build_preprocessing_pipeline().fit_transform(x)
    assert transformed.shape == x.shape
    assert not np.isnan(transformed).any()


def test_validation_selection_is_recorded(linear_models: list[TrainedModel]) -> None:
    for m in linear_models:
        if m.model_name == "linear_regression":
            continue
        sel = m.training_metadata["validation_selection"]
        assert sel["selection_split"] == VALIDATION_SPLIT
        scores = [t[f"validation_{sel['selection_metric']}"] for t in sel["grid"]]
        assert sel["selected_validation_score"] == pytest.approx(min(scores))


def test_logistic_regression_predicts_valid_probabilities(
    linear_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    logit = next(m for m in linear_models if m.model_name == "logistic_regression")
    val = synthetic_modeling_data.get(VALIDATION_SPLIT)
    p = logit.predict(val.X)
    assert p.shape == (val.n_games,)
    assert ((p > 0) & (p < 1)).all()
    assert "class_imbalance_investigation" in logit.training_metadata
    # The synthetic margin signal is learnable: predictions must beat a coin flip.
    acc = (logit.predict_labels(val.X) == val.target("home_win").to_numpy()).mean()
    assert acc > 0.6


def test_predictions_handle_missing_values_and_column_order(
    linear_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    val_x = synthetic_modeling_data.get(VALIDATION_SPLIT).X
    assert val_x.isna().any().any()
    model = next(m for m in linear_models if m.model_id == "lasso_regression__total_pts")
    preds = model.predict(val_x)
    assert np.isfinite(preds).all()
    shuffled = val_x[list(reversed(val_x.columns))]
    assert np.allclose(model.predict(shuffled), preds)
    with pytest.raises(ValueError, match="missing"):
        model.predict(val_x.drop(columns=["home_hist_pts_avg"]))


def test_saved_linear_models_reload_for_inference(
    linear_models: list[TrainedModel], synthetic_modeling_data: ModelingData, tmp_path: Path
) -> None:
    val_x = synthetic_modeling_data.get(VALIDATION_SPLIT).X
    for model in linear_models:
        path, meta = save_model_artifact(model, tmp_path)
        assert meta.is_file()
        loaded = load_model_artifact(path)
        assert loaded.feature_names == model.feature_names
        assert np.allclose(loaded.predict(val_x), model.predict(val_x))


def test_coefficients_and_report_sections(
    linear_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    ridge = next(m for m in linear_models if m.model_id == "ridge_regression__point_margin")
    coefs = standardized_coefficients(ridge)
    assert len(coefs) == 242
    assert coefs.loc[0, "feature"] == "home_hist_net_rtg"  # synthetic signal feature
    text = "\n".join(linear_report_sections(linear_models, synthetic_modeling_data))
    assert "Class imbalance investigation" in text
    assert "Regularization selection" in text


def test_linear_family_report_compares_with_baselines(
    synthetic_modeling_data: ModelingData, tmp_path: Path
) -> None:
    reports = tmp_path / "reports"
    run_model_family("baseline", synthetic_modeling_data, reports)
    out = run_model_family("linear", synthetic_modeling_data, reports, tmp_path / "models")
    text = out.written_files["report"].read_text()
    assert "Δ MAE vs best baseline" in text
    assert "Δ log loss vs best baseline" in text
