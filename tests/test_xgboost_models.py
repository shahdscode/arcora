"""Tests for XGBoost regression and classification models (Issue #22)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcora.models.artifacts import load_model_artifact, save_model_artifact
from arcora.models.base import TrainedModel
from arcora.models.data import ModelingData
from arcora.models.experiment import run_model_family
from arcora.models.modeling_strategy import MODEL_FEATURE_COLUMNS, MODEL_TARGETS, RANDOM_SEED
from arcora.models.splits import TRAIN_SPLIT, VALIDATION_SPLIT
from arcora.models.xgboost_models import (
    XGB_CLASSIFIER_PARAMS,
    XGB_REGRESSOR_PARAMS,
    train_xgboost_models,
    xgboost_version,
)

FAST = {"n_estimators": 300, "learning_rate": 0.1, "early_stopping_rounds": 20, "n_jobs": 1}


@pytest.fixture(scope="module")
def xgb_models(synthetic_modeling_data: ModelingData) -> list[TrainedModel]:
    return train_xgboost_models(
        synthetic_modeling_data, regressor_params=FAST, classifier_params=FAST
    )


def test_xgboost_dependency_is_available() -> None:
    version = xgboost_version()
    assert version is not None
    assert int(version.split(".")[0]) >= 2


def test_xgboost_trains_for_every_target(xgb_models: list[TrainedModel]) -> None:
    assert [m.target for m in xgb_models] == list(MODEL_TARGETS)
    for m in xgb_models:
        assert m.feature_names == MODEL_FEATURE_COLUMNS
        assert m.params["random_state"] == RANDOM_SEED
        meta = m.training_metadata
        assert meta["fit_split"] == TRAIN_SPLIT
        assert meta["early_stopping_split"] == VALIDATION_SPLIT
        assert 0 <= meta["best_iteration"] < meta["rounds_trained"] <= FAST["n_estimators"]


def test_default_configuration_uses_early_stopping() -> None:
    for params in (XGB_REGRESSOR_PARAMS, XGB_CLASSIFIER_PARAMS):
        assert params["early_stopping_rounds"] == 100
        assert params["random_state"] == RANDOM_SEED
        assert params["tree_method"] == "hist"
    assert XGB_CLASSIFIER_PARAMS["objective"] == "binary:logistic"


def test_early_stopping_metadata_records_chronological_splits(
    synthetic_modeling_data: ModelingData,
) -> None:
    """Fitting uses the training split and early stopping the validation split only."""
    data = synthetic_modeling_data
    model = train_xgboost_models(data, ["point_margin"], FAST, FAST)[0]
    assert model.training_metadata["n_early_stopping"] == data.get(VALIDATION_SPLIT).n_games
    assert model.training_metadata["n_train"] == data.get(TRAIN_SPLIT).n_games


def test_xgboost_probabilities_and_native_missing_values(
    xgb_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    val = synthetic_modeling_data.get(VALIDATION_SPLIT)
    assert val.X.isna().any().any()
    clf = next(m for m in xgb_models if m.target == "home_win")
    p = clf.predict(val.X)
    assert p.shape == (val.n_games,)
    assert ((p > 0) & (p < 1)).all()
    for m in xgb_models:
        assert np.isfinite(m.predict(val.X)).all()


def test_xgboost_uses_training_feature_order(
    xgb_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    val_x = synthetic_modeling_data.get(VALIDATION_SPLIT).X
    model = xgb_models[0]
    assert list(model.estimator.get_booster().feature_names) == list(MODEL_FEATURE_COLUMNS)
    shuffled = val_x[list(reversed(val_x.columns))]
    assert np.allclose(model.predict(shuffled), model.predict(val_x))


def test_xgboost_is_reproducible(synthetic_modeling_data: ModelingData) -> None:
    val_x = synthetic_modeling_data.get(VALIDATION_SPLIT).X
    a = train_xgboost_models(synthetic_modeling_data, ["home_win"], FAST, FAST)[0]
    b = train_xgboost_models(synthetic_modeling_data, ["home_win"], FAST, FAST)[0]
    assert np.array_equal(a.predict(val_x), b.predict(val_x))


def test_xgboost_artifacts_reload(
    xgb_models: list[TrainedModel], synthetic_modeling_data: ModelingData, tmp_path: Path
) -> None:
    val_x = synthetic_modeling_data.get(VALIDATION_SPLIT).X
    for model in xgb_models:
        path, _ = save_model_artifact(model, tmp_path)
        loaded = load_model_artifact(path)
        assert (
            loaded.training_metadata["best_iteration"] == model.training_metadata["best_iteration"]
        )
        assert np.allclose(loaded.predict(val_x), model.predict(val_x))


def test_xgboost_family_report_compares_every_family(
    synthetic_modeling_data: ModelingData, tmp_path: Path
) -> None:
    reports = tmp_path / "reports"
    run_model_family("baseline", synthetic_modeling_data, reports)
    run_model_family("linear", synthetic_modeling_data, reports)
    out = run_model_family("xgboost", synthetic_modeling_data, reports)
    text = out.written_files["report"].read_text()
    assert "Comparison with every trained family" in text
    assert "`baseline`" in text and "`linear`" in text and "`xgboost`" in text
    assert "Early stopping" in text
    assert out.written_files["feature_importance_report"].is_file()
