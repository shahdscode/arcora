"""Tests for Random Forest models and feature-importance reporting (Issue #21)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcora.features.strategy import FEATURE_CATEGORIES
from arcora.models.artifacts import load_model_artifact, save_model_artifact
from arcora.models.base import TrainedModel
from arcora.models.data import ModelingData
from arcora.models.experiment import run_model_family
from arcora.models.importance import (
    category_importance,
    grouped_permutation_importance,
    native_importance_frame,
)
from arcora.models.modeling_strategy import MODEL_FEATURE_COLUMNS, MODEL_TARGETS, RANDOM_SEED
from arcora.models.random_forest import (
    RF_CLASSIFIER_PARAMS,
    RF_REGRESSOR_PARAMS,
    train_random_forest_models,
)
from arcora.models.splits import TRAIN_SPLIT, VALIDATION_SPLIT

SMALL = {"n_estimators": 25, "n_jobs": 1}


@pytest.fixture(scope="module")
def rf_models(synthetic_modeling_data: ModelingData) -> list[TrainedModel]:
    return train_random_forest_models(
        synthetic_modeling_data, regressor_params=SMALL, classifier_params=SMALL
    )


def test_random_forest_trains_for_every_target(rf_models: list[TrainedModel]) -> None:
    assert [m.target for m in rf_models] == list(MODEL_TARGETS)
    for m in rf_models:
        assert m.feature_names == MODEL_FEATURE_COLUMNS
        assert m.params["random_state"] == RANDOM_SEED
        assert m.training_metadata["fit_split"] == TRAIN_SPLIT
        assert m.training_metadata["fit_seconds"] >= 0
    clf = next(m for m in rf_models if m.target == "home_win")
    assert clf.model_name == "random_forest_classifier"
    assert clf.params["max_features"] == RF_CLASSIFIER_PARAMS["max_features"]


def test_default_configuration_is_recorded_and_seeded() -> None:
    for params in (RF_REGRESSOR_PARAMS, RF_CLASSIFIER_PARAMS):
        assert params["random_state"] == RANDOM_SEED
        assert params["n_estimators"] == 300
        assert params["min_samples_leaf"] == 25


def test_random_forest_probabilities_and_missing_values(
    rf_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    val = synthetic_modeling_data.get(VALIDATION_SPLIT)
    assert val.X.isna().any().any()
    clf = next(m for m in rf_models if m.target == "home_win")
    p = clf.predict(val.X)
    assert p.shape == (val.n_games,)
    assert ((p >= 0) & (p <= 1)).all()
    for m in rf_models:
        assert np.isfinite(m.predict(val.X)).all()


def test_random_forest_is_reproducible(synthetic_modeling_data: ModelingData) -> None:
    val_x = synthetic_modeling_data.get(VALIDATION_SPLIT).X
    first = train_random_forest_models(synthetic_modeling_data, ["point_margin"], SMALL, SMALL)
    second = train_random_forest_models(synthetic_modeling_data, ["point_margin"], SMALL, SMALL)
    assert np.array_equal(first[0].predict(val_x), second[0].predict(val_x))


def test_random_forest_artifacts_reload(
    rf_models: list[TrainedModel], synthetic_modeling_data: ModelingData, tmp_path: Path
) -> None:
    val_x = synthetic_modeling_data.get(VALIDATION_SPLIT).X
    for model in rf_models:
        path, _ = save_model_artifact(model, tmp_path)
        assert np.allclose(load_model_artifact(path).predict(val_x), model.predict(val_x))


def test_native_importance_normalized_and_grouped(rf_models: list[TrainedModel]) -> None:
    margin = next(m for m in rf_models if m.target == "point_margin")
    frame = native_importance_frame(margin)
    assert len(frame) == 242
    assert frame["importance"].sum() == pytest.approx(1.0)
    assert frame.loc[0, "feature"] == "home_hist_net_rtg"  # the synthetic signal
    cats = category_importance(frame)
    assert set(cats["category"]) == set(FEATURE_CATEGORIES)
    assert cats["importance"].sum() == pytest.approx(1.0)


def test_grouped_permutation_importance_identifies_signal_group(
    rf_models: list[TrainedModel], synthetic_modeling_data: ModelingData
) -> None:
    margin = next(m for m in rf_models if m.target == "point_margin")
    perm = grouped_permutation_importance(
        margin, synthetic_modeling_data.get(VALIDATION_SPLIT), n_repeats=2
    )
    assert len(perm) == len(FEATURE_CATEGORIES)
    assert perm.iloc[0]["category"] == "Historical Team Performance"
    assert perm.iloc[0]["mean_increase"] > 0


def test_random_forest_family_run_writes_importance_report(
    synthetic_modeling_data: ModelingData, tmp_path: Path
) -> None:
    out = run_model_family("random_forest", synthetic_modeling_data, tmp_path / "reports")
    files = out.written_files
    for key in (
        "feature_importance_csv",
        "permutation_importance_csv",
        "feature_importance_figure",
        "feature_importance_report",
    ):
        assert files[key].is_file()
    text = files["feature_importance_report"].read_text()
    assert "not evidence of causation" in text
    assert "random_forest_feature_importance.md" in files["report"].read_text()
