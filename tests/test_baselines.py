"""Tests for naive prediction baselines and the shared experiment runner (Issue #19)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arcora.models.artifacts import list_model_artifacts, load_model_artifact
from arcora.models.base import load_results
from arcora.models.baselines import HistoricalReferenceModel, train_baseline_models
from arcora.models.data import ModelingData, prepare_modeling_data
from arcora.models.experiment import check_evaluation_splits, evaluate_models, run_model_family
from arcora.models.modeling_strategy import MODEL_TARGETS, REGRESSION_TARGETS
from arcora.models.splits import HOLDOUT_SPLIT, TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT
from tests.conftest import make_synthetic_feature_dataset


def test_regression_baselines_learn_from_training_only(
    synthetic_modeling_data: ModelingData,
) -> None:
    data = synthetic_modeling_data
    models = train_baseline_models(data, REGRESSION_TARGETS)
    train_y = data.get(TRAIN_SPLIT).y
    val_x = data.get(VALIDATION_SPLIT).X
    for target in REGRESSION_TARGETS:
        mean_model = next(m for m in models if m.model_id == f"mean_baseline__{target}")
        median_model = next(m for m in models if m.model_id == f"median_baseline__{target}")
        assert np.allclose(mean_model.predict(val_x), train_y[target].mean())
        assert np.allclose(median_model.predict(val_x), train_y[target].median())
        assert mean_model.training_metadata["n_train"] == data.get(TRAIN_SPLIT).n_games


def test_classification_baselines_produce_probabilities(
    synthetic_modeling_data: ModelingData,
) -> None:
    data = synthetic_modeling_data
    models = train_baseline_models(data, ("home_win",))
    names = {m.model_name for m in models}
    assert names == {"most_frequent_baseline", "prior_baseline", "historical_reference"}
    val_x = data.get(VALIDATION_SPLIT).X
    train_rate = data.get(TRAIN_SPLIT).target("home_win").mean()
    prior = next(m for m in models if m.model_name == "prior_baseline")
    assert np.allclose(prior.predict(val_x), train_rate)
    most_frequent = next(m for m in models if m.model_name == "most_frequent_baseline")
    probs = most_frequent.predict(val_x)
    assert set(np.unique(probs)) <= {0.0, 1.0}
    for model in models:
        p = model.predict(val_x)
        assert p.shape == (len(val_x),)
        assert ((p >= 0) & (p <= 1)).all()
        assert set(np.unique(model.predict_labels(val_x))) <= {0, 1}


def test_historical_reference_is_scoreline_consistent(
    synthetic_modeling_data: ModelingData,
) -> None:
    data = synthetic_modeling_data
    train, val = data.get(TRAIN_SPLIT), data.get(VALIDATION_SPLIT)
    preds = {
        t: HistoricalReferenceModel(t).fit(train.X, train.y).predict(val.X)
        for t in REGRESSION_TARGETS
    }
    assert np.allclose(preds["total_pts"], preds["home_pts"] + preds["away_pts"])
    assert np.allclose(preds["point_margin"], preds["home_pts"] - preds["away_pts"])
    # Missing history falls back to the training league mean.
    missing = val.X["home_hist_pts_avg"].isna().to_numpy()
    assert missing.any()
    model = HistoricalReferenceModel("home_pts").fit(train.X, train.y)
    assert np.allclose(preds["home_pts"][missing], model.league_home_pts_)
    assert model.league_home_pts_ == pytest.approx(train.target("home_pts").mean())


def test_historical_reference_requires_fit(synthetic_modeling_data: ModelingData) -> None:
    with pytest.raises(RuntimeError):
        HistoricalReferenceModel("home_pts").predict(synthetic_modeling_data.get(TRAIN_SPLIT).X)
    with pytest.raises(ValueError):
        HistoricalReferenceModel("not_a_target")


def test_baselines_unaffected_by_validation_targets() -> None:
    """Changing validation and test outcomes must not change any learned baseline statistic."""
    original = make_synthetic_feature_dataset()
    tampered = original.copy()
    later = tampered["season"] >= 2019
    tampered.loc[later, "home_pts"] += 50
    tampered["total_pts"] = tampered["home_pts"] + tampered["away_pts"]
    tampered["point_margin"] = tampered["home_pts"] - tampered["away_pts"]
    tampered["home_win"] = tampered["home_pts"] > tampered["away_pts"]

    before = train_baseline_models(prepare_modeling_data(original))
    after = train_baseline_models(prepare_modeling_data(tampered))
    assert [m.training_metadata for m in before] == [m.training_metadata for m in after]


def test_locked_splits_are_refused() -> None:
    for split in (TEST_SPLIT, HOLDOUT_SPLIT):
        with pytest.raises(PermissionError):
            check_evaluation_splits([split])
    check_evaluation_splits([TEST_SPLIT], allow_locked=True)
    with pytest.raises(ValueError):
        check_evaluation_splits(["random"])


def test_evaluation_produces_consistent_records(synthetic_modeling_data: ModelingData) -> None:
    data = synthetic_modeling_data
    models = train_baseline_models(data)
    results, predictions = evaluate_models(models, data)
    assert len(results) == 2 * len(models)
    assert {r.split for r in results} == {TRAIN_SPLIT, VALIDATION_SPLIT}
    for r in results:
        assert r.reproducibility["dataset_fingerprint"] == data.dataset_fingerprint
        assert r.reproducibility["n_features"] == 242
        assert r.n_games == data.get(r.split).n_games
    assert len(predictions) == sum(data.get(r.split).n_games for r in results)


def test_run_model_family_writes_reproducible_outputs(
    synthetic_modeling_data: ModelingData, tmp_path: Path
) -> None:
    out1 = run_model_family(
        "baseline", synthetic_modeling_data, tmp_path / "reports", tmp_path / "models"
    )
    out2 = run_model_family(
        "baseline", synthetic_modeling_data, tmp_path / "reports2", tmp_path / "models2"
    )
    m1 = {(r.model_id, r.split): r.metrics for r in out1.results}
    m2 = {(r.model_id, r.split): r.metrics for r in out2.results}
    assert m1 == m2

    loaded = load_results(out1.written_files["results_json"])
    assert {(r.model_id, r.split) for r in loaded} == set(m1)
    assert out1.written_files["report"].read_text().startswith("# Arcora Week 3")
    assert (tmp_path / "models" / "feature_list.json").is_file()

    artifacts = list_model_artifacts(tmp_path / "models", "baseline")
    assert len(artifacts) == 3 * len(MODEL_TARGETS)
    val = synthetic_modeling_data.get(VALIDATION_SPLIT)
    for path in artifacts:
        model = load_model_artifact(path)
        original = next(m for m in out1.models if m.model_id == model.model_id)
        assert np.allclose(model.predict(val.X), original.predict(val.X))
