"""Random Forest regression and classification models (Issue #21).

Random Forests capture nonlinear effects and feature interactions (for example rest × recent
form) that linear models cannot. Each model is a ``Pipeline`` of a training-fitted median
imputer followed by the forest, so missing early-season values are handled exactly as in the
linear models.

A single, deliberately conservative configuration is used (no hyperparameter search):
moderately deep trees with large leaves to limit variance on noisy game outcomes. All
hyperparameters and the random seed are recorded with every model.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from arcora.models.base import TrainedModel, validate_training_inputs
from arcora.models.data import ModelingData
from arcora.models.importance import write_importance_report
from arcora.models.modeling_strategy import (
    CLASSIFICATION_TARGETS,
    MODEL_TARGETS,
    RANDOM_SEED,
    REGRESSION_TARGETS,
)
from arcora.models.splits import TRAIN_SPLIT, VALIDATION_SPLIT

MODEL_FAMILY: str = "random_forest"

FAMILY_DESCRIPTION: str = (
    "Random Forest ensembles (training-fitted median imputation → forest) for every target. "
    "One fixed, conservative configuration is used — 300 trees, maximum depth 14, at least 25 "
    "games per leaf — to capture nonlinear effects and interactions without an expensive "
    "hyperparameter search. Seeds and hyperparameters are recorded with each model."
)

RF_REGRESSOR_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 14,
    "min_samples_leaf": 25,
    "max_features": 0.33,
    "bootstrap": True,
    "criterion": "squared_error",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

RF_CLASSIFIER_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 14,
    "min_samples_leaf": 25,
    "max_features": "sqrt",
    "bootstrap": True,
    "criterion": "gini",
    "class_weight": None,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}


def make_random_forest_pipeline(estimator: Any) -> Pipeline:
    """Training-fitted median imputation followed by a forest."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", estimator),
        ]
    )


def train_random_forest_models(
    data: ModelingData,
    targets: Sequence[str] = MODEL_TARGETS,
    regressor_params: dict[str, Any] | None = None,
    classifier_params: dict[str, Any] | None = None,
) -> list[TrainedModel]:
    """Fit one Random Forest per target on the training split."""
    train = data.get(TRAIN_SPLIT)
    feature_names = validate_training_inputs(train.X)
    reg_params = {**RF_REGRESSOR_PARAMS, **(regressor_params or {})}
    clf_params = {**RF_CLASSIFIER_PARAMS, **(classifier_params or {})}
    models: list[TrainedModel] = []

    for target in targets:
        if target in REGRESSION_TARGETS:
            name, params = "random_forest_regressor", reg_params
            estimator: Any = RandomForestRegressor(**params)
        elif target in CLASSIFICATION_TARGETS:
            name, params = "random_forest_classifier", clf_params
            estimator = RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown target '{target}'")

        pipeline = make_random_forest_pipeline(estimator)
        start = time.perf_counter()
        pipeline.fit(train.X, train.target(target))
        seconds = time.perf_counter() - start
        forest = pipeline.named_steps["model"]
        depths = [tree.get_depth() for tree in forest.estimators_]
        leaves = [tree.get_n_leaves() for tree in forest.estimators_]
        models.append(
            TrainedModel(
                model_name=name,
                model_family=MODEL_FAMILY,
                target=target,
                estimator=pipeline,
                feature_names=feature_names,
                params={
                    "estimator": type(estimator).__name__,
                    "preprocessing": "SimpleImputer(median)",
                    **params,
                },
                training_metadata={
                    "fit_split": TRAIN_SPLIT,
                    "n_train": train.n_games,
                    "fit_seconds": seconds,
                    "hyperparameter_search": "none (fixed configuration)",
                    "mean_tree_depth": float(sum(depths) / len(depths)),
                    "mean_tree_leaves": float(sum(leaves) / len(leaves)),
                },
            )
        )
    return models


def random_forest_post_run(
    models: list[TrainedModel], data: ModelingData, results_dir: Path
) -> dict[str, Path]:
    """Write the exploratory Random Forest feature-importance report."""
    return write_importance_report(
        models,
        data.get(VALIDATION_SPLIT),
        family=MODEL_FAMILY,
        title="Arcora Week 3 — Random Forest Feature Importance",
        importance_label="Mean decrease in impurity (MDI)",
        results_dir=results_dir,
        figures_dir=results_dir.parent / "figures" / "models",
    )


def random_forest_report_sections(models: list[TrainedModel], data: ModelingData) -> list[str]:
    """Extra Markdown section: fit cost and tree statistics."""
    lines = [
        "## Training statistics",
        "",
        "| Model | Target | Fit seconds | Mean tree depth | Mean leaves per tree |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    for m in models:
        meta = m.training_metadata
        lines.append(
            f"| `{m.model_name}` | `{m.target}` | {meta['fit_seconds']:.1f} | "
            f"{meta['mean_tree_depth']:.1f} | {meta['mean_tree_leaves']:.0f} |"
        )
    lines.extend(
        [
            "",
            "Feature importance (exploratory): see "
            "[`random_forest_feature_importance.md`](random_forest_feature_importance.md).",
        ]
    )
    return lines
