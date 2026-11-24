"""XGBoost gradient-boosted tree models (Issue #22).

XGBoost handles missing values natively (each split learns a default direction for ``NaN``), so
the 242 pregame features are passed in unimputed; boolean missing-value indicators are kept as
0/1 inputs. Boosting is fitted on the training split and **early stopping monitors the
chronological validation split only** — the locked test and partial-season holdout never
enter training or selection. Because validation decides the number of boosting rounds, the
validation scores are mildly optimistic, which is why the locked test split exists.

A single baseline configuration is used (no extensive search in this milestone). Supported
XGBoost versions: ``>=2.1,<4`` (developed and verified with 3.4.1). Saved artifacts should be
loaded with the XGBoost version recorded in their metadata.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from xgboost import XGBClassifier, XGBRegressor

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

MODEL_FAMILY: str = "xgboost"
SUPPORTED_XGBOOST_VERSIONS: str = ">=2.1,<4"

FAMILY_DESCRIPTION: str = (
    "XGBoost gradient-boosted trees (`tree_method='hist'`) with native missing-value handling. "
    "Models are fitted on the training split; early stopping (100 rounds of patience) monitors "
    "the chronological validation split to choose the number of boosting rounds. One baseline "
    "configuration is used: learning rate 0.03, depth 4, row subsample 0.8, column subsample 0.6."
)

XGB_COMMON_PARAMS: dict[str, Any] = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "early_stopping_rounds": 100,
    "importance_type": "gain",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}
XGB_REGRESSOR_PARAMS: dict[str, Any] = {
    **XGB_COMMON_PARAMS,
    "objective": "reg:squarederror",
    "eval_metric": "mae",
}
XGB_CLASSIFIER_PARAMS: dict[str, Any] = {
    **XGB_COMMON_PARAMS,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}


def xgboost_version() -> str | None:
    """Return the installed XGBoost version."""
    try:
        return importlib_metadata.version("xgboost")
    except importlib_metadata.PackageNotFoundError:
        return None


def train_xgboost_models(
    data: ModelingData,
    targets: Sequence[str] = MODEL_TARGETS,
    regressor_params: dict[str, Any] | None = None,
    classifier_params: dict[str, Any] | None = None,
) -> list[TrainedModel]:
    """Fit one XGBoost model per target with validation-based early stopping."""
    train, val = data.get(TRAIN_SPLIT), data.get(VALIDATION_SPLIT)
    feature_names = validate_training_inputs(train.X)
    if tuple(val.X.columns) != feature_names:
        raise ValueError("Validation features must match the training feature order")
    reg_params = {**XGB_REGRESSOR_PARAMS, **(regressor_params or {})}
    clf_params = {**XGB_CLASSIFIER_PARAMS, **(classifier_params or {})}
    models: list[TrainedModel] = []

    for target in targets:
        if target in REGRESSION_TARGETS:
            name, params = "xgboost_regressor", reg_params
            estimator: Any = XGBRegressor(**params)
        elif target in CLASSIFICATION_TARGETS:
            name, params = "xgboost_classifier", clf_params
            estimator = XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown target '{target}'")

        start = time.perf_counter()
        estimator.fit(
            train.X,
            train.target(target),
            eval_set=[(val.X, val.target(target))],
            verbose=False,
        )
        seconds = time.perf_counter() - start
        history = estimator.evals_result().get("validation_0", {}).get(params["eval_metric"], [])
        best_iteration = int(estimator.best_iteration)
        models.append(
            TrainedModel(
                model_name=name,
                model_family=MODEL_FAMILY,
                target=target,
                estimator=estimator,
                feature_names=feature_names,
                params={"estimator": type(estimator).__name__, **params},
                training_metadata={
                    "fit_split": TRAIN_SPLIT,
                    "early_stopping_split": VALIDATION_SPLIT,
                    "n_train": train.n_games,
                    "n_early_stopping": val.n_games,
                    "fit_seconds": seconds,
                    "best_iteration": best_iteration,
                    "n_trees_used": best_iteration + 1,
                    "rounds_trained": len(history),
                    "best_validation_score": float(estimator.best_score),
                    "early_stopping_metric": params["eval_metric"],
                    "missing_value_handling": "native (learned default split direction)",
                    "hyperparameter_search": "none (baseline configuration)",
                    "xgboost_version": xgboost_version(),
                },
            )
        )
    return models


def xgboost_post_run(
    models: list[TrainedModel], data: ModelingData, results_dir: Path
) -> dict[str, Path]:
    """Write the exploratory XGBoost (gain) feature-importance report."""
    return write_importance_report(
        models,
        data.get(VALIDATION_SPLIT),
        family=MODEL_FAMILY,
        title="Arcora Week 3 — XGBoost Feature Importance",
        importance_label="Total gain",
        results_dir=results_dir,
        figures_dir=results_dir.parent / "figures" / "models",
    )


def xgboost_report_sections(models: list[TrainedModel], data: ModelingData) -> list[str]:
    """Extra Markdown section: early-stopping outcome per model."""
    lines = [
        "## Early stopping (validation split)",
        "",
        "| Model | Target | Metric | Best iteration | Trees used | Rounds trained | "
        "Best validation score | Fit seconds |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in models:
        meta = m.training_metadata
        lines.append(
            f"| `{m.model_name}` | `{m.target}` | {meta['early_stopping_metric']} | "
            f"{meta['best_iteration']} | {meta['n_trees_used']} | {meta['rounds_trained']} | "
            f"{meta['best_validation_score']:.4f} | {meta['fit_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Early stopping uses only the validation split, so validation metrics for XGBoost are "
            "mildly optimistic; the locked 2021-22 test split remains untouched for an unbiased "
            "final estimate.",
            "",
            "Feature importance (exploratory): see "
            "[`xgboost_feature_importance.md`](xgboost_feature_importance.md).",
        ]
    )
    return lines
