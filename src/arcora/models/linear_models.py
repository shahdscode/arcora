"""Linear, regularized, and logistic regression models (Issue #20).

Every model is a scikit-learn ``Pipeline``::

    SimpleImputer(median) -> StandardScaler -> estimator

so imputation medians and scaling statistics are learned from the training split only and are
saved with the estimator. The approved missing-value indicators are ordinary inputs and are kept.

Ridge and Lasso penalties and the logistic-regression ``C`` are chosen from small fixed grids by
validation performance (``mae`` for regression, ``log_loss`` for classification). Every
candidate is fitted on the training split only; the full grid is recorded in the model's
training metadata.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arcora.evaluation.metrics import compute_metrics
from arcora.features.strategy import FEATURE_SPECIFICATIONS
from arcora.models.base import TrainedModel, validate_training_inputs
from arcora.models.data import ModelingData
from arcora.models.modeling_strategy import (
    CLASSIFICATION_TARGETS,
    CLASSIFICATION_TASK,
    MODEL_TARGETS,
    PRIMARY_METRIC,
    RANDOM_SEED,
    REGRESSION_TARGETS,
)
from arcora.models.splits import TRAIN_SPLIT, VALIDATION_SPLIT

MODEL_FAMILY: str = "linear"

FAMILY_DESCRIPTION: str = (
    "Interpretable linear models in scikit-learn pipelines (median imputation → "
    "standardization → estimator). Ordinary least squares, Ridge (L2), and Lasso (L1) regression "
    "are trained for every regression target; L2-regularized logistic regression predicts "
    "home-win probabilities. Imputers and scalers are fitted on the training split only."
)

RIDGE_ALPHAS: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0, 30000.0, 100000.0, 300000.0)
LASSO_ALPHAS: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
LOGISTIC_CS: tuple[float, ...] = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1)

_CATEGORY_BY_FEATURE = {f.name: f.category for f in FEATURE_SPECIFICATIONS}


def build_preprocessing_pipeline(scale: bool = True) -> Pipeline:
    """Return the shared preprocessing steps (training-fitted median imputation, scaling).

    ``keep_empty_features=True`` guarantees the output width always equals the input width, so
    feature ordering is stable even if a column were entirely missing in training.
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def make_pipeline(estimator: Any, scale: bool = True) -> Pipeline:
    """Append an estimator to the shared preprocessing steps."""
    preprocessing = build_preprocessing_pipeline(scale)
    return Pipeline([*preprocessing.steps, ("model", estimator)])


def _score(pipeline: Pipeline, task: str, X: pd.DataFrame, y: pd.Series) -> float | None:
    pred = pipeline.predict_proba(X)[:, 1] if task == CLASSIFICATION_TASK else pipeline.predict(X)
    return compute_metrics(task, y.to_numpy(dtype="float64"), pred)[PRIMARY_METRIC[task]]


def _fit(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        pipeline.fit(X, y)
    return time.perf_counter() - start


def select_by_validation(
    build: Callable[[float], Pipeline],
    grid: Sequence[float],
    param_name: str,
    task: str,
    data: ModelingData,
    target: str,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit one pipeline per grid value on training data; keep the best on validation.

    Lower is better for both primary metrics (MAE and log loss).
    """
    train, val = data.get(TRAIN_SPLIT), data.get(VALIDATION_SPLIT)
    y_train, y_val = train.target(target), val.target(target)
    metric = PRIMARY_METRIC[task]
    trials: list[dict[str, Any]] = []
    best: tuple[float, Pipeline, float, float] | None = None
    for value in grid:
        pipeline = build(value)
        seconds = _fit(pipeline, train.X, y_train)
        score = _score(pipeline, task, val.X, y_val)
        trials.append({param_name: value, f"validation_{metric}": score, "fit_seconds": seconds})
        if score is not None and (best is None or score < best[0]):
            best = (score, pipeline, value, seconds)
    if best is None:
        raise RuntimeError(f"No valid candidate found for {target}")
    return best[1], {
        "selection_split": VALIDATION_SPLIT,
        "selection_metric": metric,
        "param": param_name,
        "grid": trials,
        "selected_value": best[2],
        "selected_validation_score": best[0],
        "fit_seconds": best[3],
    }


def _model(
    name: str,
    target: str,
    pipeline: Pipeline,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
    metadata: dict[str, Any],
) -> TrainedModel:
    return TrainedModel(
        model_name=name,
        model_family=MODEL_FAMILY,
        target=target,
        estimator=pipeline,
        feature_names=feature_names,
        params={"preprocessing": "SimpleImputer(median)+StandardScaler", **params},
        training_metadata={"fit_split": TRAIN_SPLIT, **metadata},
    )


def _class_imbalance_investigation(
    data: ModelingData, C: float, base_metrics: dict[str, Any]
) -> dict[str, Any]:
    """Compare the selected logistic model against a ``class_weight='balanced'`` variant."""
    train, val = data.get(TRAIN_SPLIT), data.get(VALIDATION_SPLIT)
    balanced = make_pipeline(
        LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=RANDOM_SEED)
    )
    _fit(balanced, train.X, train.target("home_win"))
    balanced_metrics = compute_metrics(
        CLASSIFICATION_TASK,
        val.target("home_win").to_numpy(dtype="float64"),
        balanced.predict_proba(val.X)[:, 1],
    )
    return {
        "train_home_win_rate": float(train.target("home_win").mean()),
        "validation_home_win_rate": float(val.target("home_win").mean()),
        "unweighted_validation_metrics": base_metrics,
        "balanced_validation_metrics": balanced_metrics,
        "decision": (
            "Keep class_weight=None: the ~60/40 split is mild and balancing distorts "
            "probability calibration (higher log loss / Brier) for a small balanced-accuracy gain."
            if (balanced_metrics["log_loss"] or np.inf) >= (base_metrics["log_loss"] or np.inf)
            else "Balanced weighting improved validation log loss; consider adopting it."
        ),
    }


def train_linear_models(
    data: ModelingData,
    targets: Sequence[str] = MODEL_TARGETS,
) -> list[TrainedModel]:
    """Train OLS, Ridge, Lasso (regression) and logistic regression (classification)."""
    train, val = data.get(TRAIN_SPLIT), data.get(VALIDATION_SPLIT)
    feature_names = validate_training_inputs(train.X)
    models: list[TrainedModel] = []

    for target in targets:
        if target in REGRESSION_TARGETS:
            ols = make_pipeline(LinearRegression())
            seconds = _fit(ols, train.X, train.target(target))
            models.append(
                _model(
                    "linear_regression",
                    target,
                    ols,
                    feature_names,
                    {"estimator": "LinearRegression", "fit_intercept": True},
                    {"fit_seconds": seconds},
                )
            )
            ridge, ridge_sel = select_by_validation(
                lambda a: make_pipeline(Ridge(alpha=a, random_state=RANDOM_SEED)),
                RIDGE_ALPHAS,
                "alpha",
                "regression",
                data,
                target,
            )
            models.append(
                _model(
                    "ridge_regression",
                    target,
                    ridge,
                    feature_names,
                    {
                        "estimator": "Ridge",
                        "alpha": ridge_sel["selected_value"],
                        "alpha_grid": list(RIDGE_ALPHAS),
                        "random_state": RANDOM_SEED,
                    },
                    {"validation_selection": ridge_sel, "fit_seconds": ridge_sel["fit_seconds"]},
                )
            )
            lasso, lasso_sel = select_by_validation(
                lambda a: make_pipeline(Lasso(alpha=a, max_iter=20_000, random_state=RANDOM_SEED)),
                LASSO_ALPHAS,
                "alpha",
                "regression",
                data,
                target,
            )
            n_nonzero = int(np.count_nonzero(lasso.named_steps["model"].coef_))
            models.append(
                _model(
                    "lasso_regression",
                    target,
                    lasso,
                    feature_names,
                    {
                        "estimator": "Lasso",
                        "alpha": lasso_sel["selected_value"],
                        "alpha_grid": list(LASSO_ALPHAS),
                        "max_iter": 20_000,
                        "random_state": RANDOM_SEED,
                    },
                    {
                        "validation_selection": lasso_sel,
                        "fit_seconds": lasso_sel["fit_seconds"],
                        "n_nonzero_coefficients": n_nonzero,
                    },
                )
            )
        elif target in CLASSIFICATION_TARGETS:
            logit, logit_sel = select_by_validation(
                lambda c: make_pipeline(
                    LogisticRegression(C=c, max_iter=5000, random_state=RANDOM_SEED)
                ),
                LOGISTIC_CS,
                "C",
                CLASSIFICATION_TASK,
                data,
                target,
            )
            base_metrics = compute_metrics(
                CLASSIFICATION_TASK,
                val.target(target).to_numpy(dtype="float64"),
                logit.predict_proba(val.X)[:, 1],
            )
            models.append(
                _model(
                    "logistic_regression",
                    target,
                    logit,
                    feature_names,
                    {
                        "estimator": "LogisticRegression",
                        "penalty": "l2",
                        "C": logit_sel["selected_value"],
                        "C_grid": list(LOGISTIC_CS),
                        "class_weight": None,
                        "solver": "lbfgs",
                        "max_iter": 5000,
                        "random_state": RANDOM_SEED,
                    },
                    {
                        "validation_selection": logit_sel,
                        "fit_seconds": logit_sel["fit_seconds"],
                        "class_imbalance_investigation": _class_imbalance_investigation(
                            data, logit_sel["selected_value"], base_metrics
                        ),
                    },
                )
            )
        else:
            raise ValueError(f"Unknown target '{target}'")
    return models


def standardized_coefficients(model: TrainedModel) -> pd.DataFrame:
    """Return coefficients on the standardized scale, sorted by absolute magnitude."""
    estimator = model.estimator.named_steps["model"]
    coef = np.asarray(estimator.coef_, dtype="float64").reshape(-1)
    frame = pd.DataFrame(
        {
            "feature": list(model.feature_names),
            "category": [_CATEGORY_BY_FEATURE.get(f, "") for f in model.feature_names],
            "coefficient": coef,
        }
    )
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def linear_report_sections(models: list[TrainedModel], data: ModelingData) -> list[str]:
    """Extra Markdown sections: validation grids, class-imbalance check, top coefficients."""
    lines = [
        "## Regularization selection (validation)",
        "",
        "| Model | Target | Parameter | Selected | Validation score | Grid |",
        "| :--- | :--- | :--- | ---: | ---: | :--- |",
    ]
    for m in models:
        sel = m.training_metadata.get("validation_selection")
        if not sel:
            continue
        metric = sel["selection_metric"]
        grid = ", ".join(
            f"{t[sel['param']]}→{t[f'validation_{metric}']:.4f}"
            for t in sel["grid"]
            if t[f"validation_{metric}"] is not None
        )
        lines.append(
            f"| `{m.model_name}` | `{m.target}` | `{sel['param']}` | {sel['selected_value']} | "
            f"{sel['selected_validation_score']:.4f} ({metric}) | {grid} |"
        )

    logit = next((m for m in models if m.model_name == "logistic_regression"), None)
    if logit is not None:
        inv = logit.training_metadata["class_imbalance_investigation"]
        base, bal = inv["unweighted_validation_metrics"], inv["balanced_validation_metrics"]
        lines.extend(
            [
                "",
                "## Class imbalance investigation (`home_win`)",
                "",
                f"Training home-win rate: **{inv['train_home_win_rate']:.3f}**; validation "
                f"home-win rate: **{inv['validation_home_win_rate']:.3f}**.",
                "",
                "| Variant | Accuracy | Balanced acc. | Recall | ROC-AUC | Log loss | Brier |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, m_ in (("class_weight=None", base), ("class_weight='balanced'", bal)):
            lines.append(
                f"| {label} | {m_['accuracy']:.4f} | {m_['balanced_accuracy']:.4f} | "
                f"{m_['recall']:.4f} | {m_['roc_auc']:.4f} | {m_['log_loss']:.4f} | "
                f"{m_['brier']:.4f} |"
            )
        lines.extend(["", f"Decision: {inv['decision']}"])

    for name, target in (("ridge_regression", "point_margin"), ("logistic_regression", "home_win")):
        model = next((m for m in models if m.model_name == name and m.target == target), None)
        if model is None:
            continue
        top = standardized_coefficients(model).head(10)
        lines.extend(
            [
                "",
                f"## Top standardized coefficients — `{name}` / `{target}`",
                "",
                "Coefficients are on the standardized feature scale. Many features are highly "
                "collinear (e.g. home, away, and difference features), so individual "
                "coefficients are not causal effects.",
                "",
                "| Feature | Category | Coefficient |",
                "| :--- | :--- | ---: |",
            ]
        )
        for row in top.itertuples():
            lines.append(f"| `{row.feature}` | {row.category} | {row.coefficient:+.4f} |")
    return lines
