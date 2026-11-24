"""Naive prediction baselines that every Arcora model must beat (Issue #19).

Baselines
---------
* ``mean_baseline`` / ``median_baseline`` — ``DummyRegressor`` predicting the training mean or
  median of each regression target.
* ``most_frequent_baseline`` — ``DummyClassifier`` always predicting the most common training
  outcome (home win) with a degenerate 0/1 probability.
* ``prior_baseline`` — ``DummyClassifier`` predicting the training home-win rate as the
  probability for every game.
* ``historical_reference`` — a transparent, non-ML team-strength reference built from each
  team's season-to-date averages, a training-learned home-court offset, and league-level
  training means (average home score, average away score, home-win rate) as the cold-start
  fallback.

Every statistic is learned from the training split only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from arcora.models.base import TrainedModel, validate_training_inputs
from arcora.models.data import ModelingData
from arcora.models.modeling_strategy import (
    CLASSIFICATION_TARGETS,
    MODEL_TARGETS,
    REGRESSION_TARGETS,
)
from arcora.models.splits import TRAIN_SPLIT

MODEL_FAMILY: str = "baseline"

FAMILY_DESCRIPTION: str = (
    "Naive reference models learned exclusively from the 2003-04 to 2018-19 training split. "
    "Regression baselines predict a constant (training mean or median); classification "
    "baselines predict the majority class or the training home-win rate. The historical "
    "reference combines each team's season-to-date scoring averages with a training-learned "
    "home-court offset and falls back to league training means when a team has no history."
)

# Pregame features used by the historical reference (all part of the approved allowlist).
_HOME_FOR = "home_hist_pts_avg"
_HOME_AGAINST = "home_hist_pts_allowed_avg"
_AWAY_FOR = "away_hist_pts_avg"
_AWAY_AGAINST = "away_hist_pts_allowed_avg"
HISTORICAL_REFERENCE_FEATURES: tuple[str, ...] = (
    _HOME_FOR,
    _HOME_AGAINST,
    _AWAY_FOR,
    _AWAY_AGAINST,
)


def _normal_cdf(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return np.asarray(0.5 * (1.0 + erf(values / math.sqrt(2.0))), dtype="float64")


def _raw_expected_scores(
    X: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Average a team's points-for with its opponent's points-allowed (NaN if no history)."""
    home = ((X[_HOME_FOR] + X[_AWAY_AGAINST]) / 2.0).to_numpy(dtype="float64")
    away = ((X[_AWAY_FOR] + X[_HOME_AGAINST]) / 2.0).to_numpy(dtype="float64")
    return home, away


class HistoricalReferenceModel:
    """Team-strength reference model with league-level cold-start fallbacks.

    ``fit`` receives the full training target frame so the scoreline stays internally
    consistent: ``total = home + away`` and ``margin = home - away``. Home-win probability is
    ``Phi(predicted_margin / sigma)`` where ``sigma`` is the training residual standard
    deviation of the predicted margin.
    """

    def __init__(self, target: str) -> None:
        if target not in MODEL_TARGETS:
            raise ValueError(f"Unknown target '{target}'")
        self.target = target
        self.is_fitted_ = False

    def fit(self, X: pd.DataFrame, y_frame: pd.DataFrame) -> HistoricalReferenceModel:
        raw_home, raw_away = _raw_expected_scores(X)
        home = y_frame["home_pts"].to_numpy(dtype="float64")
        away = y_frame["away_pts"].to_numpy(dtype="float64")

        self.league_home_pts_ = float(home.mean())
        self.league_away_pts_ = float(away.mean())
        self.league_home_win_rate_ = float((home > away).mean())

        home_ok, away_ok = ~np.isnan(raw_home), ~np.isnan(raw_away)
        self.home_offset_ = (
            float((home[home_ok] - raw_home[home_ok]).mean()) if home_ok.any() else 0.0
        )
        self.away_offset_ = (
            float((away[away_ok] - raw_away[away_ok]).mean()) if away_ok.any() else 0.0
        )

        pred_home, pred_away = self._expected_scores(X)
        residual = (home - away) - (pred_home - pred_away)
        self.margin_sigma_ = float(max(residual.std(ddof=1), 1e-6))
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
        return self

    def _expected_scores(
        self, X: pd.DataFrame
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        raw_home, raw_away = _raw_expected_scores(X)
        home = np.where(np.isnan(raw_home), self.league_home_pts_, raw_home + self.home_offset_)
        away = np.where(np.isnan(raw_away), self.league_away_pts_, raw_away + self.away_offset_)
        return home.astype("float64"), away.astype("float64")

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        if not self.is_fitted_:
            raise RuntimeError("HistoricalReferenceModel is not fitted")
        home, away = self._expected_scores(X)
        if self.target == "home_pts":
            return home
        if self.target == "away_pts":
            return away
        if self.target == "total_pts":
            return home + away
        if self.target == "point_margin":
            return home - away
        return (self.predict_proba(X)[:, 1] >= 0.5).astype("float64")

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        if not self.is_fitted_:
            raise RuntimeError("HistoricalReferenceModel is not fitted")
        home, away = self._expected_scores(X)
        p_home = _normal_cdf((home - away) / self.margin_sigma_)
        return np.column_stack([1.0 - p_home, p_home])

    def get_params(self) -> dict[str, Any]:
        return {"target": self.target}

    def learned_statistics(self) -> dict[str, float]:
        return {
            "league_home_pts": self.league_home_pts_,
            "league_away_pts": self.league_away_pts_,
            "league_home_win_rate": self.league_home_win_rate_,
            "home_offset": self.home_offset_,
            "away_offset": self.away_offset_,
            "margin_sigma": self.margin_sigma_,
        }


def _dummy_statistics(estimator: Any, target: str) -> dict[str, Any]:
    if target in CLASSIFICATION_TARGETS:
        return {
            "classes": [int(c) for c in estimator.classes_],
            "class_prior": [float(p) for p in estimator.class_prior_],
        }
    return {"constant": float(np.asarray(estimator.constant_).reshape(-1)[0])}


def train_baseline_models(
    data: ModelingData,
    targets: Sequence[str] = MODEL_TARGETS,
) -> list[TrainedModel]:
    """Fit every applicable baseline for each requested target on the training split."""
    train = data.get(TRAIN_SPLIT)
    feature_names = validate_training_inputs(train.X)
    models: list[TrainedModel] = []

    for target in targets:
        y = train.target(target)
        candidates: list[tuple[str, Any, dict[str, Any]]] = []
        if target in REGRESSION_TARGETS:
            candidates = [
                ("mean_baseline", DummyRegressor(strategy="mean"), {"strategy": "mean"}),
                ("median_baseline", DummyRegressor(strategy="median"), {"strategy": "median"}),
            ]
        elif target in CLASSIFICATION_TARGETS:
            candidates = [
                (
                    "most_frequent_baseline",
                    DummyClassifier(strategy="most_frequent"),
                    {"strategy": "most_frequent"},
                ),
                ("prior_baseline", DummyClassifier(strategy="prior"), {"strategy": "prior"}),
            ]
        else:
            raise ValueError(f"Unknown target '{target}'")

        for name, estimator, params in candidates:
            estimator.fit(train.X, y)
            models.append(
                TrainedModel(
                    model_name=name,
                    model_family=MODEL_FAMILY,
                    target=target,
                    estimator=estimator,
                    feature_names=feature_names,
                    params={**params, "estimator": type(estimator).__name__},
                    training_metadata={
                        "fit_split": TRAIN_SPLIT,
                        "n_train": train.n_games,
                        "learned_statistics": _dummy_statistics(estimator, target),
                    },
                )
            )

        reference = HistoricalReferenceModel(target).fit(train.X, train.y)
        models.append(
            TrainedModel(
                model_name="historical_reference",
                model_family=MODEL_FAMILY,
                target=target,
                estimator=reference,
                feature_names=feature_names,
                params={
                    "estimator": "HistoricalReferenceModel",
                    "features_used": list(HISTORICAL_REFERENCE_FEATURES),
                },
                training_metadata={
                    "fit_split": TRAIN_SPLIT,
                    "n_train": train.n_games,
                    "learned_statistics": reference.learned_statistics(),
                },
            )
        )
    return models
