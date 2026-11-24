"""Model comparison across tasks, families, splits, and metrics (Issue #23).

Comparisons are only meaningful when every model was scored on identical games with identical
target and feature definitions. :func:`validate_comparability` enforces that by checking the
dataset, split, and feature fingerprints stored in every :class:`ModelResult`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from arcora.evaluation.metrics import CLASSIFICATION_METRIC_NAMES, REGRESSION_METRIC_NAMES
from arcora.models.base import ModelResult
from arcora.models.modeling_strategy import (
    CLASSIFICATION_TASK,
    METRIC_SPECS_BY_NAME,
    MODEL_TARGETS,
    PRIMARY_METRIC,
    PROBABILITY_CLIP_EPSILON,
    RANDOM_SEED,
)
from arcora.models.splits import SPLIT_ORDER

BASELINE_FAMILY: str = "baseline"
FAMILY_ORDER: tuple[str, ...] = ("baseline", "linear", "random_forest", "xgboost")
ALL_METRICS: tuple[str, ...] = REGRESSION_METRIC_NAMES + CLASSIFICATION_METRIC_NAMES


class ComparabilityError(ValueError):
    """Raised when results scored on different data or definitions are compared."""


def validate_comparability(results: Iterable[ModelResult]) -> None:
    """Ensure results for each split share dataset, split, and feature fingerprints."""
    by_split: dict[str, dict[str, set[Any]]] = {}
    for r in results:
        rep = r.reproducibility
        entry = by_split.setdefault(
            r.split,
            {"dataset": set(), "split": set(), "features": set(), "n_games": set()},
        )
        entry["dataset"].add(rep.get("dataset_fingerprint"))
        entry["split"].add(rep.get("split_game_id_sha256"))
        entry["features"].add(rep.get("feature_fingerprint"))
        entry["n_games"].add(r.n_games)
    problems = [
        f"split '{split}' mixes {len(values)} different {key} values"
        for split, checks in by_split.items()
        for key, values in checks.items()
        if len(values) > 1
    ]
    if problems:
        raise ComparabilityError("Results are not comparable: " + "; ".join(problems))


def comparison_table(results: Sequence[ModelResult]) -> pd.DataFrame:
    """Tidy table with one row per (model, target, split) and every metric as a column."""
    validate_comparability(results)
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "model_id": r.model_id,
            "model_name": r.model_name,
            "model_family": r.model_family,
            "target": r.target,
            "task": r.task,
            "split": r.split,
            "n_games": r.n_games,
            "primary_metric": PRIMARY_METRIC[r.task],
            "primary_score": r.metrics.get(PRIMARY_METRIC[r.task]),
        }
        for metric in ALL_METRICS:
            row[metric] = r.metrics.get(metric)
        row["undefined_metrics"] = ",".join(r.undefined_metrics)
        row["dataset_fingerprint"] = r.reproducibility.get("dataset_fingerprint")
        row["split_game_id_sha256"] = r.reproducibility.get("split_game_id_sha256")
        row["feature_fingerprint"] = r.reproducibility.get("feature_fingerprint")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    split_rank = {name: i for i, name in enumerate(SPLIT_ORDER)}
    target_rank = {name: i for i, name in enumerate(MODEL_TARGETS)}
    family_rank = {name: i for i, name in enumerate(FAMILY_ORDER)}
    frame = frame.assign(
        _split=frame["split"].map(split_rank).fillna(len(split_rank)),
        _family=frame["model_family"].map(family_rank).fillna(len(family_rank)),
        _target=frame["target"].map(target_rank).fillna(len(target_rank)),
    )
    frame = frame.sort_values(["_split", "_family", "model_name", "_target"], kind="stable")
    return frame.drop(columns=["_split", "_family", "_target"]).reset_index(drop=True)


def _sort_key(metric: str) -> bool:
    """True when ascending order ranks best first."""
    return not METRIC_SPECS_BY_NAME[metric].greater_is_better


def rank_models(table: pd.DataFrame, split: str) -> pd.DataFrame:
    """Rank models within each target on the primary metric for one split.

    Adds ``rank``, the best naive baseline's score, and the absolute and relative
    improvement over it (positive = better than the best baseline).
    """
    frames = []
    for target, group in table[table["split"] == split].groupby("target", sort=False):
        metric = str(group["primary_metric"].iloc[0])
        ascending = _sort_key(metric)
        ranked = group.sort_values("primary_score", ascending=ascending, na_position="last").copy()
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        baselines = ranked[ranked["model_family"] == BASELINE_FAMILY]["primary_score"].dropna()
        best_baseline = (
            (baselines.min() if ascending else baselines.max()) if not baselines.empty else np.nan
        )
        sign = 1.0 if ascending else -1.0
        ranked["best_baseline_score"] = best_baseline
        ranked["improvement_vs_baseline"] = sign * (best_baseline - ranked["primary_score"])
        ranked["improvement_vs_baseline_pct"] = (
            100.0 * ranked["improvement_vs_baseline"] / abs(best_baseline)
            if best_baseline and not np.isnan(best_baseline)
            else np.nan
        )
        ranked["target"] = target
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True) if frames else table.iloc[0:0].copy()


def best_models(ranked: pd.DataFrame) -> pd.DataFrame:
    """Return the top-ranked model per target."""
    return ranked[ranked["rank"] == 1].reset_index(drop=True)


def best_per_family(ranked: pd.DataFrame) -> pd.DataFrame:
    """Return the best model of each family for each target."""
    return (
        ranked.sort_values("rank")
        .groupby(["target", "model_family"], as_index=False, sort=False)
        .first()
        .sort_values(["target", "rank"])
        .reset_index(drop=True)
    )


def per_game_loss(task: str, y_true: Any, y_pred: Any) -> npt.NDArray[np.float64]:
    """Per-game loss whose mean is the primary metric (absolute error or log loss)."""
    true = np.asarray(y_true, dtype="float64")
    pred = np.asarray(y_pred, dtype="float64")
    if task == CLASSIFICATION_TASK:
        p = np.clip(pred, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
        return np.asarray(-(true * np.log(p) + (1.0 - true) * np.log(1.0 - p)), dtype="float64")
    return np.asarray(np.abs(true - pred), dtype="float64")


@dataclass(frozen=True)
class PairedComparison:
    """Paired bootstrap comparison of two models' primary-metric losses."""

    target: str
    split: str
    model_a: str
    model_b: str
    metric: str
    mean_difference: float
    ci_low: float
    ci_high: float
    n_games: int
    n_bootstrap: int

    @property
    def significant(self) -> bool:
        """True when the 95% interval excludes zero."""
        return self.ci_high < 0.0 or self.ci_low > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "target": self.target,
            "split": self.split,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "metric": self.metric,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_games": self.n_games,
            "n_bootstrap": self.n_bootstrap,
            "significant": self.significant,
        }


def paired_bootstrap(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    target: str,
    split: str,
    n_bootstrap: int = 2000,
    random_state: int = RANDOM_SEED,
) -> PairedComparison:
    """Bootstrap the mean per-game loss difference ``loss(a) - loss(b)`` over games.

    Negative values mean model ``a`` is better. Both models must have predictions for the
    same games.
    """
    sub = predictions[(predictions["target"] == target) & (predictions["split"] == split)]
    a = sub[sub["model_id"] == model_a].set_index("game_id").sort_index()
    b = sub[sub["model_id"] == model_b].set_index("game_id").sort_index()
    if a.empty or b.empty:
        raise ValueError(f"Missing predictions for {model_a} or {model_b} on {target}/{split}")
    if not a.index.equals(b.index):
        raise ComparabilityError(f"{model_a} and {model_b} were scored on different games")
    task = str(a["task"].iloc[0])
    diff = per_game_loss(task, a["y_true"], a["y_pred"]) - per_game_loss(
        task, b["y_true"], b["y_pred"]
    )
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, len(diff), size=(n_bootstrap, len(diff)))
    boot = diff[idx].mean(axis=1)
    return PairedComparison(
        target=target,
        split=split,
        model_a=model_a,
        model_b=model_b,
        metric=PRIMARY_METRIC[task],
        mean_difference=float(diff.mean()),
        ci_low=float(np.quantile(boot, 0.025)),
        ci_high=float(np.quantile(boot, 0.975)),
        n_games=len(diff),
        n_bootstrap=n_bootstrap,
    )
