"""Shared trained-model wrapper, evaluation records, and reproducibility metadata.

Every model family (baselines, linear, Random Forest, XGBoost) returns :class:`TrainedModel`
objects and is scored into :class:`ModelResult` records with one common JSON format, so all
results can be compared directly.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from arcora.evaluation.metrics import MetricDict, compute_metrics, undefined_metrics
from arcora.models.data import ModelingData, SplitData
from arcora.models.modeling_strategy import (
    CLASSIFICATION_TASK,
    CLASSIFICATION_THRESHOLD,
    RANDOM_SEED,
    assert_feature_allowlist,
    feature_list_fingerprint,
    task_for_target,
)

RESULTS_SCHEMA_VERSION: str = "1.0"
_TRACKED_LIBRARIES: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "xgboost", "joblib")


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def library_versions() -> dict[str, str | None]:
    """Return installed versions of the libraries that influence model outputs."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for lib in _TRACKED_LIBRARIES:
        try:
            versions[lib] = importlib_metadata.version(lib)
        except importlib_metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def to_jsonable(value: Any) -> Any:
    """Convert numpy / pandas scalars and containers to JSON-native types."""
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


@dataclass
class TrainedModel:
    """A fitted estimator plus everything needed to reproduce and reuse it."""

    model_name: str
    model_family: str
    target: str
    estimator: Any
    feature_names: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)
    training_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> str:
        """Learning task for the model's target."""
        return task_for_target(self.target)

    @property
    def model_id(self) -> str:
        """Unique identifier ``<model_name>__<target>``."""
        return f"{self.model_name}__{self.target}"

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing {len(missing)} model features: {missing[:5]}")
        return X.loc[:, list(self.feature_names)]

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return point predictions (regression) or home-win probabilities (classification).

        Columns are reordered to the training feature order before prediction.
        """
        features = self._prepare(X)
        if self.task == CLASSIFICATION_TASK:
            classes = [int(c) for c in getattr(self.estimator, "classes_", [0, 1])]
            proba = np.asarray(self.estimator.predict_proba(features), dtype="float64")
            if 1 not in classes:
                return np.zeros(len(features), dtype="float64")
            return proba[:, classes.index(1)]
        return np.asarray(self.estimator.predict(features), dtype="float64").reshape(-1)

    def predict_labels(
        self, X: pd.DataFrame, threshold: float = CLASSIFICATION_THRESHOLD
    ) -> npt.NDArray[np.int64]:
        """Return 0/1 home-win labels using ``p >= threshold`` (classification only)."""
        if self.task != CLASSIFICATION_TASK:
            raise ValueError("predict_labels is only available for classification models")
        return (self.predict(X) >= threshold).astype("int64")


@dataclass
class ModelResult:
    """Evaluation of one trained model on one dataset split."""

    model_name: str
    model_family: str
    target: str
    task: str
    split: str
    n_games: int
    metrics: MetricDict
    undefined_metrics: list[str]
    params: dict[str, Any]
    training_metadata: dict[str, Any]
    reproducibility: dict[str, Any]
    evaluated_at: str

    @property
    def model_id(self) -> str:
        """Unique identifier ``<model_name>__<target>``."""
        return f"{self.model_name}__{self.target}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload: dict[str, Any] = to_jsonable(asdict(self))
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelResult:
        """Rebuild a result from :meth:`to_dict` output."""
        return cls(**{k: payload[k] for k in cls.__dataclass_fields__})


def reproducibility_metadata(model: TrainedModel, data: ModelingData, split: str) -> dict[str, Any]:
    """Return the identifiers needed to reproduce an evaluation exactly."""
    split_info = data.split_manifest["splits"][split]
    return {
        "random_seed": model.params.get("random_state", RANDOM_SEED),
        "dataset_fingerprint": data.dataset_fingerprint,
        "split_game_id_sha256": split_info["game_id_sha256"],
        "train_game_id_sha256": data.split_manifest["splits"]["train"]["game_id_sha256"],
        "feature_fingerprint": feature_list_fingerprint(model.feature_names),
        "n_features": len(model.feature_names),
        "library_versions": library_versions(),
    }


def evaluate_trained_model(
    model: TrainedModel,
    split: SplitData,
    data: ModelingData,
) -> tuple[ModelResult, npt.NDArray[np.float64]]:
    """Score a trained model on a split; return the result and its raw predictions."""
    predictions = model.predict(split.X)
    y_true = split.target(model.target).to_numpy(dtype="float64")
    metrics = compute_metrics(model.task, y_true, predictions)
    result = ModelResult(
        model_name=model.model_name,
        model_family=model.model_family,
        target=model.target,
        task=model.task,
        split=split.name,
        n_games=split.n_games,
        metrics=metrics,
        undefined_metrics=undefined_metrics(metrics),
        params=to_jsonable(model.params),
        training_metadata=to_jsonable(model.training_metadata),
        reproducibility=reproducibility_metadata(model, data, split.name),
        evaluated_at=utc_timestamp(),
    )
    return result, predictions


def validate_training_inputs(X: pd.DataFrame) -> tuple[str, ...]:
    """Assert a training matrix uses exactly the approved feature allowlist."""
    return assert_feature_allowlist(tuple(str(c) for c in X.columns))


def write_results(results: Iterable[ModelResult], path: Path) -> Path:
    """Write model results to JSON using the shared results schema."""
    records = [r.to_dict() for r in results]
    payload = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at": utc_timestamp(),
        "n_results": len(records),
        "results": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_results(path: Path) -> list[ModelResult]:
    """Load model results written by :func:`write_results`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULTS_SCHEMA_VERSION:
        raise ValueError(f"Unsupported results schema in {path}: {payload.get('schema_version')}")
    return [ModelResult.from_dict(r) for r in payload["results"]]


def results_to_frame(results: Iterable[ModelResult]) -> pd.DataFrame:
    """Flatten results into a tidy one-row-per-(model, target, split) DataFrame."""
    rows = []
    for r in results:
        row: dict[str, Any] = {
            "model_name": r.model_name,
            "model_family": r.model_family,
            "target": r.target,
            "task": r.task,
            "split": r.split,
            "n_games": r.n_games,
        }
        row.update(r.metrics)
        rows.append(row)
    return pd.DataFrame(rows)
