"""Validate saved model artifacts and training reproducibility (Issue #24).

For every saved artifact the checks confirm that it reloads (with checksum verification), uses
the approved feature list in canonical order, produces one finite prediction per game, returns
valid probabilities for classifiers, handles missing values, matches the cached predictions from
training, and is insensitive to input column order. A separate check retrains selected families
twice with the same configuration and verifies identical predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arcora.models.artifacts import list_model_artifacts, load_model_artifact
from arcora.models.data import ModelingData
from arcora.models.experiment import get_model_families
from arcora.models.modeling_strategy import CLASSIFICATION_TASK, MODEL_FEATURE_COLUMNS
from arcora.models.splits import VALIDATION_SPLIT

PREDICTION_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class ArtifactCheck:
    """Outcome of validating one saved model artifact."""

    model_id: str
    model_family: str
    loads: bool
    feature_order_ok: bool
    prediction_shape_ok: bool
    predictions_finite: bool
    probabilities_valid: bool | None
    missing_values_handled: bool
    column_order_invariant: bool
    matches_cached_predictions: bool | None
    error: str | None = None

    @property
    def passed(self) -> bool:
        """True when every applicable check passed."""
        flags = [
            self.loads,
            self.feature_order_ok,
            self.prediction_shape_ok,
            self.predictions_finite,
            self.missing_values_handled,
            self.column_order_invariant,
        ]
        flags += [
            f for f in (self.probabilities_valid, self.matches_cached_predictions) if f is not None
        ]
        return all(flags) and self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True)
class ReproducibilityCheck:
    """Outcome of retraining a family twice with identical configuration."""

    model_family: str
    target: str
    max_abs_difference: float
    reproducible: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)


def _cached_predictions(models_dir: Path, family: str) -> pd.DataFrame | None:
    path = models_dir / family / "predictions.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def validate_artifact(
    path: Path, data: ModelingData, cached: pd.DataFrame | None = None
) -> ArtifactCheck:
    """Run every artifact check for one saved model."""
    family = path.parent.name
    try:
        model = load_model_artifact(path)
    except Exception as exc:  # noqa: BLE001 - report any load failure as a failed check
        return ArtifactCheck(
            path.stem, family, False, False, False, False, None, False, False, None, str(exc)
        )

    split = data.get(VALIDATION_SPLIT)
    preds = model.predict(split.X)
    shape_ok = preds.shape == (split.n_games,)
    finite = bool(np.isfinite(preds).all())
    probs_ok: bool | None = None
    if model.task == CLASSIFICATION_TASK:
        probs_ok = bool(((preds >= 0.0) & (preds <= 1.0)).all())

    all_missing = pd.DataFrame(
        np.nan, index=[0], columns=list(model.feature_names), dtype="float64"
    )
    missing_preds = model.predict(all_missing)
    missing_ok = bool(np.isfinite(missing_preds).all())

    reversed_x = split.X[list(reversed(split.X.columns))]
    order_invariant = bool(np.allclose(model.predict(reversed_x), preds, atol=PREDICTION_TOLERANCE))

    matches: bool | None = None
    if cached is not None:
        ref = cached[(cached["model_id"] == model.model_id) & (cached["split"] == VALIDATION_SPLIT)]
        if not ref.empty:
            ordered = ref.set_index("game_id").reindex(split.metadata["game_id"])["y_pred"]
            matches = bool(
                np.allclose(ordered.to_numpy(dtype="float64"), preds, atol=PREDICTION_TOLERANCE)
            )
    return ArtifactCheck(
        model_id=model.model_id,
        model_family=model.model_family,
        loads=True,
        feature_order_ok=model.feature_names == MODEL_FEATURE_COLUMNS,
        prediction_shape_ok=shape_ok,
        predictions_finite=finite,
        probabilities_valid=probs_ok,
        missing_values_handled=missing_ok,
        column_order_invariant=order_invariant,
        matches_cached_predictions=matches,
    )


def validate_saved_artifacts(
    models_dir: Path, data: ModelingData, families: Sequence[str] | None = None
) -> list[ArtifactCheck]:
    """Validate every saved artifact (optionally restricted to some families)."""
    checks: list[ArtifactCheck] = []
    for family in families or list(get_model_families()):
        cached = _cached_predictions(models_dir, family)
        for path in list_model_artifacts(models_dir, family):
            checks.append(validate_artifact(path, data, cached))
    return checks


def check_training_reproducibility(
    data: ModelingData,
    families: Sequence[str],
    targets: Sequence[str] = ("home_win",),
) -> list[ReproducibilityCheck]:
    """Retrain each family twice for the given targets and compare validation predictions."""
    registry = get_model_families()
    val_x = data.get(VALIDATION_SPLIT).X
    checks: list[ReproducibilityCheck] = []
    for family in families:
        first = registry[family].trainer(data, targets)
        second = registry[family].trainer(data, targets)
        for a, b in zip(first, second, strict=True):
            diff = float(np.max(np.abs(a.predict(val_x) - b.predict(val_x))))
            checks.append(
                ReproducibilityCheck(
                    model_family=family,
                    target=f"{a.model_name}__{a.target}",
                    max_abs_difference=diff,
                    reproducible=diff <= PREDICTION_TOLERANCE,
                )
            )
    return checks
