"""Persist and reload trained model artifacts.

Each model is stored as ``<models_dir>/<family>/<model_name>__<target>.joblib`` (the complete
:class:`~arcora.models.base.TrainedModel`, including any preprocessing pipeline and the
ordered feature list) with a human-readable ``.json`` metadata sidecar.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib

from arcora.models.base import TrainedModel, library_versions, to_jsonable, utc_timestamp
from arcora.models.modeling_strategy import feature_list_fingerprint

ARTIFACT_FORMAT_VERSION: str = "1.0"
FEATURE_LIST_FILENAME: str = "feature_list.json"


def artifact_paths(models_dir: Path, model_family: str, model_id: str) -> tuple[Path, Path]:
    """Return the ``(joblib_path, metadata_path)`` for a model."""
    base = models_dir / model_family / model_id
    return base.with_suffix(".joblib"), base.with_suffix(".json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model_artifact(
    model: TrainedModel,
    models_dir: Path,
    extra_metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save a trained model and its metadata sidecar; return both paths."""
    joblib_path, meta_path = artifact_paths(models_dir, model.model_family, model.model_id)
    joblib_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, joblib_path, compress=3)

    metadata: dict[str, Any] = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "model_id": model.model_id,
        "model_name": model.model_name,
        "model_family": model.model_family,
        "target": model.target,
        "task": model.task,
        "estimator_class": type(model.estimator).__name__,
        "params": model.params,
        "training_metadata": model.training_metadata,
        "n_features": len(model.feature_names),
        "feature_fingerprint": feature_list_fingerprint(model.feature_names),
        "feature_list_file": FEATURE_LIST_FILENAME,
        "library_versions": library_versions(),
        "artifact_file": joblib_path.name,
        "artifact_sha256": _sha256_file(joblib_path),
        "artifact_bytes": joblib_path.stat().st_size,
        "saved_at": utc_timestamp(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    meta_path.write_text(json.dumps(to_jsonable(metadata), indent=2) + "\n", encoding="utf-8")
    return joblib_path, meta_path


def write_feature_list(feature_names: tuple[str, ...], models_dir: Path) -> Path:
    """Write the ordered model input feature list shared by every artifact."""
    path = models_dir / FEATURE_LIST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_features": len(feature_names),
        "feature_fingerprint": feature_list_fingerprint(feature_names),
        "feature_names": list(feature_names),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_model_artifact(path: Path, verify_checksum: bool = True) -> TrainedModel:
    """Load a saved model, optionally verifying it against its metadata sidecar."""
    joblib_path = path.with_suffix(".joblib")
    if not joblib_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {joblib_path}")
    meta_path = joblib_path.with_suffix(".json")
    metadata: dict[str, Any] | None = None
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if verify_checksum and metadata.get("artifact_sha256") != _sha256_file(joblib_path):
            raise ValueError(f"Checksum mismatch for model artifact {joblib_path}")

    model = joblib.load(joblib_path)
    if not isinstance(model, TrainedModel):
        raise TypeError(f"{joblib_path} does not contain a TrainedModel")
    if metadata is not None and metadata.get("feature_fingerprint") != feature_list_fingerprint(
        model.feature_names
    ):
        raise ValueError(f"Feature list mismatch between {joblib_path} and its metadata")
    return model


def list_model_artifacts(models_dir: Path, model_family: str | None = None) -> list[Path]:
    """List saved ``.joblib`` artifacts, optionally restricted to one family."""
    root = models_dir / model_family if model_family else models_dir
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.joblib"))
