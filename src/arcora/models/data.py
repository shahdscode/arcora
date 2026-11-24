"""Load the Week 2 feature dataset and prepare chronological modeling splits (Issue #18)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from arcora.config import get_project_root
from arcora.features.leakage import LeakageCheckResult
from arcora.features.strategy import METADATA_COLUMNS
from arcora.models.modeling_strategy import (
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGETS,
    derive_targets,
    select_model_features,
    validate_feature_schema,
)
from arcora.models.splits import (
    SPLIT_ORDER,
    assert_valid_splits,
    build_split_manifest,
    load_split_manifest,
    split_dataset,
    verify_splits_against_manifest,
    write_split_manifest,
)


@dataclass(frozen=True)
class SplitData:
    """Metadata, model inputs, and targets for one chronological split."""

    name: str
    metadata: pd.DataFrame
    X: pd.DataFrame
    y: pd.DataFrame

    @property
    def n_games(self) -> int:
        """Number of games in the split."""
        return len(self.X)

    def target(self, name: str) -> pd.Series:
        """Return one target column."""
        if name not in self.y.columns:
            raise KeyError(f"Unknown target '{name}'. Available: {list(self.y.columns)}")
        return self.y[name]


@dataclass(frozen=True)
class ModelingData:
    """All chronological splits plus the manifest that describes them."""

    splits: dict[str, SplitData]
    feature_names: tuple[str, ...]
    dataset_fingerprint: str
    split_manifest: dict[str, Any]
    split_checks: list[LeakageCheckResult] = field(default_factory=list)

    def get(self, name: str) -> SplitData:
        """Return a split by name."""
        if name not in self.splits:
            raise KeyError(f"Unknown split '{name}'. Available: {list(self.splits)}")
        return self.splits[name]


def compute_dataset_fingerprint(dataset_df: pd.DataFrame) -> str:
    """Return a content-based SHA-256 fingerprint of a feature dataset.

    Rows are ordered by ``game_id`` and columns by name, so the fingerprint is independent of
    on-disk Parquet encoding and row order.
    """
    ordered = dataset_df.sort_values("game_id").reset_index(drop=True)
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    row_hashes = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    digest = hashlib.sha256(row_hashes.tobytes())
    digest.update(",".join(ordered.columns).encode("utf-8"))
    return digest.hexdigest()


def load_feature_dataset(path: Path) -> pd.DataFrame:
    """Load ``nba_game_features.parquet`` and validate its schema."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. Run `python scripts/build_features.py` first."
        )
    dataset_df = pd.read_parquet(path)
    validate_feature_schema(dataset_df)
    return dataset_df


def _to_split_data(name: str, part: pd.DataFrame) -> SplitData:
    return SplitData(
        name=name,
        metadata=part.loc[:, list(METADATA_COLUMNS)].copy(),
        X=select_model_features(part),
        y=derive_targets(part),
    )


def prepare_modeling_data(
    dataset_df: pd.DataFrame,
    source_dataset: str = "data/features/nba_game_features.parquet",
    manifest_path: Path | None = None,
    reuse_manifest: bool = False,
) -> ModelingData:
    """Validate, split, and package a feature dataset for model training.

    When ``reuse_manifest`` is True the splits are verified against the persisted manifest at
    ``manifest_path`` (raising on any mismatch). Otherwise, if ``manifest_path`` is given, a
    fresh manifest is written there.
    """
    validate_feature_schema(dataset_df)
    raw_splits = split_dataset(dataset_df)
    checks = assert_valid_splits(raw_splits)
    fingerprint = compute_dataset_fingerprint(dataset_df)
    manifest = build_split_manifest(raw_splits, fingerprint, source_dataset)

    if reuse_manifest:
        if manifest_path is None:
            raise ValueError("reuse_manifest=True requires manifest_path")
        persisted = load_split_manifest(manifest_path)
        verify_splits_against_manifest(raw_splits, persisted)
        manifest = persisted
    elif manifest_path is not None:
        write_split_manifest(manifest, manifest_path)

    splits = {name: _to_split_data(name, raw_splits[name]) for name in SPLIT_ORDER}
    for split in splits.values():
        if tuple(split.X.columns) != MODEL_FEATURE_COLUMNS:
            raise ValueError(f"Feature order mismatch in split '{split.name}'")
        if tuple(split.y.columns) != MODEL_TARGETS:
            raise ValueError(f"Target mismatch in split '{split.name}'")

    return ModelingData(
        splits=splits,
        feature_names=MODEL_FEATURE_COLUMNS,
        dataset_fingerprint=fingerprint,
        split_manifest=manifest,
        split_checks=checks,
    )


def load_modeling_data(
    features_path: Path,
    manifest_path: Path | None = None,
    reuse_manifest: bool = False,
) -> ModelingData:
    """Load the feature dataset from disk and prepare chronological modeling splits."""
    dataset_df = load_feature_dataset(features_path)
    try:
        source = features_path.resolve().relative_to(get_project_root()).as_posix()
    except ValueError:
        source = features_path.name
    return prepare_modeling_data(
        dataset_df,
        source_dataset=source,
        manifest_path=manifest_path,
        reuse_manifest=reuse_manifest,
    )
