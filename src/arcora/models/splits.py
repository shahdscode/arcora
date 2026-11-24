"""Chronological season-based dataset splits and split manifest (Issue #18).

Arcora never samples games randomly. Each split is defined by a contiguous block of NBA
seasons, so every game on a given calendar date lands in exactly one split and every training
date strictly precedes every validation date (and so on). The split assignment is persisted in
a JSON manifest containing per-split game counts, date ranges, and SHA-256 fingerprints of the
member ``game_id`` values so later runs can verify they evaluate on identical data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from arcora.features.leakage import LeakageCheckResult

SPLIT_MANIFEST_VERSION: str = "1.0"
SPLIT_MANIFEST_FILENAME: str = "split_manifest.json"

TRAIN_SPLIT: str = "train"
VALIDATION_SPLIT: str = "validation"
TEST_SPLIT: str = "test"
HOLDOUT_SPLIT: str = "holdout"


@dataclass(frozen=True)
class SplitDefinition:
    """Season-based definition of a single chronological dataset split."""

    name: str
    role: str
    seasons: tuple[int, ...]
    description: str
    used_for_fitting: bool
    used_for_model_selection: bool
    locked: bool

    @property
    def season_labels(self) -> tuple[str, ...]:
        """Return formatted NBA season labels (e.g. ``2003-04``) for the split."""
        return tuple(season_label(s) for s in self.seasons)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the definition to a JSON-compatible dictionary."""
        payload = asdict(self)
        payload["seasons"] = list(self.seasons)
        payload["season_labels"] = list(self.season_labels)
        return payload


def season_label(season: int) -> str:
    """Format an NBA season start year as its conventional label (``2003`` -> ``2003-04``)."""
    return f"{season}-{(season + 1) % 100:02d}"


SPLIT_DEFINITIONS: tuple[SplitDefinition, ...] = (
    SplitDefinition(
        name=TRAIN_SPLIT,
        role="Model fitting",
        seasons=tuple(range(2003, 2019)),
        description="16 complete seasons (2003-04 to 2018-19) used to fit every estimator, "
        "imputer, scaler, and baseline statistic.",
        used_for_fitting=True,
        used_for_model_selection=False,
        locked=False,
    ),
    SplitDefinition(
        name=VALIDATION_SPLIT,
        role="Model comparison, early stopping, and hyperparameter selection",
        seasons=(2019, 2020),
        description="2 complete seasons (2019-20 to 2020-21) used for every model-selection "
        "decision in Week 3.",
        used_for_fitting=False,
        used_for_model_selection=True,
        locked=False,
    ),
    SplitDefinition(
        name=TEST_SPLIT,
        role="Primary locked evaluation",
        seasons=(2021,),
        description="The complete 2021-22 season. Locked: never used for tuning or selection.",
        used_for_fitting=False,
        used_for_model_selection=False,
        locked=True,
    ),
    SplitDefinition(
        name=HOLDOUT_SPLIT,
        role="Secondary out-of-time partial-season holdout",
        seasons=(2022,),
        description="The partial 2022-23 season (through 2022-12-22). Locked and reported "
        "separately; never aggregated with complete seasons.",
        used_for_fitting=False,
        used_for_model_selection=False,
        locked=True,
    ),
)

SPLIT_ORDER: tuple[str, ...] = tuple(d.name for d in SPLIT_DEFINITIONS)
SPLIT_DEFINITIONS_BY_NAME: dict[str, SplitDefinition] = {d.name: d for d in SPLIT_DEFINITIONS}

# Verified game counts for the Week 2 feature dataset (24,848 eligible non-preseason games).
EXPECTED_SPLIT_GAME_COUNTS: dict[str, int] = {
    TRAIN_SPLIT: 20_734,
    VALIDATION_SPLIT: 2_314,
    TEST_SPLIT: 1_323,
    HOLDOUT_SPLIT: 477,
}


def assign_splits(
    dataset_df: pd.DataFrame,
    definitions: tuple[SplitDefinition, ...] = SPLIT_DEFINITIONS,
) -> pd.Series:
    """Return a Series mapping each row to its split name based solely on ``season``.

    Raises ValueError if a season belongs to more than one split or to none at all.
    """
    season_to_split: dict[int, str] = {}
    for definition in definitions:
        for season in definition.seasons:
            if season in season_to_split:
                raise ValueError(
                    f"Season {season} assigned to both '{season_to_split[season]}' "
                    f"and '{definition.name}'"
                )
            season_to_split[season] = definition.name

    seasons = dataset_df["season"].astype("int64")
    assigned = seasons.map(season_to_split)
    unassigned = sorted(seasons[assigned.isna()].unique().tolist())
    if unassigned:
        raise ValueError(f"Seasons not covered by any split definition: {unassigned}")
    return assigned.astype("string").rename("split")


def split_dataset(
    dataset_df: pd.DataFrame,
    definitions: tuple[SplitDefinition, ...] = SPLIT_DEFINITIONS,
) -> dict[str, pd.DataFrame]:
    """Split a game-level dataset into chronologically ordered season blocks.

    Each returned frame is sorted by ``(game_date, game_id)`` with a fresh RangeIndex so row
    order is deterministic across runs.
    """
    assigned = assign_splits(dataset_df, definitions)
    splits: dict[str, pd.DataFrame] = {}
    for definition in definitions:
        part = dataset_df.loc[assigned == definition.name]
        splits[definition.name] = part.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    return splits


def game_id_fingerprint(game_ids: Iterable[Any]) -> str:
    """Return a SHA-256 fingerprint of a set of game identifiers (order-insensitive)."""
    normalized = sorted(int(g) for g in game_ids)
    payload = ",".join(str(g) for g in normalized).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check(check_id: str, name: str, violations: int, details: str) -> LeakageCheckResult:
    return LeakageCheckResult(
        check_id=check_id,
        check_name=name,
        passed=violations == 0,
        violations_detected=violations,
        details=details,
    )


def validate_splits(
    splits: Mapping[str, pd.DataFrame],
    definitions: tuple[SplitDefinition, ...] = SPLIT_DEFINITIONS,
) -> list[LeakageCheckResult]:
    """Run chronological integrity checks (SPLIT-001 to SPLIT-007) over a split mapping."""
    results: list[LeakageCheckResult] = []
    ordered = [d for d in definitions if d.name in splits]

    # SPLIT-001: every split is present and non-empty.
    empty = [d.name for d in definitions if d.name not in splits or splits[d.name].empty]
    results.append(
        _check(
            "SPLIT-001",
            "All Splits Present and Non-Empty",
            len(empty),
            f"Empty or missing splits: {empty}" if empty else "All splits contain games.",
        )
    )

    # SPLIT-002: no duplicated game_id inside a split.
    dup_within = sum(int(splits[d.name]["game_id"].duplicated().sum()) for d in ordered)
    results.append(
        _check(
            "SPLIT-002",
            "No Duplicate Games Within a Split",
            dup_within,
            f"{dup_within} duplicated game_id rows found within splits.",
        )
    )

    # SPLIT-003: no game_id appears in more than one split.
    membership = pd.concat(
        [splits[d.name][["game_id"]].assign(split=d.name) for d in ordered], ignore_index=True
    )
    cross = membership.groupby("game_id")["split"].nunique()
    cross_count = int((cross > 1).sum())
    results.append(
        _check(
            "SPLIT-003",
            "No Game Appears in Multiple Splits",
            cross_count,
            f"{cross_count} game_ids assigned to more than one split.",
        )
    )

    # SPLIT-004: strict chronological ordering between consecutive splits.
    order_violations: list[str] = []
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        e_df, l_df = splits[earlier.name], splits[later.name]
        if e_df.empty or l_df.empty:
            continue
        e_max, l_min = e_df["game_date"].max(), l_df["game_date"].min()
        if not e_max < l_min:
            order_violations.append(f"{earlier.name} max {e_max} >= {later.name} min {l_min}")
    results.append(
        _check(
            "SPLIT-004",
            "Strict Chronological Ordering Across Splits",
            len(order_violations),
            "; ".join(order_violations)
            if order_violations
            else "Every split's latest game_date precedes the next split's earliest game_date.",
        )
    )

    # SPLIT-005: games played on the same calendar date never cross split boundaries.
    dates = pd.concat(
        [splits[d.name][["game_date"]].assign(split=d.name) for d in ordered], ignore_index=True
    )
    dates_per_split = dates.groupby("game_date")["split"].nunique()
    shared_dates = int((dates_per_split > 1).sum())
    results.append(
        _check(
            "SPLIT-005",
            "Same-Day Games Share a Single Split",
            shared_dates,
            f"{shared_dates} calendar dates have games in more than one split.",
        )
    )

    # SPLIT-006: every row's season matches its split definition.
    season_mismatch = 0
    for d in ordered:
        season_mismatch += int((~splits[d.name]["season"].isin(d.seasons)).sum())
    results.append(
        _check(
            "SPLIT-006",
            "Split Membership Matches Season Definitions",
            season_mismatch,
            f"{season_mismatch} rows fall outside their split's season definition.",
        )
    )

    # SPLIT-007: the partial 2022-23 season is isolated in the locked holdout.
    partial_leak = 0
    holdout_complete = 0
    for d in ordered:
        if "is_partial_season" not in splits[d.name].columns:
            continue
        partial = splits[d.name]["is_partial_season"].astype(bool)
        if d.name == HOLDOUT_SPLIT:
            holdout_complete += int((~partial).sum())
        else:
            partial_leak += int(partial.sum())
    holdout_violations = partial_leak + holdout_complete
    results.append(
        _check(
            "SPLIT-007",
            "Partial-Season Holdout Isolation",
            holdout_violations,
            f"{partial_leak} partial-season rows outside the holdout; "
            f"{holdout_complete} complete-season rows inside the holdout.",
        )
    )
    return results


def assert_valid_splits(
    splits: Mapping[str, pd.DataFrame],
    definitions: tuple[SplitDefinition, ...] = SPLIT_DEFINITIONS,
) -> list[LeakageCheckResult]:
    """Validate splits and raise ValueError listing every failed check."""
    results = validate_splits(splits, definitions)
    failed = [r for r in results if not r.passed]
    if failed:
        summary = "; ".join(f"{r.check_id} {r.check_name}: {r.details}" for r in failed)
        raise ValueError(f"Chronological split validation failed: {summary}")
    return results


def _split_summary(definition: SplitDefinition, part: pd.DataFrame) -> dict[str, Any]:
    game_types = (
        part["game_type"].astype("string").value_counts().sort_index().to_dict()
        if "game_type" in part.columns
        else {}
    )
    return {
        "name": definition.name,
        "role": definition.role,
        "seasons": list(definition.seasons),
        "season_labels": list(definition.season_labels),
        "n_games": len(part),
        "date_min": part["game_date"].min().date().isoformat() if len(part) else None,
        "date_max": part["game_date"].max().date().isoformat() if len(part) else None,
        "n_game_dates": int(part["game_date"].nunique()),
        "game_type_counts": {str(k): int(v) for k, v in game_types.items()},
        "used_for_fitting": definition.used_for_fitting,
        "used_for_model_selection": definition.used_for_model_selection,
        "locked": definition.locked,
        "game_id_sha256": game_id_fingerprint(part["game_id"]),
    }


def build_split_manifest(
    splits: Mapping[str, pd.DataFrame],
    dataset_fingerprint: str,
    source_dataset: str,
    definitions: tuple[SplitDefinition, ...] = SPLIT_DEFINITIONS,
) -> dict[str, Any]:
    """Build a deterministic, JSON-serializable split manifest.

    The manifest intentionally omits wall-clock timestamps so that regenerating it from the
    same dataset produces byte-identical output.
    """
    checks = validate_splits(splits, definitions)
    return {
        "manifest_version": SPLIT_MANIFEST_VERSION,
        "split_strategy": "season_block_chronological",
        "source_dataset": source_dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "total_games": int(sum(len(splits[d.name]) for d in definitions)),
        "split_order": [d.name for d in definitions],
        "splits": {d.name: _split_summary(d, splits[d.name]) for d in definitions},
        "integrity_checks": [r.to_dict() for r in checks],
        "all_checks_passed": all(r.passed for r in checks),
    }


def write_split_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    """Persist a split manifest as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def load_split_manifest(path: Path) -> dict[str, Any]:
    """Load a persisted split manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def verify_splits_against_manifest(
    splits: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any],
) -> None:
    """Raise ValueError if the given splits differ from a persisted manifest.

    Compares split names, season definitions, game counts, and game_id fingerprints so that a
    reused manifest guarantees every model is evaluated on identical games.
    """
    expected_splits: Mapping[str, Any] = manifest.get("splits", {})
    if list(expected_splits) != list(splits):
        raise ValueError(
            f"Split names differ from manifest: {list(splits)} != {list(expected_splits)}"
        )
    problems: list[str] = []
    for name, part in splits.items():
        expected = expected_splits[name]
        definition = SPLIT_DEFINITIONS_BY_NAME.get(name)
        if definition is not None and list(definition.seasons) != expected["seasons"]:
            problems.append(f"{name}: season definition changed")
        if len(part) != expected["n_games"]:
            problems.append(f"{name}: n_games {len(part)} != {expected['n_games']}")
        if game_id_fingerprint(part["game_id"]) != expected["game_id_sha256"]:
            problems.append(f"{name}: game_id fingerprint mismatch")
    if problems:
        raise ValueError("Splits do not match the persisted manifest: " + "; ".join(problems))
