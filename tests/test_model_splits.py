"""Tests for chronological model splits and the split manifest (Issue #18)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from arcora.config import ArcoraConfig
from arcora.features.leakage import LeakageCheckResult
from arcora.models.data import compute_dataset_fingerprint, load_modeling_data
from arcora.models.splits import (
    EXPECTED_SPLIT_GAME_COUNTS,
    HOLDOUT_SPLIT,
    SPLIT_DEFINITIONS,
    SPLIT_ORDER,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    SplitDefinition,
    assert_valid_splits,
    assign_splits,
    build_split_manifest,
    game_id_fingerprint,
    load_split_manifest,
    season_label,
    split_dataset,
    validate_splits,
    verify_splits_against_manifest,
    write_split_manifest,
)

FEATURES_PATH = ArcoraConfig.from_env().features_data_dir / "nba_game_features.parquet"
requires_features = pytest.mark.skipif(
    not FEATURES_PATH.is_file(), reason="Week 2 feature dataset not built"
)


def _synthetic_games() -> pd.DataFrame:
    """Two games per season for 2003-2022 with season-appropriate dates."""
    rows = []
    game_id = 1
    for season in range(2003, 2023):
        for offset in (0, 30):
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": pd.Timestamp(f"{season}-11-01") + pd.Timedelta(days=offset),
                    "season": season,
                    "game_type": "Regular Season",
                    "is_partial_season": season == 2022,
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def _check(results: list[LeakageCheckResult], check_id: str) -> bool:
    return next(r for r in results if r.check_id == check_id).passed


def test_split_definitions_cover_expected_seasons() -> None:
    by_name = {d.name: d for d in SPLIT_DEFINITIONS}
    assert SPLIT_ORDER == (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT, HOLDOUT_SPLIT)
    assert by_name[TRAIN_SPLIT].season_labels[0] == "2003-04"
    assert by_name[TRAIN_SPLIT].season_labels[-1] == "2018-19"
    assert by_name[VALIDATION_SPLIT].season_labels == ("2019-20", "2020-21")
    assert by_name[TEST_SPLIT].season_labels == ("2021-22",)
    assert by_name[HOLDOUT_SPLIT].season_labels == ("2022-23",)
    assert by_name[VALIDATION_SPLIT].used_for_model_selection
    assert by_name[TEST_SPLIT].locked and by_name[HOLDOUT_SPLIT].locked
    assert not by_name[TEST_SPLIT].used_for_model_selection
    assert not by_name[HOLDOUT_SPLIT].used_for_model_selection


def test_season_label_formatting() -> None:
    assert season_label(2003) == "2003-04"
    assert season_label(1999) == "1999-00"


def test_split_dataset_is_chronological_and_valid() -> None:
    splits = split_dataset(_synthetic_games())
    assert [len(splits[s]) for s in SPLIT_ORDER] == [32, 4, 2, 2]
    results = assert_valid_splits(splits)
    assert all(r.passed for r in results)


def test_assign_splits_rejects_uncovered_season() -> None:
    df = _synthetic_games()
    df.loc[0, "season"] = 1999
    with pytest.raises(ValueError, match="not covered"):
        assign_splits(df)


def test_assign_splits_rejects_overlapping_definitions() -> None:
    overlapping = (
        SplitDefinition("a", "r", (2003, 2004), "", True, False, False),
        SplitDefinition("b", "r", (2004,), "", False, True, False),
    )
    with pytest.raises(ValueError, match="assigned to both"):
        assign_splits(_synthetic_games(), overlapping)


def test_detects_game_in_multiple_splits() -> None:
    splits = split_dataset(_synthetic_games())
    leaked = splits[TRAIN_SPLIT].iloc[[0]]
    splits[VALIDATION_SPLIT] = pd.concat([splits[VALIDATION_SPLIT], leaked], ignore_index=True)
    results = validate_splits(splits)
    assert not _check(results, "SPLIT-003")
    with pytest.raises(ValueError, match="SPLIT-003"):
        assert_valid_splits(splits)


def test_detects_same_day_games_crossing_boundary() -> None:
    splits = split_dataset(_synthetic_games())
    first_val_date = splits[VALIDATION_SPLIT]["game_date"].min()
    splits[TRAIN_SPLIT].loc[len(splits[TRAIN_SPLIT]) - 1, "game_date"] = first_val_date
    results = validate_splits(splits)
    assert not _check(results, "SPLIT-005")
    assert not _check(results, "SPLIT-004")


def test_detects_chronological_order_violation() -> None:
    splits = split_dataset(_synthetic_games())
    splits[TEST_SPLIT].loc[0, "game_date"] = pd.Timestamp("2010-01-01")
    assert not _check(validate_splits(splits), "SPLIT-004")


def test_detects_partial_season_leak_outside_holdout() -> None:
    splits = split_dataset(_synthetic_games())
    splits[TEST_SPLIT].loc[0, "is_partial_season"] = True
    assert not _check(validate_splits(splits), "SPLIT-007")


def test_game_id_fingerprint_is_order_insensitive() -> None:
    assert game_id_fingerprint([3, 1, 2]) == game_id_fingerprint([1, 2, 3])
    assert game_id_fingerprint([1, 2]) != game_id_fingerprint([1, 2, 3])


def test_manifest_round_trip_and_verification(tmp_path: Path) -> None:
    df = _synthetic_games()
    splits = split_dataset(df)
    manifest = build_split_manifest(splits, compute_dataset_fingerprint(df), "synthetic")
    assert manifest["all_checks_passed"]
    path = write_split_manifest(manifest, tmp_path / "split_manifest.json")
    loaded = load_split_manifest(path)
    assert loaded == manifest
    verify_splits_against_manifest(splits, loaded)

    # Manifest is deterministic (no timestamps)
    again = build_split_manifest(split_dataset(df), compute_dataset_fingerprint(df), "synthetic")
    assert again == manifest

    # Tampering with membership is detected
    changed = split_dataset(df.iloc[1:])
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_splits_against_manifest(changed, loaded)


@requires_features
def test_real_dataset_split_counts_and_integrity(tmp_path: Path) -> None:
    data = load_modeling_data(FEATURES_PATH, manifest_path=tmp_path / "manifest.json")
    counts = {name: split.n_games for name, split in data.splits.items()}
    assert counts == EXPECTED_SPLIT_GAME_COUNTS
    assert sum(counts.values()) == 24_848
    assert all(r.passed for r in data.split_checks)

    train_max = data.get(TRAIN_SPLIT).metadata["game_date"].max()
    val = data.get(VALIDATION_SPLIT).metadata["game_date"]
    test = data.get(TEST_SPLIT).metadata["game_date"]
    assert train_max < val.min()
    assert val.max() < test.min()

    holdout = data.get(HOLDOUT_SPLIT).metadata
    assert holdout["is_partial_season"].all()
    for name in (TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT):
        assert not data.get(name).metadata["is_partial_season"].any()

    # Reusing the persisted manifest succeeds and returns identical fingerprints.
    reused = load_modeling_data(
        FEATURES_PATH, manifest_path=tmp_path / "manifest.json", reuse_manifest=True
    )
    assert reused.split_manifest == data.split_manifest


@requires_features
def test_committed_manifest_matches_dataset() -> None:
    manifest_path = ArcoraConfig.from_env().splits_dir / "split_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("Split manifest not generated")
    data = load_modeling_data(FEATURES_PATH, manifest_path=manifest_path, reuse_manifest=True)
    assert data.split_manifest["all_checks_passed"]
