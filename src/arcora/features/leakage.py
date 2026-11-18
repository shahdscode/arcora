"""Automated temporal leakage detection and feature integrity checks (Issue #14)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from arcora.features.matchup_features import (
    build_matchup_feature_dataset,
    join_asof_standings_features,
)
from arcora.features.strategy import (
    FEATURE_COLUMN_NAMES,
    FORBIDDEN_POSTGAME_COLUMNS,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)


@dataclass(frozen=True)
class LeakageCheckResult:
    """Result of executing a single automated temporal leakage or integrity check."""

    check_id: str
    check_name: str
    passed: bool
    violations_detected: int
    details: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize check result to a JSON-compatible dictionary."""
        return asdict(self)


def split_features_and_targets(
    dataset_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate a full game feature dataset into (metadata_df, X_features_df, y_targets_df).

    Raises ValueError if any target column or forbidden post-game column leaks into X.
    """
    meta_cols = [c for c in METADATA_COLUMNS if c in dataset_df.columns]
    target_cols = [c for c in TARGET_COLUMNS if c in dataset_df.columns]
    feature_cols = [
        c
        for c in dataset_df.columns
        if c not in set(METADATA_COLUMNS) and c not in set(TARGET_COLUMNS)
    ]

    leaked_targets = set(feature_cols) & set(TARGET_COLUMNS)
    if leaked_targets:
        raise ValueError(f"Target leakage detected in input feature matrix X: {leaked_targets}")

    leaked_postgame = set(feature_cols) & set(FORBIDDEN_POSTGAME_COLUMNS)
    if leaked_postgame:
        raise ValueError(f"Forbidden post-game box-score columns detected in X: {leaked_postgame}")

    return (
        dataset_df[meta_cols].copy(),
        dataset_df[feature_cols].copy(),
        dataset_df[target_cols].copy(),
    )


def check_target_and_postgame_exclusion(dataset_df: pd.DataFrame) -> LeakageCheckResult:
    """Verify X contains all expected features and zero target or post-game columns."""
    try:
        _, x_df, _ = split_features_and_targets(dataset_df)
        missing_feats = set(FEATURE_COLUMN_NAMES) - set(x_df.columns)
        unexpected_feats = set(x_df.columns) - set(FEATURE_COLUMN_NAMES)
        violations = len(missing_feats) + len(unexpected_feats)
        if violations == 0:
            msg = (
                f"Input feature matrix X cleanly separates {len(x_df.columns)} pregame "
                "features from all 5 targets and all post-game box-score columns."
            )
        else:
            msg = f"Feature matrix mismatch: missing={missing_feats}, unexpected={unexpected_feats}"
        return LeakageCheckResult(
            check_id="LEAK-001",
            check_name="target_and_postgame_column_exclusion",
            passed=(violations == 0),
            violations_detected=violations,
            details=msg,
        )
    except ValueError as exc:
        return LeakageCheckResult(
            check_id="LEAK-001",
            check_name="target_and_postgame_column_exclusion",
            passed=False,
            violations_detected=1,
            details=str(exc),
        )


def check_preseason_exclusion(dataset_df: pd.DataFrame) -> LeakageCheckResult:
    """Verify zero Preseason games exist in the engineered dataset."""
    preseason_cnt = int((dataset_df["game_type"] == "Preseason").sum())
    return LeakageCheckResult(
        check_id="LEAK-002",
        check_name="preseason_games_excluded",
        passed=(preseason_cnt == 0),
        violations_detected=preseason_cnt,
        details=(
            f"All {len(dataset_df):,} rows belong to eligible stages "
            "(Regular Season, Playoffs, Play-In); 0 Preseason games present."
            if preseason_cnt == 0
            else f"Detected {preseason_cnt} Preseason games in feature dataset."
        ),
    )


def check_current_game_exclusion_on_openers(dataset_df: pd.DataFrame) -> LeakageCheckResult:
    """Verify season openers have NaN historical/rolling/EWMA stats (confirming shift(1))."""
    home_openers = dataset_df[dataset_df["home_hist_games_played"] == 0]
    away_openers = dataset_df[dataset_df["away_hist_games_played"] == 0]

    home_leaks = int(
        home_openers[["home_hist_pts_avg", "home_rolling_pts_avg_5", "home_ewma_pts"]]
        .notna()
        .any(axis=1)
        .sum()
    )
    away_leaks = int(
        away_openers[["away_hist_pts_avg", "away_rolling_pts_avg_5", "away_ewma_pts"]]
        .notna()
        .any(axis=1)
        .sum()
    )
    total_leaks = home_leaks + away_leaks
    return LeakageCheckResult(
        check_id="LEAK-003",
        check_name="current_game_excluded_via_shift_1",
        passed=(total_leaks == 0),
        violations_detected=total_leaks,
        details=(
            f"Verified {len(home_openers):,} home openers and {len(away_openers):,} away "
            "openers have strictly null prior scoring averages (0 current-game leaks)."
            if total_leaks == 0
            else f"Detected {total_leaks} season-opener rows with leaked non-null stats."
        ),
    )


def check_future_information_invariance() -> LeakageCheckResult:
    """Verify mutating future game scores, adding future games, or shuffling input order
    leaves all historical pregame features 100% invariant.
    """
    dates = pd.date_range("2022-10-18", periods=8, freq="2D")
    base_games = pd.DataFrame(
        {
            "game_id": list(range(101, 109)),
            "game_date": dates,
            "season": [2022] * 8,
            "season_label": ["2022-23"] * 8,
            "game_type": ["Regular Season"] * 8,
            "game_status": ["Final"] * 8,
            "home_team_id": [10, 20, 10, 20, 10, 20, 10, 20],
            "away_team_id": [20, 10, 20, 10, 20, 10, 20, 10],
            "home_pts": [102, 108, 115, 99, 110, 104, 120, 112],
            "away_pts": [98, 101, 105, 107, 100, 109, 111, 103],
            "total_pts": [200, 209, 220, 206, 210, 213, 231, 215],
            "point_margin": [4, 7, 10, -8, 10, -5, 9, 9],
            "home_win": [True, True, True, False, True, False, True, True],
            "home_fg_pct": [0.48] * 8,
            "home_fg3_pct": [0.36] * 8,
            "home_ft_pct": [0.80] * 8,
            "home_ast": [24] * 8,
            "home_reb": [44] * 8,
            "away_fg_pct": [0.45] * 8,
            "away_fg3_pct": [0.34] * 8,
            "away_ft_pct": [0.77] * 8,
            "away_ast": [21] * 8,
            "away_reb": [41] * 8,
        }
    )

    # 1. Baseline on first 6 games
    ds_first6 = build_matchup_feature_dataset(base_games.iloc[:6].copy())

    # 2. Full 8 games with extreme mutated scores on games 7 and 8, plus shuffled input order
    perturbed = base_games.copy()
    perturbed.loc[perturbed["game_id"] >= 107, "home_pts"] = 299
    perturbed.loc[perturbed["game_id"] >= 107, "away_pts"] = 40
    perturbed.loc[perturbed["game_id"] >= 107, "total_pts"] = 339
    perturbed.loc[perturbed["game_id"] >= 107, "point_margin"] = 259
    perturbed = perturbed.sample(frac=1.0, random_state=42).reset_index(drop=True)

    ds_perturbed = build_matchup_feature_dataset(perturbed).iloc[:6].reset_index(drop=True)

    feat_cols = list(FEATURE_COLUMN_NAMES)
    try:
        pd.testing.assert_frame_equal(ds_first6[feat_cols], ds_perturbed[feat_cols])
        return LeakageCheckResult(
            check_id="LEAK-004",
            check_name="future_game_and_shuffle_invariance",
            passed=True,
            violations_detected=0,
            details=(
                "Pregame features for games 1..6 are 100% invariant to adding games 7..8, "
                "mutating future scores, and shuffling input row order."
            ),
        )
    except AssertionError as exc:
        return LeakageCheckResult(
            check_id="LEAK-004",
            check_name="future_game_and_shuffle_invariance",
            passed=False,
            violations_detected=1,
            details=f"Future information invariance failed: {exc}",
        )


def check_standings_asof_temporal_safety() -> LeakageCheckResult:
    """Verify standings join strictly enforces standings_date < game_date."""
    games = pd.DataFrame(
        {
            "game_id": [1, 2],
            "game_date": pd.to_datetime(["2022-10-20", "2022-10-22"]),
            "season": [2022, 2022],
            "home_team_id": [10, 10],
            "away_team_id": [20, 20],
        }
    )
    standings = pd.DataFrame(
        {
            "team_id": [10, 10, 20],
            "season": [2022, 2022, 2022],
            "season_type": ["Regular Season"] * 3,
            # Snapshot on 2022-10-20 is same-day for Game 1 (must be rejected for Game 1,
            # accepted for Game 2 on 2022-10-22). Snapshot on 2022-10-22 is same-day for Game 2.
            "standings_date": pd.to_datetime(["2022-10-20", "2022-10-22", "2022-10-22"]),
            "games_played": [5, 82, 82],
            "wins": [4, 60, 50],
            "losses": [1, 22, 32],
            "win_pct": [0.800, 0.732, 0.610],
            "home_wins": [2, 35, 30],
            "home_losses": [0, 6, 11],
            "road_wins": [2, 25, 20],
            "road_losses": [1, 16, 21],
        }
    )
    joined = join_asof_standings_features(games, standings)
    g1 = joined.loc[joined["game_id"] == 1].iloc[0]
    g2 = joined.loc[joined["game_id"] == 2].iloc[0]

    violations = 0
    if g1["home_standings_games_played"] != 0 or not np.isnan(g1["home_standings_win_pct"]):
        violations += 1
    if g2["home_standings_games_played"] != 5 or not np.isclose(g2["home_standings_win_pct"], 0.8):
        violations += 1
    if g2["away_standings_games_played"] != 0 or not np.isnan(g2["away_standings_win_pct"]):
        violations += 1

    return LeakageCheckResult(
        check_id="LEAK-005",
        check_name="standings_strict_backward_asof_join",
        passed=(violations == 0),
        violations_detected=violations,
        details=(
            "Standings join strictly enforces standings_date < game_date and rejects "
            "same-day and future/end-of-season snapshots."
        ),
    )


def check_feature_integrity_and_bounds(dataset_df: pd.DataFrame) -> LeakageCheckResult:
    """Validate unique game_ids, valid ranges, and exact difference consistency."""
    dup_ids = int(dataset_df["game_id"].duplicated().sum())
    same_teams = int((dataset_df["home_team_id"] == dataset_df["away_team_id"]).sum())

    win_pct_cols = [c for c in FEATURE_COLUMN_NAMES if "win_pct" in c and not c.endswith("_diff")]
    bad_win_pct = 0
    for c in win_pct_cols:
        s = dataset_df[c].dropna()
        bad_win_pct += int(((s < -1e-6) | (s > 1.0 + 1e-6)).sum())

    bad_rest = 0
    for side in ("home", "away"):
        r = dataset_df[f"{side}_rest_days"].dropna()
        bad_rest += int(((r < 1.0) | (r > 14.0)).sum())

    diff_check = (
        dataset_df["rolling_pts_avg_5_diff"]
        - (dataset_df["home_rolling_pts_avg_5"] - dataset_df["away_rolling_pts_avg_5"])
    ).dropna()
    bad_diff = int((diff_check.abs() > 1e-6).sum())

    total_violations = dup_ids + same_teams + bad_win_pct + bad_rest + bad_diff
    return LeakageCheckResult(
        check_id="LEAK-006",
        check_name="feature_integrity_bounds_and_differences",
        passed=(total_violations == 0),
        violations_detected=total_violations,
        details=(
            f"All {len(dataset_df):,} game rows have unique game_ids, valid win_pct "
            "bounds [0, 1], valid rest_days [1, 14], and exact home-away differentials."
            if total_violations == 0
            else (
                f"Violations found: dup_ids={dup_ids}, same_teams={same_teams}, "
                f"bad_win_pct={bad_win_pct}, bad_rest={bad_rest}, bad_diff={bad_diff}"
            )
        ),
    )


def run_leakage_audit_suite(dataset_df: pd.DataFrame) -> list[LeakageCheckResult]:
    """Run all automated leakage and feature integrity checks on dataset_df."""
    return [
        check_target_and_postgame_exclusion(dataset_df),
        check_preseason_exclusion(dataset_df),
        check_current_game_exclusion_on_openers(dataset_df),
        check_future_information_invariance(),
        check_standings_asof_temporal_safety(),
        check_feature_integrity_and_bounds(dataset_df),
    ]
