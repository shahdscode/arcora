"""Automated temporal leakage detection tests (Issue #14)."""

from __future__ import annotations

import pandas as pd
import pytest

from arcora.features.leakage import (
    check_current_game_exclusion_on_openers,
    check_feature_integrity_and_bounds,
    check_future_information_invariance,
    check_preseason_exclusion,
    check_standings_asof_temporal_safety,
    check_target_and_postgame_exclusion,
    split_features_and_targets,
)
from arcora.features.matchup_features import build_matchup_feature_dataset
from arcora.features.strategy import FORBIDDEN_POSTGAME_COLUMNS, TARGET_COLUMNS


def _make_sample_games() -> pd.DataFrame:
    dates = pd.date_range("2022-10-18", periods=6, freq="2D")
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5, 6],
            "game_date": dates,
            "season": [2022] * 6,
            "season_label": ["2022-23"] * 6,
            "game_type": ["Regular Season"] * 6,
            "game_status": ["Final"] * 6,
            "home_team_id": [10, 20, 10, 20, 10, 20],
            "away_team_id": [20, 10, 20, 10, 20, 10],
            "home_pts": [105, 112, 98, 115, 108, 102],
            "away_pts": [99, 104, 101, 109, 100, 107],
            "total_pts": [204, 216, 199, 224, 208, 209],
            "point_margin": [6, 8, -3, 6, 8, -5],
            "home_win": [True, True, False, True, True, False],
            "home_fg_pct": [0.48] * 6,
            "home_fg3_pct": [0.36] * 6,
            "home_ft_pct": [0.80] * 6,
            "home_ast": [25] * 6,
            "home_reb": [44] * 6,
            "away_fg_pct": [0.45] * 6,
            "away_fg3_pct": [0.34] * 6,
            "away_ft_pct": [0.78] * 6,
            "away_ast": [22] * 6,
            "away_reb": [41] * 6,
        }
    )


def test_target_and_postgame_exclusion_from_feature_matrix() -> None:
    """Verify targets and post-game columns are excluded from X and caught if injected."""
    ds = build_matchup_feature_dataset(_make_sample_games())
    meta_df, x_df, y_df = split_features_and_targets(ds)

    assert len(meta_df) == 6
    assert len(x_df) == 6
    assert len(y_df) == 6
    assert list(y_df.columns) == list(TARGET_COLUMNS)
    for t_col in TARGET_COLUMNS:
        assert t_col not in x_df.columns
    for pg_col in FORBIDDEN_POSTGAME_COLUMNS:
        assert pg_col not in x_df.columns

    res = check_target_and_postgame_exclusion(ds)
    assert res.passed is True

    # Inject a forbidden post-game column and verify split_features_and_targets raises ValueError
    corrupted = ds.copy()
    corrupted["home_fg_pct"] = 0.50
    with pytest.raises(ValueError, match="Forbidden post-game"):
        split_features_and_targets(corrupted)
    assert check_target_and_postgame_exclusion(corrupted).passed is False


def test_current_game_exclusion_and_opener_checks() -> None:
    """Verify current-game stats never leak into opener features and violations are caught."""
    ds = build_matchup_feature_dataset(_make_sample_games())
    res = check_current_game_exclusion_on_openers(ds)
    assert res.passed is True
    assert res.violations_detected == 0

    # Deliberately leak current game score into opener's rolling average
    leaked = ds.copy()
    leaked.loc[leaked["home_hist_games_played"] == 0, "home_rolling_pts_avg_5"] = 105.0
    bad_res = check_current_game_exclusion_on_openers(leaked)
    assert bad_res.passed is False
    assert bad_res.violations_detected > 0


def test_future_information_and_standings_leakage_checks() -> None:
    """Verify future-game invariance, standings as-of check, preseason check, and bounds."""
    ds = build_matchup_feature_dataset(_make_sample_games())

    assert check_preseason_exclusion(ds).passed is True
    assert check_future_information_invariance().passed is True
    assert check_standings_asof_temporal_safety().passed is True
    assert check_feature_integrity_and_bounds(ds).passed is True
