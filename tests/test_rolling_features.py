"""Tests for rolling-window (5g, 10g), variability, and EWMA features (Issue #11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arcora.features.rolling_features import (
    attach_rolling_features_to_games,
    compute_team_rolling_and_ewma_features,
)
from arcora.features.team_performance import build_team_game_logs


def _make_seven_game_series() -> pd.DataFrame:
    """Build a 7-game synthetic schedule between Team 10 and Team 20."""
    dates = pd.date_range("2022-10-18", periods=7, freq="2D")
    home_pts = [100, 110, 120, 90, 105, 115, 130]
    away_pts = [95, 100, 105, 100, 95, 110, 120]
    return pd.DataFrame(
        {
            "game_id": list(range(1, 8)),
            "game_date": dates,
            "season": [2022] * 7,
            "season_label": ["2022-23"] * 7,
            "game_type": ["Regular Season"] * 7,
            "game_status": ["Final"] * 7,
            "home_team_id": [10] * 7,
            "away_team_id": [20] * 7,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "total_pts": [h + a for h, a in zip(home_pts, away_pts, strict=False)],
            "point_margin": [h - a for h, a in zip(home_pts, away_pts, strict=False)],
            "home_win": [h > a for h, a in zip(home_pts, away_pts, strict=False)],
            "home_fg_pct": [0.48] * 7,
            "home_fg3_pct": [0.36] * 7,
            "home_ft_pct": [0.80] * 7,
            "home_ast": [25] * 7,
            "home_reb": [44] * 7,
            "away_fg_pct": [0.45] * 7,
            "away_fg3_pct": [0.34] * 7,
            "away_ft_pct": [0.78] * 7,
            "away_ast": [22] * 7,
            "away_reb": [41] * 7,
        }
    )


def test_rolling_5_and_10_and_std_manual_verification() -> None:
    """Verify rolling 5g, 10g, std, and EWMA against manual calculations."""
    games = _make_seven_game_series()
    res = attach_rolling_features_to_games(games)

    # Game 1: 0 prior games -> NaN for rolling means/stds/ewma, insufficient=True
    g1 = res.loc[res["game_id"] == 1].iloc[0]
    assert bool(g1["home_rolling_insufficient_5"]) is True
    assert bool(g1["home_rolling_insufficient_10"]) is True
    assert np.isnan(g1["home_rolling_pts_avg_5"])
    assert np.isnan(g1["home_rolling_pts_std_5"])
    assert np.isnan(g1["home_ewma_pts"])

    # Game 2: 1 prior game (100 pts) -> mean=100.0, std=NaN (needs >=2 prior games)
    g2 = res.loc[res["game_id"] == 2].iloc[0]
    assert pytest.approx(g2["home_rolling_pts_avg_5"]) == 100.0
    assert np.isnan(g2["home_rolling_pts_std_5"])
    assert pytest.approx(g2["home_ewma_pts"]) == 100.0

    # Game 6: 5 prior games [100, 110, 120, 90, 105] -> mean = 525 / 5 = 105.0
    g6 = res.loc[res["game_id"] == 6].iloc[0]
    assert bool(g6["home_rolling_insufficient_5"]) is False
    assert bool(g6["home_rolling_insufficient_10"]) is True
    assert pytest.approx(g6["home_rolling_pts_avg_5"]) == 105.0
    expected_std_g6 = float(np.std([100, 110, 120, 90, 105], ddof=1))
    assert pytest.approx(g6["home_rolling_pts_std_5"]) == expected_std_g6

    # Game 7: 6 prior games -> rolling 5g uses games 2..6 [110, 120, 90, 105, 115] -> mean = 108.0
    # while rolling 10g uses all 6 prior games [100, 110, 120, 90, 105, 115] -> mean = 640 / 6
    g7 = res.loc[res["game_id"] == 7].iloc[0]
    assert pytest.approx(g7["home_rolling_pts_avg_5"]) == 108.0
    assert pytest.approx(g7["home_rolling_pts_avg_10"]) == 640.0 / 6.0

    # EWMA weights recent games more heavily than older games:
    # After game 6 (115 pts), ewma_pts on game 7 should exceed the simple 6-game mean (106.67)
    assert g7["home_ewma_pts"] > g7["home_rolling_pts_avg_10"]


def test_rolling_features_future_invariance_and_same_day_protection() -> None:
    """Verify future game mutations and same-day games never leak into rolling features."""
    games = _make_seven_game_series()
    games_mut = games.copy()
    games_mut.loc[games_mut["game_id"] == 7, "home_pts"] = 300

    r1 = attach_rolling_features_to_games(games)
    r2 = attach_rolling_features_to_games(games_mut)
    roll_cols = [c for c in r1.columns if "_rolling_" in c or "_ewma_" in c]
    pd.testing.assert_frame_equal(r1.iloc[:6][roll_cols], r2.iloc[:6][roll_cols])

    # Same-day test: set games 3 and 4 to the same date
    same_day_games = games.iloc[:4].copy()
    same_day_games.loc[same_day_games["game_id"] == 4, "game_date"] = same_day_games.loc[
        same_day_games["game_id"] == 3, "game_date"
    ].iloc[0]
    logs = build_team_game_logs(same_day_games)
    roll_df = compute_team_rolling_and_ewma_features(logs)
    t10 = roll_df[roll_df["team_id"] == 10]
    assert pytest.approx(t10.iloc[2]["rolling_pts_avg_5"]) == 105.0  # mean(100, 110)
    assert (
        pytest.approx(t10.iloc[3]["rolling_pts_avg_5"]) == 105.0
    )  # same date -> still mean(100, 110)
