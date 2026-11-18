"""Tests for rest, back-to-back, schedule density, and streak features (Issue #12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arcora.features.schedule_features import attach_schedule_features_to_games


def _make_schedule_games() -> pd.DataFrame:
    """Create a synthetic schedule testing openers, B2Bs, 7d/14d density, and streaks."""
    dates = pd.to_datetime(
        [
            "2022-10-18",  # Game 1: 10 (H) vs 20 (A) -> opener for both
            "2022-10-19",  # Game 2: 10 (H) vs 30 (A) -> 10 on B2B (rest=1), 30 opener
            "2022-10-20",  # Game 3: 30 (H) vs 20 (A) -> 30 on B2B (rest=1), 20 rest=2
            "2022-10-21",  # Game 4: 20 (H) vs 30 (A) -> BOTH on B2B (rest=1 for both!)
            "2022-10-24",  # Game 5: 10 (H) vs 20 (A) -> 10 has 2 prior home games, rest=5
        ]
    )
    home_ids = [10, 10, 30, 20, 10]
    away_ids = [20, 30, 20, 30, 20]
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4, 5],
            "game_date": dates,
            "season": [2022] * 5,
            "season_label": ["2022-23"] * 5,
            "game_type": ["Regular Season"] * 5,
            "game_status": ["Final"] * 5,
            "home_team_id": home_ids,
            "away_team_id": away_ids,
            "home_pts": [105] * 5,
            "away_pts": [100] * 5,
            "total_pts": [205] * 5,
            "point_margin": [5] * 5,
            "home_win": [True] * 5,
            "home_fg_pct": [0.48] * 5,
            "home_fg3_pct": [0.36] * 5,
            "home_ft_pct": [0.80] * 5,
            "home_ast": [25] * 5,
            "home_reb": [44] * 5,
            "away_fg_pct": [0.45] * 5,
            "away_fg3_pct": [0.34] * 5,
            "away_ft_pct": [0.78] * 5,
            "away_ast": [22] * 5,
            "away_reb": [41] * 5,
        }
    )


def test_rest_days_back_to_back_and_density_calculations() -> None:
    """Verify opener flags, rest days, B2B flags, 7d/14d counts, and streaks."""
    games = _make_schedule_games()
    res = attach_schedule_features_to_games(games)

    # Game 1: Season opener for Team 10 and Team 20
    g1 = res.loc[res["game_id"] == 1].iloc[0]
    assert bool(g1["home_is_season_opener"]) is True
    assert bool(g1["away_is_season_opener"]) is True
    assert np.isnan(g1["home_rest_days"])
    assert np.isnan(g1["rest_days_diff"])
    assert bool(g1["home_is_back_to_back"]) is False
    assert g1["home_games_last_7d"] == 0
    assert g1["home_consecutive_home_games"] == 0

    # Game 2 (2022-10-19): Team 10 on B2B (rest=1), Team 30 on opener
    g2 = res.loc[res["game_id"] == 2].iloc[0]
    assert pytest.approx(g2["home_rest_days"]) == 1.0
    assert bool(g2["home_is_back_to_back"]) is True
    assert bool(g2["away_is_season_opener"]) is True
    assert bool(g2["both_teams_back_to_back"]) is False
    assert g2["home_games_last_7d"] == 1
    assert g2["home_consecutive_home_games"] == 1

    # Game 4 (2022-10-21): Both Team 20 and Team 30 played on 2022-10-20 -> both B2B!
    g4 = res.loc[res["game_id"] == 4].iloc[0]
    assert pytest.approx(g4["home_rest_days"]) == 1.0
    assert pytest.approx(g4["away_rest_days"]) == 1.0
    assert pytest.approx(g4["rest_days_diff"]) == 0.0
    assert bool(g4["both_teams_back_to_back"]) is True
    # Team 20 played away on Oct 18 and Oct 20 -> 2 consecutive away games entering Game 4
    assert g4["home_consecutive_away_games"] == 2
    assert g4["home_consecutive_home_games"] == 0

    # Game 5 (2022-10-24): Team 10 played Oct 18 (H) and Oct 19 (H) -> rest=5, 2 consec home
    g5 = res.loc[res["game_id"] == 5].iloc[0]
    assert pytest.approx(g5["home_rest_days"]) == 5.0
    assert pytest.approx(g5["away_rest_days"]) == 3.0  # Team 20 last played Oct 21
    assert pytest.approx(g5["rest_days_diff"]) == 2.0
    assert g5["home_games_last_7d"] == 2
    assert g5["home_consecutive_home_games"] == 2
    assert g5["away_games_last_7d"] == 3


def test_season_boundary_resets_schedule_features() -> None:
    """Verify a new season resets opener flags, rest days, and streaks."""
    games = _make_schedule_games().iloc[:2].copy()
    games.loc[1, "season"] = 2023
    games.loc[1, "game_date"] = pd.Timestamp("2023-10-24")
    res = attach_schedule_features_to_games(games)

    g2 = res.loc[res["game_id"] == 2].iloc[0]
    assert bool(g2["home_is_season_opener"]) is True
    assert np.isnan(g2["home_rest_days"])
    assert g2["home_games_last_7d"] == 0
    assert g2["home_consecutive_home_games"] == 0
