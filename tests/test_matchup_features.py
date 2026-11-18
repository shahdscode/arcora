"""Tests for game-level matchup, head-to-head, and as-of standings features (Issue #13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arcora.features.matchup_features import (
    build_matchup_feature_dataset,
    compute_head_to_head_features,
    join_asof_standings_features,
)


def _make_matchup_games() -> pd.DataFrame:
    """Create a 4-game synthetic schedule with team swaps for H2H and matchup testing."""
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "game_date": pd.to_datetime(["2022-10-18", "2022-10-20", "2022-10-22", "2022-10-25"]),
            "season": [2022, 2022, 2022, 2022],
            "season_label": ["2022-23"] * 4,
            "game_type": ["Regular Season", "Regular Season", "Play-In", "Playoffs"],
            "game_status": ["Final"] * 4,
            "home_team_id": [10, 20, 10, 20],
            "away_team_id": [20, 10, 20, 10],
            "home_pts": [110, 105, 100, 115],
            "away_pts": [100, 95, 108, 105],
            "total_pts": [210, 200, 208, 220],
            "point_margin": [10, 10, -8, 10],
            "home_win": [True, True, False, True],
            "home_fg_pct": [0.50, 0.49, 0.46, 0.51],
            "home_fg3_pct": [0.38, 0.37, 0.34, 0.39],
            "home_ft_pct": [0.80, 0.79, 0.77, 0.82],
            "home_ast": [25, 24, 22, 27],
            "home_reb": [45, 44, 41, 46],
            "away_fg_pct": [0.45, 0.44, 0.48, 0.46],
            "away_fg3_pct": [0.34, 0.33, 0.36, 0.35],
            "away_ft_pct": [0.76, 0.75, 0.81, 0.78],
            "away_ast": [21, 20, 24, 22],
            "away_reb": [40, 39, 43, 41],
        }
    )


def test_head_to_head_orientation_across_home_away_swaps() -> None:
    """Verify H2H features correctly orient win % and margin when teams swap venues."""
    games = _make_matchup_games()
    h2h = compute_head_to_head_features(games)

    # Game 1 (10 H vs 20 A): 0 prior meetings
    g1 = h2h.loc[h2h["game_id"] == 1].iloc[0]
    assert g1["h2h_prior_meetings"] == 0
    assert np.isnan(g1["h2h_home_team_win_pct"])

    # Game 2 (20 H vs 10 A): 1 prior meeting (Game 1 where 10 beat 20 by +10).
    # Since Team 20 is NOW home in Game 2, Team 20's prior H2H win % is 0.0 and margin is -10.0!
    g2 = h2h.loc[h2h["game_id"] == 2].iloc[0]
    assert g2["h2h_prior_meetings"] == 1
    assert pytest.approx(g2["h2h_home_team_win_pct"]) == 0.0
    assert pytest.approx(g2["h2h_home_team_margin_avg"]) == -10.0
    assert pytest.approx(g2["h2h_total_pts_avg"]) == 210.0

    # Game 3 (10 H vs 20 A): 2 prior meetings (Game 1: 10 won +10; Game 2: 20 won +10 -> 0.0)
    g3 = h2h.loc[h2h["game_id"] == 3].iloc[0]
    assert g3["h2h_prior_meetings"] == 2
    assert pytest.approx(g3["h2h_home_team_win_pct"]) == 0.5
    assert pytest.approx(g3["h2h_home_team_margin_avg"]) == 0.0
    assert pytest.approx(g3["h2h_total_pts_avg"]) == 205.0


def test_asof_standings_join_strictly_rejects_same_day_and_future_snapshots() -> None:
    """Verify standings join uses standings_date < game_date and rejects same-day/future rows."""
    games = _make_matchup_games().iloc[:2].copy()
    standings = pd.DataFrame(
        {
            "team_id": [10, 10, 10, 20, 20],
            "season": [2022] * 5,
            "season_type": ["Regular Season"] * 5,
            "standings_date": pd.to_datetime(
                [
                    "2022-10-18",  # Same day as Game 1 -> MUST NOT be used for Game 1!
                    "2022-10-19",  # Strictly before Game 2 (2022-10-20) -> used for Game 2!
                    "2022-10-20",  # Same day as Game 2 -> MUST NOT be used for Game 2!
                    "2022-10-19",
                    "2022-10-25",  # Future snapshot -> MUST NOT be used for Game 2!
                ]
            ),
            "games_played": [1, 1, 2, 1, 10],
            "wins": [1, 1, 1, 0, 9],
            "losses": [0, 0, 1, 1, 1],
            "win_pct": [1.0, 1.0, 0.5, 0.0, 0.9],
            "home_wins": [1, 1, 1, 0, 5],
            "home_losses": [0, 0, 0, 0, 0],
            "road_wins": [0, 0, 0, 0, 4],
            "road_losses": [0, 0, 1, 1, 1],
        }
    )

    st_joined = join_asof_standings_features(games, standings)

    # Game 1 on 2022-10-18: no snapshot strictly before 2022-10-18 -> games_played=0, win_pct=NaN
    g1 = st_joined.loc[st_joined["game_id"] == 1].iloc[0]
    assert g1["home_standings_games_played"] == 0
    assert np.isnan(g1["home_standings_win_pct"])

    # Game 2 on 2022-10-20 (H=20, A=10): uses 2022-10-19 snapshot (not 2022-10-20 or 2022-10-25)
    g2 = st_joined.loc[st_joined["game_id"] == 2].iloc[0]
    assert g2["home_standings_games_played"] == 1
    assert pytest.approx(g2["home_standings_win_pct"]) == 0.0
    assert g2["away_standings_games_played"] == 1
    assert pytest.approx(g2["away_standings_win_pct"]) == 1.0
    assert pytest.approx(g2["standings_win_pct_diff"]) == -1.0


def test_build_matchup_feature_dataset_integrates_all_columns_and_differences() -> None:
    """Verify full matchup dataset has 1 row per game and exact home - away differences."""
    games = _make_matchup_games()
    ds = build_matchup_feature_dataset(games)

    assert len(ds) == 4
    assert ds["game_id"].duplicated().sum() == 0

    g3 = ds.loc[ds["game_id"] == 3].iloc[0]
    assert pytest.approx(g3["rolling_pts_avg_5_diff"]) == (
        g3["home_rolling_pts_avg_5"] - g3["away_rolling_pts_avg_5"]
    )
    assert pytest.approx(g3["hist_net_rtg_diff"]) == (
        g3["home_hist_net_rtg"] - g3["away_hist_net_rtg"]
    )
    assert bool(g3["is_play_in_game"]) is True
    assert bool(g3["is_playoff_game"]) is False
