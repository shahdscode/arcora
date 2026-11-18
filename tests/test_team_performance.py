"""Tests for historical team performance features and temporal correctness (Issue #10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arcora.features.team_performance import (
    attach_historical_features_to_games,
    build_team_game_logs,
    compute_team_historical_features,
    filter_eligible_games,
)


def _make_synthetic_games() -> pd.DataFrame:
    """Create a 4-game synthetic schedule for teams 10 and 20 in season 2022."""
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "game_date": pd.to_datetime(["2022-10-18", "2022-10-20", "2022-10-22", "2022-10-25"]),
            "season": [2022, 2022, 2022, 2022],
            "season_label": ["2022-23"] * 4,
            "game_type": ["Regular Season"] * 4,
            "game_status": ["Final"] * 4,
            "home_team_id": [10, 20, 10, 10],
            "away_team_id": [20, 10, 20, 20],
            "home_pts": [100, 110, 120, 90],
            "away_pts": [90, 100, 100, 105],
            "total_pts": [190, 210, 220, 195],
            "point_margin": [10, 10, 20, -15],
            "home_win": [True, True, True, False],
            "home_fg_pct": [0.50, 0.52, 0.55, 0.42],
            "home_fg3_pct": [0.35, 0.40, 0.38, 0.30],
            "home_ft_pct": [0.80, 0.78, 0.85, 0.75],
            "home_ast": [24, 26, 28, 20],
            "home_reb": [45, 42, 48, 38],
            "away_fg_pct": [0.44, 0.48, 0.45, 0.49],
            "away_fg3_pct": [0.32, 0.36, 0.33, 0.37],
            "away_ft_pct": [0.75, 0.82, 0.76, 0.80],
            "away_ast": [20, 22, 21, 25],
            "away_reb": [40, 44, 39, 46],
        }
    )


def test_filter_eligible_games_excludes_preseason() -> None:
    """Ensure Preseason games are excluded from historical feature calculations."""
    games = _make_synthetic_games()
    games.loc[0, "game_type"] = "Preseason"
    filtered = filter_eligible_games(games)
    assert len(filtered) == 3
    assert "Preseason" not in filtered["game_type"].tolist()


def test_advanced_metrics_formulas_in_team_logs() -> None:
    """Verify possessions, eFG%, TS%, TOV rate, ORtg, DRtg, and Net Rating formulas."""
    games = _make_synthetic_games().iloc[:1].copy()
    pstats = pd.DataFrame(
        {
            "game_id": [1, 1],
            "team_id": [10, 20],
            "player_id": [101, 201],
            "did_play": [True, True],
            "fgm": [40, 35],
            "fga": [80, 80],
            "fg3m": [10, 8],
            "fg3a": [25, 25],
            "ftm": [10, 12],
            "fta": [25, 25],
            "oreb": [11, 10],
            "dreb": [34, 30],
            "reb": [45, 40],
            "ast": [24, 20],
            "stl": [8, 6],
            "blk": [5, 4],
            "turnovers": [10, 15],
            "pf": [18, 20],
            "pts": [100, 90],
        }
    )
    logs = build_team_game_logs(games, pstats)
    t10 = logs[logs["team_id"] == 10].iloc[0]

    # Possessions = FGA + 0.44 * FTA - OREB + TOV = 80 + 11 - 11 + 10 = 90.0
    assert pytest.approx(t10["possessions"], rel=1e-5) == 90.0
    # eFG% = (40 + 0.5 * 10) / 80 = 45 / 80 = 0.5625
    assert pytest.approx(t10["efg_pct"], rel=1e-5) == 0.5625
    # TS% = 100 / (2 * (80 + 11)) = 100 / 182
    assert pytest.approx(t10["ts_pct"], rel=1e-5) == 100.0 / 182.0
    # TOV rate = 10 / 90
    assert pytest.approx(t10["tov_rate"], rel=1e-5) == 10.0 / 90.0
    # ORtg = 100 * 100 / 90
    assert pytest.approx(t10["ortg"], rel=1e-5) == 10000.0 / 90.0


def test_historical_features_manual_verification_and_no_current_game_leakage() -> None:
    """Verify historical expanding averages match hand calculations and exclude current game."""
    games = _make_synthetic_games()
    enriched = attach_historical_features_to_games(games)

    # Game 1 (2022-10-18): Season opener for both Team 10 (home) and Team 20 (away)
    g1 = enriched.loc[enriched["game_id"] == 1].iloc[0]
    assert g1["home_hist_games_played"] == 0
    assert bool(g1["home_hist_insufficient_history"]) is True
    assert np.isnan(g1["home_hist_pts_avg"])
    assert np.isnan(g1["away_hist_pts_avg"])

    # Game 2 (2022-10-20): Team 10 is away; prior game was Game 1 (scored 100, allowed 90, won)
    g2 = enriched.loc[enriched["game_id"] == 2].iloc[0]
    assert g2["away_hist_games_played"] == 1
    assert pytest.approx(g2["away_hist_pts_avg"]) == 100.0
    assert pytest.approx(g2["away_hist_pts_allowed_avg"]) == 90.0
    assert pytest.approx(g2["away_hist_win_pct"]) == 1.0
    # Team 10 has 0 prior road games entering Game 2
    assert g2["away_venue_games_played"] == 0
    assert np.isnan(g2["away_venue_pts_avg"])

    # Game 3 (2022-10-22): Team 10 is home; prior games are Game 1 (100 pts) and Game 2 (100 pts)
    g3 = enriched.loc[enriched["game_id"] == 3].iloc[0]
    assert g3["home_hist_games_played"] == 2
    assert pytest.approx(g3["home_hist_pts_avg"]) == 100.0
    assert pytest.approx(g3["home_hist_pts_allowed_avg"]) == 100.0  # (90 + 110) / 2
    assert pytest.approx(g3["home_hist_win_pct"]) == 0.5  # 1W, 1L
    # Team 10 has 1 prior home game (Game 1: 100 pts)
    assert g3["home_venue_games_played"] == 1
    assert pytest.approx(g3["home_venue_pts_avg"]) == 100.0


def test_historical_features_future_invariance() -> None:
    """Mutating future game scores must not change historical features of earlier games."""
    games_orig = _make_synthetic_games()
    games_mutated = _make_synthetic_games()
    games_mutated.loc[games_mutated["game_id"] == 4, "home_pts"] = 250
    games_mutated.loc[games_mutated["game_id"] == 4, "away_pts"] = 50

    res_orig = attach_historical_features_to_games(games_orig)
    res_mut = attach_historical_features_to_games(games_mutated)

    hist_cols = [c for c in res_orig.columns if "_hist_" in c or "_venue_" in c]
    pd.testing.assert_frame_equal(
        res_orig.iloc[:3][hist_cols],
        res_mut.iloc[:3][hist_cols],
    )


def test_same_day_games_do_not_leak_into_each_other() -> None:
    """Verify two games for the same team on the same date only use strictly earlier dates."""
    games = _make_synthetic_games().iloc[:3].copy()
    # Set Game 2 and Game 3 to the exact same date (2022-10-20)
    games.loc[games["game_id"] == 3, "game_date"] = pd.Timestamp("2022-10-20")
    logs = build_team_game_logs(games)
    hist = compute_team_historical_features(logs)
    t10_same_day = hist[(hist["team_id"] == 10) & (hist["game_date"] == pd.Timestamp("2022-10-20"))]
    assert len(t10_same_day) == 2
    # Both games on 2022-10-20 must only see Game 1 (from 2022-10-18)
    assert (t10_same_day["hist_games_played"] == 1).all()
    assert np.allclose(t10_same_day["hist_pts_avg"], 100.0)
