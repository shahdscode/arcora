"""Tests for automated NBA data quality validation checks (Issue #6)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arcora.config import ArcoraConfig
from arcora.validation.checks import (
    check_distinct_home_and_away_teams,
    check_foreign_key_reference,
    check_game_dates_and_seasons,
    check_game_scores_and_numeric_types,
    check_game_status_and_type,
    check_missing_critical_values,
    check_no_duplicate_primary_keys,
    check_player_dnp_and_rebound_consistency,
    check_player_game_team_matches_game_participants,
    check_player_stats_numeric_integrity,
    check_players_career_game_counts,
    check_standings_integrity,
    run_validation_suite,
    write_validation_artifacts,
)


def test_processed_datasets_pass_all_quality_checks(tmp_path: Path) -> None:
    """Verify that the canonical processed Parquet datasets pass all 25 quality checks."""
    cfg = ArcoraConfig.from_env()
    summary = run_validation_suite(cfg.processed_data_dir)

    assert summary.overall_passed is True
    assert summary.total_checks == 25
    assert summary.passed_checks == 25
    assert summary.failed_checks == 0

    for res in summary.results:
        assert res.passed is True, f"Check {res.rule_id} failed: {res.message}"

    artifacts = write_validation_artifacts(summary, tmp_path / "reports")
    assert artifacts["report_markdown"].is_file()
    assert artifacts["summary_json"].is_file()


def test_validation_detects_duplicate_game_ids() -> None:
    """Verify duplicate game_id check fails and reports offending keys."""
    df = pd.DataFrame({"game_id": [101, 101, 102]})
    res = check_no_duplicate_primary_keys(df, "games", ("game_id",), "VAL-001")
    assert res.passed is False
    assert res.failed_records == 1
    assert "duplicate primary key" in res.message


def test_validation_detects_invalid_dates_and_seasons() -> None:
    """Verify date and season validation detects out-of-range seasons and mismatched labels."""
    df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "game_date": pd.to_datetime(["2022-10-20", "1990-01-01"]),
            "season": [2022, 1990],
            "season_label": ["2022-23", "1990-99"],
        }
    )
    res = check_game_dates_and_seasons(df)
    assert res.passed is False
    assert res.failed_records == 1
    assert "invalid dates or inconsistent season identifiers" in res.message


def test_validation_detects_identical_home_and_away_teams() -> None:
    """Verify check catches games where home_team_id == away_team_id."""
    df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "home_team_id": [1610612737, 1610612738],
            "away_team_id": [1610612737, 1610612739],
        }
    )
    res = check_distinct_home_and_away_teams(df)
    assert res.passed is False
    assert res.failed_records == 1
    assert "home_team_id == away_team_id" in res.message


def test_validation_detects_invalid_scores_and_percentages() -> None:
    """Verify score validation detects negative points, bad margins, or percentages > 1."""
    df = pd.DataFrame(
        {
            "game_id": [1, 2],
            "season": [2022, 2022],
            "home_team_id": [1, 2],
            "away_team_id": [3, 4],
            "home_pts": [110, -5],
            "away_pts": [100, 98],
            "total_pts": [210, 93],
            "point_margin": [10, -103],
            "home_win": [True, False],
            "home_fg_pct": [0.48, 1.25],
            "home_ft_pct": [0.80, 0.75],
            "home_fg3_pct": [0.35, 0.33],
            "home_ast": [25, 20],
            "home_reb": [44, 40],
            "away_fg_pct": [0.45, 0.44],
            "away_ft_pct": [0.78, 0.77],
            "away_fg3_pct": [0.34, 0.31],
            "away_ast": [22, 19],
            "away_reb": [41, 39],
        }
    )
    res = check_game_scores_and_numeric_types(df)
    assert res.passed is False
    assert res.failed_records == 1


def test_validation_detects_missing_critical_values_and_foreign_keys() -> None:
    """Verify missing value check and foreign key checks catch corrupt records."""
    games = pd.DataFrame({"game_id": [1, 2], "home_team_id": [10, 99], "away_team_id": [20, 10]})
    teams = pd.DataFrame({"team_id": [10, 20], "city": ["Boston", None]})

    null_res = check_missing_critical_values(teams, "teams", ["team_id", "city"], "VAL-010")
    assert null_res.passed is False
    assert null_res.failed_records == 1

    fk_res = check_foreign_key_reference(
        games, "games", "home_team_id", teams, "teams", "team_id", "VAL-013"
    )
    assert fk_res.passed is False
    assert fk_res.failed_records == 1
    assert "99" in fk_res.message

    pstats = pd.DataFrame(
        {
            "game_id": [1],
            "team_id": [999],
            "player_id": [500],
            "minutes_played": [-2.0],
            "fgm": [6],
            "fga": [5],
            "fg_pct": [0.5],
            "fg3m": [1],
            "fg3a": [2],
            "fg3_pct": [0.5],
            "ftm": [1],
            "fta": [2],
            "ft_pct": [0.5],
            "oreb": [1],
            "dreb": [2],
            "reb": [3],
            "ast": [1],
            "stl": [0],
            "blk": [0],
            "turnovers": [1],
            "pf": [2],
            "pts": [14],
        }
    )
    pnum_res = check_player_stats_numeric_integrity(pstats)
    assert pnum_res.passed is False
    assert pnum_res.failed_records == 1

    part_res = check_player_game_team_matches_game_participants(pstats, games)
    assert part_res.passed is False
    assert part_res.failed_records == 1


def test_validation_detects_new_canonical_rule_violations() -> None:
    """Verify checks for career counts, game status/type, DNP/rebounds, and standings."""
    players = pd.DataFrame(
        {
            "player_id": [1],
            "first_season": [2020],
            "last_season": [2021],
            "career_games_played": [50],
            "career_games_logged": [40],
        }
    )
    assert check_players_career_game_counts(players).passed is False

    bad_games = pd.DataFrame(
        {"game_id": [1], "game_status": ["Scheduled"], "game_type": ["Invalid"]}
    )
    assert check_game_status_and_type(bad_games).passed is False

    bad_pstats = pd.DataFrame(
        {
            "game_id": [1],
            "player_id": [10],
            "did_play": [False],
            "minutes_played": [5.0],
            "pts": [2],
            "oreb": [1],
            "dreb": [2],
            "reb": [5],
        }
    )
    assert check_player_dnp_and_rebound_consistency(bad_pstats).passed is False

    bad_standings = pd.DataFrame(
        {
            "team_id": [1610612737],
            "standings_date": pd.to_datetime(["2022-12-01"]),
            "games_played": [20],
            "wins": [10],
            "losses": [9],
            "win_pct": [0.5],
            "conference": ["North"],
            "season_type": ["Regular Season"],
        }
    )
    assert check_standings_integrity(bad_standings).passed is False
