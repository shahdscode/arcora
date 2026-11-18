"""Feature engineering strategy, target definitions, and feature specifications (Issue #9)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TARGET_COLUMNS: tuple[str, ...] = (
    "home_pts",
    "away_pts",
    "point_margin",
    "total_pts",
    "home_win",
)

METADATA_COLUMNS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "season",
    "season_label",
    "game_type",
    "is_partial_season",
    "home_team_id",
    "away_team_id",
)

FORBIDDEN_POSTGAME_COLUMNS: tuple[str, ...] = (
    "home_fg_pct",
    "home_ft_pct",
    "home_fg3_pct",
    "home_ast",
    "home_reb",
    "away_fg_pct",
    "away_ft_pct",
    "away_fg3_pct",
    "away_ast",
    "away_reb",
    "game_status",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "turnovers",
    "pf",
    "pts",
    "plus_minus",
)

FEATURE_CATEGORIES: tuple[str, ...] = (
    "Historical Team Performance",
    "Home and Away Performance",
    "Recent Team Form",
    "Rolling Offensive and Defensive Statistics",
    "Rest and Schedule-Related Features",
    "Head-to-Head Historical Performance",
    "Standings and Season Context",
    "Home-Away Matchup Differences",
)


@dataclass(frozen=True)
class TargetSpec:
    """Specification for a model prediction target."""

    name: str
    task_type: str
    dtype: str
    formula: str
    description: str


@dataclass(frozen=True)
class FeatureSpec:
    """Complete specification for an engineered pregame model feature."""

    name: str
    category: str
    description: str
    formula: str
    source_columns: str
    dtype: str
    lookback_window: str
    pregame_available: bool
    missing_value_strategy: str
    leakage_prevention_note: str


PREDICTION_TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        name="home_pts",
        task_type="regression",
        dtype="int64",
        formula="games.home_pts",
        description="Final points scored by the home team.",
    ),
    TargetSpec(
        name="away_pts",
        task_type="regression",
        dtype="int64",
        formula="games.away_pts",
        description="Final points scored by the away team.",
    ),
    TargetSpec(
        name="point_margin",
        task_type="regression",
        dtype="int64",
        formula="home_pts - away_pts",
        description="Final signed point differential from the home team perspective.",
    ),
    TargetSpec(
        name="total_pts",
        task_type="regression",
        dtype="int64",
        formula="home_pts + away_pts",
        description="Combined final points scored by both teams.",
    ),
    TargetSpec(
        name="home_win",
        task_type="binary_classification",
        dtype="bool",
        formula="home_pts > away_pts",
        description="Binary outcome indicator (True if home team wins).",
    ),
)


def build_feature_specifications() -> list[FeatureSpec]:
    """Build the complete ordered catalog of engineered pregame features."""
    specs: list[FeatureSpec] = []

    def _add(
        name: str,
        category: str,
        description: str,
        formula: str,
        source_columns: str,
        dtype: str = "float64",
        lookback_window: str = "Season-to-date prior games (strictly < game_date)",
        missing_value_strategy: str = (
            "NaN when team has 0 prior games in season; "
            "flagged via insufficient_history indicator."
        ),
        leakage_prevention_note: str = (
            "Computed after chronological sort and shift(1); "
            "excludes current and same-day games."
        ),
    ) -> None:
        specs.append(
            FeatureSpec(
                name=name,
                category=category,
                description=description,
                formula=formula,
                source_columns=source_columns,
                dtype=dtype,
                lookback_window=lookback_window,
                pregame_available=True,
                missing_value_strategy=missing_value_strategy,
                leakage_prevention_note=leakage_prevention_note,
            )
        )

    # 1. Historical Team Performance (per side: home, away)
    hist_metrics = [
        (
            "hist_games_played",
            "int64",
            "Count of prior completed season games",
            "COUNT(prior_games)",
            "games.game_id",
            "0 on season opener.",
        ),
        (
            "hist_insufficient_history",
            "bool",
            "Indicator that team has fewer than 5 prior season games",
            "hist_games_played < 5",
            "games.game_id",
            "False when >= 5 prior games, True otherwise.",
        ),
        (
            "hist_pts_avg",
            "float64",
            "Historical season-to-date average points scored",
            "MEAN(pts_for)",
            "games.home_pts, games.away_pts",
            "NaN on season opener.",
        ),
        (
            "hist_pts_allowed_avg",
            "float64",
            "Historical season-to-date average points conceded",
            "MEAN(pts_against)",
            "games.home_pts, games.away_pts",
            "NaN on season opener.",
        ),
        (
            "hist_total_pts_avg",
            "float64",
            "Historical season-to-date average combined game points",
            "MEAN(pts_for + pts_against)",
            "games.total_pts",
            "NaN on season opener.",
        ),
        (
            "hist_margin_avg",
            "float64",
            "Historical season-to-date average point margin",
            "MEAN(pts_for - pts_against)",
            "games.point_margin",
            "NaN on season opener.",
        ),
        (
            "hist_win_pct",
            "float64",
            "Historical season-to-date winning percentage",
            "MEAN(won_game)",
            "games.home_win",
            "NaN on season opener.",
        ),
        (
            "hist_reb_avg",
            "float64",
            "Historical season-to-date average total rebounds",
            "MEAN(reb)",
            "player_game_stats.reb",
            "NaN on season opener.",
        ),
        (
            "hist_oreb_avg",
            "float64",
            "Historical season-to-date average offensive rebounds",
            "MEAN(oreb)",
            "player_game_stats.oreb",
            "NaN on season opener.",
        ),
        (
            "hist_dreb_avg",
            "float64",
            "Historical season-to-date average defensive rebounds",
            "MEAN(dreb)",
            "player_game_stats.dreb",
            "NaN on season opener.",
        ),
        (
            "hist_ast_avg",
            "float64",
            "Historical season-to-date average assists",
            "MEAN(ast)",
            "player_game_stats.ast",
            "NaN on season opener.",
        ),
        (
            "hist_stl_avg",
            "float64",
            "Historical season-to-date average steals",
            "MEAN(stl)",
            "player_game_stats.stl",
            "NaN on season opener.",
        ),
        (
            "hist_blk_avg",
            "float64",
            "Historical season-to-date average blocks",
            "MEAN(blk)",
            "player_game_stats.blk",
            "NaN on season opener.",
        ),
        (
            "hist_tov_avg",
            "float64",
            "Historical season-to-date average turnovers",
            "MEAN(turnovers)",
            "player_game_stats.turnovers",
            "NaN on season opener.",
        ),
        (
            "hist_fg_pct",
            "float64",
            "Historical season-to-date field goal percentage",
            "MEAN(fgm / fga)",
            "player_game_stats.fgm, fga",
            "NaN on season opener.",
        ),
        (
            "hist_fg3_pct",
            "float64",
            "Historical season-to-date 3-point percentage",
            "MEAN(fg3m / fg3a)",
            "player_game_stats.fg3m, fg3a",
            "NaN on season opener.",
        ),
        (
            "hist_ft_pct",
            "float64",
            "Historical season-to-date free-throw percentage",
            "MEAN(ftm / fta)",
            "player_game_stats.ftm, fta",
            "NaN on season opener.",
        ),
        (
            "hist_efg_pct",
            "float64",
            "Historical season-to-date Effective Field Goal Percentage (eFG%)",
            "MEAN((fgm + 0.5 * fg3m) / fga)",
            "player_game_stats.fgm, fg3m, fga",
            "NaN on season opener.",
        ),
        (
            "hist_ts_pct",
            "float64",
            "Historical season-to-date True Shooting Percentage (TS%)",
            "MEAN(pts / (2 * (fga + 0.44 * fta)))",
            "player_game_stats.pts, fga, fta",
            "NaN on season opener.",
        ),
        (
            "hist_tov_rate",
            "float64",
            "Historical season-to-date turnover rate per possession",
            "MEAN(tov / possessions)",
            "player_game_stats.turnovers, fga, fta, oreb",
            "NaN on season opener.",
        ),
        (
            "hist_possessions_avg",
            "float64",
            "Historical season-to-date estimated possessions per game",
            "MEAN(fga + 0.44 * fta - oreb + tov)",
            "player_game_stats.fga, fta, oreb, turnovers",
            "NaN on season opener.",
        ),
        (
            "hist_ortg",
            "float64",
            "Historical season-to-date Offensive Rating (points per 100 possessions)",
            "MEAN(100 * pts_for / possessions)",
            "games.pts, player_game_stats",
            "NaN on season opener.",
        ),
        (
            "hist_drtg",
            "float64",
            "Historical season-to-date Defensive Rating (points allowed per 100 possessions)",
            "MEAN(100 * pts_against / opp_possessions)",
            "games.pts, player_game_stats",
            "NaN on season opener.",
        ),
        (
            "hist_net_rtg",
            "float64",
            "Historical season-to-date Net Rating (ORtg - DRtg)",
            "hist_ortg - hist_drtg",
            "games.pts, player_game_stats",
            "NaN on season opener.",
        ),
    ]
    for side in ("home", "away"):
        for col_suffix, dtype, desc, formula, src, mv_note in hist_metrics:
            _add(
                name=f"{side}_{col_suffix}",
                category="Historical Team Performance",
                description=f"{side.capitalize()} team: {desc}.",
                formula=formula,
                source_columns=src,
                dtype=dtype,
                missing_value_strategy=mv_note,
            )

    # 2. Home and Away Venue Performance Splits
    venue_specs = [
        (
            "home_venue_games_played",
            "int64",
            "Home team's prior home games played in season",
            "COUNT(prior_home_games)",
            "0 on first home game.",
        ),
        (
            "home_venue_pts_avg",
            "float64",
            "Home team's average points scored in prior home games",
            "MEAN(home_pts in prior home games)",
            "NaN on first home game.",
        ),
        (
            "home_venue_pts_allowed_avg",
            "float64",
            "Home team's average points conceded in prior home games",
            "MEAN(away_pts in prior home games)",
            "NaN on first home game.",
        ),
        (
            "home_venue_margin_avg",
            "float64",
            "Home team's average scoring margin in prior home games",
            "MEAN(margin in prior home games)",
            "NaN on first home game.",
        ),
        (
            "home_venue_win_pct",
            "float64",
            "Home team's win percentage in prior home games",
            "MEAN(win in prior home games)",
            "NaN on first home game.",
        ),
        (
            "away_venue_games_played",
            "int64",
            "Away team's prior road games played in season",
            "COUNT(prior_away_games)",
            "0 on first road game.",
        ),
        (
            "away_venue_pts_avg",
            "float64",
            "Away team's average points scored in prior road games",
            "MEAN(away_pts in prior road games)",
            "NaN on first road game.",
        ),
        (
            "away_venue_pts_allowed_avg",
            "float64",
            "Away team's average points conceded in prior road games",
            "MEAN(home_pts in prior road games)",
            "NaN on first road game.",
        ),
        (
            "away_venue_margin_avg",
            "float64",
            "Away team's average scoring margin in prior road games",
            "MEAN(margin in prior road games)",
            "NaN on first road game.",
        ),
        (
            "away_venue_win_pct",
            "float64",
            "Away team's win percentage in prior road games",
            "MEAN(win in prior road games)",
            "NaN on first road game.",
        ),
    ]
    for name, dtype, desc, formula, mv_note in venue_specs:
        _add(
            name=name,
            category="Home and Away Performance",
            description=desc,
            formula=formula,
            source_columns="games.home_pts, games.away_pts, games.home_win",
            dtype=dtype,
            lookback_window="Season-to-date prior venue-specific games (< game_date)",
            missing_value_strategy=mv_note,
        )

    # 3 & 4. Rolling Windows (5 and 10) and EWMA
    rolling_metrics = [
        ("pts_avg", "Recent Team Form", "average points scored", "MEAN(pts_for)"),
        ("pts_allowed_avg", "Recent Team Form", "average points conceded", "MEAN(pts_against)"),
        ("margin_avg", "Recent Team Form", "average scoring margin", "MEAN(margin)"),
        ("win_pct", "Recent Team Form", "winning percentage", "MEAN(win)"),
        (
            "reb_avg",
            "Rolling Offensive and Defensive Statistics",
            "average total rebounds",
            "MEAN(reb)",
        ),
        ("ast_avg", "Rolling Offensive and Defensive Statistics", "average assists", "MEAN(ast)"),
        ("tov_avg", "Rolling Offensive and Defensive Statistics", "average turnovers", "MEAN(tov)"),
        (
            "fg_pct",
            "Rolling Offensive and Defensive Statistics",
            "field goal percentage",
            "MEAN(fg_pct)",
        ),
        (
            "fg3_pct",
            "Rolling Offensive and Defensive Statistics",
            "3-point percentage",
            "MEAN(fg3_pct)",
        ),
        (
            "ft_pct",
            "Rolling Offensive and Defensive Statistics",
            "free-throw percentage",
            "MEAN(ft_pct)",
        ),
        (
            "efg_pct",
            "Rolling Offensive and Defensive Statistics",
            "Effective FG% (eFG%)",
            "MEAN(efg_pct)",
        ),
        (
            "ts_pct",
            "Rolling Offensive and Defensive Statistics",
            "True Shooting % (TS%)",
            "MEAN(ts_pct)",
        ),
        (
            "possessions_avg",
            "Rolling Offensive and Defensive Statistics",
            "average estimated possessions",
            "MEAN(possessions)",
        ),
        (
            "ortg",
            "Rolling Offensive and Defensive Statistics",
            "Offensive Rating (ORtg)",
            "MEAN(ortg)",
        ),
        (
            "drtg",
            "Rolling Offensive and Defensive Statistics",
            "Defensive Rating (DRtg)",
            "MEAN(drtg)",
        ),
        (
            "net_rtg",
            "Rolling Offensive and Defensive Statistics",
            "Net Rating (ORtg - DRtg)",
            "MEAN(net_rtg)",
        ),
    ]
    std_metrics = [
        ("pts_std", "Recent Team Form", "standard deviation of points scored", "STD(pts_for)"),
        (
            "pts_allowed_std",
            "Recent Team Form",
            "standard deviation of points conceded",
            "STD(pts_against)",
        ),
        ("margin_std", "Recent Team Form", "standard deviation of scoring margin", "STD(margin)"),
    ]
    for window in (5, 10):
        for side in ("home", "away"):
            _add(
                name=f"{side}_rolling_insufficient_{window}",
                category="Recent Team Form",
                description=(
                    f"{side.capitalize()} team has fewer than {window} "
                    "prior games in the season."
                ),
                formula=f"prior_games_count < {window}",
                source_columns="games.game_id",
                dtype="bool",
                lookback_window=f"Last {window} prior games (< game_date)",
                missing_value_strategy="Always non-null boolean.",
            )
            for m_key, cat, desc, formula in rolling_metrics:
                _add(
                    name=f"{side}_rolling_{m_key}_{window}",
                    category=cat,
                    description=f"{side.capitalize()} team rolling {window}-game {desc}.",
                    formula=f"ROLLING_{window}({formula}) after shift(1)",
                    source_columns="games, player_game_stats",
                    dtype="float64",
                    lookback_window=f"Last {window} prior games (min_periods=1, < game_date)",
                    missing_value_strategy=(
                        f"Uses available prior games (1..{window}) with min_periods=1; "
                        "NaN only on season opener (0 prior games)."
                    ),
                )
            for s_key, cat, desc, formula in std_metrics:
                _add(
                    name=f"{side}_rolling_{s_key}_{window}",
                    category=cat,
                    description=f"{side.capitalize()} team rolling {window}-game {desc}.",
                    formula=f"ROLLING_{window}({formula}, ddof=1) after shift(1)",
                    source_columns="games",
                    dtype="float64",
                    lookback_window=f"Last {window} prior games (min_periods=2, < game_date)",
                    missing_value_strategy="NaN when team has fewer than 2 prior games in season.",
                )

    ewma_metrics = [
        ("ewma_pts", "Recent Team Form", "EWMA (span=5) of points scored", "EWMA(pts_for, span=5)"),
        (
            "ewma_pts_allowed",
            "Recent Team Form",
            "EWMA (span=5) of points conceded",
            "EWMA(pts_against, span=5)",
        ),
        (
            "ewma_margin",
            "Recent Team Form",
            "EWMA (span=5) of scoring margin",
            "EWMA(margin, span=5)",
        ),
        ("ewma_win_pct", "Recent Team Form", "EWMA (span=5) of win indicator", "EWMA(win, span=5)"),
        (
            "ewma_efg_pct",
            "Rolling Offensive and Defensive Statistics",
            "EWMA (span=5) of Effective FG%",
            "EWMA(efg_pct, span=5)",
        ),
        (
            "ewma_ortg",
            "Rolling Offensive and Defensive Statistics",
            "EWMA (span=5) of Offensive Rating",
            "EWMA(ortg, span=5)",
        ),
        (
            "ewma_drtg",
            "Rolling Offensive and Defensive Statistics",
            "EWMA (span=5) of Defensive Rating",
            "EWMA(drtg, span=5)",
        ),
        (
            "ewma_net_rtg",
            "Rolling Offensive and Defensive Statistics",
            "EWMA (span=5) of Net Rating",
            "EWMA(net_rtg, span=5)",
        ),
    ]
    for side in ("home", "away"):
        for e_key, cat, desc, formula in ewma_metrics:
            _add(
                name=f"{side}_{e_key}",
                category=cat,
                description=f"{side.capitalize()} team {desc} (decay alpha=1/3).",
                formula=f"{formula} after shift(1)",
                source_columns="games, player_game_stats",
                dtype="float64",
                lookback_window="Exponentially weighted prior season games (span=5, < game_date)",
                missing_value_strategy="NaN only on season opener (0 prior games).",
            )

    # 5. Rest and Schedule-Related Features
    sched_team_metrics = [
        (
            "rest_days",
            "float64",
            "Days elapsed since previous game in season (capped at 14.0)",
            "MIN((game_date - prev_game_date).days, 14.0)",
            "NaN on season opener (flagged by is_season_opener).",
        ),
        (
            "is_season_opener",
            "bool",
            "True if first game of the season for the team",
            "prior_games_count == 0",
            "Always non-null boolean.",
        ),
        (
            "is_back_to_back",
            "bool",
            "True if team played on the previous calendar day (rest_days == 1)",
            "rest_days == 1.0",
            "False on season opener.",
        ),
        (
            "games_last_7d",
            "int64",
            "Number of games completed by team in the previous 7 calendar days",
            "COUNT(games in [game_date - 7d, game_date - 1d])",
            "0 on season opener.",
        ),
        (
            "games_last_14d",
            "int64",
            "Number of games completed by team in the previous 14 calendar days",
            "COUNT(games in [game_date - 14d, game_date - 1d])",
            "0 on season opener.",
        ),
        (
            "recent_home_games_10g",
            "int64",
            "Number of home games in team's previous up-to-10 games",
            "SUM(is_home over last 10 prior games)",
            "0 on season opener.",
        ),
        (
            "recent_away_games_10g",
            "int64",
            "Number of away games in team's previous up-to-10 games",
            "SUM(is_away over last 10 prior games)",
            "0 on season opener.",
        ),
        (
            "consecutive_home_games",
            "int64",
            "Streak of consecutive prior home games entering this game",
            "Streak count of prior home games",
            "0 on season opener or after road game.",
        ),
        (
            "consecutive_away_games",
            "int64",
            "Streak of consecutive prior road games entering this game",
            "Streak count of prior road games",
            "0 on season opener or after home game.",
        ),
    ]
    for side in ("home", "away"):
        for s_key, dtype, desc, formula, mv_note in sched_team_metrics:
            _add(
                name=f"{side}_{s_key}",
                category="Rest and Schedule-Related Features",
                description=f"{side.capitalize()} team: {desc}.",
                formula=formula,
                source_columns="games.game_date, games.home_team_id, games.away_team_id",
                dtype=dtype,
                lookback_window="Prior season schedule (< game_date)",
                missing_value_strategy=mv_note,
            )

    _add(
        name="both_teams_back_to_back",
        category="Rest and Schedule-Related Features",
        description=(
            "True when both home and away teams are playing on the "
            "second night of a back-to-back."
        ),
        formula="home_is_back_to_back AND away_is_back_to_back",
        source_columns="games.game_date",
        dtype="bool",
        lookback_window="Prior calendar day (< game_date)",
        missing_value_strategy="Always non-null boolean.",
    )
    _add(
        name="rest_days_diff",
        category="Rest and Schedule-Related Features",
        description="Home team rest days minus away team rest days.",
        formula="home_rest_days - away_rest_days",
        source_columns="games.game_date",
        dtype="float64",
        lookback_window="Prior game dates (< game_date)",
        missing_value_strategy="NaN if either team is playing its season opener.",
    )

    # 6. Head-to-Head Historical Performance
    h2h_specs = [
        (
            "h2h_prior_meetings",
            "int64",
            "Number of prior completed non-preseason games between the two franchises",
            "COUNT(prior H2H games < game_date)",
            "0 when teams have no prior meetings in lookback.",
        ),
        (
            "h2h_home_team_win_pct",
            "float64",
            "Current home franchise's win percentage in prior head-to-head meetings",
            "MEAN(current_home_team_won in prior H2H)",
            "NaN when h2h_prior_meetings == 0.",
        ),
        (
            "h2h_home_team_margin_avg",
            "float64",
            "Current home franchise's average point margin in prior head-to-head meetings",
            "MEAN(current_home_pts - current_away_pts in prior H2H)",
            "NaN when h2h_prior_meetings == 0.",
        ),
        (
            "h2h_total_pts_avg",
            "float64",
            "Average combined total points in prior head-to-head meetings",
            "MEAN(total_pts in prior H2H)",
            "NaN when h2h_prior_meetings == 0.",
        ),
    ]
    for name, dtype, desc, formula, mv_note in h2h_specs:
        _add(
            name=name,
            category="Head-to-Head Historical Performance",
            description=desc,
            formula=formula,
            source_columns="games.home_team_id, games.away_team_id, games.home_pts, games.away_pts",
            dtype=dtype,
            lookback_window="All prior completed non-preseason H2H games strictly before game_date",
            missing_value_strategy=mv_note,
        )

    # 7. Standings and Season Context
    standings_specs = [
        (
            "home_standings_games_played",
            "int64",
            "Home team cumulative games played from latest standings before game_date",
            "asof(standings.games_played, standings_date < game_date)",
            "0 before first season game.",
        ),
        (
            "home_standings_wins",
            "int64",
            "Home team cumulative wins from latest standings before game_date",
            "asof(standings.wins, standings_date < game_date)",
            "0 before first season game.",
        ),
        (
            "home_standings_losses",
            "int64",
            "Home team cumulative losses from latest standings before game_date",
            "asof(standings.losses, standings_date < game_date)",
            "0 before first season game.",
        ),
        (
            "home_standings_win_pct",
            "float64",
            "Home team winning percentage from latest standings before game_date",
            "asof(standings.win_pct, standings_date < game_date)",
            "NaN when no prior standings snapshot in season.",
        ),
        (
            "home_standings_home_win_pct",
            "float64",
            "Home team home-record win percentage from latest standings before game_date",
            "home_wins / (home_wins + home_losses)",
            "NaN when 0 prior home games in standings.",
        ),
        (
            "away_standings_games_played",
            "int64",
            "Away team cumulative games played from latest standings before game_date",
            "asof(standings.games_played, standings_date < game_date)",
            "0 before first season game.",
        ),
        (
            "away_standings_wins",
            "int64",
            "Away team cumulative wins from latest standings before game_date",
            "asof(standings.wins, standings_date < game_date)",
            "0 before first season game.",
        ),
        (
            "away_standings_losses",
            "int64",
            "Away team cumulative losses from latest standings before game_date",
            "asof(standings.losses, standings_date < game_date)",
            "0 before first season game.",
        ),
        (
            "away_standings_win_pct",
            "float64",
            "Away team winning percentage from latest standings before game_date",
            "asof(standings.win_pct, standings_date < game_date)",
            "NaN when no prior standings snapshot in season.",
        ),
        (
            "away_standings_road_win_pct",
            "float64",
            "Away team road-record win percentage from latest standings before game_date",
            "road_wins / (road_wins + road_losses)",
            "NaN when 0 prior road games in standings.",
        ),
        (
            "standings_win_pct_diff",
            "float64",
            "Home standings win percentage minus away standings win percentage",
            "home_standings_win_pct - away_standings_win_pct",
            "NaN if either team lacks a prior standings snapshot.",
        ),
        (
            "home_court_indicator",
            "int64",
            "Constant indicator (1) representing standard home-court assignment",
            "1",
            "Always 1.",
        ),
        (
            "is_playoff_game",
            "bool",
            "True if competition stage is Playoffs",
            "game_type == 'Playoffs'",
            "Always non-null boolean.",
        ),
        (
            "is_play_in_game",
            "bool",
            "True if competition stage is Play-In Tournament",
            "game_type == 'Play-In'",
            "Always non-null boolean.",
        ),
    ]
    for name, dtype, desc, formula, mv_note in standings_specs:
        _add(
            name=name,
            category="Standings and Season Context",
            description=desc,
            formula=formula,
            source_columns="standings.*, games.game_type",
            dtype=dtype,
            lookback_window=(
                "Latest daily standings snapshot strictly before game_date "
                "(standings_date < game_date)"
            ),
            missing_value_strategy=mv_note,
            leakage_prevention_note=(
                "Enforces strict backward as-of join (standings_date < game_date) "
                "matching team_id and season; rejects same-day and future snapshots."
            ),
        )

    # 8. Home-Away Matchup Differences
    diff_specs = [
        (
            "hist_pts_avg_diff",
            "home_hist_pts_avg - away_hist_pts_avg",
            "Historical points scored average differential",
        ),
        (
            "hist_pts_allowed_avg_diff",
            "home_hist_pts_allowed_avg - away_hist_pts_allowed_avg",
            "Historical points conceded average differential",
        ),
        (
            "hist_margin_avg_diff",
            "home_hist_margin_avg - away_hist_margin_avg",
            "Historical scoring margin average differential",
        ),
        (
            "hist_win_pct_diff",
            "home_hist_win_pct - away_hist_win_pct",
            "Historical winning percentage differential",
        ),
        (
            "hist_ortg_diff",
            "home_hist_ortg - away_hist_ortg",
            "Historical Offensive Rating differential",
        ),
        (
            "hist_drtg_diff",
            "home_hist_drtg - away_hist_drtg",
            "Historical Defensive Rating differential",
        ),
        (
            "hist_net_rtg_diff",
            "home_hist_net_rtg - away_hist_net_rtg",
            "Historical Net Rating differential",
        ),
        (
            "hist_efg_pct_diff",
            "home_hist_efg_pct - away_hist_efg_pct",
            "Historical Effective FG% differential",
        ),
        (
            "hist_ts_pct_diff",
            "home_hist_ts_pct - away_hist_ts_pct",
            "Historical True Shooting % differential",
        ),
        (
            "hist_reb_avg_diff",
            "home_hist_reb_avg - away_hist_reb_avg",
            "Historical rebounds average differential",
        ),
        (
            "hist_ast_avg_diff",
            "home_hist_ast_avg - away_hist_ast_avg",
            "Historical assists average differential",
        ),
        (
            "hist_tov_avg_diff",
            "home_hist_tov_avg - away_hist_tov_avg",
            "Historical turnovers average differential",
        ),
        (
            "venue_win_pct_diff",
            "home_venue_win_pct - away_venue_win_pct",
            "Home team home win % minus away team road win %",
        ),
        (
            "venue_margin_avg_diff",
            "home_venue_margin_avg - away_venue_margin_avg",
            "Home team home margin minus away team road margin",
        ),
    ]
    for window in (5, 10):
        for m_key, _, desc, _ in rolling_metrics:
            if m_key in ("ts_pct", "possessions_avg"):
                continue
            diff_specs.append(
                (
                    f"rolling_{m_key}_{window}_diff",
                    f"home_rolling_{m_key}_{window} - away_rolling_{m_key}_{window}",
                    f"Rolling {window}-game {desc} differential (home minus away)",
                )
            )
    ewma_diff_keys = [
        ("ewma_pts_diff", "home_ewma_pts - away_ewma_pts", "EWMA points scored differential"),
        (
            "ewma_pts_allowed_diff",
            "home_ewma_pts_allowed - away_ewma_pts_allowed",
            "EWMA points conceded differential",
        ),
        (
            "ewma_margin_diff",
            "home_ewma_margin - away_ewma_margin",
            "EWMA scoring margin differential",
        ),
        (
            "ewma_win_pct_diff",
            "home_ewma_win_pct - away_ewma_win_pct",
            "EWMA win percentage differential",
        ),
        (
            "ewma_net_rtg_diff",
            "home_ewma_net_rtg - away_ewma_net_rtg",
            "EWMA Net Rating differential",
        ),
        (
            "ewma_efg_pct_diff",
            "home_ewma_efg_pct - away_ewma_efg_pct",
            "EWMA Effective FG% differential",
        ),
        (
            "home_ortg_vs_away_drtg_10",
            "home_rolling_ortg_10 - away_rolling_drtg_10",
            "Home 10-game Offensive Rating minus Away 10-game Defensive Rating",
        ),
        (
            "away_ortg_vs_home_drtg_10",
            "away_rolling_ortg_10 - home_rolling_drtg_10",
            "Away 10-game Offensive Rating minus Home 10-game Defensive Rating",
        ),
    ]
    diff_specs.extend(ewma_diff_keys)

    for name, formula, desc in diff_specs:
        _add(
            name=name,
            category="Home-Away Matchup Differences",
            description=f"{desc} (Home minus Away).",
            formula=formula,
            source_columns="Engineered pregame home and away feature columns",
            dtype="float64",
            lookback_window="Same lookback window as constituent home and away pregame features",
            missing_value_strategy="NaN if either constituent home or away pregame feature is NaN.",
        )

    return specs


FEATURE_SPECIFICATIONS: tuple[FeatureSpec, ...] = tuple(build_feature_specifications())
FEATURE_COLUMN_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURE_SPECIFICATIONS)


def render_feature_strategy_markdown() -> str:
    """Render docs/feature_engineering_strategy.md."""
    lines = [
        "# Arcora Week 2 — Feature Engineering Strategy & Leakage Prevention Specification",
        "",
        "## 1. Objective",
        "",
        "Transform Arcora's validated canonical NBA datasets (`games.parquet`, "
        "`player_game_stats.parquet`, `standings.parquet`, `teams.parquet`) into a "
        "reproducible, leakage-free, game-level machine learning dataset "
        "(`data/features/nba_game_features.parquet`) using **strictly pre-tip-off information**.",
        "",
        "---",
        "",
        "## 2. Prediction Targets vs. Input Features",
        "",
        "Arcora models five game-level outcomes. Target columns are retained in the Parquet "
        "dataset for supervised training and evaluation, but are **strictly separated** from "
        "the model input feature matrix `X`:",
        "",
        "| Target Column | Task Type | Dtype | Formula | Description |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for t in PREDICTION_TARGETS:
        lines.append(
            f"| `{t.name}` | `{t.task_type}` | `{t.dtype}` | `{t.formula}` | {t.description} |"
        )

    lines.extend(
        [
            "",
            "### Forbidden Post-Game Columns (Never Permitted in `X`)",
            "Same-game box-score outcomes (`home_fg_pct`, `home_ft_pct`, `home_fg3_pct`, "
            "`home_ast`, `home_reb`, `away_fg_pct`, `away_ft_pct`, `away_fg3_pct`, "
            "`away_ast`, `away_reb`, and player-level counting stats from the current game) "
            "are post-game realizations and are explicitly banned from the input feature matrix.",
            "",
            "---",
            "",
            "## 3. Feature Categories Overview",
            "",
            "| Category | Feature Count | Purpose |",
            "| :--- | ---: | :--- |",
        ]
    )
    for cat in FEATURE_CATEGORIES:
        cnt = sum(1 for f in FEATURE_SPECIFICATIONS if f.category == cat)
        lines.append(f"| **{cat}** | {cnt} | Captures pregame {cat.lower()} signals. |")
    lines.append(f"| **Total Pregame Features** | **{len(FEATURE_SPECIFICATIONS)}** | — |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Strict Temporal & Leakage Prevention Rules",
            "",
            "1. **Strict Pregame Availability (`t < game_date`)**: Every feature for a game on "
            "`game_date` is computed exclusively from games completed on dates strictly prior "
            "to `game_date` (`prior_game_date < game_date`).",
            "2. **Chronological Sorting & `shift(1)`**: All team-game logs are sorted by "
            "`(team_id, season, game_date, game_id)` and shifted by 1 game (`shift(1)`) before "
            "computing expanding or rolling windows. If multiple games ever occur on the same "
            "`game_date` for a team, all games on date `d` receive the pre-date-`d` state so "
            "same-day results never leak across games.",
            "3. **Strict Backward As-Of Standings Join (`standings_date < game_date`)**: "
            "Standings features are joined via `pd.merge_asof(..., allow_exact_matches=False)` "
            "so same-day and end-of-season standings are never used.",
            "4. **Preseason Exclusion**: Preseason games (`game_type == 'Preseason'`, "
            "`1,675` games) are excluded from the ML dataset and from contributing to "
            "regular-season, play-in, or playoff feature histories (`24,848` eligible games).",
            "5. **Season Isolation & Partial `2022-23` Flagging**: Historical and rolling team "
            "statistics are computed within each NBA `season` so offseason roster turnover does "
            "not corrupt early-season metrics, and `is_partial_season` explicitly flags the "
            "truncated `2022-23` season (cutoff `2022-12-22`).",
            "6. **No Future Imputation**: Early-season `NaN` values (such as a team's season "
            "opener with 0 prior games) are never imputed with seasonal or dataset means; "
            "instead, explicit boolean indicators (`home_hist_insufficient_history`, "
            "`home_rolling_insufficient_5`, `home_is_season_opener`, etc.) are provided.",
            "",
            "---",
            "",
            "## 5. Final Dataset Structure (`data/features/nba_game_features.parquet`)",
            "",
            "- **Grain**: One row per eligible non-preseason completed NBA game (`24,848` rows).",
            f"- **Metadata Columns ({len(METADATA_COLUMNS)})**: "
            + ", ".join(f"`{c}`" for c in METADATA_COLUMNS),
            f"- **Engineered Input Feature Columns ({len(FEATURE_COLUMN_NAMES)})**: "
            "Grouped into 8 categories with `home_`, `away_`, `h2h_`, and `_diff` conventions.",
            f"- **Target Columns ({len(TARGET_COLUMNS)})**: "
            + ", ".join(f"`{c}`" for c in TARGET_COLUMNS),
            "",
        ]
    )
    return "\n".join(lines)


def render_feature_dictionary_markdown() -> str:
    """Render docs/feature_dictionary.md with full specifications for every feature."""
    lines = [
        "# Arcora NBA Game Feature Dictionary",
        "",
        f"This dictionary documents all **{len(FEATURE_SPECIFICATIONS)}** pregame input "
        f"features, **{len(METADATA_COLUMNS)}** metadata columns, and "
        f"**{len(TARGET_COLUMNS)}** target columns in "
        "`data/features/nba_game_features.parquet`.",
        "",
        "## 1. Identifier & Metadata Columns",
        "",
        "| Column | Dtype | Role | Description |",
        "| :--- | :--- | :--- | :--- |",
        "| `game_id` | `int64` | Primary Key | Unique NBA game identifier. |",
        "| `game_date` | `datetime64[ns]` | Metadata | Calendar date of game (EST). |",
        "| `season` | `int64` | Metadata | NBA season start year (`2003`–`2022`). |",
        "| `season_label` | `string` | Metadata | Formatted season string (`2003-04`–`2022-23`). |",
        "| `game_type` | `string` | Metadata | "
        "Competition stage (`Regular Season`, `Playoffs`, `Play-In`). |",
        "| `is_partial_season` | `bool` | Metadata | "
        "True for the partial `2022-23` season (`season == 2022`). |",
        "| `home_team_id` | `int64` | Foreign Key | Home team franchise ID (`teams.team_id`). |",
        "| `away_team_id` | `int64` | Foreign Key | Away team franchise ID (`teams.team_id`). |",
        "",
        "## 2. Prediction Target Columns (Excluded from Input Matrix `X`)",
        "",
        "| Target Column | Task Type | Dtype | Formula | Description |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for t in PREDICTION_TARGETS:
        lines.append(
            f"| `{t.name}` | `{t.task_type}` | `{t.dtype}` | `{t.formula}` | {t.description} |"
        )

    lines.extend(
        [
            "",
            "## 3. Engineered Pregame Input Features",
            "",
        ]
    )
    for cat in FEATURE_CATEGORIES:
        cat_specs = [f for f in FEATURE_SPECIFICATIONS if f.category == cat]
        lines.extend(
            [
                f"### {cat} ({len(cat_specs)} features)",
                "",
                "| Feature Name | Dtype | Formula | Source | Lookback Window | "
                "Pregame | Missing-Value Handling | Leakage Prevention | Description |",
                "| :--- | :--- | :--- | :--- | :--- | :---: | :--- | :--- | :--- |",
            ]
        )
        for f in cat_specs:
            pg = "Yes" if f.pregame_available else "No"
            lines.append(
                f"| `{f.name}` | `{f.dtype}` | `{f.formula}` | `{f.source_columns}` | "
                f"{f.lookback_window} | {pg} | {f.missing_value_strategy} | "
                f"{f.leakage_prevention_note} | {f.description} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_strategy_documentation(docs_dir: Path) -> dict[str, Path]:
    """Write feature_engineering_strategy.md and feature_dictionary.md."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = docs_dir / "feature_engineering_strategy.md"
    strategy_path.write_text(render_feature_strategy_markdown(), encoding="utf-8")

    dict_path = docs_dir / "feature_dictionary.md"
    dict_path.write_text(render_feature_dictionary_markdown(), encoding="utf-8")

    catalog_path = docs_dir / "predictive_features_catalog.md"
    catalog_path.write_text(render_feature_dictionary_markdown(), encoding="utf-8")

    return {
        "strategy": strategy_path,
        "feature_dictionary": dict_path,
        "predictive_features_catalog": catalog_path,
    }
