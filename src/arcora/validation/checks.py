"""Automated NBA data quality and integrity validation checks (Issue #6)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from arcora.models.schema import (
    CANONICAL_ENTITIES,
    GAMES_ENTITY,
    PLAYER_GAME_STATS_ENTITY,
    PLAYERS_ENTITY,
    STANDINGS_ENTITY,
    TEAMS_ENTITY,
)


@dataclass(frozen=True)
class ValidationCheckResult:
    """Result of executing a single automated data quality rule."""

    rule_id: str
    rule_name: str
    entity: str
    passed: bool
    failed_records: int
    total_records: int
    message: str
    rule_description: str


@dataclass(frozen=True)
class DataQualitySummary:
    """Aggregate summary of all automated data quality checks."""

    evaluated_at_utc: str
    processed_data_dir: str
    overall_passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    entity_row_counts: dict[str, int]
    results: list[ValidationCheckResult]

    def to_dict(self) -> dict[str, Any]:
        """Serialize summary to a JSON-compatible dictionary."""
        return asdict(self)


def check_no_duplicate_primary_keys(
    df: pd.DataFrame,
    entity_name: str,
    pk_cols: tuple[str, ...],
    rule_id: str,
) -> ValidationCheckResult:
    """Verify that an entity contains no duplicate primary key combinations."""
    total = int(len(df))
    dup_mask = df.duplicated(subset=list(pk_cols), keep=False)
    dup_count = int(df.duplicated(subset=list(pk_cols)).sum())
    pk_label = ", ".join(pk_cols)
    if dup_count == 0:
        msg = f"All {total:,} records in '{entity_name}' have unique ({pk_label}) keys."
    else:
        sample_keys = df.loc[dup_mask, list(pk_cols)].head(3).to_dict(orient="records")
        msg = (
            f"Detected {dup_count:,} duplicate primary key records on ({pk_label}) in "
            f"'{entity_name}'; sample offending keys: {sample_keys}"
        )
    return ValidationCheckResult(
        rule_id=rule_id,
        rule_name=f"unique_pk_{entity_name}",
        entity=entity_name,
        passed=(dup_count == 0),
        failed_records=dup_count,
        total_records=total,
        message=msg,
        rule_description=(
            f"Primary key ({pk_label}) in `{entity_name}` must contain zero duplicates."
        ),
    )


def check_missing_critical_values(
    df: pd.DataFrame,
    entity_name: str,
    required_cols: list[str],
    rule_id: str,
) -> ValidationCheckResult:
    """Verify that no required/non-nullable columns contain null or NaN values."""
    total = int(len(df))
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return ValidationCheckResult(
            rule_id=rule_id,
            rule_name=f"non_null_critical_{entity_name}",
            entity=entity_name,
            passed=False,
            failed_records=total,
            total_records=total,
            message=f"Missing required columns in '{entity_name}': {missing_cols}",
            rule_description=(
                f"All non-nullable columns in `{entity_name}` must be present and non-null."
            ),
        )

    null_mask = df[required_cols].isna().any(axis=1)
    failed_cnt = int(null_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"Zero missing values across {len(required_cols)} critical columns in '{entity_name}'."
        )
    else:
        col_nulls = {c: int(df[c].isna().sum()) for c in required_cols if df[c].isna().any()}
        msg = (
            f"Detected {failed_cnt:,} rows with missing critical values in '{entity_name}': "
            f"{col_nulls}"
        )
    return ValidationCheckResult(
        rule_id=rule_id,
        rule_name=f"non_null_critical_{entity_name}",
        entity=entity_name,
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=f"Critical columns in `{entity_name}` must not contain null/NaN values.",
    )


def check_game_dates_and_seasons(games_df: pd.DataFrame) -> ValidationCheckResult:
    """Validate game_date timestamps, season integer range, and season_label format."""
    total = int(len(games_df))
    if not is_datetime64_any_dtype(games_df["game_date"]):
        dtype_str = str(games_df["game_date"].dtype)
        return ValidationCheckResult(
            rule_id="VAL-005",
            rule_name="valid_game_dates_and_seasons",
            entity="games",
            passed=False,
            failed_records=total,
            total_records=total,
            message=f"Column 'game_date' has invalid dtype '{dtype_str}' (expected datetime64).",
            rule_description=(
                "`game_date` must be valid timestamps within 2003-2023 and `season` must "
                "match `YYYY` (2003-2022) and `season_label` (`YYYY-YY`)."
            ),
        )

    dates = pd.to_datetime(games_df["game_date"], errors="coerce")
    seasons = games_df["season"]
    expected_labels = seasons.map(lambda s: f"{int(s)}-{(int(s) + 1) % 100:02d}")

    invalid_mask = (
        dates.isna()
        | (dates < pd.Timestamp("2003-01-01"))
        | (dates > pd.Timestamp("2023-12-31"))
        | (seasons < 2003)
        | (seasons > 2022)
        | (games_df["season_label"] != expected_labels)
        | ((dates.dt.year != seasons) & (dates.dt.year != seasons + 1))
    )
    failed_cnt = int(invalid_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} games have valid dates ({dates.min().date()} to {dates.max().date()}) "
            "and consistent season identifiers (2003 to 2022)."
        )
    else:
        bad_ids = games_df.loc[invalid_mask, "game_id"].head(5).tolist()
        msg = (
            f"Found {failed_cnt:,} games with invalid dates or inconsistent season identifiers; "
            f"sample game_ids: {bad_ids}"
        )
    return ValidationCheckResult(
        rule_id="VAL-005",
        rule_name="valid_game_dates_and_seasons",
        entity="games",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "`game_date` must be valid timestamps within the season window and `season` / "
            "`season_label` must be consistent."
        ),
    )


def check_distinct_home_and_away_teams(games_df: pd.DataFrame) -> ValidationCheckResult:
    """Ensure home_team_id and away_team_id are never identical in any game."""
    total = int(len(games_df))
    same_team_mask = games_df["home_team_id"] == games_df["away_team_id"]
    failed_cnt = int(same_team_mask.sum())
    if failed_cnt == 0:
        msg = f"All {total:,} games have distinct home_team_id and away_team_id values."
    else:
        bad_ids = games_df.loc[same_team_mask, "game_id"].head(5).tolist()
        msg = (
            f"Found {failed_cnt:,} games where home_team_id == away_team_id; "
            f"sample game_ids: {bad_ids}"
        )
    return ValidationCheckResult(
        rule_id="VAL-006",
        rule_name="distinct_home_and_away_teams",
        entity="games",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description="In `games`, `home_team_id` must never equal `away_team_id`.",
    )


def check_game_scores_and_numeric_types(games_df: pd.DataFrame) -> ValidationCheckResult:
    """Validate score columns, shooting percentages, derived totals, and numeric dtypes."""
    total = int(len(games_df))
    int_cols = [
        "game_id",
        "season",
        "home_team_id",
        "away_team_id",
        "home_pts",
        "away_pts",
        "total_pts",
        "point_margin",
        "home_ast",
        "home_reb",
        "away_ast",
        "away_reb",
    ]
    pct_cols = [
        "home_fg_pct",
        "home_ft_pct",
        "home_fg3_pct",
        "away_fg_pct",
        "away_ft_pct",
        "away_fg3_pct",
    ]

    bad_dtypes = [c for c in int_cols if not is_integer_dtype(games_df[c])]
    bad_dtypes.extend(c for c in pct_cols if not is_numeric_dtype(games_df[c]))
    if not is_bool_dtype(games_df["home_win"]):
        bad_dtypes.append("home_win")

    if bad_dtypes:
        return ValidationCheckResult(
            rule_id="VAL-007",
            rule_name="valid_game_scores_and_types",
            entity="games",
            passed=False,
            failed_records=total,
            total_records=total,
            message=f"Invalid column data types in 'games': {bad_dtypes}",
            rule_description=(
                "Score columns must be positive integers, percentages in [0, 1], and "
                "`total_pts`, `point_margin`, and `home_win` mathematically consistent."
            ),
        )

    pct_out_of_bounds = (games_df[pct_cols] < 0.0).any(axis=1) | (
        games_df[pct_cols] > 1.0
    ).any(axis=1)
    invalid_score_mask = (
        (games_df["home_pts"] <= 0)
        | (games_df["away_pts"] <= 0)
        | (games_df["total_pts"] != games_df["home_pts"] + games_df["away_pts"])
        | (games_df["point_margin"] != games_df["home_pts"] - games_df["away_pts"])
        | (games_df["home_win"] != (games_df["home_pts"] > games_df["away_pts"]))
        | (games_df["home_ast"] < 0)
        | (games_df["home_reb"] < 0)
        | (games_df["away_ast"] < 0)
        | (games_df["away_reb"] < 0)
        | pct_out_of_bounds
    )
    failed_cnt = int(invalid_score_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} games have valid integer scores, percentages in [0, 1], "
            "and consistent total_pts, point_margin, and home_win values."
        )
    else:
        bad_ids = games_df.loc[invalid_score_mask, "game_id"].head(5).tolist()
        msg = (
            f"Found {failed_cnt:,} games with invalid scores, percentages, or derived outcome "
            f"inconsistencies; sample game_ids: {bad_ids}"
        )
    return ValidationCheckResult(
        rule_id="VAL-007",
        rule_name="valid_game_scores_and_types",
        entity="games",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "Score columns must be positive integers, percentages in [0, 1], and "
            "`total_pts`, `point_margin`, and `home_win` mathematically consistent."
        ),
    )


def check_player_stats_numeric_integrity(pstats_df: pd.DataFrame) -> ValidationCheckResult:
    """Validate player_game_stats numeric bounds, made <= attempted, and non-negative minutes."""
    total = int(len(pstats_df))
    non_neg_cols = [
        "minutes_played",
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
    ]
    pct_cols = ["fg_pct", "fg3_pct", "ft_pct"]

    invalid_mask = (
        (pstats_df[non_neg_cols] < 0).any(axis=1)
        | (pstats_df[pct_cols] < 0.0).any(axis=1)
        | (pstats_df[pct_cols] > 1.0).any(axis=1)
        | (pstats_df["fgm"] > pstats_df["fga"])
        | (pstats_df["fg3m"] > pstats_df["fg3a"])
        | (pstats_df["ftm"] > pstats_df["fta"])
    )
    failed_cnt = int(invalid_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} player_game_stats records satisfy non-negativity, percentage "
            "bounds [0, 1], and made <= attempted shooting constraints."
        )
    else:
        bad_rows = (
            pstats_df.loc[invalid_mask, ["game_id", "player_id"]].head(3).to_dict(orient="records")
        )
        msg = (
            f"Found {failed_cnt:,} player_game_stats rows violating numeric or shooting "
            f"constraints; sample rows: {bad_rows}"
        )
    return ValidationCheckResult(
        rule_id="VAL-008",
        rule_name="valid_player_stats_numeric_bounds",
        entity="player_game_stats",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "Player counting stats and `minutes_played` must be >= 0, percentages in [0, 1], "
            "and made shots <= attempted shots."
        ),
    )


def check_foreign_key_reference(
    source_df: pd.DataFrame,
    source_entity: str,
    source_col: str,
    target_df: pd.DataFrame,
    target_entity: str,
    target_col: str,
    rule_id: str,
) -> ValidationCheckResult:
    """Validate that every foreign key in source_df[source_col] exists in target_df[target_col]."""
    total = int(len(source_df))
    valid_targets = set(target_df[target_col].dropna().tolist())
    unmatched_mask = ~source_df[source_col].isin(valid_targets)
    failed_cnt = int(unmatched_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"100% of '{source_entity}.{source_col}' values ({total:,} rows) reference valid "
            f"'{target_entity}.{target_col}' keys."
        )
    else:
        sample_bad = source_df.loc[unmatched_mask, source_col].unique()[:5].tolist()
        msg = (
            f"Found {failed_cnt:,} orphan rows in '{source_entity}.{source_col}' not found in "
            f"'{target_entity}.{target_col}'; sample invalid IDs: {sample_bad}"
        )
    return ValidationCheckResult(
        rule_id=rule_id,
        rule_name=f"fk_{source_entity}_{source_col}_to_{target_entity}",
        entity=source_entity,
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            f"Foreign key `{source_entity}.{source_col}` must reference an existing "
            f"`{target_entity}.{target_col}`."
        ),
    )


def check_player_game_team_matches_game_participants(
    pstats_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> ValidationCheckResult:
    """Verify a player's team_id in a game is either the game's home_team_id or away_team_id."""
    total = int(len(pstats_df))
    game_teams = games_df[["game_id", "home_team_id", "away_team_id"]]
    merged = pstats_df[["game_id", "team_id", "player_id"]].merge(
        game_teams, on="game_id", how="left"
    )
    invalid_mask = (merged["team_id"] != merged["home_team_id"]) & (
        merged["team_id"] != merged["away_team_id"]
    )
    failed_cnt = int(invalid_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} player_game_stats rows belong to either the home or away team "
            "of their referenced game."
        )
    else:
        sample_bad = (
            merged.loc[invalid_mask, ["game_id", "team_id", "player_id"]]
            .head(3)
            .to_dict(orient="records")
        )
        msg = (
            f"Found {failed_cnt:,} player_game_stats rows where team_id is neither the "
            f"home_team_id nor away_team_id of game_id; sample: {sample_bad}"
        )
    return ValidationCheckResult(
        rule_id="VAL-015",
        rule_name="player_team_matches_game_participants",
        entity="player_game_stats",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "Every `player_game_stats.team_id` must match either `home_team_id` or "
            "`away_team_id` of the corresponding `game_id`."
        ),
    )


def check_players_career_game_counts(players_df: pd.DataFrame) -> ValidationCheckResult:
    """Verify career_games_played <= career_games_logged and non-negative."""
    total = int(len(players_df))
    invalid_mask = (
        (players_df["career_games_played"] < 0)
        | (players_df["career_games_logged"] < 0)
        | (players_df["career_games_played"] > players_df["career_games_logged"])
        | (players_df["last_season"] < players_df["first_season"])
    )
    failed_cnt = int(invalid_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} players have valid season spans and "
            "0 <= career_games_played <= career_games_logged."
        )
    else:
        bad_ids = players_df.loc[invalid_mask, "player_id"].head(5).tolist()
        msg = (
            f"Found {failed_cnt:,} players with invalid season span or career game counts; "
            f"sample player_ids: {bad_ids}"
        )
    return ValidationCheckResult(
        rule_id="VAL-019",
        rule_name="valid_player_career_counts_and_seasons",
        entity="players",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "Players must have `last_season >= first_season` and "
            "`0 <= career_games_played <= career_games_logged`."
        ),
    )


def check_game_status_and_type(games_df: pd.DataFrame) -> ValidationCheckResult:
    """Verify all canonical games have game_status == 'Final' and a valid game_type."""
    total = int(len(games_df))
    valid_types = {"Preseason", "Regular Season", "All-Star", "Playoffs", "Play-In"}
    invalid_mask = (games_df["game_status"] != "Final") | (
        ~games_df["game_type"].isin(valid_types)
    )
    failed_cnt = int(invalid_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} games have game_status == 'Final' and valid game_type classifications."
        )
    else:
        bad_ids = games_df.loc[invalid_mask, "game_id"].head(5).tolist()
        msg = (
            f"Found {failed_cnt:,} games with non-Final status or invalid game_type; "
            f"sample game_ids: {bad_ids}"
        )
    return ValidationCheckResult(
        rule_id="VAL-020",
        rule_name="valid_game_status_and_type",
        entity="games",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "All canonical `games` must have `game_status == 'Final'` and a valid `game_type`."
        ),
    )


def check_player_dnp_and_rebound_consistency(pstats_df: pd.DataFrame) -> ValidationCheckResult:
    """Verify DNP rows have zero minutes/points and reb == oreb + dreb across all rows."""
    total = int(len(pstats_df))
    dnp_violation = (~pstats_df["did_play"]) & (
        (pstats_df["minutes_played"] != 0.0) | (pstats_df["pts"] != 0)
    )
    reb_violation = pstats_df["reb"] != (pstats_df["oreb"] + pstats_df["dreb"])
    invalid_mask = dnp_violation | reb_violation
    failed_cnt = int(invalid_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} player_game_stats rows satisfy DNP zero-stat consistency and "
            "reb == oreb + dreb."
        )
    else:
        bad_rows = (
            pstats_df.loc[invalid_mask, ["game_id", "player_id"]].head(3).to_dict(orient="records")
        )
        msg = (
            f"Found {failed_cnt:,} player_game_stats rows failing DNP or rebound consistency; "
            f"sample: {bad_rows}"
        )
    return ValidationCheckResult(
        rule_id="VAL-021",
        rule_name="player_dnp_and_rebound_consistency",
        entity="player_game_stats",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "DNP rows (`did_play == False`) must have 0 minutes/points, and `reb == oreb + dreb`."
        ),
    )


def check_standings_integrity(standings_df: pd.DataFrame) -> ValidationCheckResult:
    """Verify standings games_played == wins + losses, win_pct in [0, 1], and valid labels."""
    total = int(len(standings_df))
    invalid_mask = (
        (standings_df["games_played"] != standings_df["wins"] + standings_df["losses"])
        | (standings_df["wins"] < 0)
        | (standings_df["losses"] < 0)
        | (standings_df["win_pct"] < 0.0)
        | (standings_df["win_pct"] > 1.0)
        | (~standings_df["conference"].isin(["East", "West"]))
        | (~standings_df["season_type"].isin(["Preseason", "Regular Season"]))
    )
    failed_cnt = int(invalid_mask.sum())
    if failed_cnt == 0:
        msg = (
            f"All {total:,} standings rows satisfy games_played == wins + losses, "
            "win_pct in [0, 1], and valid conference/season_type labels."
        )
    else:
        bad_rows = (
            standings_df.loc[invalid_mask, ["team_id", "standings_date"]]
            .head(3)
            .to_dict(orient="records")
        )
        msg = (
            f"Found {failed_cnt:,} standings rows failing record or domain checks; "
            f"sample: {bad_rows}"
        )
    return ValidationCheckResult(
        rule_id="VAL-025",
        rule_name="valid_standings_records_and_domains",
        entity="standings",
        passed=(failed_cnt == 0),
        failed_records=failed_cnt,
        total_records=total,
        message=msg,
        rule_description=(
            "Standings must satisfy `games_played == wins + losses`, `win_pct` in [0, 1], "
            "and valid `conference` and `season_type`."
        ),
    )


def run_validation_suite_on_frames(
    teams_df: pd.DataFrame,
    players_df: pd.DataFrame,
    games_df: pd.DataFrame,
    pstats_df: pd.DataFrame,
    standings_df: pd.DataFrame | None = None,
    processed_data_dir: str = "data/processed",
) -> DataQualitySummary:
    """Execute all automated validation rules on the four canonical DataFrames."""
    results: list[ValidationCheckResult] = [
        # Primary Key Uniqueness Checks (VAL-001 to VAL-004)
        check_no_duplicate_primary_keys(
            games_df, "games", GAMES_ENTITY.primary_key, "VAL-001"
        ),
        check_no_duplicate_primary_keys(
            teams_df, "teams", TEAMS_ENTITY.primary_key, "VAL-002"
        ),
        check_no_duplicate_primary_keys(
            players_df, "players", PLAYERS_ENTITY.primary_key, "VAL-003"
        ),
        check_no_duplicate_primary_keys(
            pstats_df,
            "player_game_stats",
            PLAYER_GAME_STATS_ENTITY.primary_key,
            "VAL-004",
        ),
        # Domain, Date, Season, Team, and Numeric Integrity Checks (VAL-005 to VAL-008)
        check_game_dates_and_seasons(games_df),
        check_distinct_home_and_away_teams(games_df),
        check_game_scores_and_numeric_types(games_df),
        check_player_stats_numeric_integrity(pstats_df),
        # Missing Critical Values Checks (VAL-009 to VAL-012)
        check_missing_critical_values(
            games_df, "games", GAMES_ENTITY.non_nullable_columns, "VAL-009"
        ),
        check_missing_critical_values(
            teams_df, "teams", TEAMS_ENTITY.non_nullable_columns, "VAL-010"
        ),
        check_missing_critical_values(
            players_df, "players", PLAYERS_ENTITY.non_nullable_columns, "VAL-011"
        ),
        check_missing_critical_values(
            pstats_df,
            "player_game_stats",
            PLAYER_GAME_STATS_ENTITY.non_nullable_columns,
            "VAL-012",
        ),
        # Foreign Key & Entity Reference Checks (VAL-013 to VAL-018)
        check_foreign_key_reference(
            games_df, "games", "home_team_id", teams_df, "teams", "team_id", "VAL-013"
        ),
        check_foreign_key_reference(
            games_df, "games", "away_team_id", teams_df, "teams", "team_id", "VAL-014"
        ),
        check_player_game_team_matches_game_participants(pstats_df, games_df),
        check_foreign_key_reference(
            pstats_df,
            "player_game_stats",
            "game_id",
            games_df,
            "games",
            "game_id",
            "VAL-016",
        ),
        check_foreign_key_reference(
            pstats_df,
            "player_game_stats",
            "team_id",
            teams_df,
            "teams",
            "team_id",
            "VAL-017",
        ),
        check_foreign_key_reference(
            pstats_df,
            "player_game_stats",
            "player_id",
            players_df,
            "players",
            "player_id",
            "VAL-018",
        ),
        check_players_career_game_counts(players_df),
        check_game_status_and_type(games_df),
        check_player_dnp_and_rebound_consistency(pstats_df),
    ]

    entity_counts: dict[str, int] = {
        "teams": int(len(teams_df)),
        "players": int(len(players_df)),
        "games": int(len(games_df)),
        "player_game_stats": int(len(pstats_df)),
    }

    if standings_df is not None:
        results.extend(
            [
                check_no_duplicate_primary_keys(
                    standings_df,
                    "standings",
                    STANDINGS_ENTITY.primary_key,
                    "VAL-022",
                ),
                check_missing_critical_values(
                    standings_df,
                    "standings",
                    STANDINGS_ENTITY.non_nullable_columns,
                    "VAL-023",
                ),
                check_foreign_key_reference(
                    standings_df,
                    "standings",
                    "team_id",
                    teams_df,
                    "teams",
                    "team_id",
                    "VAL-024",
                ),
                check_standings_integrity(standings_df),
            ]
        )
        entity_counts["standings"] = int(len(standings_df))

    passed_cnt = sum(1 for r in results if r.passed)
    failed_cnt = len(results) - passed_cnt
    evaluated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    return DataQualitySummary(
        evaluated_at_utc=evaluated_at,
        processed_data_dir=processed_data_dir,
        overall_passed=(failed_cnt == 0),
        total_checks=len(results),
        passed_checks=passed_cnt,
        failed_checks=failed_cnt,
        entity_row_counts=entity_counts,
        results=results,
    )


def run_validation_suite(processed_dir: Path) -> DataQualitySummary:
    """Load processed canonical Parquet datasets and execute all validation checks."""
    for entity_name, spec in CANONICAL_ENTITIES.items():
        fpath = processed_dir / spec.storage_file
        if not fpath.is_file():
            raise FileNotFoundError(
                f"Missing processed Parquet file for '{entity_name}': {fpath}"
            )

    teams_df = pd.read_parquet(processed_dir / TEAMS_ENTITY.storage_file)
    players_df = pd.read_parquet(processed_dir / PLAYERS_ENTITY.storage_file)
    games_df = pd.read_parquet(processed_dir / GAMES_ENTITY.storage_file)
    pstats_df = pd.read_parquet(processed_dir / PLAYER_GAME_STATS_ENTITY.storage_file)
    standings_df = pd.read_parquet(processed_dir / STANDINGS_ENTITY.storage_file)

    try:
        rel_dir = processed_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_dir = processed_dir.as_posix()

    return run_validation_suite_on_frames(
        teams_df=teams_df,
        players_df=players_df,
        games_df=games_df,
        pstats_df=pstats_df,
        standings_df=standings_df,
        processed_data_dir=rel_dir,
    )


def render_data_quality_report_markdown(summary: DataQualitySummary) -> str:
    """Render a comprehensive markdown Data Quality Report from validation results."""
    status_str = "PASSED" if summary.overall_passed else "FAILED"
    lines = [
        "# Arcora Automated NBA Data Quality Report",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Overall Validation Status** | **{status_str}** |",
        f"| **Evaluated At (UTC)** | `{summary.evaluated_at_utc}` |",
        f"| **Total Validation Rules Executed** | {summary.total_checks} |",
        f"| **Rules Passed** | {summary.passed_checks} |",
        f"| **Rules Failed** | {summary.failed_checks} |",
        f"| **`teams` Validated Rows** | {summary.entity_row_counts['teams']:,} |",
        f"| **`players` Validated Rows** | {summary.entity_row_counts['players']:,} |",
        f"| **`games` Validated Rows** | {summary.entity_row_counts['games']:,} |",
        f"| **`player_game_stats` Validated Rows** | "
        f"{summary.entity_row_counts['player_game_stats']:,} |",
        f"| **`standings` Validated Rows** | {summary.entity_row_counts.get('standings', 0):,} |",
        "",
        "---",
        "",
        "## 2. Automated Validation Rule Results",
        "",
        "| Rule ID | Entity | Rule Name | Status | Failed / Total | Validation Result Details |",
        "| :---: | :--- | :--- | :---: | ---: | :--- |",
    ]

    for res in summary.results:
        badge = "PASS" if res.passed else "FAIL"
        lines.append(
            f"| `{res.rule_id}` | `{res.entity}` | `{res.rule_name}` | **{badge}** | "
            f"{res.failed_records:,} / {res.total_records:,} | {res.message} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Documented Validation Rules & Catalog",
            "",
            "| Rule ID | Entity | Rule Specification |",
            "| :---: | :--- | :--- |",
        ]
    )
    for res in summary.results:
        lines.append(f"| `{res.rule_id}` | `{res.entity}` | {res.rule_description} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Detected Raw Issues Resolved Prior to Validation",
            "",
            "- **Duplicate Game IDs (`VAL-001`)**: 29 duplicate `GAME_ID` records in raw "
            "`games.csv` (caused by floating-point rounding discrepancies in December 2020 "
            "games) were deduplicated during canonical ingestion.",
            "- **Unscored 2003 Preseason Games (`VAL-007`, `VAL-009`)**: 99 rows in raw "
            "`games.csv` with all-null box scores were removed so that 100% of canonical "
            "`games` rows have valid positive integer scores.",
            "- **Duplicate Player Box-Score Keys (`VAL-004`)**: 289 duplicate `(GAME_ID, TEAM_ID, "
            "PLAYER_ID)` rows in raw `games_details.csv` were deduplicated.",
            "- **Malformed/Negative Minutes Played (`VAL-008`)**: 12 negative `MIN` strings (e.g. "
            "`'-9'`) and 109,690 DNP null rows in `games_details.csv` were standardized to "
            "`minutes_played >= 0.0` with an explicit `did_play` boolean indicator.",
            "- **Unmapped Historical Players (`VAL-018`)**: 918 distinct `PLAYER_ID`s present in "
            "`games_details.csv` (2003–2008 and 2020–2022) but absent from `players.csv` (which "
            "only covered 2009–2019) were unified into canonical `players.parquet`, achieving "
            "100% foreign-key referential integrity.",
            "",
        ]
    )
    return "\n".join(lines)


def write_validation_artifacts(
    summary: DataQualitySummary,
    reports_dir: Path,
) -> dict[str, Path]:
    """Write data_quality_report.md and data_quality_summary.json."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    md_path = reports_dir / "data_quality_report.md"
    md_path.write_text(render_data_quality_report_markdown(summary), encoding="utf-8")

    json_path = reports_dir / "data_quality_summary.json"
    json_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {
        "report_markdown": md_path,
        "summary_json": json_path,
    }
