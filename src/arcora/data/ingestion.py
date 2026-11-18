"""Reproducible NBA data ingestion pipeline transforming raw CSVs to Parquet (Issue #5)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arcora.data.acquisition import verify_raw_files_present
from arcora.models.schema import (
    GAMES_ENTITY,
    PLAYER_GAME_STATS_ENTITY,
    PLAYERS_ENTITY,
    STANDINGS_ENTITY,
    TEAMS_ENTITY,
)


def _portable_path(path: Path, anchor: Path | None = None) -> str:
    """Return a project-relative POSIX path when under anchor, else path name/str."""
    resolved = path.resolve()
    if anchor is not None:
        try:
            return resolved.relative_to(anchor.resolve()).as_posix()
        except ValueError:
            pass
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd).as_posix()
    except ValueError:
        return resolved.as_posix()


@dataclass(frozen=True)
class EntityIngestionStats:
    """Row and cleaning statistics for a single canonical entity."""

    entity_name: str
    output_file: str
    raw_rows_read: int
    rows_written: int
    duplicates_removed: int
    malformed_or_missing_handled: int
    columns: list[str]
    corrections_applied: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionPipelineResult:
    """Summary of a full canonical ingestion pipeline execution."""

    executed_at_utc: str
    raw_data_dir: str
    processed_data_dir: str
    log_file: str
    entities: dict[str, EntityIngestionStats]

    def to_dict(self) -> dict[str, Any]:
        """Serialize ingestion result to a JSON-compatible dictionary."""
        return asdict(self)


def configure_ingestion_logger(log_path: Path) -> logging.Logger:
    """Configure a file and stream logger for the ingestion pipeline."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("arcora.ingestion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def format_season_label(season: int) -> str:
    """Format a 4-digit NBA season start year (e.g., 2022) into '2022-23'."""
    return f"{int(season)}-{(int(season) + 1) % 100:02d}"


def classify_game_type(game_id: int) -> str:
    """Classify an NBA game's stage from the leading digit of its GAME_ID."""
    prefix = str(int(game_id))[0]
    mapping = {
        "1": "Preseason",
        "2": "Regular Season",
        "3": "All-Star",
        "4": "Playoffs",
        "5": "Play-In",
    }
    return mapping.get(prefix, "Regular Season")


def parse_minutes_value(val: str | float | int | None) -> tuple[float, bool]:
    """Parse a raw MIN string into decimal minutes and a flag indicating if malformed.

    Returns:
        Tuple of (minutes_float >= 0.0, was_malformed).
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0, False
    text = str(val).strip()
    if not text:
        return 0.0, False

    try:
        if ":" in text:
            parts = text.split(":")
            if len(parts) != 2:
                return 0.0, True
            mins = float(parts[0])
            secs = float(parts[1])
            if mins < 0 or secs < 0 or secs >= 60:
                return 0.0, True
            return round(mins + (secs / 60.0), 4), False
        numeric = float(text)
        if numeric < 0:
            return 0.0, True
        return round(numeric, 4), False
    except (ValueError, TypeError):
        return 0.0, True


def transform_teams(
    teams_raw: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, EntityIngestionStats]:
    """Transform raw teams.csv into the canonical teams entity."""
    raw_count = int(len(teams_raw))
    df = teams_raw.copy()

    dups_before = int(df.duplicated(subset=["TEAM_ID"]).sum())
    if dups_before > 0:
        logger.warning("Removing %d duplicate TEAM_ID rows from teams.csv", dups_before)
        df = df.drop_duplicates(subset=["TEAM_ID"], keep="first")

    missing_arena = int(df["ARENACAPACITY"].isna().sum())
    if missing_arena > 0:
        logger.info(
            "Handling %d missing ARENACAPACITY values in teams.csv using nullable Int64",
            missing_arena,
        )

    city_clean = df["CITY"].astype(str).str.strip()
    nick_clean = df["NICKNAME"].astype(str).str.strip()

    out = pd.DataFrame(
        {
            "team_id": df["TEAM_ID"].astype("int64"),
            "league_id": df["LEAGUE_ID"].astype("int64").map(lambda x: f"{x:02d}").astype("string"),
            "abbreviation": df["ABBREVIATION"].astype(str).str.strip().str.upper().astype("string"),
            "nickname": nick_clean.astype("string"),
            "city": city_clean.astype("string"),
            "full_name": (city_clean + " " + nick_clean).astype("string"),
            "year_founded": df["YEARFOUNDED"].astype("int64"),
            "min_year": df["MIN_YEAR"].astype("int64"),
            "max_year": df["MAX_YEAR"].astype("int64"),
            "arena": df["ARENA"].astype(str).str.strip().astype("string"),
            "arena_capacity": pd.to_numeric(df["ARENACAPACITY"], errors="coerce")
            .round()
            .astype("Int64"),
            "owner": df["OWNER"].astype(str).str.strip().astype("string"),
            "general_manager": df["GENERALMANAGER"].astype(str).str.strip().astype("string"),
            "head_coach": df["HEADCOACH"].astype(str).str.strip().astype("string"),
            "dleague_affiliation": df["DLEAGUEAFFILIATION"]
            .astype(str)
            .str.strip()
            .astype("string"),
        }
    )
    out = out[TEAMS_ENTITY.column_names].sort_values("team_id").reset_index(drop=True)

    stats = EntityIngestionStats(
        entity_name="teams",
        output_file=TEAMS_ENTITY.storage_file,
        raw_rows_read=raw_count,
        rows_written=int(len(out)),
        duplicates_removed=dups_before,
        malformed_or_missing_handled=missing_arena,
        columns=list(out.columns),
        corrections_applied={"missing_arena_capacity_preserved_null": missing_arena},
    )
    logger.info("Transformed teams: %d raw -> %d clean rows", raw_count, len(out))
    return out, stats


def transform_games(
    games_raw: pd.DataFrame,
    valid_team_ids: set[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, EntityIngestionStats]:
    """Transform raw games.csv into the canonical games entity."""
    raw_count = int(len(games_raw))
    df = games_raw.copy()

    non_final_mask = df["GAME_STATUS_TEXT"].astype(str).str.strip() != "Final"
    non_final_count = int(non_final_mask.sum())
    if non_final_count > 0:
        logger.warning(
            "Dropping %d non-Final game rows (GAME_STATUS_TEXT != 'Final')",
            non_final_count,
        )
        df = df[~non_final_mask].copy()

    critical_score_cols = [
        "PTS_home",
        "PTS_away",
        "FG_PCT_home",
        "FT_PCT_home",
        "FG3_PCT_home",
        "AST_home",
        "REB_home",
        "FG_PCT_away",
        "FT_PCT_away",
        "FG3_PCT_away",
        "AST_away",
        "REB_away",
    ]
    missing_mask = df[critical_score_cols].isna().any(axis=1)
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        logger.warning(
            "Dropping %d unscored/incomplete game rows with missing box-score totals",
            missing_count,
        )
        df = df[~missing_mask].copy()

    invalid_teams_mask = (
        (~df["HOME_TEAM_ID"].isin(valid_team_ids))
        | (~df["VISITOR_TEAM_ID"].isin(valid_team_ids))
        | (df["HOME_TEAM_ID"] == df["VISITOR_TEAM_ID"])
    )
    invalid_teams_count = int(invalid_teams_mask.sum())
    if invalid_teams_count > 0:
        logger.warning("Dropping %d games with invalid or identical team IDs", invalid_teams_count)
        df = df[~invalid_teams_mask].copy()

    df = df.sort_values(["GAME_DATE_EST", "GAME_ID"]).reset_index(drop=True)
    dups_count = int(df.duplicated(subset=["GAME_ID"]).sum())
    if dups_count > 0:
        logger.warning("Removing %d duplicate GAME_ID rows from games.csv", dups_count)
        df = df.drop_duplicates(subset=["GAME_ID"], keep="first").copy()

    pct_raw_cols = [
        "FG_PCT_home",
        "FT_PCT_home",
        "FG3_PCT_home",
        "FG_PCT_away",
        "FT_PCT_away",
        "FG3_PCT_away",
    ]
    pct_clipped_count = 0
    for pcol in pct_raw_cols:
        s_num = df[pcol].astype("float64")
        pct_clipped_count += int(((s_num < 0.0) | (s_num > 1.0)).sum())
    if pct_clipped_count > 0:
        logger.warning(
            "Clipped %d out-of-range shooting percentage values to [0.0, 1.0] in games",
            pct_clipped_count,
        )

    home_pts = df["PTS_home"].round().astype("int64")
    away_pts = df["PTS_away"].round().astype("int64")
    seasons = df["SEASON"].astype("int64")
    game_ids = df["GAME_ID"].astype("int64")

    out = pd.DataFrame(
        {
            "game_id": game_ids,
            "game_date": pd.to_datetime(
                df["GAME_DATE_EST"].astype(str).str.strip(), format="%Y-%m-%d"
            ),
            "season": seasons,
            "season_label": seasons.map(format_season_label).astype("string"),
            "game_type": game_ids.map(classify_game_type).astype("string"),
            "game_status": df["GAME_STATUS_TEXT"].astype(str).str.strip().astype("string"),
            "home_team_id": df["HOME_TEAM_ID"].astype("int64"),
            "away_team_id": df["VISITOR_TEAM_ID"].astype("int64"),
            "home_pts": home_pts,
            "away_pts": away_pts,
            "total_pts": (home_pts + away_pts).astype("int64"),
            "point_margin": (home_pts - away_pts).astype("int64"),
            "home_win": (home_pts > away_pts).astype("bool"),
            "home_fg_pct": df["FG_PCT_home"].astype("float64").round(3).clip(0.0, 1.0),
            "home_ft_pct": df["FT_PCT_home"].astype("float64").round(3).clip(0.0, 1.0),
            "home_fg3_pct": df["FG3_PCT_home"].astype("float64").round(3).clip(0.0, 1.0),
            "home_ast": df["AST_home"].round().astype("int64"),
            "home_reb": df["REB_home"].round().astype("int64"),
            "away_fg_pct": df["FG_PCT_away"].astype("float64").round(3).clip(0.0, 1.0),
            "away_ft_pct": df["FT_PCT_away"].astype("float64").round(3).clip(0.0, 1.0),
            "away_fg3_pct": df["FG3_PCT_away"].astype("float64").round(3).clip(0.0, 1.0),
            "away_ast": df["AST_away"].round().astype("int64"),
            "away_reb": df["REB_away"].round().astype("int64"),
        }
    )
    out = out[GAMES_ENTITY.column_names].sort_values(["game_date", "game_id"]).reset_index(
        drop=True
    )

    stats = EntityIngestionStats(
        entity_name="games",
        output_file=GAMES_ENTITY.storage_file,
        raw_rows_read=raw_count,
        rows_written=int(len(out)),
        duplicates_removed=dups_count,
        malformed_or_missing_handled=non_final_count + missing_count + invalid_teams_count,
        columns=list(out.columns),
        corrections_applied={
            "non_final_games_dropped": non_final_count,
            "unscored_games_dropped": missing_count,
            "invalid_team_games_dropped": invalid_teams_count,
            "shooting_pct_clipped_to_unit_interval": pct_clipped_count,
        },
    )
    logger.info(
        "Transformed games: %d raw -> %d clean rows (removed %d duplicates, %d unscored)",
        raw_count,
        len(out),
        dups_count,
        missing_count,
    )
    return out, stats


def transform_player_game_stats(
    details_raw: pd.DataFrame,
    games_df: pd.DataFrame,
    valid_team_ids: set[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, EntityIngestionStats]:
    """Transform raw games_details.csv into the canonical player_game_stats entity."""
    raw_count = int(len(details_raw))
    df = details_raw.copy()

    valid_game_teams = set(
        zip(
            games_df["game_id"].astype(int).tolist(),
            games_df["home_team_id"].astype(int).tolist(),
            strict=False,
        )
    ) | set(
        zip(
            games_df["game_id"].astype(int).tolist(),
            games_df["away_team_id"].astype(int).tolist(),
            strict=False,
        )
    )
    pair_series = list(
        zip(
            df["GAME_ID"].astype(int).tolist(),
            df["TEAM_ID"].astype(int).tolist(),
            strict=False,
        )
    )
    fk_mask = pd.Series(
        [pair in valid_game_teams for pair in pair_series],
        index=df.index,
        dtype="bool",
    ) & df["TEAM_ID"].isin(valid_team_ids)
    orphan_count = int((~fk_mask).sum())
    if orphan_count > 0:
        logger.warning(
            "Filtering %d player_game_stats rows with unmapped (GAME_ID, TEAM_ID) pair",
            orphan_count,
        )
        df = df[fk_mask].copy()

    pk_cols = ["GAME_ID", "TEAM_ID", "PLAYER_ID"]
    dups_count = int(df.duplicated(subset=pk_cols).sum())
    if dups_count > 0:
        logger.warning(
            "Removing %d duplicate (GAME_ID, TEAM_ID, PLAYER_ID) rows from games_details.csv",
            dups_count,
        )
        df = df.drop_duplicates(subset=pk_cols, keep="first").copy()

    did_play = df["PTS"].notna()
    dnp_count = int((~did_play).sum())

    parsed_mins = [parse_minutes_value(v) for v in df["MIN"].tolist()]
    minutes_floats = [m for m, _ in parsed_mins]
    malformed_min_count = sum(1 for _, bad in parsed_mins if bad)
    if malformed_min_count > 0:
        logger.warning(
            "Normalized %d malformed or negative MIN values to 0.0 in player_game_stats",
            malformed_min_count,
        )

    minutes_series = pd.Series(minutes_floats, index=df.index, dtype="float64")
    minutes_series = minutes_series.where(did_play, 0.0)

    raw_pos = df["START_POSITION"].astype("string").str.strip().str.upper()
    valid_pos = raw_pos.where(raw_pos.isin(["F", "C", "G"]), None)
    is_starter = valid_pos.notna()

    raw_comment = df["COMMENT"].astype("string").str.strip()
    clean_comment = raw_comment.where(raw_comment.notna() & (raw_comment != ""), None)

    def _int_stat(col_name: str) -> pd.Series:
        return df[col_name].fillna(0.0).round().clip(lower=0).astype("int64")

    def _pct_stat(col_name: str) -> tuple[pd.Series, int]:
        raw_s = df[col_name].fillna(0.0).astype("float64")
        clipped_cnt = int(((raw_s < 0.0) | (raw_s > 1.0)).sum())
        return raw_s.round(3).clip(0.0, 1.0), clipped_cnt

    fgm = _int_stat("FGM")
    raw_fga = _int_stat("FGA")
    fga_fixed_cnt = int((raw_fga < fgm).sum())
    fga = pd.Series(np.maximum(raw_fga, fgm), index=df.index, dtype="int64")

    fg3m = _int_stat("FG3M")
    raw_fg3a = _int_stat("FG3A")
    fg3a_fixed_cnt = int((raw_fg3a < fg3m).sum())
    fg3a = pd.Series(np.maximum(raw_fg3a, fg3m), index=df.index, dtype="int64")

    ftm = _int_stat("FTM")
    raw_fta = _int_stat("FTA")
    fta_fixed_cnt = int((raw_fta < ftm).sum())
    fta = pd.Series(np.maximum(raw_fta, ftm), index=df.index, dtype="int64")

    fg_pct_s, fg_pct_clipped = _pct_stat("FG_PCT")
    fg3_pct_s, fg3_pct_clipped = _pct_stat("FG3_PCT")
    ft_pct_s, ft_pct_clipped = _pct_stat("FT_PCT")

    oreb = _int_stat("OREB")
    dreb = _int_stat("DREB")
    reb = _int_stat("REB")

    out = pd.DataFrame(
        {
            "game_id": df["GAME_ID"].astype("int64"),
            "team_id": df["TEAM_ID"].astype("int64"),
            "player_id": df["PLAYER_ID"].astype("int64"),
            "player_name": df["PLAYER_NAME"].astype(str).str.strip().astype("string"),
            "team_abbreviation": df["TEAM_ABBREVIATION"]
            .astype(str)
            .str.strip()
            .str.upper()
            .astype("string"),
            "team_city": df["TEAM_CITY"].astype(str).str.strip().astype("string"),
            "is_starter": is_starter.astype("bool"),
            "start_position": valid_pos.astype("string"),
            "did_play": did_play.astype("bool"),
            "comment": clean_comment.astype("string"),
            "minutes_played": minutes_series,
            "fgm": fgm,
            "fga": fga,
            "fg_pct": fg_pct_s,
            "fg3m": fg3m,
            "fg3a": fg3a,
            "fg3_pct": fg3_pct_s,
            "ftm": ftm,
            "fta": fta,
            "ft_pct": ft_pct_s,
            "oreb": oreb,
            "dreb": dreb,
            "reb": reb,
            "ast": _int_stat("AST"),
            "stl": _int_stat("STL"),
            "blk": _int_stat("BLK"),
            "turnovers": _int_stat("TO"),
            "pf": _int_stat("PF"),
            "pts": _int_stat("PTS"),
            "plus_minus": pd.to_numeric(df["PLUS_MINUS"], errors="coerce").astype("float64"),
        }
    )
    out = (
        out[PLAYER_GAME_STATS_ENTITY.column_names]
        .sort_values(["game_id", "team_id", "player_id"])
        .reset_index(drop=True)
    )

    corrections = {
        "orphan_rows_dropped": orphan_count,
        "dnp_rows_zero_filled": dnp_count,
        "malformed_minutes_reset_to_zero": malformed_min_count,
        "fga_raised_to_fgm": fga_fixed_cnt,
        "fg3a_raised_to_fg3m": fg3a_fixed_cnt,
        "fta_raised_to_ftm": fta_fixed_cnt,
        "shooting_pct_clipped_to_unit_interval": (
            fg_pct_clipped + fg3_pct_clipped + ft_pct_clipped
        ),
    }

    stats = EntityIngestionStats(
        entity_name="player_game_stats",
        output_file=PLAYER_GAME_STATS_ENTITY.storage_file,
        raw_rows_read=raw_count,
        rows_written=int(len(out)),
        duplicates_removed=dups_count,
        malformed_or_missing_handled=(
            dnp_count
            + malformed_min_count
            + orphan_count
            + fga_fixed_cnt
            + fg3a_fixed_cnt
            + fta_fixed_cnt
        ),
        columns=list(out.columns),
        corrections_applied=corrections,
    )
    logger.info(
        "Transformed player_game_stats: %d raw -> %d clean rows "
        "(removed %d duplicates, handled %d DNP rows, %d malformed MIN)",
        raw_count,
        len(out),
        dups_count,
        dnp_count,
        malformed_min_count,
    )
    return out, stats


def transform_players(
    players_raw: pd.DataFrame,
    player_stats_df: pd.DataFrame,
    games_df: pd.DataFrame,
    valid_team_ids: set[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, EntityIngestionStats]:
    """Build canonical players entity unifying players.csv and player_game_stats."""
    raw_count = int(len(players_raw))
    roster_df = players_raw.copy()
    roster_df = roster_df[roster_df["TEAM_ID"].isin(valid_team_ids)].copy()

    roster_records = pd.DataFrame(
        {
            "player_id": roster_df["PLAYER_ID"].astype("int64"),
            "player_name": roster_df["PLAYER_NAME"].astype(str).str.strip(),
            "team_id": roster_df["TEAM_ID"].astype("int64"),
            "season": roster_df["SEASON"].astype("int64"),
        }
    )

    game_seasons = games_df[["game_id", "season"]].copy()
    stats_with_season = player_stats_df[
        ["player_id", "player_name", "team_id", "game_id"]
    ].merge(game_seasons, on="game_id", how="left")

    box_records = pd.DataFrame(
        {
            "player_id": stats_with_season["player_id"].astype("int64"),
            "player_name": stats_with_season["player_name"].astype(str).str.strip(),
            "team_id": stats_with_season["team_id"].astype("int64"),
            "season": stats_with_season["season"].fillna(2003).astype("int64"),
        }
    )

    combined = pd.concat([roster_records, box_records], ignore_index=True)
    combined = combined.sort_values(["player_id", "season"]).reset_index(drop=True)

    grouped = combined.groupby("player_id", sort=True)
    first_season = grouped["season"].min().astype("int64")
    last_season = grouped["season"].max().astype("int64")
    latest_rows = combined.drop_duplicates(subset=["player_id"], keep="last").set_index(
        "player_id"
    )

    games_logged = (
        player_stats_df.groupby("player_id")["game_id"]
        .nunique()
        .reindex(first_season.index, fill_value=0)
        .astype("int64")
    )
    games_played = (
        player_stats_df[player_stats_df["did_play"]]
        .groupby("player_id")["game_id"]
        .nunique()
        .reindex(first_season.index, fill_value=0)
        .astype("int64")
    )

    out = pd.DataFrame(
        {
            "player_id": first_season.index.astype("int64").to_numpy(),
            "player_name": latest_rows["player_name"].astype("string").to_numpy(),
            "first_season": first_season.to_numpy(),
            "last_season": last_season.to_numpy(),
            "career_games_played": games_played.to_numpy(),
            "career_games_logged": games_logged.to_numpy(),
        }
    )
    out = out[PLAYERS_ENTITY.column_names].sort_values("player_id").reset_index(drop=True)

    roster_unique = int(roster_df["PLAYER_ID"].nunique())
    backfilled_players = int(len(out)) - roster_unique
    dups_collapsed = int(len(roster_df)) - roster_unique

    stats = EntityIngestionStats(
        entity_name="players",
        output_file=PLAYERS_ENTITY.storage_file,
        raw_rows_read=raw_count,
        rows_written=int(len(out)),
        duplicates_removed=dups_collapsed,
        malformed_or_missing_handled=backfilled_players,
        columns=list(out.columns),
        corrections_applied={
            "historical_players_backfilled_from_box_scores": backfilled_players,
        },
    )
    logger.info(
        "Transformed players: %d roster rows + box-score appearances -> %d unique players "
        "(backfilled %d historical players missing from players.csv)",
        raw_count,
        len(out),
        backfilled_players,
    )
    return out, stats


def _split_record_column(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split 'W-L' record strings into integer wins and losses Series."""
    parts = series.astype(str).str.strip().str.split("-", expand=True)
    wins = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype("int64")
    losses = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype("int64")
    return wins, losses


def transform_standings(
    ranking_raw: pd.DataFrame,
    valid_team_ids: set[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, EntityIngestionStats]:
    """Transform raw ranking.csv into the canonical standings entity."""
    raw_count = int(len(ranking_raw))
    df = ranking_raw.copy()

    invalid_team_mask = ~df["TEAM_ID"].isin(valid_team_ids)
    invalid_team_count = int(invalid_team_mask.sum())
    if invalid_team_count > 0:
        logger.warning(
            "Dropping %d standings rows with unmapped TEAM_ID",
            invalid_team_count,
        )
        df = df[~invalid_team_mask].copy()

    pk_raw = ["TEAM_ID", "SEASON_ID", "STANDINGSDATE"]
    dups_count = int(df.duplicated(subset=pk_raw).sum())
    if dups_count > 0:
        logger.warning(
            "Removing %d duplicate (TEAM_ID, SEASON_ID, STANDINGSDATE) rows from ranking.csv",
            dups_count,
        )
        df = df.drop_duplicates(subset=pk_raw, keep="first").copy()

    sid_str = df["SEASON_ID"].astype(str).str.strip()
    season_series = sid_str.str[-4:].astype("int64")
    season_type_series = (
        sid_str.str[0]
        .map({"1": "Preseason", "2": "Regular Season"})
        .fillna("Regular Season")
        .astype("string")
    )

    home_wins, home_losses = _split_record_column(df["HOME_RECORD"])
    road_wins, road_losses = _split_record_column(df["ROAD_RECORD"])

    raw_wpct = df["W_PCT"].astype("float64")
    wpct_clipped = int(((raw_wpct < 0.0) | (raw_wpct > 1.0)).sum())

    out = pd.DataFrame(
        {
            "team_id": df["TEAM_ID"].astype("int64"),
            "season": season_series,
            "season_label": season_series.map(format_season_label).astype("string"),
            "season_type": season_type_series,
            "standings_date": pd.to_datetime(
                df["STANDINGSDATE"].astype(str).str.strip(), format="%Y-%m-%d"
            ),
            "conference": df["CONFERENCE"].astype(str).str.strip().astype("string"),
            "team_name": df["TEAM"].astype(str).str.strip().astype("string"),
            "games_played": df["G"].astype("int64"),
            "wins": df["W"].astype("int64"),
            "losses": df["L"].astype("int64"),
            "win_pct": raw_wpct.round(3).clip(0.0, 1.0),
            "home_wins": home_wins,
            "home_losses": home_losses,
            "road_wins": road_wins,
            "road_losses": road_losses,
            "return_to_play": pd.to_numeric(df["RETURNTOPLAY"], errors="coerce").astype(
                "float64"
            ),
        }
    )
    out = (
        out[STANDINGS_ENTITY.column_names]
        .sort_values(["standings_date", "season", "season_type", "team_id"])
        .reset_index(drop=True)
    )

    stats = EntityIngestionStats(
        entity_name="standings",
        output_file=STANDINGS_ENTITY.storage_file,
        raw_rows_read=raw_count,
        rows_written=int(len(out)),
        duplicates_removed=dups_count,
        malformed_or_missing_handled=invalid_team_count + wpct_clipped,
        columns=list(out.columns),
        corrections_applied={
            "invalid_team_rows_dropped": invalid_team_count,
            "win_pct_clipped_to_unit_interval": wpct_clipped,
        },
    )
    logger.info(
        "Transformed standings: %d raw -> %d clean rows (removed %d duplicates)",
        raw_count,
        len(out),
        dups_count,
    )
    return out, stats


def export_parquet_idempotent(df: pd.DataFrame, output_path: Path) -> None:
    """Write a DataFrame to Parquet atomically so re-runs overwrite cleanly."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False, engine="pyarrow")
    tmp_path.replace(output_path)


def run_ingestion_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    logs_dir: Path,
    reports_dir: Path | None = None,
) -> IngestionPipelineResult:
    """Execute the end-to-end canonical NBA data ingestion pipeline."""
    log_file = logs_dir / "ingestion.log"
    logger = configure_ingestion_logger(log_file)
    logger.info("Starting Arcora NBA canonical data ingestion pipeline")

    present, missing = verify_raw_files_present(raw_dir)
    if not present:
        msg = f"Cannot run ingestion; missing raw files in {raw_dir}: {', '.join(missing)}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    processed_dir.mkdir(parents=True, exist_ok=True)

    teams_raw = pd.read_csv(raw_dir / "teams.csv", low_memory=False)
    games_raw = pd.read_csv(raw_dir / "games.csv", low_memory=False)
    details_raw = pd.read_csv(raw_dir / "games_details.csv", low_memory=False)
    players_raw = pd.read_csv(raw_dir / "players.csv", low_memory=False)
    ranking_raw = pd.read_csv(raw_dir / "ranking.csv", low_memory=False)

    teams_df, teams_stats = transform_teams(teams_raw, logger)
    valid_team_ids = set(teams_df["team_id"].astype(int).tolist())

    games_df, games_stats = transform_games(games_raw, valid_team_ids, logger)

    player_stats_df, pstats_stats = transform_player_game_stats(
        details_raw, games_df, valid_team_ids, logger
    )

    players_df, players_stats = transform_players(
        players_raw, player_stats_df, games_df, valid_team_ids, logger
    )

    standings_df, standings_stats = transform_standings(ranking_raw, valid_team_ids, logger)

    export_parquet_idempotent(teams_df, processed_dir / TEAMS_ENTITY.storage_file)
    export_parquet_idempotent(players_df, processed_dir / PLAYERS_ENTITY.storage_file)
    export_parquet_idempotent(games_df, processed_dir / GAMES_ENTITY.storage_file)
    export_parquet_idempotent(
        player_stats_df, processed_dir / PLAYER_GAME_STATS_ENTITY.storage_file
    )
    export_parquet_idempotent(standings_df, processed_dir / STANDINGS_ENTITY.storage_file)

    executed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    result = IngestionPipelineResult(
        executed_at_utc=executed_at,
        raw_data_dir=_portable_path(raw_dir),
        processed_data_dir=_portable_path(processed_dir),
        log_file=_portable_path(log_file),
        entities={
            "teams": teams_stats,
            "players": players_stats,
            "games": games_stats,
            "player_game_stats": pstats_stats,
            "standings": standings_stats,
        },
    )

    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = reports_dir / "ingestion_manifest.json"
        manifest_path.write_text(
            json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    logger.info("Ingestion pipeline completed successfully")
    return result
