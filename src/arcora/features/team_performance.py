"""Historical team performance features using strictly pregame data (Issue #10)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ELIGIBLE_GAME_TYPES: frozenset[str] = frozenset({"Regular Season", "Playoffs", "Play-In"})
MIN_HISTORY_GAMES: int = 5


def filter_eligible_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to completed non-preseason games sorted chronologically."""
    df = games_df.copy()
    if "game_status" in df.columns:
        df = df[df["game_status"] == "Final"].copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(ELIGIBLE_GAME_TYPES)].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def build_team_game_logs(
    games_df: pd.DataFrame,
    pstats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construct a chronological team-game log (two rows per game: home and away).

    Calculates single-game team box-score totals and advanced metrics:
    - Estimated possessions: ``FGA + 0.44 * FTA - OREB + TOV``
    - Effective FG% (eFG%): ``(FGM + 0.5 * FG3M) / FGA``
    - True Shooting % (TS%): ``PTS / (2 * (FGA + 0.44 * FTA))``
    - Turnover Rate: ``TOV / Possessions``
    - Offensive Rating (ORtg): ``100 * PTS / Possessions``
    - Defensive Rating (DRtg): ``100 * Opp_PTS / Opp_Possessions``
    - Net Rating: ``ORtg - DRtg``
    """
    elig = filter_eligible_games(games_df)

    box_cols = [
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
    ]

    if pstats_df is not None and len(pstats_df) > 0:
        active = pstats_df
        if "did_play" in pstats_df.columns:
            active = pstats_df[pstats_df["did_play"]]
        team_box = (
            active.groupby(["game_id", "team_id"], as_index=False)[box_cols]
            .sum()
            .rename(columns={"turnovers": "tov"})
        )
    else:
        team_box = pd.DataFrame(
            columns=["game_id", "team_id"] + [c if c != "turnovers" else "tov" for c in box_cols]
        )

    home_log = pd.DataFrame(
        {
            "game_id": elig["game_id"].astype("int64"),
            "game_date": elig["game_date"],
            "season": elig["season"].astype("int64"),
            "game_type": elig["game_type"].astype("string"),
            "team_id": elig["home_team_id"].astype("int64"),
            "opponent_id": elig["away_team_id"].astype("int64"),
            "is_home": True,
            "pts_for": elig["home_pts"].astype("float64"),
            "pts_against": elig["away_pts"].astype("float64"),
            "total_pts": elig["total_pts"].astype("float64"),
            "margin": (elig["home_pts"] - elig["away_pts"]).astype("float64"),
            "win": elig["home_win"].astype("float64"),
            "game_fg_pct": elig["home_fg_pct"].astype("float64"),
            "game_fg3_pct": elig["home_fg3_pct"].astype("float64"),
            "game_ft_pct": elig["home_ft_pct"].astype("float64"),
            "game_reb": elig["home_reb"].astype("float64"),
            "game_ast": elig["home_ast"].astype("float64"),
        }
    )

    away_log = pd.DataFrame(
        {
            "game_id": elig["game_id"].astype("int64"),
            "game_date": elig["game_date"],
            "season": elig["season"].astype("int64"),
            "game_type": elig["game_type"].astype("string"),
            "team_id": elig["away_team_id"].astype("int64"),
            "opponent_id": elig["home_team_id"].astype("int64"),
            "is_home": False,
            "pts_for": elig["away_pts"].astype("float64"),
            "pts_against": elig["home_pts"].astype("float64"),
            "total_pts": elig["total_pts"].astype("float64"),
            "margin": (elig["away_pts"] - elig["home_pts"]).astype("float64"),
            "win": (~elig["home_win"]).astype("float64"),
            "game_fg_pct": elig["away_fg_pct"].astype("float64"),
            "game_fg3_pct": elig["away_fg3_pct"].astype("float64"),
            "game_ft_pct": elig["away_ft_pct"].astype("float64"),
            "game_reb": elig["away_reb"].astype("float64"),
            "game_ast": elig["away_ast"].astype("float64"),
        }
    )

    combined = pd.concat([home_log, away_log], ignore_index=True)
    combined = combined.merge(team_box, on=["game_id", "team_id"], how="left")

    # Fallback when pstats_df is not provided or a synthetic test omits player stats
    combined["reb"] = combined["reb"].fillna(combined["game_reb"]).astype("float64")
    combined["ast"] = combined["ast"].fillna(combined["game_ast"]).astype("float64")
    combined["fga"] = combined["fga"].fillna(85.0).astype("float64")
    combined["fgm"] = (
        combined["fgm"]
        .fillna((combined["game_fg_pct"] * combined["fga"]).round())
        .astype("float64")
    )
    combined["fg3a"] = combined["fg3a"].fillna(25.0).astype("float64")
    combined["fg3m"] = (
        combined["fg3m"]
        .fillna((combined["game_fg3_pct"] * combined["fg3a"]).round())
        .astype("float64")
    )
    combined["fta"] = combined["fta"].fillna(22.0).astype("float64")
    combined["ftm"] = (
        combined["ftm"]
        .fillna((combined["game_ft_pct"] * combined["fta"]).round())
        .astype("float64")
    )
    combined["oreb"] = combined["oreb"].fillna((combined["reb"] * 0.25).round()).astype("float64")
    combined["dreb"] = combined["dreb"].fillna(combined["reb"] - combined["oreb"]).astype("float64")
    combined["stl"] = combined["stl"].fillna(7.5).astype("float64")
    combined["blk"] = combined["blk"].fillna(4.8).astype("float64")
    combined["tov"] = combined["tov"].fillna(14.0).astype("float64")

    fga_safe = combined["fga"].replace(0.0, np.nan)
    fg3a_safe = combined["fg3a"].replace(0.0, np.nan)
    fta_safe = combined["fta"].replace(0.0, np.nan)

    combined["fg_pct"] = (combined["fgm"] / fga_safe).fillna(combined["game_fg_pct"]).clip(0.0, 1.0)
    combined["fg3_pct"] = (
        (combined["fg3m"] / fg3a_safe).fillna(combined["game_fg3_pct"]).clip(0.0, 1.0)
    )
    combined["ft_pct"] = (combined["ftm"] / fta_safe).fillna(combined["game_ft_pct"]).clip(0.0, 1.0)

    combined["efg_pct"] = ((combined["fgm"] + 0.5 * combined["fg3m"]) / fga_safe).clip(0.0, 1.5)
    tsa_denom = 2.0 * (combined["fga"] + 0.44 * combined["fta"])
    combined["ts_pct"] = (combined["pts_for"] / tsa_denom.replace(0.0, np.nan)).clip(0.0, 1.5)

    # Estimated possessions: FGA + 0.44 * FTA - OREB + TOV
    poss = combined["fga"] + 0.44 * combined["fta"] - combined["oreb"] + combined["tov"]
    combined["possessions"] = poss.clip(lower=1.0)

    # Join opponent possessions for the same game to compute accurate DRtg
    opp_poss = combined[["game_id", "team_id", "possessions"]].rename(
        columns={"team_id": "opponent_id", "possessions": "opp_possessions"}
    )
    combined = combined.merge(opp_poss, on=["game_id", "opponent_id"], how="left")
    combined["opp_possessions"] = combined["opp_possessions"].fillna(combined["possessions"])

    combined["tov_rate"] = (combined["tov"] / combined["possessions"]).clip(0.0, 1.0)
    combined["ortg"] = 100.0 * (combined["pts_for"] / combined["possessions"])
    combined["drtg"] = 100.0 * (combined["pts_against"] / combined["opp_possessions"])
    combined["net_rtg"] = combined["ortg"] - combined["drtg"]

    return combined.sort_values(["team_id", "season", "game_date", "game_id"]).reset_index(
        drop=True
    )


def compute_team_historical_features(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly pregame season-to-date expanding historical features per team.

    Guarantees:
    - Sorted chronologically by ``(team_id, season, game_date, game_id)``.
    - Applies ``shift(1)`` within ``(team_id, season)`` before ``expanding().mean()``.
    - If multiple games share the same ``(team_id, season, game_date)``, all games on
      that date receive the pre-date state so same-day games never leak into one another.
    """
    df = team_logs.sort_values(["team_id", "season", "game_date", "game_id"]).reset_index(drop=True)

    stat_map = {
        "pts_for": "hist_pts_avg",
        "pts_against": "hist_pts_allowed_avg",
        "total_pts": "hist_total_pts_avg",
        "margin": "hist_margin_avg",
        "win": "hist_win_pct",
        "reb": "hist_reb_avg",
        "oreb": "hist_oreb_avg",
        "dreb": "hist_dreb_avg",
        "ast": "hist_ast_avg",
        "stl": "hist_stl_avg",
        "blk": "hist_blk_avg",
        "tov": "hist_tov_avg",
        "fg_pct": "hist_fg_pct",
        "fg3_pct": "hist_fg3_pct",
        "ft_pct": "hist_ft_pct",
        "efg_pct": "hist_efg_pct",
        "ts_pct": "hist_ts_pct",
        "tov_rate": "hist_tov_rate",
        "possessions": "hist_possessions_avg",
        "ortg": "hist_ortg",
        "drtg": "hist_drtg",
    }

    grp = df.groupby(["team_id", "season"], sort=False)
    df["hist_games_played"] = grp.cumcount().astype("int64")

    for src_col, out_col in stat_map.items():
        shifted = grp[src_col].shift(1)
        exp_mean = (
            shifted.groupby([df["team_id"], df["season"]], sort=False)
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        df[out_col] = exp_mean.astype("float64")

    df["hist_net_rtg"] = (df["hist_ortg"] - df["hist_drtg"]).astype("float64")

    # Enforce same-day invariance: all games on the same date get the first pre-date row
    same_day_cols = ["hist_games_played"] + list(stat_map.values()) + ["hist_net_rtg"]
    date_grp = df.groupby(["team_id", "season", "game_date"], sort=False)
    for col in same_day_cols:
        df[col] = date_grp[col].transform("first")

    df["hist_games_played"] = df["hist_games_played"].astype("int64")
    df["hist_insufficient_history"] = (df["hist_games_played"] < MIN_HISTORY_GAMES).astype("bool")

    # Venue-specific expanding splits (home games only vs. road games only within season)
    venue_grp = df.groupby(["team_id", "season", "is_home"], sort=False)
    df["venue_games_played"] = venue_grp.cumcount().astype("int64")
    venue_stat_map = {
        "pts_for": "venue_pts_avg",
        "pts_against": "venue_pts_allowed_avg",
        "margin": "venue_margin_avg",
        "win": "venue_win_pct",
    }
    for src_col, out_col in venue_stat_map.items():
        v_shifted = venue_grp[src_col].shift(1)
        v_exp = (
            v_shifted.groupby([df["team_id"], df["season"], df["is_home"]], sort=False)
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
        )
        df[out_col] = v_exp.astype("float64")

    v_date_grp = df.groupby(["team_id", "season", "is_home", "game_date"], sort=False)
    for col in ["venue_games_played"] + list(venue_stat_map.values()):
        df[col] = v_date_grp[col].transform("first")
    df["venue_games_played"] = df["venue_games_played"].astype("int64")

    keep_cols = (
        [
            "game_id",
            "team_id",
            "season",
            "game_date",
            "is_home",
            "hist_games_played",
            "hist_insufficient_history",
        ]
        + list(stat_map.values())
        + ["hist_net_rtg", "venue_games_played"]
        + list(venue_stat_map.values())
    )
    return df[keep_cols].copy()


def attach_historical_features_to_games(
    games_df: pd.DataFrame,
    pstats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute and attach home and away historical and venue split features to games_df."""
    elig = filter_eligible_games(games_df)
    logs = build_team_game_logs(elig, pstats_df)
    hist = compute_team_historical_features(logs)

    hist_base_cols = [c for c in hist.columns if c.startswith("hist_")]
    venue_base_cols = [c for c in hist.columns if c.startswith("venue_")]

    home_hist = hist[hist["is_home"]].copy()
    home_rename = {c: f"home_{c}" for c in hist_base_cols + venue_base_cols}
    home_hist = home_hist[["game_id", "team_id"] + hist_base_cols + venue_base_cols].rename(
        columns={"team_id": "home_team_id", **home_rename}
    )

    away_hist = hist[~hist["is_home"]].copy()
    away_rename = {c: f"away_{c}" for c in hist_base_cols + venue_base_cols}
    away_hist = away_hist[["game_id", "team_id"] + hist_base_cols + venue_base_cols].rename(
        columns={"team_id": "away_team_id", **away_rename}
    )

    merged = elig.merge(home_hist, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_hist, on=["game_id", "away_team_id"], how="left")
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)
