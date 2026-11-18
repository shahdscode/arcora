"""Game-level matchup, head-to-head, and as-of standings features (Issue #13)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arcora.features.rolling_features import compute_team_rolling_and_ewma_features
from arcora.features.schedule_features import compute_team_schedule_features
from arcora.features.strategy import (
    FEATURE_COLUMN_NAMES,
    METADATA_COLUMNS,
    TARGET_COLUMNS,
)
from arcora.features.team_performance import (
    build_team_game_logs,
    compute_team_historical_features,
    filter_eligible_games,
)


def compute_head_to_head_features(elig_games: pd.DataFrame) -> pd.DataFrame:
    """Compute strictly pregame head-to-head (H2H) features for each game.

    Groups games by unordered franchise pair ``(min(home, away), max(home, away))``
    and computes prior meeting counts, current-home-team win %, current-home-team
    scoring margin, and average total points using only games with
    ``prior_game_date < game_date``.
    """
    df = elig_games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    n = len(df)

    h2h_meetings = np.zeros(n, dtype="int64")
    h2h_win_pct = np.full(n, np.nan, dtype="float64")
    h2h_margin = np.full(n, np.nan, dtype="float64")
    h2h_total_pts = np.full(n, np.nan, dtype="float64")

    home_ids = df["home_team_id"].to_numpy(dtype="int64")
    away_ids = df["away_team_id"].to_numpy(dtype="int64")
    dates_days = df["game_date"].to_numpy(dtype="datetime64[D]").astype("int64")
    home_pts = df["home_pts"].to_numpy(dtype="float64")
    away_pts = df["away_pts"].to_numpy(dtype="float64")
    total_pts = df["total_pts"].to_numpy(dtype="float64")

    pair_a = np.minimum(home_ids, away_ids)
    pair_b = np.maximum(home_ids, away_ids)

    # Group indices by unordered pair (pair_a, pair_b)
    pair_df = pd.DataFrame({"pair_a": pair_a, "pair_b": pair_b, "idx": np.arange(n)})
    for _, grp in pair_df.groupby(["pair_a", "pair_b"], sort=False):
        idxs = grp["idx"].to_numpy()
        m = len(idxs)
        if m <= 1:
            continue

        # Advance pointer only when date strictly increases (no same-day leakage)
        prior_ptr = 0
        # Track cumulative stats oriented toward team_a (the smaller team_id)
        team_a_id = pair_a[idxs[0]]
        cum_meetings = 0
        cum_a_wins = 0.0
        cum_a_margin = 0.0
        cum_tot_pts = 0.0

        for pos in range(m):
            cur_i = idxs[pos]
            cur_date = dates_days[cur_i]
            while prior_ptr < pos and dates_days[idxs[prior_ptr]] < cur_date:
                prev_i = idxs[prior_ptr]
                if home_ids[prev_i] == team_a_id:
                    a_margin = home_pts[prev_i] - away_pts[prev_i]
                else:
                    a_margin = away_pts[prev_i] - home_pts[prev_i]
                cum_meetings += 1
                cum_a_wins += 1.0 if a_margin > 0 else 0.0
                cum_a_margin += a_margin
                cum_tot_pts += total_pts[prev_i]
                prior_ptr += 1

            h2h_meetings[cur_i] = cum_meetings
            if cum_meetings > 0:
                a_win_rate = cum_a_wins / cum_meetings
                a_avg_margin = cum_a_margin / cum_meetings
                h2h_total_pts[cur_i] = cum_tot_pts / cum_meetings
                if home_ids[cur_i] == team_a_id:
                    h2h_win_pct[cur_i] = a_win_rate
                    h2h_margin[cur_i] = a_avg_margin
                else:
                    h2h_win_pct[cur_i] = 1.0 - a_win_rate
                    h2h_margin[cur_i] = -a_avg_margin

    return pd.DataFrame(
        {
            "game_id": df["game_id"].astype("int64"),
            "h2h_prior_meetings": h2h_meetings,
            "h2h_home_team_win_pct": h2h_win_pct,
            "h2h_home_team_margin_avg": h2h_margin,
            "h2h_total_pts_avg": h2h_total_pts,
        }
    )


def join_asof_standings_features(
    elig_games: pd.DataFrame,
    standings_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach pregame standings features using strict backward as-of join.

    Enforces ``standings_date < game_date`` (``allow_exact_matches=False``) within matching
    ``(team_id, season)`` on regular-season standings snapshots.
    """
    base = elig_games[["game_id", "game_date", "season", "home_team_id", "away_team_id"]].copy()
    base["game_date"] = pd.to_datetime(base["game_date"])

    if standings_df is None or len(standings_df) == 0:
        out = pd.DataFrame({"game_id": base["game_id"].astype("int64")})
        for side in ("home", "away"):
            out[f"{side}_standings_games_played"] = np.int64(0)
            out[f"{side}_standings_wins"] = np.int64(0)
            out[f"{side}_standings_losses"] = np.int64(0)
            out[f"{side}_standings_win_pct"] = np.nan
        out["home_standings_home_win_pct"] = np.nan
        out["away_standings_road_win_pct"] = np.nan
        out["standings_win_pct_diff"] = np.nan
        return out

    st = standings_df.copy()
    if "season_type" in st.columns:
        st = st[st["season_type"] == "Regular Season"].copy()
    st["standings_date"] = pd.to_datetime(st["standings_date"])
    st["team_id"] = st["team_id"].astype("int64")
    st["season"] = st["season"].astype("int64")

    # Deduplicate on (team_id, season, standings_date) and sort by standings_date for merge_asof
    st = (
        st.drop_duplicates(subset=["team_id", "season", "standings_date"], keep="last")
        .sort_values("standings_date")
        .reset_index(drop=True)
    )

    home_games_denom = (st["home_wins"] + st["home_losses"]).replace(0, np.nan)
    road_games_denom = (st["road_wins"] + st["road_losses"]).replace(0, np.nan)
    st["st_home_win_pct"] = (st["home_wins"] / home_games_denom).astype("float64")
    st["st_road_win_pct"] = (st["road_wins"] / road_games_denom).astype("float64")

    st_cols = [
        "team_id",
        "season",
        "standings_date",
        "games_played",
        "wins",
        "losses",
        "win_pct",
        "st_home_win_pct",
        "st_road_win_pct",
    ]
    st_clean = st[st_cols].sort_values("standings_date").reset_index(drop=True)

    # Sort base by game_date for pd.merge_asof
    base_sorted = base.sort_values("game_date").reset_index(drop=True)

    home_asof = pd.merge_asof(
        base_sorted[["game_id", "game_date", "season", "home_team_id"]].rename(
            columns={"home_team_id": "team_id"}
        ),
        st_clean,
        left_on="game_date",
        right_on="standings_date",
        by=["team_id", "season"],
        direction="backward",
        allow_exact_matches=False,
    )

    away_asof = pd.merge_asof(
        base_sorted[["game_id", "game_date", "season", "away_team_id"]].rename(
            columns={"away_team_id": "team_id"}
        ),
        st_clean,
        left_on="game_date",
        right_on="standings_date",
        by=["team_id", "season"],
        direction="backward",
        allow_exact_matches=False,
    )

    out = pd.DataFrame(
        {
            "game_id": home_asof["game_id"].astype("int64"),
            "home_standings_games_played": home_asof["games_played"].fillna(0).astype("int64"),
            "home_standings_wins": home_asof["wins"].fillna(0).astype("int64"),
            "home_standings_losses": home_asof["losses"].fillna(0).astype("int64"),
            "home_standings_win_pct": home_asof["win_pct"].astype("float64"),
            "home_standings_home_win_pct": home_asof["st_home_win_pct"].astype("float64"),
            "away_standings_games_played": away_asof["games_played"].fillna(0).astype("int64"),
            "away_standings_wins": away_asof["wins"].fillna(0).astype("int64"),
            "away_standings_losses": away_asof["losses"].fillna(0).astype("int64"),
            "away_standings_win_pct": away_asof["win_pct"].astype("float64"),
            "away_standings_road_win_pct": away_asof["st_road_win_pct"].astype("float64"),
        }
    )
    out["standings_win_pct_diff"] = (
        out["home_standings_win_pct"] - out["away_standings_win_pct"]
    ).astype("float64")
    return out


def build_matchup_feature_dataset(
    games_df: pd.DataFrame,
    pstats_df: pd.DataFrame | None = None,
    standings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete game-level matchup dataset integrating all feature groups."""
    elig = filter_eligible_games(games_df)
    logs = build_team_game_logs(elig, pstats_df)

    hist_df = compute_team_historical_features(logs)
    roll_df = compute_team_rolling_and_ewma_features(logs)
    sched_df = compute_team_schedule_features(logs)

    # Combine team-level feature tables on (game_id, team_id, is_home)
    team_feats = hist_df.merge(
        roll_df.drop(columns=["season", "game_date"]),
        on=["game_id", "team_id", "is_home"],
        how="inner",
    ).merge(
        sched_df.drop(columns=["season", "game_date"]),
        on=["game_id", "team_id", "is_home"],
        how="inner",
    )

    per_team_cols = [
        c
        for c in team_feats.columns
        if c not in ("game_id", "team_id", "season", "game_date", "is_home")
    ]

    home_feats = team_feats[team_feats["is_home"]][["game_id", "team_id"] + per_team_cols].rename(
        columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in per_team_cols}}
    )
    away_feats = team_feats[~team_feats["is_home"]][["game_id", "team_id"] + per_team_cols].rename(
        columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in per_team_cols}}
    )

    # Drop unnecessary venue columns that belong only to the opposite side
    home_feats = home_feats.drop(
        columns=[c for c in home_feats.columns if c.startswith("away_venue_")],
        errors="ignore",
    )
    away_feats = away_feats.drop(
        columns=[c for c in away_feats.columns if c.startswith("home_venue_")],
        errors="ignore",
    )

    merged = elig.merge(home_feats, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_feats, on=["game_id", "away_team_id"], how="left")

    # Schedule interaction features
    merged["both_teams_back_to_back"] = (
        merged["home_is_back_to_back"] & merged["away_is_back_to_back"]
    ).astype("bool")
    merged["rest_days_diff"] = (merged["home_rest_days"] - merged["away_rest_days"]).astype(
        "float64"
    )

    # Head-to-head features
    h2h_df = compute_head_to_head_features(elig)
    merged = merged.merge(h2h_df, on="game_id", how="left")

    # Standings as-of features
    st_df = join_asof_standings_features(elig, standings_df)
    merged = merged.merge(st_df, on="game_id", how="left")

    # Context indicators
    merged["is_partial_season"] = (merged["season"] == 2022).astype("bool")
    merged["home_court_indicator"] = np.int64(1)
    merged["is_playoff_game"] = (merged["game_type"] == "Playoffs").astype("bool")
    merged["is_play_in_game"] = (merged["game_type"] == "Play-In").astype("bool")

    # Matchup difference features (home - away)
    hist_diff_pairs = [
        ("hist_pts_avg_diff", "home_hist_pts_avg", "away_hist_pts_avg"),
        ("hist_pts_allowed_avg_diff", "home_hist_pts_allowed_avg", "away_hist_pts_allowed_avg"),
        ("hist_margin_avg_diff", "home_hist_margin_avg", "away_hist_margin_avg"),
        ("hist_win_pct_diff", "home_hist_win_pct", "away_hist_win_pct"),
        ("hist_ortg_diff", "home_hist_ortg", "away_hist_ortg"),
        ("hist_drtg_diff", "home_hist_drtg", "away_hist_drtg"),
        ("hist_net_rtg_diff", "home_hist_net_rtg", "away_hist_net_rtg"),
        ("hist_efg_pct_diff", "home_hist_efg_pct", "away_hist_efg_pct"),
        ("hist_ts_pct_diff", "home_hist_ts_pct", "away_hist_ts_pct"),
        ("hist_reb_avg_diff", "home_hist_reb_avg", "away_hist_reb_avg"),
        ("hist_ast_avg_diff", "home_hist_ast_avg", "away_hist_ast_avg"),
        ("hist_tov_avg_diff", "home_hist_tov_avg", "away_hist_tov_avg"),
        ("venue_win_pct_diff", "home_venue_win_pct", "away_venue_win_pct"),
        ("venue_margin_avg_diff", "home_venue_margin_avg", "away_venue_margin_avg"),
    ]
    for out_col, h_col, a_col in hist_diff_pairs:
        merged[out_col] = (merged[h_col] - merged[a_col]).astype("float64")

    rolling_diff_keys = [
        "pts_avg",
        "pts_allowed_avg",
        "margin_avg",
        "win_pct",
        "reb_avg",
        "ast_avg",
        "tov_avg",
        "fg_pct",
        "fg3_pct",
        "ft_pct",
        "efg_pct",
        "ortg",
        "drtg",
        "net_rtg",
    ]
    for w in (5, 10):
        for k in rolling_diff_keys:
            merged[f"rolling_{k}_{w}_diff"] = (
                merged[f"home_rolling_{k}_{w}"] - merged[f"away_rolling_{k}_{w}"]
            ).astype("float64")

    ewma_diff_pairs = [
        ("ewma_pts_diff", "home_ewma_pts", "away_ewma_pts"),
        ("ewma_pts_allowed_diff", "home_ewma_pts_allowed", "away_ewma_pts_allowed"),
        ("ewma_margin_diff", "home_ewma_margin", "away_ewma_margin"),
        ("ewma_win_pct_diff", "home_ewma_win_pct", "away_ewma_win_pct"),
        ("ewma_net_rtg_diff", "home_ewma_net_rtg", "away_ewma_net_rtg"),
        ("ewma_efg_pct_diff", "home_ewma_efg_pct", "away_ewma_efg_pct"),
        ("home_ortg_vs_away_drtg_10", "home_rolling_ortg_10", "away_rolling_drtg_10"),
        ("away_ortg_vs_home_drtg_10", "away_rolling_ortg_10", "home_rolling_drtg_10"),
    ]
    for out_col, h_col, a_col in ewma_diff_pairs:
        merged[out_col] = (merged[h_col] - merged[a_col]).astype("float64")

    ordered_cols = list(METADATA_COLUMNS) + list(FEATURE_COLUMN_NAMES) + list(TARGET_COLUMNS)
    return merged[ordered_cols].sort_values(["game_date", "game_id"]).reset_index(drop=True)
