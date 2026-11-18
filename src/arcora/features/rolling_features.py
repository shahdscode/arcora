"""Rolling-window (5g, 10g) and EWMA performance features (Issue #11)."""

from __future__ import annotations

import pandas as pd

from arcora.features.team_performance import build_team_game_logs, filter_eligible_games

ROLLING_WINDOWS: tuple[int, ...] = (5, 10)
EWMA_SPAN: int = 5


def compute_team_rolling_and_ewma_features(
    team_logs: pd.DataFrame,
    windows: tuple[int, ...] = ROLLING_WINDOWS,
    ewma_span: int = EWMA_SPAN,
) -> pd.DataFrame:
    """Compute rolling 5-game, 10-game, variability (std), and EWMA features per team.

    Temporal & Leakage Guarantees:
    - Games are sorted by ``(team_id, season, game_date, game_id)``.
    - ``shift(1)`` is applied within ``(team_id, season)`` before any rolling or EWMA
      aggregation so the current game is strictly excluded.
    - Same-day invariance is enforced by propagating the pre-date feature values across
      all games occurring on the same ``(team_id, season, game_date)``.
    """
    df = team_logs.sort_values(["team_id", "season", "game_date", "game_id"]).reset_index(drop=True)

    mean_metric_map = {
        "pts_for": "pts_avg",
        "pts_against": "pts_allowed_avg",
        "margin": "margin_avg",
        "win": "win_pct",
        "reb": "reb_avg",
        "ast": "ast_avg",
        "tov": "tov_avg",
        "fg_pct": "fg_pct",
        "fg3_pct": "fg3_pct",
        "ft_pct": "ft_pct",
        "efg_pct": "efg_pct",
        "ts_pct": "ts_pct",
        "possessions": "possessions_avg",
        "ortg": "ortg",
        "drtg": "drtg",
        "net_rtg": "net_rtg",
    }
    std_metric_map = {
        "pts_for": "pts_std",
        "pts_against": "pts_allowed_std",
        "margin": "margin_std",
    }
    ewma_metric_map = {
        "pts_for": "ewma_pts",
        "pts_against": "ewma_pts_allowed",
        "margin": "ewma_margin",
        "win": "ewma_win_pct",
        "efg_pct": "ewma_efg_pct",
        "ortg": "ewma_ortg",
        "drtg": "ewma_drtg",
        "net_rtg": "ewma_net_rtg",
    }

    grp = df.groupby(["team_id", "season"], sort=False)
    prior_games_count = grp.cumcount()

    # Pre-shift all source series by 1 game within (team_id, season)
    shifted_cols: dict[str, pd.Series] = {}
    for src_col in set(mean_metric_map) | set(std_metric_map) | set(ewma_metric_map):
        shifted_cols[src_col] = grp[src_col].shift(1)

    generated_cols: list[str] = []

    for w in windows:
        insuff_col = f"rolling_insufficient_{w}"
        df[insuff_col] = (prior_games_count < w).astype("bool")
        generated_cols.append(insuff_col)

        for src_col, suffix in mean_metric_map.items():
            out_col = f"rolling_{suffix}_{w}"
            roll_mean = (
                shifted_cols[src_col]
                .groupby([df["team_id"], df["season"]], sort=False)
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            df[out_col] = roll_mean.astype("float64")
            generated_cols.append(out_col)

        for src_col, suffix in std_metric_map.items():
            out_col = f"rolling_{suffix}_{w}"
            roll_std = (
                shifted_cols[src_col]
                .groupby([df["team_id"], df["season"]], sort=False)
                .rolling(window=w, min_periods=2)
                .std(ddof=1)
                .reset_index(level=[0, 1], drop=True)
            )
            df[out_col] = roll_std.astype("float64")
            generated_cols.append(out_col)

    for src_col, out_col in ewma_metric_map.items():
        ewma_vals = (
            shifted_cols[src_col]
            .groupby([df["team_id"], df["season"]], sort=False)
            .transform(lambda s: s.ewm(span=ewma_span, min_periods=1, adjust=True).mean())
        )
        df[out_col] = ewma_vals.astype("float64")
        generated_cols.append(out_col)

    # Same-day game invariance: propagate first pre-date value across same-day games
    date_grp = df.groupby(["team_id", "season", "game_date"], sort=False)
    for col in generated_cols:
        df[col] = date_grp[col].transform("first")

    for w in windows:
        df[f"rolling_insufficient_{w}"] = df[f"rolling_insufficient_{w}"].astype("bool")

    keep_cols = ["game_id", "team_id", "season", "game_date", "is_home"] + generated_cols
    return df[keep_cols].copy()


def attach_rolling_features_to_games(
    games_df: pd.DataFrame,
    pstats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute and attach home and away rolling (5g, 10g) and EWMA features to games_df."""
    elig = filter_eligible_games(games_df)
    logs = build_team_game_logs(elig, pstats_df)
    roll = compute_team_rolling_and_ewma_features(logs)

    feat_cols = [c for c in roll.columns if c.startswith("rolling_") or c.startswith("ewma_")]

    home_roll = roll[roll["is_home"]].copy()
    home_roll = home_roll[["game_id", "team_id"] + feat_cols].rename(
        columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in feat_cols}}
    )

    away_roll = roll[~roll["is_home"]].copy()
    away_roll = away_roll[["game_id", "team_id"] + feat_cols].rename(
        columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in feat_cols}}
    )

    merged = elig.merge(home_roll, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_roll, on=["game_id", "away_team_id"], how="left")
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)
