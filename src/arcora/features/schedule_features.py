"""Rest, back-to-back, schedule density, and streak features (Issue #12)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arcora.features.team_performance import build_team_game_logs, filter_eligible_games

MAX_REST_DAYS_CAP: float = 14.0


def compute_team_schedule_features(team_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute pregame rest days, B2B flags, 7d/14d density, and home/away streaks per team.

    Guarantees:
    - Only games with ``prior_game_date < game_date`` in the same ``(team_id, season)``
      are used.
    - Current game and any same-day games are strictly excluded.
    """
    df = team_logs.sort_values(["team_id", "season", "game_date", "game_id"]).reset_index(drop=True)

    n = len(df)
    rest_days = np.full(n, np.nan, dtype="float64")
    is_season_opener = np.zeros(n, dtype="bool")
    is_back_to_back = np.zeros(n, dtype="bool")
    games_last_7d = np.zeros(n, dtype="int64")
    games_last_14d = np.zeros(n, dtype="int64")
    recent_home_10g = np.zeros(n, dtype="int64")
    recent_away_10g = np.zeros(n, dtype="int64")
    consec_home = np.zeros(n, dtype="int64")
    consec_away = np.zeros(n, dtype="int64")

    team_ids = df["team_id"].to_numpy()
    seasons = df["season"].to_numpy()
    dates_ns = df["game_date"].to_numpy(dtype="datetime64[D]").astype("int64")
    is_home_arr = df["is_home"].to_numpy(dtype="bool")

    # Process each (team_id, season) slice in linear time
    start_idx = 0
    while start_idx < n:
        t_id = team_ids[start_idx]
        s_id = seasons[start_idx]
        end_idx = start_idx + 1
        while end_idx < n and team_ids[end_idx] == t_id and seasons[end_idx] == s_id:
            end_idx += 1

        # Within [start_idx, end_idx), iterate chronologically, advancing prior_ptr only
        # when date strictly increases (preventing same-day leakage)
        prior_end = start_idx
        for i in range(start_idx, end_idx):
            cur_date = dates_ns[i]
            while prior_end < i and dates_ns[prior_end] < cur_date:
                prior_end += 1

            if prior_end == start_idx:
                is_season_opener[i] = True
                continue

            prev_date = dates_ns[prior_end - 1]
            diff_days = float(cur_date - prev_date)
            capped_rest = min(diff_days, MAX_REST_DAYS_CAP)
            rest_days[i] = capped_rest
            is_back_to_back[i] = diff_days == 1.0

            prior_dates = dates_ns[start_idx:prior_end]
            days_ago = cur_date - prior_dates
            games_last_7d[i] = int(np.sum((days_ago >= 1) & (days_ago <= 7)))
            games_last_14d[i] = int(np.sum((days_ago >= 1) & (days_ago <= 14)))

            win_start = max(start_idx, prior_end - 10)
            recent_venues = is_home_arr[win_start:prior_end]
            h_cnt = int(np.sum(recent_venues))
            recent_home_10g[i] = h_cnt
            recent_away_10g[i] = len(recent_venues) - h_cnt

            # Consecutive prior home or away streak
            h_streak = 0
            a_streak = 0
            for k in range(prior_end - 1, start_idx - 1, -1):
                if is_home_arr[k]:
                    if a_streak > 0:
                        break
                    h_streak += 1
                else:
                    if h_streak > 0:
                        break
                    a_streak += 1
            consec_home[i] = h_streak
            consec_away[i] = a_streak

        start_idx = end_idx

    out = pd.DataFrame(
        {
            "game_id": df["game_id"].astype("int64"),
            "team_id": df["team_id"].astype("int64"),
            "season": df["season"].astype("int64"),
            "game_date": df["game_date"],
            "is_home": df["is_home"].astype("bool"),
            "rest_days": rest_days,
            "is_season_opener": is_season_opener,
            "is_back_to_back": is_back_to_back,
            "games_last_7d": games_last_7d,
            "games_last_14d": games_last_14d,
            "recent_home_games_10g": recent_home_10g,
            "recent_away_games_10g": recent_away_10g,
            "consecutive_home_games": consec_home,
            "consecutive_away_games": consec_away,
        }
    )
    return out


def attach_schedule_features_to_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach home/away rest, B2B, schedule density, and streak features."""
    elig = filter_eligible_games(games_df)
    logs = build_team_game_logs(elig, pstats_df=None)
    sched = compute_team_schedule_features(logs)

    sched_cols = [
        "rest_days",
        "is_season_opener",
        "is_back_to_back",
        "games_last_7d",
        "games_last_14d",
        "recent_home_games_10g",
        "recent_away_games_10g",
        "consecutive_home_games",
        "consecutive_away_games",
    ]

    home_s = sched[sched["is_home"]][["game_id", "team_id"] + sched_cols].rename(
        columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in sched_cols}}
    )
    away_s = sched[~sched["is_home"]][["game_id", "team_id"] + sched_cols].rename(
        columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in sched_cols}}
    )

    merged = elig.merge(home_s, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_s, on=["game_id", "away_team_id"], how="left")

    merged["both_teams_back_to_back"] = (
        merged["home_is_back_to_back"] & merged["away_is_back_to_back"]
    ).astype("bool")
    merged["rest_days_diff"] = (merged["home_rest_days"] - merged["away_rest_days"]).astype(
        "float64"
    )
    return merged.sort_values(["game_date", "game_id"]).reset_index(drop=True)
