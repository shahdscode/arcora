# Arcora Week 2 — Feature Engineering Strategy & Leakage Prevention Specification

## 1. Objective

Transform Arcora's validated canonical NBA datasets (`games.parquet`, `player_game_stats.parquet`, `standings.parquet`, `teams.parquet`) into a reproducible, leakage-free, game-level machine learning dataset (`data/features/nba_game_features.parquet`) using **strictly pre-tip-off information**.

---

## 2. Prediction Targets vs. Input Features

Arcora models five game-level outcomes. Target columns are retained in the Parquet dataset for supervised training and evaluation, but are **strictly separated** from the model input feature matrix `X`:

| Target Column | Task Type | Dtype | Formula | Description |
| :--- | :--- | :--- | :--- | :--- |
| `home_pts` | `regression` | `int64` | `games.home_pts` | Final points scored by the home team. |
| `away_pts` | `regression` | `int64` | `games.away_pts` | Final points scored by the away team. |
| `point_margin` | `regression` | `int64` | `home_pts - away_pts` | Final signed point differential from the home team perspective. |
| `total_pts` | `regression` | `int64` | `home_pts + away_pts` | Combined final points scored by both teams. |
| `home_win` | `binary_classification` | `bool` | `home_pts > away_pts` | Binary outcome indicator (True if home team wins). |

### Forbidden Post-Game Columns (Never Permitted in `X`)
Same-game box-score outcomes (`home_fg_pct`, `home_ft_pct`, `home_fg3_pct`, `home_ast`, `home_reb`, `away_fg_pct`, `away_ft_pct`, `away_fg3_pct`, `away_ast`, `away_reb`, and player-level counting stats from the current game) are post-game realizations and are explicitly banned from the input feature matrix.

---

## 3. Feature Categories Overview

| Category | Feature Count | Purpose |
| :--- | ---: | :--- |
| **Historical Team Performance** | 48 | Captures pregame historical team performance signals. |
| **Home and Away Performance** | 10 | Captures pregame home and away performance signals. |
| **Recent Team Form** | 40 | Captures pregame recent team form signals. |
| **Rolling Offensive and Defensive Statistics** | 56 | Captures pregame rolling offensive and defensive statistics signals. |
| **Rest and Schedule-Related Features** | 20 | Captures pregame rest and schedule-related features signals. |
| **Head-to-Head Historical Performance** | 4 | Captures pregame head-to-head historical performance signals. |
| **Standings and Season Context** | 14 | Captures pregame standings and season context signals. |
| **Home-Away Matchup Differences** | 50 | Captures pregame home-away matchup differences signals. |
| **Total Pregame Features** | **242** | — |

---

## 4. Strict Temporal & Leakage Prevention Rules

1. **Strict Pregame Availability (`t < game_date`)**: Every feature for a game on `game_date` is computed exclusively from games completed on dates strictly prior to `game_date` (`prior_game_date < game_date`).
2. **Chronological Sorting & `shift(1)`**: All team-game logs are sorted by `(team_id, season, game_date, game_id)` and shifted by 1 game (`shift(1)`) before computing expanding or rolling windows. If multiple games ever occur on the same `game_date` for a team, all games on date `d` receive the pre-date-`d` state so same-day results never leak across games.
3. **Strict Backward As-Of Standings Join (`standings_date < game_date`)**: Standings features are joined via `pd.merge_asof(..., allow_exact_matches=False)` so same-day and end-of-season standings are never used.
4. **Preseason Exclusion**: Preseason games (`game_type == 'Preseason'`, `1,675` games) are excluded from the ML dataset and from contributing to regular-season, play-in, or playoff feature histories (`24,848` eligible games).
5. **Season Isolation & Partial `2022-23` Flagging**: Historical and rolling team statistics are computed within each NBA `season` so offseason roster turnover does not corrupt early-season metrics, and `is_partial_season` explicitly flags the truncated `2022-23` season (cutoff `2022-12-22`).
6. **No Future Imputation**: Early-season `NaN` values (such as a team's season opener with 0 prior games) are never imputed with seasonal or dataset means; instead, explicit boolean indicators (`home_hist_insufficient_history`, `home_rolling_insufficient_5`, `home_is_season_opener`, etc.) are provided.

---

## 5. Final Dataset Structure (`data/features/nba_game_features.parquet`)

- **Grain**: One row per eligible non-preseason completed NBA game (`24,848` rows).
- **Metadata Columns (8)**: `game_id`, `game_date`, `season`, `season_label`, `game_type`, `is_partial_season`, `home_team_id`, `away_team_id`
- **Engineered Input Feature Columns (242)**: Grouped into 8 categories with `home_`, `away_`, `h2h_`, and `_diff` conventions.
- **Target Columns (5)**: `home_pts`, `away_pts`, `point_margin`, `total_pts`, `home_win`
