# Arcora Week 2 — ML Feature Dataset Quality & Leakage Report

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Overall Validation & Leakage Status** | **PASSED** |
| **Generated At (UTC)** | `2026-11-18T13:28:21Z` |
| **Output Parquet Dataset** | `data/features/nba_game_features.parquet` |
| **Total Eligible Games (`rows`)** | 24,848 |
| **Metadata / Identifier Columns** | 8 |
| **Engineered Pregame Input Features (`X`)** | 242 |
| **Prediction Target Columns (`y`)** | 5 |
| **Duplicate `game_id` Count** | 0 |
| **Seasons Covered** | 20 (19 complete + 1 partial `[2022]`) |
| **Features with Unexpected Missing Values** | 0 |

---

## 2. Competition Stage Distribution

| Competition Stage (`game_type`) | Game Count | Share (%) | Notes |
| :--- | ---: | ---: | :--- |
| `Regular Season` | 23,244 | 93.54% | Standard 82-game (or lockout/COVID/partial) schedule |
| `Playoffs` | 1,591 | 6.40% | Postseason elimination series |
| `Play-In` | 13 | 0.05% | Play-In Tournament games (introduced 2019-20) |

### 2.1 Exact Eligible Game Reconciliation (`26,523` -> `24,848`)

| Competition Stage (`game_type`) | Source Count (`games.parquet`) | Excluded Count | Retained Count (`nba_game_features.parquet`) | Explanation |
| :--- | ---: | ---: | ---: | :--- |
| **Preseason (`GAME_ID` prefix `'1'`)** | **`1,675`** | **`-1,675`** | **`0`** | **100% of the 1,675 excluded games are Preseason games** (exhibition rotations; excluded from ML dataset and all rolling/historical lookbacks). |
| **Regular Season (`GAME_ID` prefix `'2'`)** | `23,244` | `0` | **`23,244`** | All completed regular-season games retained (`93.54%` of eligible dataset). |
| **Playoffs (`GAME_ID` prefix `'4'`)** | `1,591` | `0` | **`1,591`** | All completed playoff games retained (`6.40%` of eligible dataset). |
| **Play-In (`GAME_ID` prefix `'5'`)** | `13` | `0` | **`13`** | All completed play-in tournament games retained (`0.05%` of eligible dataset). |
| **Total Games** | **`26,523`** | **`-1,675`** | **`24,848`** | **`26,523 - 1,675 = 24,848` (`23,244 + 1,591 + 13 = 24,848`).** |

### 2.2 Partial `2022–23` Season (`season == 2022`) Breakdown

| Competition Stage (`season == 2022`) | Source Count (`games.parquet`) | Excluded Count | Retained Count (`nba_game_features.parquet`) |
| :--- | ---: | ---: | ---: |
| **Regular Season** | `477` | `0` | **`477`** |
| **Preseason** | `65` | `-65` | **`0`** |
| **Playoffs / Play-In** | `0` | `0` | **`0`** |
| **Total (`2022–23` through `2022-12-22`)** | **`542`** | **`-65`** | **`477`** |

---

## 3. Feature Categories & Data Types

| Feature Category | Feature Count |
| :--- | ---: |
| **Historical Team Performance** | 48 |
| **Home and Away Performance** | 10 |
| **Recent Team Form** | 40 |
| **Rolling Offensive and Defensive Statistics** | 56 |
| **Rest and Schedule-Related Features** | 20 |
| **Head-to-Head Historical Performance** | 4 |
| **Standings and Season Context** | 14 |
| **Home-Away Matchup Differences** | 50 |
| **Total Input Features (`X`)** | **242** |

### Feature Data Type Breakdown
| Pandas / Arrow Dtype | Column Count |
| :--- | ---: |
| `int64` | 24 |
| `bool` | 13 |
| `float64` | 205 |

---

## 4. Prediction Target Distributions (`y`)

| Target Column | Mean | Std Dev | Min | Median | Max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `home_pts` | 103.6413 | 13.2697 | 59.00 | 103.00 | 168.00 |
| `away_pts` | 100.7657 | 13.4196 | 54.00 | 100.00 | 168.00 |
| `point_margin` | 2.8757 | 13.4687 | -58.00 | 4.00 | 73.00 |
| `total_pts` | 204.4070 | 23.0421 | 124.00 | 203.00 | 329.00 |
| `home_win` | 0.5910 | 0.4917 | 0.00 | 1.00 | 1.00 |

---

## 5. Automated Temporal Leakage Audit Results

| Check ID | Check Name | Status | Violations | Details |
| :---: | :--- | :---: | ---: | :--- |
| `LEAK-001` | `target_and_postgame_column_exclusion` | **PASS** | 0 | Input feature matrix X cleanly separates 242 pregame features from all 5 targets and all post-game box-score columns. |
| `LEAK-002` | `preseason_games_excluded` | **PASS** | 0 | All 24,848 rows belong to eligible stages (Regular Season, Playoffs, Play-In); 0 Preseason games present. |
| `LEAK-003` | `current_game_excluded_via_shift_1` | **PASS** | 0 | Verified 317 home openers and 282 away openers have strictly null prior scoring averages (0 current-game leaks). |
| `LEAK-004` | `future_game_and_shuffle_invariance` | **PASS** | 0 | Pregame features for games 1..6 are 100% invariant to adding games 7..8, mutating future scores, and shuffling input row order. |
| `LEAK-005` | `standings_strict_backward_asof_join` | **PASS** | 0 | Standings join strictly enforces standings_date < game_date and rejects same-day and future/end-of-season snapshots. |
| `LEAK-006` | `feature_integrity_bounds_and_differences` | **PASS** | 0 | All 24,848 game rows have unique game_ids, valid win_pct bounds [0, 1], valid rest_days [1, 14], and exact home-away differentials. |
| `LEAK-007` | `same_day_game_temporal_isolation` | **PASS** | 0 | Same-day games share identical pre-date feature states (strictly prior_date < game_date); earlier same-date results never leak into later same-date games. |

---

## 6. Missing-Value Analysis & Cold-Start Accounting

- **Zero Unexpected Missingness**: `0` features have unexpected missing values.
- **Expected Cold-Start Missingness**:
  - **Season Openers (`0` prior games in season)**: `600` team-season openers across 20 seasons have `NaN` for expanding/rolling means, EWMA, and `rest_days` (explicitly flagged by `home_is_season_opener`, `away_is_season_opener`, and `*_hist_insufficient_history`).
  - **Rolling Standard Deviations (`< 2` prior games)**: Require at least 2 prior games (`ddof=1`), resulting in expected `NaN` values on games 1 and 2 of each team's season.
  - **Venue Splits (`0` prior home or road games)**: First home game or first road game of a season has `NaN` venue averages (`home_venue_games_played == 0`).
  - **First Historical Head-to-Head Meeting (`h2h_prior_meetings == 0`)**: Initial franchise meetings in 2003-04 have `NaN` prior H2H win % and margin.

| Feature Name | Category | Dtype | Null Count | Null % | Missingness Classification |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `home_hist_games_played` | Historical Team Performance | `int64` | 0 | 0.00% | `none` |
| `home_hist_insufficient_history` | Historical Team Performance | `bool` | 0 | 0.00% | `none` |
| `home_hist_pts_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_pts_allowed_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_total_pts_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_margin_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_win_pct` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_reb_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_oreb_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_dreb_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_ast_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_stl_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_blk_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_tov_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_fg_pct` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_fg3_pct` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_ft_pct` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_efg_pct` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_ts_pct` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_tov_rate` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_possessions_avg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_ortg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_drtg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_hist_net_rtg` | Historical Team Performance | `float64` | 317 | 1.28% | `expected_early_history` |
| `away_hist_games_played` | Historical Team Performance | `int64` | 0 | 0.00% | `none` |
| `away_hist_insufficient_history` | Historical Team Performance | `bool` | 0 | 0.00% | `none` |
| `away_hist_pts_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_pts_allowed_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_total_pts_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_margin_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_win_pct` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_reb_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_oreb_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_dreb_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_ast_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_stl_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_blk_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_tov_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_fg_pct` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_fg3_pct` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_ft_pct` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_efg_pct` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_ts_pct` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_tov_rate` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_possessions_avg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_ortg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_drtg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_hist_net_rtg` | Historical Team Performance | `float64` | 282 | 1.13% | `expected_early_history` |
| `home_venue_games_played` | Home and Away Performance | `int64` | 0 | 0.00% | `none` |
| `home_venue_pts_avg` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `home_venue_pts_allowed_avg` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `home_venue_margin_avg` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `home_venue_win_pct` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `away_venue_games_played` | Home and Away Performance | `int64` | 0 | 0.00% | `none` |
| `away_venue_pts_avg` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `away_venue_pts_allowed_avg` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `away_venue_margin_avg` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `away_venue_win_pct` | Home and Away Performance | `float64` | 599 | 2.41% | `expected_early_history` |
| `home_rolling_insufficient_5` | Recent Team Form | `bool` | 0 | 0.00% | `none` |
| `home_rolling_pts_avg_5` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_pts_allowed_avg_5` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_margin_avg_5` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_win_pct_5` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_reb_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ast_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_tov_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_fg_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_fg3_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ft_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_efg_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ts_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_possessions_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ortg_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_drtg_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_net_rtg_5` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_pts_std_5` | Recent Team Form | `float64` | 601 | 2.42% | `expected_early_history` |
| `home_rolling_pts_allowed_std_5` | Recent Team Form | `float64` | 601 | 2.42% | `expected_early_history` |
| `home_rolling_margin_std_5` | Recent Team Form | `float64` | 601 | 2.42% | `expected_early_history` |
| `away_rolling_insufficient_5` | Recent Team Form | `bool` | 0 | 0.00% | `none` |
| `away_rolling_pts_avg_5` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_pts_allowed_avg_5` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_margin_avg_5` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_win_pct_5` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_reb_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ast_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_tov_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_fg_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_fg3_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ft_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_efg_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ts_pct_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_possessions_avg_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ortg_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_drtg_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_net_rtg_5` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_pts_std_5` | Recent Team Form | `float64` | 597 | 2.40% | `expected_early_history` |
| `away_rolling_pts_allowed_std_5` | Recent Team Form | `float64` | 597 | 2.40% | `expected_early_history` |
| `away_rolling_margin_std_5` | Recent Team Form | `float64` | 597 | 2.40% | `expected_early_history` |
| `home_rolling_insufficient_10` | Recent Team Form | `bool` | 0 | 0.00% | `none` |
| `home_rolling_pts_avg_10` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_pts_allowed_avg_10` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_margin_avg_10` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_win_pct_10` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_reb_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ast_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_tov_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_fg_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_fg3_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ft_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_efg_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ts_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_possessions_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_ortg_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_drtg_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_net_rtg_10` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_rolling_pts_std_10` | Recent Team Form | `float64` | 601 | 2.42% | `expected_early_history` |
| `home_rolling_pts_allowed_std_10` | Recent Team Form | `float64` | 601 | 2.42% | `expected_early_history` |
| `home_rolling_margin_std_10` | Recent Team Form | `float64` | 601 | 2.42% | `expected_early_history` |
| `away_rolling_insufficient_10` | Recent Team Form | `bool` | 0 | 0.00% | `none` |
| `away_rolling_pts_avg_10` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_pts_allowed_avg_10` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_margin_avg_10` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_win_pct_10` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_reb_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ast_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_tov_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_fg_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_fg3_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ft_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_efg_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ts_pct_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_possessions_avg_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_ortg_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_drtg_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_net_rtg_10` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_rolling_pts_std_10` | Recent Team Form | `float64` | 597 | 2.40% | `expected_early_history` |
| `away_rolling_pts_allowed_std_10` | Recent Team Form | `float64` | 597 | 2.40% | `expected_early_history` |
| `away_rolling_margin_std_10` | Recent Team Form | `float64` | 597 | 2.40% | `expected_early_history` |
| `home_ewma_pts` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_ewma_pts_allowed` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_ewma_margin` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_ewma_win_pct` | Recent Team Form | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_ewma_efg_pct` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_ewma_ortg` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_ewma_drtg` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_ewma_net_rtg` | Rolling Offensive and Defensive Statistics | `float64` | 317 | 1.28% | `expected_early_history` |
| `away_ewma_pts` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_ewma_pts_allowed` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_ewma_margin` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_ewma_win_pct` | Recent Team Form | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_ewma_efg_pct` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_ewma_ortg` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_ewma_drtg` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_ewma_net_rtg` | Rolling Offensive and Defensive Statistics | `float64` | 282 | 1.13% | `expected_early_history` |
| `home_rest_days` | Rest and Schedule-Related Features | `float64` | 317 | 1.28% | `expected_early_history` |
| `home_is_season_opener` | Rest and Schedule-Related Features | `bool` | 0 | 0.00% | `none` |
| `home_is_back_to_back` | Rest and Schedule-Related Features | `bool` | 0 | 0.00% | `none` |
| `home_games_last_7d` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `home_games_last_14d` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `home_recent_home_games_10g` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `home_recent_away_games_10g` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `home_consecutive_home_games` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `home_consecutive_away_games` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `away_rest_days` | Rest and Schedule-Related Features | `float64` | 282 | 1.13% | `expected_early_history` |
| `away_is_season_opener` | Rest and Schedule-Related Features | `bool` | 0 | 0.00% | `none` |
| `away_is_back_to_back` | Rest and Schedule-Related Features | `bool` | 0 | 0.00% | `none` |
| `away_games_last_7d` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `away_games_last_14d` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `away_recent_home_games_10g` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `away_recent_away_games_10g` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `away_consecutive_home_games` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `away_consecutive_away_games` | Rest and Schedule-Related Features | `int64` | 0 | 0.00% | `none` |
| `both_teams_back_to_back` | Rest and Schedule-Related Features | `bool` | 0 | 0.00% | `none` |
| `rest_days_diff` | Rest and Schedule-Related Features | `float64` | 342 | 1.38% | `expected_early_history` |
| `h2h_prior_meetings` | Head-to-Head Historical Performance | `int64` | 0 | 0.00% | `none` |
| `h2h_home_team_win_pct` | Head-to-Head Historical Performance | `float64` | 435 | 1.75% | `expected_early_history` |
| `h2h_home_team_margin_avg` | Head-to-Head Historical Performance | `float64` | 435 | 1.75% | `expected_early_history` |
| `h2h_total_pts_avg` | Head-to-Head Historical Performance | `float64` | 435 | 1.75% | `expected_early_history` |
| `home_standings_games_played` | Standings and Season Context | `int64` | 0 | 0.00% | `none` |
| `home_standings_wins` | Standings and Season Context | `int64` | 0 | 0.00% | `none` |
| `home_standings_losses` | Standings and Season Context | `int64` | 0 | 0.00% | `none` |
| `home_standings_win_pct` | Standings and Season Context | `float64` | 57 | 0.23% | `expected_early_history_or_standings_snapshot_start` |
| `home_standings_home_win_pct` | Standings and Season Context | `float64` | 599 | 2.41% | `expected_early_history_or_standings_snapshot_start` |
| `away_standings_games_played` | Standings and Season Context | `int64` | 0 | 0.00% | `none` |
| `away_standings_wins` | Standings and Season Context | `int64` | 0 | 0.00% | `none` |
| `away_standings_losses` | Standings and Season Context | `int64` | 0 | 0.00% | `none` |
| `away_standings_win_pct` | Standings and Season Context | `float64` | 57 | 0.23% | `expected_early_history_or_standings_snapshot_start` |
| `away_standings_road_win_pct` | Standings and Season Context | `float64` | 599 | 2.41% | `expected_early_history_or_standings_snapshot_start` |
| `standings_win_pct_diff` | Standings and Season Context | `float64` | 57 | 0.23% | `expected_early_history_or_standings_snapshot_start` |
| `home_court_indicator` | Standings and Season Context | `int64` | 0 | 0.00% | `none` |
| `is_playoff_game` | Standings and Season Context | `bool` | 0 | 0.00% | `none` |
| `is_play_in_game` | Standings and Season Context | `bool` | 0 | 0.00% | `none` |
| `hist_pts_avg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_pts_allowed_avg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_margin_avg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_win_pct_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_ortg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_drtg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_net_rtg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_efg_pct_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_ts_pct_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_reb_avg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_ast_avg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `hist_tov_avg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `venue_win_pct_diff` | Home-Away Matchup Differences | `float64` | 772 | 3.11% | `expected_early_history` |
| `venue_margin_avg_diff` | Home-Away Matchup Differences | `float64` | 772 | 3.11% | `expected_early_history` |
| `rolling_pts_avg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_pts_allowed_avg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_margin_avg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_win_pct_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_reb_avg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_ast_avg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_tov_avg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_fg_pct_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_fg3_pct_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_ft_pct_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_efg_pct_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_ortg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_drtg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_net_rtg_5_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_pts_avg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_pts_allowed_avg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_margin_avg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_win_pct_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_reb_avg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_ast_avg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_tov_avg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_fg_pct_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_fg3_pct_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_ft_pct_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_efg_pct_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_ortg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_drtg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `rolling_net_rtg_10_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `ewma_pts_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `ewma_pts_allowed_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `ewma_margin_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `ewma_win_pct_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `ewma_net_rtg_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `ewma_efg_pct_diff` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `home_ortg_vs_away_drtg_10` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |
| `away_ortg_vs_home_drtg_10` | Home-Away Matchup Differences | `float64` | 342 | 1.38% | `expected_early_history` |

---

## 7. Known Limitations & Week 3 Modeling Recommendations

1. **Partial `2022-23` Season Protocol (`is_partial_season == True`)**:
   - The `2022` (`2022-23`) season contains only `477` regular-season games through the dataset snapshot cutoff date `2022-12-22` (`0` Play-In or Playoff games, representing ~39% of a full regular season).
   - **Chronological Split Rule for Week 3**: Model training and full-season cross-validation must use the **19 complete seasons (`2003-04` through `2021-22`, `is_partial_season == False`, `24,371` games)**, e.g.:
     - *Training Split*: `2003-04` through `2018-19` (16 complete seasons, `20,762` games)
     - *Validation Split*: `2019-20` through `2020-21` (2 complete seasons, `2,282` games)
     - *Full-Season Test Split*: `2021-22` (1 complete season, `1,327` games)
     - *Out-of-Time Partial-Season Holdout*: `2022-23` (`is_partial_season == True`, `477` games) — evaluated separately as a live early-season forward-testing benchmark and never aggregated as a complete season.
2. **Cold-Start Handling for Linear / Neural Models**: Tree-based gradient boosting models (LightGBM, XGBoost, CatBoost) handle early-season `NaN` values natively alongside `*_insufficient_*` and `*_is_season_opener` flags. For linear or neural models in Week 3, fit any imputation transformers strictly on the training split.
