# Arcora Automated NBA Data Quality Report

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Overall Validation Status** | **PASSED** |
| **Evaluated At (UTC)** | `2026-11-17T11:18:04Z` |
| **Total Validation Rules Executed** | 25 |
| **Rules Passed** | 25 |
| **Rules Failed** | 0 |
| **`teams` Validated Rows** | 30 |
| **`players` Validated Rows** | 2,687 |
| **`games` Validated Rows** | 26,523 |
| **`player_game_stats` Validated Rows** | 668,338 |
| **`standings` Validated Rows** | 210,313 |

---

## 2. Automated Validation Rule Results

| Rule ID | Entity | Rule Name | Status | Failed / Total | Validation Result Details |
| :---: | :--- | :--- | :---: | ---: | :--- |
| `VAL-001` | `games` | `unique_pk_games` | **PASS** | 0 / 26,523 | All 26,523 records in 'games' have unique (game_id) keys. |
| `VAL-002` | `teams` | `unique_pk_teams` | **PASS** | 0 / 30 | All 30 records in 'teams' have unique (team_id) keys. |
| `VAL-003` | `players` | `unique_pk_players` | **PASS** | 0 / 2,687 | All 2,687 records in 'players' have unique (player_id) keys. |
| `VAL-004` | `player_game_stats` | `unique_pk_player_game_stats` | **PASS** | 0 / 668,338 | All 668,338 records in 'player_game_stats' have unique (game_id, team_id, player_id) keys. |
| `VAL-005` | `games` | `valid_game_dates_and_seasons` | **PASS** | 0 / 26,523 | All 26,523 games have valid dates (2003-10-05 to 2022-12-22) and consistent season identifiers (2003 to 2022). |
| `VAL-006` | `games` | `distinct_home_and_away_teams` | **PASS** | 0 / 26,523 | All 26,523 games have distinct home_team_id and away_team_id values. |
| `VAL-007` | `games` | `valid_game_scores_and_types` | **PASS** | 0 / 26,523 | All 26,523 games have valid integer scores, percentages in [0, 1], and consistent total_pts, point_margin, and home_win values. |
| `VAL-008` | `player_game_stats` | `valid_player_stats_numeric_bounds` | **PASS** | 0 / 668,338 | All 668,338 player_game_stats records satisfy non-negativity, percentage bounds [0, 1], and made <= attempted shooting constraints. |
| `VAL-009` | `games` | `non_null_critical_games` | **PASS** | 0 / 26,523 | Zero missing values across 23 critical columns in 'games'. |
| `VAL-010` | `teams` | `non_null_critical_teams` | **PASS** | 0 / 30 | Zero missing values across 14 critical columns in 'teams'. |
| `VAL-011` | `players` | `non_null_critical_players` | **PASS** | 0 / 2,687 | Zero missing values across 6 critical columns in 'players'. |
| `VAL-012` | `player_game_stats` | `non_null_critical_player_game_stats` | **PASS** | 0 / 668,338 | Zero missing values across 27 critical columns in 'player_game_stats'. |
| `VAL-013` | `games` | `fk_games_home_team_id_to_teams` | **PASS** | 0 / 26,523 | 100% of 'games.home_team_id' values (26,523 rows) reference valid 'teams.team_id' keys. |
| `VAL-014` | `games` | `fk_games_away_team_id_to_teams` | **PASS** | 0 / 26,523 | 100% of 'games.away_team_id' values (26,523 rows) reference valid 'teams.team_id' keys. |
| `VAL-015` | `player_game_stats` | `player_team_matches_game_participants` | **PASS** | 0 / 668,338 | All 668,338 player_game_stats rows belong to either the home or away team of their referenced game. |
| `VAL-016` | `player_game_stats` | `fk_player_game_stats_game_id_to_games` | **PASS** | 0 / 668,338 | 100% of 'player_game_stats.game_id' values (668,338 rows) reference valid 'games.game_id' keys. |
| `VAL-017` | `player_game_stats` | `fk_player_game_stats_team_id_to_teams` | **PASS** | 0 / 668,338 | 100% of 'player_game_stats.team_id' values (668,338 rows) reference valid 'teams.team_id' keys. |
| `VAL-018` | `player_game_stats` | `fk_player_game_stats_player_id_to_players` | **PASS** | 0 / 668,338 | 100% of 'player_game_stats.player_id' values (668,338 rows) reference valid 'players.player_id' keys. |
| `VAL-019` | `players` | `valid_player_career_counts_and_seasons` | **PASS** | 0 / 2,687 | All 2,687 players have valid season spans and 0 <= career_games_played <= career_games_logged. |
| `VAL-020` | `games` | `valid_game_status_and_type` | **PASS** | 0 / 26,523 | All 26,523 games have game_status == 'Final' and valid game_type classifications. |
| `VAL-021` | `player_game_stats` | `player_dnp_and_rebound_consistency` | **PASS** | 0 / 668,338 | All 668,338 player_game_stats rows satisfy DNP zero-stat consistency and reb == oreb + dreb. |
| `VAL-022` | `standings` | `unique_pk_standings` | **PASS** | 0 / 210,313 | All 210,313 records in 'standings' have unique (team_id, season, season_type, standings_date) keys. |
| `VAL-023` | `standings` | `non_null_critical_standings` | **PASS** | 0 / 210,313 | Zero missing values across 15 critical columns in 'standings'. |
| `VAL-024` | `standings` | `fk_standings_team_id_to_teams` | **PASS** | 0 / 210,313 | 100% of 'standings.team_id' values (210,313 rows) reference valid 'teams.team_id' keys. |
| `VAL-025` | `standings` | `valid_standings_records_and_domains` | **PASS** | 0 / 210,313 | All 210,313 standings rows satisfy games_played == wins + losses, win_pct in [0, 1], and valid conference/season_type labels. |

---

## 3. Documented Validation Rules & Catalog

| Rule ID | Entity | Rule Specification |
| :---: | :--- | :--- |
| `VAL-001` | `games` | Primary key (game_id) in `games` must contain zero duplicates. |
| `VAL-002` | `teams` | Primary key (team_id) in `teams` must contain zero duplicates. |
| `VAL-003` | `players` | Primary key (player_id) in `players` must contain zero duplicates. |
| `VAL-004` | `player_game_stats` | Primary key (game_id, team_id, player_id) in `player_game_stats` must contain zero duplicates. |
| `VAL-005` | `games` | `game_date` must be valid timestamps within the season window and `season` / `season_label` must be consistent. |
| `VAL-006` | `games` | In `games`, `home_team_id` must never equal `away_team_id`. |
| `VAL-007` | `games` | Score columns must be positive integers, percentages in [0, 1], and `total_pts`, `point_margin`, and `home_win` mathematically consistent. |
| `VAL-008` | `player_game_stats` | Player counting stats and `minutes_played` must be >= 0, percentages in [0, 1], and made shots <= attempted shots. |
| `VAL-009` | `games` | Critical columns in `games` must not contain null/NaN values. |
| `VAL-010` | `teams` | Critical columns in `teams` must not contain null/NaN values. |
| `VAL-011` | `players` | Critical columns in `players` must not contain null/NaN values. |
| `VAL-012` | `player_game_stats` | Critical columns in `player_game_stats` must not contain null/NaN values. |
| `VAL-013` | `games` | Foreign key `games.home_team_id` must reference an existing `teams.team_id`. |
| `VAL-014` | `games` | Foreign key `games.away_team_id` must reference an existing `teams.team_id`. |
| `VAL-015` | `player_game_stats` | Every `player_game_stats.team_id` must match either `home_team_id` or `away_team_id` of the corresponding `game_id`. |
| `VAL-016` | `player_game_stats` | Foreign key `player_game_stats.game_id` must reference an existing `games.game_id`. |
| `VAL-017` | `player_game_stats` | Foreign key `player_game_stats.team_id` must reference an existing `teams.team_id`. |
| `VAL-018` | `player_game_stats` | Foreign key `player_game_stats.player_id` must reference an existing `players.player_id`. |
| `VAL-019` | `players` | Players must have `last_season >= first_season` and `0 <= career_games_played <= career_games_logged`. |
| `VAL-020` | `games` | All canonical `games` must have `game_status == 'Final'` and a valid `game_type`. |
| `VAL-021` | `player_game_stats` | DNP rows (`did_play == False`) must have 0 minutes/points, and `reb == oreb + dreb`. |
| `VAL-022` | `standings` | Primary key (team_id, season, season_type, standings_date) in `standings` must contain zero duplicates. |
| `VAL-023` | `standings` | Critical columns in `standings` must not contain null/NaN values. |
| `VAL-024` | `standings` | Foreign key `standings.team_id` must reference an existing `teams.team_id`. |
| `VAL-025` | `standings` | Standings must satisfy `games_played == wins + losses`, `win_pct` in [0, 1], and valid `conference` and `season_type`. |

---

## 4. Raw-to-Canonical Row Reconciliation & Resolved Issues

| Canonical Entity | Raw Source Rows | Excluded / Collapsed Rows | Canonical Parquet Rows | Exact Reconciliation Breakdown |
| :--- | ---: | ---: | ---: | :--- |
| `teams` | 30 | 0 | 30 | All 30 NBA franchises preserved (`4` missing `arena_capacity` stored as nullable `Int64`). |
| `games` | 26,651 | -128 | 26,523 | `-99` unscored Oct 2003 preseason rows (`PTS_home` is NaN) and `-29` duplicate Dec 2020 `GAME_ID` rows (`26,651 - 99 - 29 = 26,523`). |
| `player_game_stats` | 668,628 | -290 | 668,338 | `-1` invalid participant-team row (`GAME_ID=10500109, PLAYER_ID=2564, TEAM_ID=1610612756` where PHX did not play in IND @ ATL) and `-289` duplicate `(GAME_ID, TEAM_ID, PLAYER_ID)` rows across 12 duplicated Dec 2020 games (`668,628 - 1 - 289 = 668,338`). |
| `players` | 7,228 | -4,541 (net) | 2,687 | `7,228` team-season roster rows (2009–2019) collapsed to `1,769` unique players (`-5,459` multi-season duplicates) plus `+918` historical players (2003–2008, 2020–2022) backfilled from `games_details.csv` (`1,769 + 918 = 2,687`). |
| `standings` | 210,342 | -29 | 210,313 | `-29` duplicate `(TEAM_ID, SEASON_ID, STANDINGSDATE)` rows removed (`210,342 - 29 = 210,313`). |

### Summary of Resolved Raw Data Anomalies
- **Duplicate Game IDs (`VAL-001`)**: 29 duplicate `GAME_ID` records in raw `games.csv` (caused by floating-point rounding discrepancies in December 2020 games) were deduplicated during canonical ingestion.
- **Unscored 2003 Preseason Games (`VAL-007`, `VAL-009`, `VAL-020`)**: 99 rows in raw `games.csv` with all-null box scores were removed so that 100% of canonical `games` rows have `game_status == 'Final'` and valid positive integer scores.
- **Player Box-Score 290-Row Reconciliation (`VAL-004`, `VAL-015`)**: Exactly 290 rows were excluded from `games_details.csv` (`668,628 -> 668,338`): 1 row where `TEAM_ID = 1610612756` (PHX) did not match either participating team in `GAME_ID = 10500109` (IND @ ATL), and 289 duplicate `(GAME_ID, TEAM_ID, PLAYER_ID)` rows.
- **Malformed/Negative Minutes Played (`VAL-008`, `VAL-021`)**: 12 negative `MIN` strings (e.g. `'-9'`), 1,463 `MM:60` rounding strings, and 109,681 DNP null rows in `games_details.csv` were standardized to `minutes_played >= 0.0` with an explicit `did_play` boolean indicator.
- **Unmapped Historical Players (`VAL-018`, `VAL-019`)**: 918 distinct `PLAYER_ID`s present in `games_details.csv` (2003–2008 and 2020–2022) but absent from `players.csv` (which only covered 2009–2019) were unified into canonical `players.parquet`, achieving 100% foreign-key referential integrity.
