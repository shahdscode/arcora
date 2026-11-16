# NBA Dataset Schema, Relationship, and Data Quality Audit Report

## 1. Executive Summary

This audit evaluates all five raw CSV files in the Nathan Lauga NBA Games dataset (`games.csv`, `games_details.csv`, `players.csv`, `ranking.csv`, `teams.csv`) prior to designing and executing Arcora's canonical ingestion pipeline.

| Metric | Value |
| :--- | ---: |
| **Total Raw Tables Inspected** | 5 |
| **Total Raw Records Across Tables** | 912,879 |
| **Unique Games (`games.csv`)** | 26,622 (out of 26,651 rows) |
| **Duplicate `GAME_ID` Rows (`games.csv`)** | 29 |
| **Games Missing Scores (`games.csv`)** | 99 |
| **Duplicate Player-Game Rows (`games_details.csv`)** | 289 |
| **DNP / Inactive Player Rows (`games_details.csv`)** | 109,690 |
| **Distinct Players (`players.csv` vs `games_details.csv`)** | 1,769 vs 2,687 |

---

## 2. Source File Schema & Column Audit

### 2.1 `games.csv`

- **Row Count**: 26,651
- **Column Count**: 21
- **Candidate Primary Key**: (`GAME_ID`)
- **Primary Key Duplicates**: 29
- **Exact Full-Row Duplicates**: 0

| Column | Inferred Dtype | Non-Null | Null Count | Null % | Unique | Sample Values |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `GAME_DATE_EST` | `str` | 26,651 | 0 | 0.00% | 4,304 | `2022-12-22`, `2022-12-21`, `2022-12-20` |
| `GAME_ID` | `int64` | 26,651 | 0 | 0.00% | 26,622 | `22200477`, `22200478`, `22200466` |
| `GAME_STATUS_TEXT` | `str` | 26,651 | 0 | 0.00% | 1 | `Final` |
| `HOME_TEAM_ID` | `int64` | 26,651 | 0 | 0.00% | 30 | `1610612740`, `1610612762`, `1610612739` |
| `VISITOR_TEAM_ID` | `int64` | 26,651 | 0 | 0.00% | 30 | `1610612759`, `1610612764`, `1610612749` |
| `SEASON` | `int64` | 26,651 | 0 | 0.00% | 20 | `2022`, `2021`, `2020` |
| `TEAM_ID_home` | `int64` | 26,651 | 0 | 0.00% | 30 | `1610612740`, `1610612762`, `1610612739` |
| `PTS_home` | `float64` | 26,552 | 99 | 0.37% | 100 | `126.0`, `120.0`, `114.0` |
| `FG_PCT_home` | `float64` | 26,552 | 99 | 0.37% | 404 | `0.484`, `0.488`, `0.482` |
| `FT_PCT_home` | `float64` | 26,552 | 99 | 0.37% | 410 | `0.926`, `0.952`, `0.786` |
| `FG3_PCT_home` | `float64` | 26,552 | 99 | 0.37% | 437 | `0.382`, `0.457`, `0.313` |
| `AST_home` | `float64` | 26,552 | 99 | 0.37% | 42 | `25.0`, `16.0`, `22.0` |
| `REB_home` | `float64` | 26,552 | 99 | 0.37% | 55 | `46.0`, `40.0`, `37.0` |
| `TEAM_ID_away` | `int64` | 26,651 | 0 | 0.00% | 30 | `1610612759`, `1610612764`, `1610612749` |
| `PTS_away` | `float64` | 26,552 | 99 | 0.37% | 106 | `117.0`, `112.0`, `106.0` |
| `FG_PCT_away` | `float64` | 26,552 | 99 | 0.37% | 393 | `0.478`, `0.561`, `0.47` |
| `FT_PCT_away` | `float64` | 26,552 | 99 | 0.37% | 387 | `0.815`, `0.765`, `0.682` |
| `FG3_PCT_away` | `float64` | 26,552 | 99 | 0.37% | 417 | `0.321`, `0.333`, `0.433` |
| `AST_away` | `float64` | 26,552 | 99 | 0.37% | 41 | `23.0`, `20.0`, `15.0` |
| `REB_away` | `float64` | 26,552 | 99 | 0.37% | 52 | `44.0`, `37.0`, `46.0` |
| `HOME_TEAM_WINS` | `int64` | 26,651 | 0 | 0.00% | 2 | `1`, `0` |

### 2.2 `games_details.csv`

- **Row Count**: 668,628
- **Column Count**: 29
- **Candidate Primary Key**: (`GAME_ID`, `TEAM_ID`, `PLAYER_ID`)
- **Primary Key Duplicates**: 289
- **Exact Full-Row Duplicates**: 0

| Column | Inferred Dtype | Non-Null | Null Count | Null % | Unique | Sample Values |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `GAME_ID` | `int64` | 668,628 | 0 | 0.00% | 26,523 | `22200477`, `22200478`, `22200466` |
| `TEAM_ID` | `int64` | 668,628 | 0 | 0.00% | 30 | `1610612759`, `1610612740`, `1610612764` |
| `TEAM_ABBREVIATION` | `str` | 668,628 | 0 | 0.00% | 34 | `SAS`, `NOP`, `WAS` |
| `TEAM_CITY` | `str` | 668,628 | 0 | 0.00% | 33 | `San Antonio`, `New Orleans`, `Washington` |
| `PLAYER_ID` | `int64` | 668,628 | 0 | 0.00% | 2,687 | `1629641`, `1631110`, `1627751` |
| `PLAYER_NAME` | `str` | 668,628 | 0 | 0.00% | 2,664 | `Romeo Langford`, `Jeremy Sochan`, `Jakob Poeltl` |
| `NICKNAME` | `str` | 53,037 | 615,591 | 92.07% | 558 | `Romeo`, `Jeremy`, `Jakob` |
| `START_POSITION` | `str` | 255,765 | 412,863 | 61.75% | 3 | `F`, `C`, `G` |
| `COMMENT` | `str` | 109,689 | 558,939 | 83.59% | 5,348 | `DNP - Coach's Decision`, `DNP - Injury/Illness`, `DND - Injury/Illness` |
| `MIN` | `str` | 558,938 | 109,690 | 16.41% | 3,802 | `18:06`, `31:01`, `21:42` |
| `FGM` | `float64` | 558,938 | 109,690 | 16.41% | 26 | `1.0`, `7.0`, `6.0` |
| `FGA` | `float64` | 558,938 | 109,690 | 16.41% | 47 | `1.0`, `14.0`, `9.0` |
| `FG_PCT` | `float64` | 558,938 | 109,690 | 16.41% | 309 | `1.0`, `0.5`, `0.667` |
| `FG3M` | `float64` | 558,938 | 109,690 | 16.41% | 15 | `0.0`, `2.0`, `1.0` |
| `FG3A` | `float64` | 558,938 | 109,690 | 16.41% | 25 | `0.0`, `4.0`, `6.0` |
| `FG3_PCT` | `float64` | 558,938 | 109,690 | 16.41% | 109 | `0.0`, `0.5`, `0.167` |
| `FTM` | `float64` | 558,938 | 109,690 | 16.41% | 27 | `0.0`, `7.0`, `1.0` |
| `FTA` | `float64` | 558,938 | 109,690 | 16.41% | 33 | `0.0`, `10.0`, `1.0` |
| `FT_PCT` | `float64` | 558,938 | 109,690 | 16.41% | 165 | `0.0`, `0.7`, `1.0` |
| `OREB` | `float64` | 558,938 | 109,690 | 16.41% | 18 | `1.0`, `6.0`, `0.0` |
| `DREB` | `float64` | 558,938 | 109,690 | 16.41% | 25 | `1.0`, `3.0`, `9.0` |
| `REB` | `float64` | 558,938 | 109,690 | 16.41% | 32 | `2.0`, `9.0`, `4.0` |
| `AST` | `float64` | 558,938 | 109,690 | 16.41% | 26 | `0.0`, `6.0`, `1.0` |
| `STL` | `float64` | 558,938 | 109,690 | 16.41% | 11 | `1.0`, `3.0`, `0.0` |
| `BLK` | `float64` | 558,938 | 109,690 | 16.41% | 13 | `0.0`, `2.0`, `1.0` |
| `TO` | `float64` | 558,938 | 109,690 | 16.41% | 13 | `2.0`, `0.0`, `1.0` |
| `PF` | `float64` | 558,938 | 109,690 | 16.41% | 8 | `5.0`, `1.0`, `4.0` |
| `PTS` | `float64` | 558,938 | 109,690 | 16.41% | 66 | `2.0`, `23.0`, `13.0` |
| `PLUS_MINUS` | `float64` | 535,277 | 133,351 | 19.94% | 108 | `-2.0`, `-14.0`, `-4.0` |

### 2.3 `players.csv`

- **Row Count**: 7,228
- **Column Count**: 4
- **Candidate Primary Key**: (`PLAYER_ID`, `TEAM_ID`, `SEASON`)
- **Primary Key Duplicates**: 0
- **Exact Full-Row Duplicates**: 0

| Column | Inferred Dtype | Non-Null | Null Count | Null % | Unique | Sample Values |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `PLAYER_NAME` | `str` | 7,228 | 0 | 0.00% | 1,749 | `Royce O'Neale`, `Bojan Bogdanovic`, `Rudy Gobert` |
| `TEAM_ID` | `int64` | 7,228 | 0 | 0.00% | 30 | `1610612762`, `1610612755`, `1610612752` |
| `PLAYER_ID` | `int64` | 7,228 | 0 | 0.00% | 1,769 | `1626220`, `202711`, `203497` |
| `SEASON` | `int64` | 7,228 | 0 | 0.00% | 11 | `2019`, `2018`, `2017` |

### 2.4 `ranking.csv`

- **Row Count**: 210,342
- **Column Count**: 13
- **Candidate Primary Key**: (`TEAM_ID`, `SEASON_ID`, `STANDINGSDATE`)
- **Primary Key Duplicates**: 29
- **Exact Full-Row Duplicates**: 0

| Column | Inferred Dtype | Non-Null | Null Count | Null % | Unique | Sample Values |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `TEAM_ID` | `int64` | 210,342 | 0 | 0.00% | 30 | `1610612743`, `1610612763`, `1610612740` |
| `LEAGUE_ID` | `int64` | 210,342 | 0 | 0.00% | 1 | `0` |
| `SEASON_ID` | `int64` | 210,342 | 0 | 0.00% | 41 | `22022`, `12022`, `22021` |
| `STANDINGSDATE` | `str` | 210,342 | 0 | 0.00% | 7,023 | `2022-12-22`, `2022-12-21`, `2022-12-20` |
| `CONFERENCE` | `str` | 210,342 | 0 | 0.00% | 2 | `West`, `East` |
| `TEAM` | `str` | 210,342 | 0 | 0.00% | 34 | `Denver`, `Memphis`, `New Orleans` |
| `G` | `int64` | 210,342 | 0 | 0.00% | 83 | `30`, `31`, `32` |
| `W` | `int64` | 210,342 | 0 | 0.00% | 74 | `19`, `17`, `16` |
| `L` | `int64` | 210,342 | 0 | 0.00% | 73 | `11`, `12`, `13` |
| `W_PCT` | `float64` | 210,342 | 0 | 0.00% | 895 | `0.633`, `0.613`, `0.594` |
| `HOME_RECORD` | `str` | 210,342 | 0 | 0.00% | 800 | `10-3`, `13-2`, `13-4` |
| `ROAD_RECORD` | `str` | 210,342 | 0 | 0.00% | 758 | `9-8`, `6-9`, `6-8` |
| `RETURNTOPLAY` | `float64` | 3,990 | 206,352 | 98.10% | 2 | `1.0`, `0.0` |

### 2.5 `teams.csv`

- **Row Count**: 30
- **Column Count**: 14
- **Candidate Primary Key**: (`TEAM_ID`)
- **Primary Key Duplicates**: 0
- **Exact Full-Row Duplicates**: 0

| Column | Inferred Dtype | Non-Null | Null Count | Null % | Unique | Sample Values |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `LEAGUE_ID` | `int64` | 30 | 0 | 0.00% | 1 | `0` |
| `TEAM_ID` | `int64` | 30 | 0 | 0.00% | 30 | `1610612737`, `1610612738`, `1610612740` |
| `MIN_YEAR` | `int64` | 30 | 0 | 0.00% | 15 | `1949`, `1946`, `2002` |
| `MAX_YEAR` | `int64` | 30 | 0 | 0.00% | 1 | `2019` |
| `ABBREVIATION` | `str` | 30 | 0 | 0.00% | 30 | `ATL`, `BOS`, `NOP` |
| `NICKNAME` | `str` | 30 | 0 | 0.00% | 30 | `Hawks`, `Celtics`, `Pelicans` |
| `YEARFOUNDED` | `int64` | 30 | 0 | 0.00% | 15 | `1949`, `1946`, `2002` |
| `CITY` | `str` | 30 | 0 | 0.00% | 29 | `Atlanta`, `Boston`, `New Orleans` |
| `ARENA` | `str` | 30 | 0 | 0.00% | 29 | `State Farm Arena`, `TD Garden`, `Smoothie King Center` |
| `ARENACAPACITY` | `float64` | 26 | 4 | 13.33% | 24 | `18729.0`, `18624.0`, `21711.0` |
| `OWNER` | `str` | 30 | 0 | 0.00% | 30 | `Tony Ressler`, `Wyc Grousbeck`, `Tom Benson` |
| `GENERALMANAGER` | `str` | 30 | 0 | 0.00% | 30 | `Travis Schlenk`, `Danny Ainge`, `Trajan Langdon` |
| `HEADCOACH` | `str` | 30 | 0 | 0.00% | 30 | `Lloyd Pierce`, `Brad Stevens`, `Alvin Gentry` |
| `DLEAGUEAFFILIATION` | `str` | 30 | 0 | 0.00% | 28 | `Erie Bayhawks`, `Maine Red Claws`, `No Affiliate` |

---

## 3. Entity Relationships & Foreign Key Audit

### 3.1 How Home and Away Teams Are Represented
- In `games.csv`, each game row represents one matchup with `HOME_TEAM_ID` (duplicated in `TEAM_ID_home`) and `VISITOR_TEAM_ID` (duplicated in `TEAM_ID_away`).
- Home team statistics carry the `_home` suffix (`PTS_home`, `FG_PCT_home`, `FT_PCT_home`, `FG3_PCT_home`, `AST_home`, `REB_home`), while away team statistics carry the `_away` suffix.
- `HOME_TEAM_WINS` is a binary target indicator (`1` if `PTS_home > PTS_away`, else `0`).
- In `games_details.csv`, each row records a single player's appearance for `TEAM_ID` in `GAME_ID`. Joining `games_details.TEAM_ID` to `games.HOME_TEAM_ID` / `games.VISITOR_TEAM_ID` determines whether the player competed at home or away.

### 3.2 Game ID Structure & Encoding
- `GAME_ID` is an 8-digit integer (standard 10-digit NBA Game ID with leading `00` stripped by integer parsing).
- The leading digit encodes game type: `1` = Preseason (1,774 rows), `2` = Regular Season (23,273 rows), `4` = Playoffs (1,591 rows), `5` = Play-In Tournament (13 rows).

### 3.3 Foreign Key Verification Summary

| Relationship | Source (`Column`) | Target (`Column`) | Unmatched Rows | Distinct Unmatched Keys | Status | Notes |
| :--- | :--- | :--- | ---: | ---: | :---: | :--- |
| `games_home_team_to_teams` | `games.csv` (`HOME_TEAM_ID`) | `teams.csv` (`TEAM_ID`) | 0 | 0 | **PASS** | Every home team ID in games.csv maps cleanly to teams.csv. |
| `games_visitor_team_to_teams` | `games.csv` (`VISITOR_TEAM_ID`) | `teams.csv` (`TEAM_ID`) | 0 | 0 | **PASS** | Every away/visitor team ID in games.csv maps cleanly to teams.csv. |
| `details_game_to_games` | `games_details.csv` (`GAME_ID`) | `games.csv` (`GAME_ID`) | 0 | 0 | **PASS** | Every player box-score record in games_details.csv links to a valid GAME_ID. |
| `games_to_details_coverage` | `games.csv` (`GAME_ID`) | `games_details.csv` (`GAME_ID`) | 99 | 99 | **WARN** | 99 games in the 2003 preseason/early-season lack box-score rows in games_details.csv and also have null score columns in games.csv. |
| `details_team_to_teams` | `games_details.csv` (`TEAM_ID`) | `teams.csv` (`TEAM_ID`) | 0 | 0 | **PASS** | All TEAM_ID values in games_details.csv exist in teams.csv. |
| `details_player_to_players_csv` | `games_details.csv` (`PLAYER_ID`) | `players.csv` (`PLAYER_ID`) | 79,093 | 918 | **WARN** | players.csv only covers 2009-2019 rosters (1,769 unique players), whereas games_details.csv spans 2003-2022 (2,687 unique players). Canonical players dimension must union players.csv and games_details.csv. |
| `ranking_team_to_teams` | `ranking.csv` (`TEAM_ID`) | `teams.csv` (`TEAM_ID`) | 0 | 0 | **PASS** | All TEAM_ID values in ranking.csv exist in teams.csv. |

---

## 4. Identified Data Quality Issues & Anomalies

| ID | Severity | Table | Affected Rows | Summary | Remediation Plan |
| :---: | :---: | :--- | ---: | :--- | :--- |
| `DQ-001` | **HIGH** | `games.csv` | 29 | Duplicate GAME_ID records with float-rounding differences: Found 29 duplicate GAME_ID rows (58 total rows across 29 games in Dec 2020) where identical games appear twice with minor IEEE-754 float representation differences (e.g., 0.343 vs 0.3429999999999999). | Deduplicate games by primary key GAME_ID during canonical ingestion after sorting deterministically and rounding percentage columns to 3 decimal places. |
| `DQ-002` | **HIGH** | `games.csv` | 99 | Unrecorded 2003 preseason/early games with all-null box scores: Found 99 game rows in season 2003 (Oct 2003) where PTS_home, PTS_away, FG_PCT, FT_PCT, FG3_PCT, AST, and REB are all NaN, and HOME_TEAM_WINS defaults to 0 despite no score. | Filter out or explicitly flag unplayed/unscored games with null PTS_home or PTS_away in the canonical games dataset so downstream models train only on valid completed games. |
| `DQ-003` | **LOW** | `games.csv` | 26,651 | Redundant team identifier columns in games.csv: HOME_TEAM_ID is 100% identical to TEAM_ID_home across all 26,651 rows, and VISITOR_TEAM_ID is 100% identical to TEAM_ID_away across all 26,651 rows. | Drop redundant TEAM_ID_home and TEAM_ID_away columns and standardize on home_team_id and away_team_id in the canonical games schema. |
| `DQ-004` | **HIGH** | `games_details.csv` | 289 | Duplicate (GAME_ID, TEAM_ID, PLAYER_ID) player box-score rows: Found 289 duplicate player-game records in games_details.csv (primarily corresponding to the duplicated Dec 2020 games). | Deduplicate player_game_stats on (game_id, team_id, player_id) during ingestion. |
| `DQ-005` | **MEDIUM** | `games_details.csv` | 109,690 | Did-Not-Play (DNP/Inactive) rows with null statistical columns: Found 109,690 rows where MIN and all 18 numeric box-score columns (FGM through PTS) are NaN because the player did not enter the game (e.g., 'DNP - Coach's Decision'). Additionally, PLUS_MINUS is null in 133,351 rows. | Add boolean did_play flag in canonical player_game_stats; preserve DNP status with 0.0 minutes and clean nullable integer/float stats. |
| `DQ-006` | **MEDIUM** | `games_details.csv` | 24,157 | Inconsistent and negative string formats in MIN (minutes played): Among 558,938 non-null MIN strings: 23,661 lack a ':SS' component (e.g. '23'), 484 contain floating-point decimals before the colon (e.g. '12.000000:00'), and 12 contain corrupted negative strings (e.g. '-9', '-5'). | Parse MIN strings into standardized numeric minutes_played (float >= 0.0), clamping or setting malformed negative minute strings to null/0.0. |
| `DQ-007` | **HIGH** | `players.csv` | 79,093 | Incomplete historical player roster coverage in players.csv: players.csv only contains seasons 2009-2019 (1,769 unique players), leaving 918 unique players (79,093 box-score rows) from 2003-2008 and 2020-2022 absent from players.csv. | Construct canonical players table by combining players.csv with distinct (PLAYER_ID, PLAYER_NAME) records from games_details.csv so 100% of player foreign keys resolve. |
| `DQ-008` | **MEDIUM** | `ranking.csv` | 29 | Duplicate daily team standings rows and prefixed SEASON_ID encoding: Found 29 duplicate (TEAM_ID, SEASON_ID, STANDINGSDATE) rows, and SEASON_ID uses a 5-digit prefix format (1YYYY for preseason, 2YYYY for regular season) instead of the 4-digit SEASON integer. | Normalize season_id into canonical season (int YYYY) and season_type ('Preseason' vs 'Regular Season') and deduplicate daily standings. |
| `DQ-009` | **LOW** | `teams.csv` | 4 | Missing ARENACAPACITY for 4 NBA franchises: ARENACAPACITY is NaN for 4 franchises: NOP (Pelicans), BKN (Nets), PHI (76ers), and PHX (Suns). | Use nullable integer Int64 or known official arena capacities for the 4 franchises during canonical team normalization. |

---

## 5. Dataset Limitations

1. No explicit game-type column in games.csv: Preseason (GAME_ID prefix '1'), Regular Season ('2'), All-Star/Special ('3'), Playoff ('4'), and Play-In ('5') games are mixed together and must be distinguished via GAME_ID prefix.
2. Partial 2022-23 season: Data collection ends on 2022-12-22 (542 games in season 2022).
3. Historical franchise relocations/renames: teams.csv stores only a single current row per TEAM_ID (e.g., OKC for Seattle SuperSonics/Thunder, NOP for New Orleans/OKC Hornets/Pelicans, BKN for New Jersey/Brooklyn Nets).
4. Missing play-by-play and quarter-by-quarter scoring: games.csv provides only final game totals and shooting percentages, without quarter splits or overtime indicators.
5. Missing betting lines, referee assignments, and injury reports prior to tip-off.
