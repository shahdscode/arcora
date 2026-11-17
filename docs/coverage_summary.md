# Historical NBA Dataset Coverage Summary

## 1. Executive Coverage Metrics

| Metric | Value |
| :--- | :--- |
| **Seasons Covered** | 20 season IDs (`2003` to `2022`): 19 complete seasons (2003-04–2021-22) + 1 partial season (2022-23 through 2022-12-22) |
| **Game Date Range (`games.csv`)** | `2003-10-05` to `2022-12-22` |
| **Standings Date Range (`ranking.csv`)** | `2003-10-01` to `2022-12-22` |
| **Total Game Records (`games.csv`)** | 26,651 rows (26,622 unique `GAME_ID`s) |
| **Unique Games with Box Scores (`games_details.csv`)** | 26,523 unique `GAME_ID`s |
| **NBA Franchises (`teams.csv`)** | 30 franchises |
| **Unique Players (`players.csv` / `games_details.csv`)** | 1,769 (roster table) / 2,687 (box-score logs) |

## 2. Season-by-Season Game Coverage

| Season ID | Season Label | Total Game Rows | Completed (`Final`) | First Game Date | Last Game Date | Active Franchises |
| :---: | :---: | ---: | ---: | :---: | :---: | ---: |
| `2003` | 2003-04 | 1,385 | 1,385 | `2003-10-05` | `2004-06-15` | 29 |
| `2004` | 2004-05 | 1,362 | 1,362 | `2004-10-22` | `2005-06-23` | 30 |
| `2005` | 2005-06 | 1,432 | 1,432 | `2005-10-10` | `2006-06-20` | 30 |
| `2006` | 2006-07 | 1,419 | 1,419 | `2006-10-09` | `2007-06-14` | 30 |
| `2007` | 2007-08 | 1,411 | 1,411 | `2007-10-06` | `2008-06-17` | 30 |
| `2008` | 2008-09 | 1,425 | 1,425 | `2008-10-05` | `2009-06-14` | 30 |
| `2009` | 2009-10 | 1,424 | 1,424 | `2009-10-01` | `2010-06-17` | 30 |
| `2010` | 2010-11 | 1,422 | 1,422 | `2010-10-04` | `2011-06-12` | 30 |
| `2011` | 2011-12 | 1,104 | 1,104 | `2011-12-16` | `2012-06-21` | 30 |
| `2012` | 2012-13 | 1,420 | 1,420 | `2012-10-06` | `2013-06-20` | 30 |
| `2013` | 2013-14 | 1,427 | 1,427 | `2013-10-05` | `2014-06-15` | 30 |
| `2014` | 2014-15 | 1,418 | 1,418 | `2014-10-04` | `2015-06-16` | 30 |
| `2015` | 2015-16 | 1,416 | 1,416 | `2015-10-02` | `2016-06-19` | 30 |
| `2016` | 2016-17 | 1,405 | 1,405 | `2016-10-01` | `2017-06-12` | 30 |
| `2017` | 2017-18 | 1,382 | 1,382 | `2017-09-30` | `2018-06-08` | 30 |
| `2018` | 2018-19 | 1,378 | 1,378 | `2018-09-28` | `2019-06-13` | 30 |
| `2019` | 2019-20 | 1,241 | 1,241 | `2019-10-04` | `2020-10-11` | 30 |
| `2020` | 2020-21 | 1,249 | 1,249 | `2020-12-11` | `2021-07-20` | 30 |
| `2021` | 2021-22 | 1,389 | 1,389 | `2021-10-03` | `2022-06-16` | 30 |
| `2022` | 2022-23 | 542 | 542 | `2022-09-30` | `2022-12-22` | 30 |

## 3. Historical Coverage Notes

- **19 Complete Seasons + 1 Partial Season (`2022-23`)**: Seasons `2003-04` through `2021-22` (19 seasons) include full regular-season and postseason schedules. Season `2022-23` (`SEASON = 2022`) is a **partial in-progress season** ending on the dataset snapshot date `2022-12-22` (`542` games: `472` Regular Season, `70` Preseason, `0` Playoffs). Downstream training/validation splits must account for this cutoff.
- **Game Stage Breakdown (26,651 Raw Rows -> 26,523 Canonical Completed Games)**: The dataset is **not** exclusively regular-season games; it spans four competition stages encoded by the leading digit of `GAME_ID`:
  - **Regular Season (`2` prefix)**: `23,273` raw rows -> `23,244` canonical games (after removing `29` duplicate December 2020 `GAME_ID` rows).
  - **Preseason (`1` prefix)**: `1,774` raw rows -> `1,675` canonical completed games (after removing `99` unscored October 2003 preseason rows).
  - **Playoffs (`4` prefix)**: `1,591` raw rows -> `1,591` canonical games.
  - **Play-In Tournament (`5` prefix)**: `13` raw rows -> `13` canonical games.
- **Expansion from 29 to 30 Franchises**: The `2003` (2003-04) season has 29 active NBA franchises; the Charlotte Bobcats (later Hornets, `TEAM_ID = 1610612766`) joined in the `2004` (2004-05) season, bringing total active franchises to 30 from 2004 onward.
- **Shortened Seasons**:
  - `2011` (2011-12): Lockout-shortened 66-game regular season starting in December 2011.
  - `2019` (2019-20) & `2020` (2020-21): COVID-19 pandemic disruptions (2019-20 suspension and Orlando Bubble; 2020-21 shortened 72-game regular season).
  - `2022` (2022-23): Partial in-progress season captured through `2022-12-22` (dataset snapshot cutoff).
- **Player Roster Table Scope (`players.csv`)**: Note that `players.csv` only covers team-season rosters from `2009` through `2019` (7,228 rows, 1,769 unique players), whereas `games_details.csv` captures box-score appearances across the entire 2003–2022 span (2,687 unique players).
