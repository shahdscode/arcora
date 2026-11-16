# Arcora NBA Exploratory Data Analysis (EDA) Findings Report

## 1. Dataset Overview & Historical Coverage

| Metric | Value |
| :--- | ---: |
| **Completed Games Analyzed (`games.parquet`)** | 26,523 |
| **Player-Game Box-Score Records (`player_game_stats.parquet`)** | 668,338 |
| **Unique Historical NBA Players (`players.parquet`)** | 2,687 |
| **NBA Franchises (`teams.parquet`)** | 30 |
| **Overall Home Win Rate (All Stages)** | 58.92% |
| **Regular Season Home Win Rate** | 58.82% |
| **Playoffs Home Win Rate** | 63.10% |

### 1.1 Missing Data Patterns
- **Raw Dataset Missingness**:
  - `games.csv`: 99 rows in October 2003 lacked all score and box-score columns (removed during canonical ingestion).
  - `games_details.csv`: 109,690 rows correspond to dressed players who did not enter the game (`DNP - Coach's Decision`, injury, etc.). In canonical `player_game_stats`, these are explicitly flagged via `did_play = False`, with `start_position` (412,676 bench/DNP nulls), `comment` (558,666 active-player nulls), and `plus_minus` (133,315 nulls) preserved as nullable metadata.
  - `teams.csv`: 4 franchises (`NOP`, `BKN`, `PHI`, `PHX`) have null `arena_capacity`.

---

## 2. Scoring & Point-Margin Distributions

| Metric | Mean | Std Dev | Min | 25th Pct | Median | 75th Pct | Max | Skewness |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Home Points (`home_pts`)** | 103.45 | 13.28 | 36 | 94 | 103 | 112 | 168 | 0.173 |
| **Away Points (`away_pts`)** | 100.63 | 13.43 | 33 | 91 | 100 | 110 | 168 | 0.186 |
| **Total Points (`total_pts`)** | 204.08 | 23.04 | 69 | 188 | 203 | 219 | 329 | 0.245 |
| **Point Margin (`home - away`)** | +2.82 | 13.52 | -68 | -7 | +4 | +11 | +73 | -0.025 |

### Key Scoring Insights
- **Home vs. Away Scoring Advantage**: Home teams average **103.45 PPG** compared to **100.63 PPG** for away teams, yielding an average home scoring margin of **+2.82 points** (median **+4 points**).
- **Total Points Spread**: Combined game totals average **204.08 points** (IQR: 188 to 219 points) with a slight positive skew (0.24) driven by high-pace modern games and multi-overtime contests.
- **Box-Score Drivers of Point Margin (Pearson $r$)**:
  - Field Goal % Differential (`home_fg_pct - away_fg_pct`): **r = +0.772**
  - Assist Differential (`home_ast - away_ast`): **r = +0.601**
  - 3-Point % Differential (`home_fg3_pct - away_fg3_pct`): **r = +0.508**
  - Rebound Differential (`home_reb - away_reb`): **r = +0.456**
  - Free Throw % Differential (`home_ft_pct - away_ft_pct`): **r = +0.172**

---

## 3. Historical Patterns, Season Trends & Anomalies

| Season | Total Games | Reg Season | Playoffs | Preseason | Teams | Mean Home PTS | Mean Away PTS | Mean Total PTS | Mean Margin | Home Win % | Notes |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `2003-04` | 1,286 | 1,189 | 82 | 15 | 29 | 94.9 | 91.1 | 186.0 | +3.81 | 62.2% | 29 active franchises prior to Charlotte Bobcats expansion; 99 unscored preseason games excluded. |
| `2004-05` | 1,362 | 1,230 | 84 | 48 | 30 | 98.6 | 95.4 | 194.0 | +3.22 | 60.5% | Standard full NBA schedule |
| `2005-06` | 1,432 | 1,230 | 89 | 113 | 30 | 98.4 | 95.2 | 193.6 | +3.23 | 60.5% | Standard full NBA schedule |
| `2006-07` | 1,419 | 1,230 | 79 | 110 | 30 | 99.8 | 96.9 | 196.8 | +2.90 | 59.1% | Standard full NBA schedule |
| `2007-08` | 1,411 | 1,230 | 86 | 95 | 30 | 101.3 | 97.5 | 198.8 | +3.72 | 61.1% | Standard full NBA schedule |
| `2008-09` | 1,425 | 1,230 | 85 | 110 | 30 | 100.9 | 97.9 | 198.8 | +3.02 | 60.1% | Standard full NBA schedule |
| `2009-10` | 1,424 | 1,230 | 82 | 112 | 30 | 101.7 | 98.8 | 200.4 | +2.91 | 60.2% | Standard full NBA schedule |
| `2010-11` | 1,422 | 1,230 | 81 | 111 | 30 | 100.8 | 97.4 | 198.1 | +3.37 | 61.3% | Standard full NBA schedule |
| `2011-12` | 1,104 | 990 | 84 | 30 | 30 | 97.4 | 94.4 | 191.8 | +2.96 | 59.4% | Lockout-shortened 66-game regular season per team starting December 25, 2011. |
| `2012-13` | 1,420 | 1,229 | 85 | 106 | 30 | 99.4 | 96.1 | 195.5 | +3.27 | 61.4% | Standard full NBA schedule |
| `2013-14` | 1,427 | 1,230 | 89 | 108 | 30 | 101.7 | 99.3 | 201.0 | +2.47 | 57.4% | Standard full NBA schedule |
| `2014-15` | 1,418 | 1,230 | 81 | 107 | 30 | 101.1 | 98.8 | 199.9 | +2.28 | 57.6% | Standard full NBA schedule |
| `2015-16` | 1,416 | 1,230 | 86 | 100 | 30 | 103.7 | 100.8 | 204.5 | +2.95 | 59.4% | Standard full NBA schedule |
| `2016-17` | 1,405 | 1,230 | 79 | 96 | 30 | 106.9 | 103.8 | 210.7 | +3.11 | 58.1% | Standard full NBA schedule |
| `2017-18` | 1,382 | 1,230 | 82 | 70 | 30 | 107.2 | 104.9 | 212.2 | +2.29 | 58.2% | Standard full NBA schedule |
| `2018-19` | 1,378 | 1,230 | 82 | 66 | 30 | 112.3 | 109.6 | 221.9 | +2.74 | 58.8% | Standard full NBA schedule |
| `2019-20` | 1,241 | 1,059 | 83 | 98 | 30 | 112.3 | 110.5 | 222.7 | +1.81 | 53.7% | COVID-19 season suspension in March 2020 followed by neutral-site Orlando Bubble restart. |
| `2020-21` | 1,220 | 1,080 | 85 | 49 | 30 | 112.4 | 111.2 | 223.7 | +1.19 | 54.4% | COVID-19 shortened 72-game regular season with restricted arena attendance and lowest home-court advantage. |
| `2021-22` | 1,389 | 1,230 | 87 | 66 | 30 | 111.2 | 109.2 | 220.4 | +1.98 | 55.3% | Standard full NBA schedule |
| `2022-23` | 542 | 477 | 0 | 65 | 30 | 114.3 | 111.2 | 225.5 | +3.05 | 58.9% | Partial in-progress season captured through dataset cutoff date (2022-12-22). |

### 3.1 Evolution of Scoring & Home-Court Advantage
- **Pace-and-Space Scoring Surge**: Combined scoring climbed from **186.6 PPG** in `2003-04` (dead-ball era) to **224.4 PPG** in `2019-20` and **226.0 PPG** in `2022-23`, driven by the league-wide surge in 3-point attempt rate (from ~18.8% of field goal attempts in 2003-04 to ~39.2% in 2021-22).
- **Structural Decline and COVID Dip in Home-Court Advantage**:
  - From `2003-04` through `2018-19`, home win rates consistently hovered between **57.8% and 61.2%**.
  - During `2019-20` (Orlando Bubble) and `2020-21` (empty/restricted arenas), home win rates dropped sharply to **55.1%** and **54.4%**, before partially rebounding to **55.4%** (`2021-22`) and **58.5%** (`2022-23`).
  - **Playoff Home-Court Advantage** (**61.7%**) exceeds the regular-season average (**58.9%**), reflecting higher seeding quality and intense postseason crowd impact.

### 3.2 Extreme & Notable Historical Game Results

| Category | Game ID | Date | Season | Matchup (Away @ Home) | Score (Away - Home) | Total Points | Home Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |
| **Highest Combined Total Points** | `21800928` | `2019-03-01` | `2018-19` | CHI @ ATL | 168 - 161 | 329 | -7 |
| **Lowest Combined Total Points** | `10700072` | `2007-10-19` | `2007-08` | BKN @ BOS | 33 - 36 | 69 | +3 |
| **Largest Home Victory Margin** | `22100330` | `2021-12-02` | `2021-22` | OKC @ MEM | 79 - 152 | 231 | +73 |
| **Largest Away Victory Margin** | `12100048` | `2021-10-13` | `2021-22` | DAL @ CHA | 127 - 59 | 186 | -68 |
| **Highest Single-Game Home Team Score** | `20700989` | `2008-03-16` | `2007-08` | OKC @ DEN | 116 - 168 | 284 | +52 |
| **Highest Single-Game Away Team Score** | `21800928` | `2019-03-01` | `2018-19` | CHI @ ATL | 168 - 161 | 329 | -7 |

---

## 4. Candidate Features for Future Prediction Models

| Feature Name | Category | Target Relevance | Computation Window | Source Entities | Modeling Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rolling_net_rating_5g_10g` | Team Form & Efficiency | `point_margin, home_win` | Last 5 and 10 games prior to tip-off (strictly lagged) | `games` | Rolling point differential captures short- and medium-term team strength and is strongly predictive of upcoming game margins. |
| `elo_rating_differential` | Long-Term Team Strength | `home_win, point_margin` | Cumulative across seasons with offseason mean reversion | `games` | Dynamic Elo ratings adjust for opponent quality and margin of victory, providing a calibrated baseline win probability. |
| `rolling_efg_pct_differential` | Shooting Efficiency (Four Factors) | `home_pts, away_pts, home_win` | Last 10 games (expanding window early in season) | `games`, `player_game_stats` | Effective field goal percentage (eFG% = (FGM + 0.5 * FG3M) / FGA) weights 3-point efficiency and has the highest correlation with winning. |
| `rest_days_and_back_to_back_flag` | Schedule & Fatigue | `home_pts, away_pts, point_margin` | Days elapsed since each team's previous game_date | `games` | Teams playing on 0 days rest (second night of a back-to-back), especially on the road, suffer a measurable drop in shooting efficiency and win rate. |
| `home_away_location_split_win_pct` | Venue & Home-Court Advantage | `home_win, point_margin` | Season-to-date home win % vs away win % | `games` | Home teams win 59.0% of historical games (+3.01 mean margin); team-specific venue splits capture altitude (e.g., DEN, UTA) and crowd effects. |
| `rolling_pace_and_total_possessions` | Pace & Tempo | `total_pts, home_pts, away_pts` | Last 10 games prior to tip-off | `games`, `player_game_stats` | Estimated possessions (~FGA - OREB + TO + 0.44 * FTA) directly govern total game scoring volume across different NBA eras. |
| `rolling_rebound_and_assist_margin` | Ball Movement & Possession Control | `point_margin, home_win` | Last 10 games prior to tip-off | `games`, `player_game_stats` | In-game rebound differential (r = +0.405 with point_margin) and assist differential (r = +0.478 with point_margin) are strong structural drivers. |
| `active_starters_minutes_and_pie_availability` | Roster & Player Availability | `home_win, point_margin` | Rolling 10-game top-5 rotation player impact | `player_game_stats`, `players` | Aggregating recent scoring and plus/minus of available starters accounts for key player injuries, trades, and load management. |
| `three_point_attempt_rate_3par` | Modern Spacing & Shot Profile | `total_pts, home_pts, away_pts` | Season-to-date and rolling 10-game FG3A / FGA | `player_game_stats` | Captures secular offensive evolution (league 3PAr rose from 18.8% in 2003 to 39.2% in 2021-22) and variance in scoring outcomes. |
| `head_to_head_season_series_margin` | Matchup History | `point_margin, home_win` | Trailing 3-season head-to-head matchups | `games` | Captures stylistic matchup advantages between specific franchises and division rivals. |

---

## 5. Generated Visualizations (`reports/figures/`)

1. `reports/figures/01_games_per_season.png`: Stacked bar chart of games per season by competition stage (`Regular Season`, `Playoffs`, `Preseason`, `Play-In`).
2. `reports/figures/02_scoring_distributions.png`: Three-panel distribution analysis of Home vs. Away Points, Total Points per Game, and Point Margin.
3. `reports/figures/03_seasonal_scoring_trends.png`: Dual-axis time series of average home/away scoring alongside the league-wide 3-point attempt rate (`FG3A / FGA`).
4. `reports/figures/04_home_court_advantage.png`: Seasonal home win rate trajectory and franchise-by-franchise Home vs. Road win percentage comparison.
5. `reports/figures/05_boxscore_correlations.png`: Correlation matrix of team box-score differentials with `point_margin` and `home_win`.
6. `reports/figures/06_extreme_games_and_margins.png`: Joint scatter plot of total points vs. home point margin across seasons.

---

## 6. Data Limitations Affecting Downstream Modeling

- **Non-Stationarity Across NBA Eras**: Because league-wide average total points increased by nearly 40 PPG between 2003 and 2022, raw scoring totals must be normalized relative to the trailing seasonal league average or modeled with era-adaptive features.
- **Preseason vs. Regular Season Noise**: Preseason games exhibit compressed starter minutes and lower predictability; models targeting regular-season and playoff outcomes should filter out or separately weight `game_type == 'Preseason'`.
- **In-Game Box-Score Leakage Prevention**: Columns like `home_fg_pct`, `home_reb`, and `home_ast` in `games.parquet` are post-game outcomes. Predictive models MUST only use strictly lagged rolling windows (`shift(1)`) computed prior to tip-off.
- **Absence of Pre-Game Injury Report Feed**: While `player_game_stats` shows who played (`did_play`) ex-post, pre-game availability must be approximated from recent rotation participation unless external injury feeds are joined.
