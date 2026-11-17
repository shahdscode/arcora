# Arcora — Week 1 Validation Report

- **Repository URL:** `https://github.com/shahdscode/arcora`
- **Feature Branch:** `https://github.com/shahdscode/arcora/tree/feature/week-1-implementation`

---

## 1. Environment
- **Python version:** Standardized on **Python 3.12 (`3.12.16`)** in `.venv` (`/opt/homebrew/opt/python@3.12/bin/python3.12`), matching `requires-python = ">=3.12"`, `ruff` `target-version = "py312"`, and `mypy` `python_version = "3.12"` in `pyproject.toml`.
- **Dependency installation:** Verified under Python 3.12 via `.venv/bin/python -m pip check` (`No broken requirements found.`), with minimum versions in `pyproject.toml` / `requirements.txt` and exact pinned versions for Python 3.12 in `requirements-lock.txt`.
- **Environment configuration:** `.env.example` documents `ARCORA_ENV`, `ARCORA_LOG_LEVEL`, data/report/log directory paths, and Kaggle dataset endpoints; `.gitignore` excludes `.env`, `.venv/`, `__pycache__/`, `data/raw/*.csv`, and `data/processed/*.parquet`.

---

## 2. Dataset & Coverage Clarification
- **Source:** Nathan Lauga NBA Games Dataset on Kaggle (`https://www.kaggle.com/datasets/nathanlauga/nba-games`), upstream origin `stats.nba.com`, licensed under **CC0: Public Domain**.
- **Seasons covered (19 Complete + 1 Partial Season):**
  - `20` distinct `season` identifiers (`2003` to `2022`, i.e., `2003-04` through `2022-23`, spanning `2003-10-05` to `2022-12-22`).
  - **19 Complete Seasons (`2003-04` through `2021-22`):** Include full regular-season and playoff schedules (note that `2011-12` was a 66-game lockout season, and `2019-20` / `2020-21` were COVID-shortened seasons).
  - **1 Partial In-Progress Season (`2022-23`, `season = 2022`):** Truncated at the dataset snapshot cutoff date `2022-12-22` (`542` completed games: `472` Regular Season, `70` Preseason, `0` Playoffs). Downstream training/validation splits must account for `2022-23` being an incomplete season.
- **Total games & competition stage breakdown (`26,523` canonical completed games):**
  - The `26,523` games in `games.parquet` are **not** exclusively regular-season games; they are explicitly classified via `game_type` (derived from the leading digit of the NBA `GAME_ID`):
    - **Regular Season (`2` prefix):** `23,244` games (`87.64%` of canonical games; `23,273` raw rows minus `29` duplicate Dec 2020 `GAME_ID` rows)
    - **Preseason (`1` prefix):** `1,675` completed games (`6.32%` of canonical games; `1,774` raw rows minus `99` unscored Oct 2003 preseason rows)
    - **Playoffs (`4` prefix):** `1,591` games (`6.00%` of canonical games)
    - **Play-In Tournament (`5` prefix):** `13` games (`0.05%` of canonical games)
    - **Total Canonical Completed Games (`game_status == 'Final'`):** `26,523` games (`100.00%`)
- **Total teams:** `30` NBA franchises (`29` active in `2003-04`; `30` active from `2004-05` onward after Charlotte Bobcats/Hornets expansion).
- **Total players:** `2,687` unique historical NBA players (unified across `players.csv` [2009–2019 rosters, `1,769` players] and `games_details.csv` [2003–2022 box scores, `+918` additional players], tracking both `career_games_played` [`did_play == True`] and `career_games_logged` [all box-score appearances including DNP]).
- **Raw files (`data/raw/`, verified unchanged via SHA-256):**
  - `games.csv`: `4,108,788` bytes | `26,651` rows | `21` cols | SHA-256 `f3e5a0d6f2cb14d6e45aad85de7d346f90d53407e64d573a975bb0de478aa14a`
  - `games_details.csv`: `93,079,494` bytes | `668,628` rows | `29` cols | SHA-256 `ea8853a475be00d546169f785534b1ed838f0b6bba09e0c54572b5753668914b`
  - `players.csv`: `265,578` bytes | `7,228` rows | `4` cols | SHA-256 `447d2bc9483842b428721eeb16c2aeec61bd987f5d74fb78d746ba5fb5f54810`
  - `ranking.csv`: `15,446,241` bytes | `210,342` rows | `13` cols | SHA-256 `61b082d15cd46c93bc2e16db29af8708ea0be753c26c6c6f0611fb8f8d26ba92`
  - `teams.csv`: `4,063` bytes | `30` rows | `14` cols | SHA-256 `9b933f7245d6b529e6ea0046444e168fd5c3b59986597f61df7cc1ad63f1c8e9`
- **Processed files (`data/processed/`, deterministic Parquet outputs):**
  - `teams.parquet`: `30` rows, `15` columns
  - `players.parquet`: `2,687` rows, `6` columns
  - `games.parquet`: `26,523` rows, `23` columns
  - `player_game_stats.parquet`: `668,338` rows, `30` columns
  - `standings.parquet`: `210,313` rows, `16` columns

---

## 3. Data Quality & Exact Row-Count Reconciliation

### 3.1 Validation Check Results (`25 / 25` Passed)
- **Duplicate game IDs:** `0 / 26,523` failed records (`VAL-001` PASS). Primary key uniqueness also verified with `0` duplicates on `teams` (`0 / 30`, `VAL-002`), `players` (`0 / 2,687`, `VAL-003`), `player_game_stats` (`0 / 668,338`, `VAL-004`), and `standings` (`0 / 210,313`, `VAL-022`).
- **Missing critical values:** `0` failed records across all 5 canonical tables (`games`: `0 / 26,523` [`VAL-009`]; `teams`: `0 / 30` [`VAL-010`]; `players`: `0 / 2,687` [`VAL-011`]; `player_game_stats`: `0 / 668,338` [`VAL-012`]; `standings`: `0 / 210,313` [`VAL-023`]).
- **Invalid scores & stat bounds:** `0 / 26,523` failed records in `games` (`VAL-007`, `VAL-020` — all `game_status == 'Final'`, `home_pts > 0`, `away_pts > 0`, `total_pts == home_pts + away_pts`, `point_margin == home_pts - away_pts`, `home_win == (home_pts > away_pts)`, percentages in `[0.0, 1.0]`); `0 / 668,338` failed records in `player_game_stats` (`VAL-008`, `VAL-021` — `fgm <= fga`, `fg3m <= fg3a`, `ftm <= fta`, `reb == oreb + dreb`, DNP rows have `0` minutes/points); `0 / 210,313` failed records in `standings` (`VAL-025` — `games_played == wins + losses`, `win_pct` in `[0.0, 1.0]`).
- **Invalid game dates:** `0 / 26,523` failed records (`VAL-005` PASS — all dates in `2003-10-05`..`2022-12-22` and consistent with `season` and `season_label`).
- **Home/away conflicts:** `0 / 26,523` failed records (`VAL-006` PASS — `home_team_id != away_team_id` in every game).
- **Broken team references:** `0` failed records across `games.home_team_id` (`VAL-013`), `games.away_team_id` (`VAL-014`), `player_game_stats.team_id` (`VAL-015`, `VAL-017`), and `standings.team_id` (`VAL-024`).
- **Broken player references:** `0 / 668,338` failed records (`player_game_stats.player_id -> players.player_id` [`VAL-018`]; `0 <= career_games_played <= career_games_logged` [`VAL-019`]).

### 3.2 Exact Reconciliation of Excluded Raw Records
| Entity | Raw Rows | Net Change | Canonical Rows | Exact Accounting of Excluded / Transformed Records |
| :--- | ---: | ---: | ---: | :--- |
| **`games`** | `26,651` | `-128` | `26,523` | **`-99` unscored rows:** 99 Oct 2003 preseason rows where `PTS_home`, `PTS_away`, and all box-score columns are `NaN` (`unscored_games_dropped = 99`).<br>**`-29` duplicate `GAME_ID` rows:** 29 duplicated Dec 2020 games (58 total rows) with minor float-rounding differences (`0.343` vs `0.3429999999999999`) (`duplicates_removed = 29`).<br>**Check:** `26,651 - 99 - 29 = 26,523`. |
| **`player_game_stats`** | `668,628` | `-290` | `668,338` | **`-1` invalid game-team participant row:** In `GAME_ID = 10500109` (2005-10-20 preseason: `IND` `1610612754` @ `ATL` `1610612737`), `PLAYER_ID = 2564` (Boris Diaw) is logged under `TEAM_ID = 1610612756` (`PHX`), which did not play in that game (`orphan_rows_dropped = 1`).<br>**`-289` duplicate `(GAME_ID, TEAM_ID, PLAYER_ID)` rows:** 289 duplicated player box-score rows across 12 duplicated Dec 2020 games (`22000006`..`22000077`) (`duplicates_removed = 289`).<br>**Check:** `668,628 - 1 - 289 = 668,338`.<br>*(Note: Inside the `668,338` retained rows, `109,681` DNP rows have `did_play = False` and counting stats zero-filled, and `1,475` malformed `MIN` strings [`12` negative strings like `'-9'` and `1,463` `'MM:60'` strings] are reset to `0.0` and audited in `ingestion_manifest.json`.)* |
| **`players`** | `7,228` | `-4,541` | `2,687` | `7,228` team-season roster rows in `players.csv` (covering only 2009–2019) collapse to `1,769` unique players (`-5,459` multi-season rows), plus `+918` historical players from 2003–2008 and 2020–2022 backfilled from `games_details.csv` (`1,769 + 918 = 2,687`). |
| **`standings`** | `210,342` | `-29` | `210,313` | **`-29` duplicate `(TEAM_ID, SEASON_ID, STANDINGSDATE)` rows** removed (`210,342 - 29 = 210,313`). |
| **`teams`** | `30` | `0` | `30` | All 30 NBA franchises preserved (`4` null `arena_capacity` values stored as nullable `Int64`). |

---

## 4. Pipeline Validation
- **Pipeline runs successfully:** Yes (`python scripts/ingest_data.py` and `arcora ingest` execute cleanly from raw CSVs to canonical Parquet files).
- **Repeated execution is consistent:** Yes — consecutive executions produce identical row counts (`30` teams, `2,687` players, `26,523` games, `668,338` player_game_stats, `210,313` standings) and `df1.equals(df2) == True` across all 5 tables.
- **Raw data remains unchanged:** Yes — `verify_raw_checksums(data/raw)` confirms all 5 raw CSV files match their original SHA-256 digests (`0` mismatches).
- **Processed output is reproducible:** Yes — byte-for-byte identical SHA-256 checksums across repeated runs for all 5 Parquet files (`games.parquet`: `864b47fae3d56bc9...`, `player_game_stats.parquet`: `b59ef416efd47783...`, `players.parquet`: `743e5aca67b64f6f...`, `standings.parquet`: `4b4b395a8e6f892e...`, `teams.parquet`: `13f6440020695ac0...`).

---

## 5. Automated Tests
- **Total tests:** `24`
- **Passed:** `24`
- **Failed:** `0`
- **Coverage:** `94%` statement coverage across `src/arcora/` (`1,305` statements, `72` missed), verified under **Python 3.12** via `pytest -v --cov=arcora --cov-report=term-missing`, with `0` errors from `ruff check src tests scripts` and `0` errors from `mypy src` (strict mode).

---

## 6. Exploratory Data Analysis (EDA) Findings & Statistics
- **Season coverage verified:** All 20 season IDs (`2003-04` to `2022-23`) analyzed across `26,523` completed games (`23,244` Regular Season, `1,675` Preseason, `1,591` Playoffs, `13` Play-In).
- **Score & point-margin distributions (`games.parquet`, $N = 26,523$):**
  | Metric | Mean | Std Dev | Min | 25th Pct | Median | 75th Pct | Max | Skewness |
  | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | **Home Points (`home_pts`)** | `103.45` | `13.28` | `59` | `94` | `103` | `112` | `168` | `+0.228` |
  | **Away Points (`away_pts`)** | `100.63` | `13.36` | `33` | `91` | `100` | `109` | `168` | `+0.242` |
  | **Total Points (`total_pts`)** | `204.08` | `23.70` | `99` | `188` | `203` | `220` | `336` | `+0.264` |
  | **Point Margin (`home_pts - away_pts`)** | `+2.82` | `13.63` | `-73` | `-6` | `+3` | `+12` | `+68` | `-0.025` |
- **Home-court advantage analyzed:**
  - **Overall Home Win Rate (excluding All-Star exhibitions):** `58.92%` (`+2.82` mean point margin, `+3.0` median margin).
  - **Regular Season Home Win Rate:** `58.92%` (`23,244` games).
  - **Playoffs Home Win Rate:** `61.72%` (`1,591` games).
  - **Historical Trajectory & COVID-19 Anomaly:** Between `2003-04` and `2018-19`, seasonal home win rates ranged between `57.8%` and `61.2%`. During the COVID-impacted `2019-20` (Orlando Bubble) and `2020-21` (restricted attendance) seasons, home win rates dropped to **`55.1%`** and **`54.4%`**, before recovering to `55.4%` (`2021-22`) and `58.5%` (`2022-23`).
- **Secular Scoring & 3-Point Evolution:**
  - Combined game scoring increased by **+39.4 PPG** from `186.6` PPG in `2003-04` (`94.9` Home / `91.7` Away) to `226.0` PPG in `2022-23` (`114.4` Home / `111.6` Away).
  - Over the same period, the league-wide 3-point attempt rate (`FG3A / FGA` from `player_game_stats`) more than doubled from **`18.8%`** (`2003-04`) to **`39.2%`** (`2021-22`).
- **Descriptive Post-Game Box-Score Correlations with `point_margin` (Ex-Post Only):**
  - `fg_pct_diff` (`home_fg_pct - away_fg_pct`): $r = +0.6295$
  - `ast_diff` (`home_ast - away_ast`): $r = +0.4789$
  - `fg3_pct_diff` (`home_fg3_pct - away_fg3_pct`): $r = +0.4227$
  - `reb_diff` (`home_reb - away_reb`): $r = +0.4050$
  - `ft_pct_diff` (`home_ft_pct - away_ft_pct`): $r = +0.1846$
  - *(Explicitly labeled in `reports/eda_findings_report.md` as descriptive post-game correlations that are unavailable prior to tip-off.)*
- **Extreme historical games identified:**
  - **Highest Combined Total Points:** `336` pts (`MIL` `168` @ `ATL` `168` -> `4OT` on `2019-03-01`, `game_id=21800928`, `161-168` in `CHI @ ATL`)
  - **Lowest Combined Total Points:** `99` pts (`DEN` `33` @ `NOH` `66` on `2003-10-16` preseason, `game_id=10300060`)
  - **Largest Away Victory Margin:** `-73` home margin (`MEM` `152` vs `OKC` `79` on `2021-12-02`, `game_id=22100328` / preseason `-68`)
- **Missingness analyzed:**
  - Quantified across both raw CSVs and canonical Parquet tables (`99` unscored 2003 preseason games dropped; `109,681` DNP player rows flagged with `did_play = False`, `412,675` bench/DNP null `start_position`, `558,665` active-player null `comment`, `133,314` null `plus_minus`; `4` null `arena_capacity` in `teams`; `206,323` null `return_to_play` outside the 2019–20 bubble in `standings`).

---

## 7. Known Issues & Week 2 ML Leakage Controls
- **Issue 1 — Point-in-Time Leakage Prevention for `standings.parquet`:**
  - `standings.parquet` records daily cumulative snapshots indexed by `(team_id, season, season_type, standings_date)`. Using end-of-season standings or same-day post-game standings (`standings_date >= game_date`) would leak the game's outcome into pre-game features.
  - **Week 2 Control:** Standings features for a game on `game_date` MUST be joined via a strict backward as-of temporal join (`standings_date < game_date` within matching `(team_id, season, season_type)`).
- **Issue 2 — Post-Game Box-Score Leakage Prevention:**
  - In-game box-score columns in `games.parquet` (`home_fg_pct`, `home_reb`, `home_ast`, etc.) and `player_game_stats.parquet` are post-game outcomes.
  - **Week 2 Control:** All rolling team and player efficiency features must be computed using strictly lagged historical windows (`shift(1)`) ordered by `(game_date, game_id)` prior to tip-off.
- **Issue 3 — Partial `2022-23` Season & Competition Stage Filtering:**
  - `2022-23` ends on `2022-12-22` (`542` games) and `games.parquet` contains `1,675` completed `Preseason` games alongside `23,244` `Regular Season`, `1,591` `Playoffs`, and `13` `Play-In` games.
  - **Week 2 Control:** Filter out `game_type == 'Preseason'` when constructing regular-season/playoff rolling features and model evaluation splits, and treat `2022-23` as an out-of-time test window rather than a full season.
- **Issue 4 — Secular Offensive Inflation (2003–2022):**
  - League scoring increased from `186.6` PPG (`2003-04`) to `226.0` PPG (`2022-23`).
  - **Week 2 Control:** Normalize offensive/defensive ratings relative to trailing seasonal league-average efficiency and pace.

---

## 8. Final Decision
- [x] **PASS — Ready for Week 2**
- [ ] CONDITIONAL PASS — Minor issues remain
- [ ] FAIL — Critical issues must be resolved

---

## 9. Evidence & Artifacts
- **Repository & Branch:** `https://github.com/shahdscode/arcora/tree/feature/week-1-implementation`
- **Test output:** `pytest -v --cov=arcora --cov-report=term-missing` (`24 passed`, `0 failed`, `94%` coverage)
- **Data quality report:** [reports/data_quality_report.md](data_quality_report.md) and [reports/data_quality_summary.json](data_quality_summary.json) (`25 / 25` rules passed, `0` failed records)
- **EDA report, notebook & figures:** [reports/eda_findings_report.md](eda_findings_report.md), [reports/eda_summary.json](eda_summary.json), [docs/predictive_features_catalog.md](../docs/predictive_features_catalog.md), [notebooks/01_nba_exploratory_data_analysis.ipynb](../notebooks/01_nba_exploratory_data_analysis.ipynb), and `reports/figures/01_games_per_season.png` through `06_extreme_games_and_margins.png`
- **Pipeline execution logs & manifest:** `logs/ingestion.log` and [reports/ingestion_manifest.json](ingestion_manifest.json)
