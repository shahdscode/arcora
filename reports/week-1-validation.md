# Arcora — Week 1 Validation Report

## 1. Environment
- **Python version:** Python `3.14.5` (`requires-python = ">=3.12"` in `pyproject.toml`, `target-version = "py312"` for `ruff`, `python_version = "3.12"` for `mypy`)
- **Dependency installation:** Verified via `.venv/bin/python -m pip check` (`No broken requirements found.`), configured in `pyproject.toml`, `requirements.txt`, and pinned in `requirements-lock.txt`
- **Environment configuration:** `.env.example` documented with `ARCORA_ENV`, `ARCORA_LOG_LEVEL`, data/report/log directory paths, and Kaggle dataset configuration; `.gitignore` excludes `.env`, `.venv/`, `__pycache__/`, `data/raw/*.csv`, and `data/processed/*.parquet`

## 2. Dataset
- **Source:** Nathan Lauga NBA Games Dataset on Kaggle (`https://www.kaggle.com/datasets/nathanlauga/nba-games`), upstream origin `stats.nba.com`, licensed under **CC0: Public Domain**
- **Seasons covered:** 20 NBA seasons (`2003` to `2022`, i.e., `2003-04` through `2022-23`, spanning `2003-10-05` to `2022-12-22`)
- **Total games:** `26,523` canonical completed games (from `26,651` raw rows in `games.csv` after removing `29` duplicate `GAME_ID` records and `99` unscored 2003 preseason games)
- **Total teams:** `30` NBA franchises
- **Total players:** `2,687` unique historical NBA players (unified across `players.csv` and `games_details.csv`, with both `career_games_played` and `career_games_logged`)
- **Raw files (`data/raw/`, verified unchanged via SHA-256):**
  - `games.csv` (`4,108,788` bytes, `26,651` rows, `21` columns, SHA-256: `f3e5a0d6f2cb14d6e45aad85de7d346f90d53407e64d573a975bb0de478aa14a`)
  - `games_details.csv` (`93,079,494` bytes, `668,628` rows, `29` columns, SHA-256: `ea8853a475be00d546169f785534b1ed838f0b6bba09e0c54572b5753668914b`)
  - `players.csv` (`265,578` bytes, `7,228` rows, `4` columns, SHA-256: `447d2bc9483842b428721eeb16c2aeec61bd987f5d74fb78d746ba5fb5f54810`)
  - `ranking.csv` (`15,446,241` bytes, `210,342` rows, `13` columns, SHA-256: `61b082d15cd46c93bc2e16db29af8708ea0be753c26c6c6f0611fb8f8d26ba92`)
  - `teams.csv` (`4,063` bytes, `30` rows, `14` columns, SHA-256: `9b933f7245d6b529e6ea0046444e168fd5c3b59986597f61df7cc1ad63f1c8e9`)
- **Processed files (`data/processed/`, deterministic Parquet outputs):**
  - `teams.parquet` (`30` rows, `15` columns, SHA-256: `13f6440020695ac0...`)
  - `players.parquet` (`2,687` rows, `6` columns, SHA-256: `743e5aca67b64f6f...`)
  - `games.parquet` (`26,523` rows, `23` columns, SHA-256: `864b47fae3d56bc9...`)
  - `player_game_stats.parquet` (`668,338` rows, `30` columns, SHA-256: `b59ef416efd47783...`)
  - `standings.parquet` (`210,313` rows, `16` columns, SHA-256: `4b4b395a8e6f892e...`)

## 3. Data Quality
- **Duplicate game IDs:** `0 / 26,523` failed records (`VAL-001` PASS; `29` raw duplicates removed during ingestion). Also `0` duplicate primary keys in `teams` (`0 / 30`, `VAL-002`), `players` (`0 / 2,687`, `VAL-003`), `player_game_stats` (`0 / 668,338`, `VAL-004`), and `standings` (`0 / 210,313`, `VAL-022`).
- **Missing critical values:** `0` failed records across all 5 canonical tables (`games`: `0 / 26,523` across 23 non-nullable columns [`VAL-009`]; `teams`: `0 / 30` across 14 non-nullable columns [`VAL-010`]; `players`: `0 / 2,687` across 6 non-nullable columns [`VAL-011`]; `player_game_stats`: `0 / 668,338` across 27 non-nullable columns [`VAL-012`]; `standings`: `0 / 210,313` across 15 non-nullable columns [`VAL-023`]).
- **Invalid scores:** `0 / 26,523` failed records (`VAL-007` PASS — all `home_pts > 0`, `away_pts > 0`, `total_pts == home_pts + away_pts`, `point_margin == home_pts - away_pts`, `home_win == (home_pts > away_pts)`, `game_status == 'Final'` [`VAL-020`], and percentages in `[0.0, 1.0]`). In `player_game_stats`, `0 / 668,338` failed records for non-negative stats, `fgm <= fga`, `fg3m <= fg3a`, `ftm <= fta` (`VAL-008`), DNP zero-stat consistency, and `reb == oreb + dreb` (`VAL-021`). In `standings`, `0 / 210,313` failed records for `games_played == wins + losses` and `win_pct` in `[0.0, 1.0]` (`VAL-025`).
- **Invalid game dates:** `0 / 26,523` failed records (`VAL-005` PASS — all `game_date` timestamps valid within `2003-10-05` to `2022-12-22` and consistent with `season` `2003`–`2022` and `season_label` `YYYY-YY`).
- **Home/away conflicts:** `0 / 26,523` failed records (`VAL-006` PASS — `home_team_id != away_team_id` for 100% of games).
- **Broken team references:** `0` failed records (`games.home_team_id -> teams.team_id`: `0 / 26,523` [`VAL-013`]; `games.away_team_id -> teams.team_id`: `0 / 26,523` [`VAL-014`]; `player_game_stats.team_id -> teams.team_id`: `0 / 668,338` [`VAL-017`]; `player_game_stats.team_id` in `{home_team_id, away_team_id}` of `game_id`: `0 / 668,338` [`VAL-015`]; `standings.team_id -> teams.team_id`: `0 / 210,313` [`VAL-024`]).
- **Broken player references:** `0 / 668,338` failed records (`player_game_stats.player_id -> players.player_id` [`VAL-018`]; `players` season span and `0 <= career_games_played <= career_games_logged`: `0 / 2,687` [`VAL-019`]).

## 4. Pipeline Validation
- **Pipeline runs successfully:** Yes — `python scripts/ingest_data.py` (and `arcora ingest`) executes end-to-end from `data/raw/*.csv` to `data/processed/*.parquet` without manual intervention.
- **Repeated execution is consistent:** Yes — verified by running `run_ingestion_pipeline()` twice consecutively and comparing row counts, DataFrames (`df1.equals(df2) == True`), and file hashes.
- **Raw data remains unchanged:** Yes — `verify_raw_checksums(data/raw)` confirms all 5 raw CSV files match their original SHA-256 digests (`0` mismatches).
- **Processed output is reproducible:** Yes — byte-for-byte identical SHA-256 checksums across repeated runs for all 5 Parquet files (`games.parquet`, `player_game_stats.parquet`, `players.parquet`, `standings.parquet`, `teams.parquet`).

## 5. Automated Tests
- **Total tests:** `24`
- **Passed:** `24`
- **Failed:** `0`
- **Coverage:** `94%` statement coverage across `src/arcora/` (`1,305` statements, `72` missed, verified via `pytest -v --cov=arcora --cov-report=term-missing`); `ruff check src tests scripts` (`0` errors) and `mypy src` (`0` errors in strict mode).

## 6. EDA
- **Season coverage verified:** Yes — all 20 seasons (`2003-04` through `2022-23`) analyzed by competition stage (`Regular Season`: `23,244` games, `Playoffs`: `1,591` games, `Preseason`: `1,675` games, `Play-In`: `13` games).
- **Score distributions reviewed:** Yes — `home_pts` (mean `103.45`, std `13.28`, median `103`), `away_pts` (mean `100.63`, std `13.36`, median `100`), `total_pts` (mean `204.08`, std `23.70`, median `203`), and `point_margin` (mean `+2.82`, std `13.63`, median `+3`).
- **Home advantage analyzed:** Yes — `58.92%` overall home win rate (excluding All-Star exhibitions), `58.92%` in Regular Season, `61.72%` in Playoffs, with the COVID-19 dip to `55.1%` (`2019-20`) and `54.4%` (`2020-21`) documented.
- **Missingness analyzed:** Yes — raw vs. processed missing value patterns documented (99 unscored 2003 preseason games excluded; 109,681 DNP player-game rows flagged with `did_play = False`; 4 null `arena_capacity` values in `teams`; 206,323 null `return_to_play` values outside the 2019-20 bubble in `standings`).
- **Anomalies documented:** Yes — lockout season (`2011-12`), COVID-19 seasons (`2019-20`, `2020-21`), partial `2022-23` cutoff (`2022-12-22`), 1,475 malformed/negative or `MM:60` raw `MIN` strings, and extreme scoring/blowout games documented.

## 7. Known Issues & ML Risk Controls
- **Issue 1 (Era Non-Stationarity):** League scoring rose from `186.6` combined PPG in `2003-04` to `226.0` PPG in `2022-23` alongside a doubling of the 3-point attempt rate (`18.8%` to `39.2%`). Week 2 rolling features should normalize relative to trailing seasonal league averages.
- **Issue 2 (Pre-Game vs. Post-Game Leakage Prevention):** In-game box-score statistics (`home_fg_pct`, `home_reb`, `home_ast`, etc.) in `games.parquet` and `player_game_stats.parquet` are post-game outcomes. All Week 2 predictive features must use strictly lagged (`shift(1)`) historical windows computed prior to tip-off, filtering out `Preseason` games when training regular-season/playoff models.

## 8. Final Decision
- [x] PASS — Ready for Week 2
- [ ] CONDITIONAL PASS — Minor issues remain
- [ ] FAIL — Critical issues must be resolved

## 9. Evidence
- **Test output:** `pytest -v --cov=arcora --cov-report=term-missing` (`24 passed in 7.41s`, `94%` total coverage)
- **Data quality report:** [reports/data_quality_report.md](data_quality_report.md) and [reports/data_quality_summary.json](data_quality_summary.json) (`25 / 25` checks passed, `0` failed records)
- **EDA figures & reports:** [reports/eda_findings_report.md](eda_findings_report.md), [reports/eda_summary.json](eda_summary.json), [notebooks/01_nba_exploratory_data_analysis.ipynb](../notebooks/01_nba_exploratory_data_analysis.ipynb), and `reports/figures/01_games_per_season.png` through `06_extreme_games_and_margins.png`
- **Pipeline execution logs:** `logs/ingestion.log` and [reports/ingestion_manifest.json](ingestion_manifest.json)
