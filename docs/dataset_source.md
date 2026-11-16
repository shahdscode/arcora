# Historical NBA Dataset Source & Provenance Documentation

## 1. Dataset Source Overview

| Attribute | Value |
| :--- | :--- |
| **Dataset Title** | NBA games data |
| **Author / Curator** | Nathan Lauga |
| **Platform** | Kaggle (`nathanlauga/nba-games`) |
| **Original URL** | https://www.kaggle.com/datasets/nathanlauga/nba-games |
| **Direct Download Endpoint** | `https://www.kaggle.com/api/v1/datasets/download/nathanlauga/nba-games` |
| **Upstream Data Origin** | NBA Official Stats API (stats.nba.com) |
| **Retrieval Date (UTC)** | `2026-11-16T15:37:46Z` |
| **Local Storage Path** | `data/raw/` |

## 2. License and Usage Restrictions

- **License**: CC0: Public Domain (https://creativecommons.org/publicdomain/zero/1.0/)
- **Usage Terms**: Released under CC0 1.0 Universal Public Domain Dedication on Kaggle for research, analytics, and educational modeling. NBA team names and trademarks remain property of the National Basketball Association.
- **Immutability Policy**: Raw files in `data/raw/` are treated as read-only source artifacts. Any cleaning, deduplication, type normalization, or feature extraction occurs downstream in the ingestion pipeline and writes to `data/processed/`.

## 3. Raw File Inventory & Checksums

| File Name | Size (Bytes) | Rows | Columns | SHA-256 Checksum |
| :--- | ---: | ---: | ---: | :--- |
| `games.csv` | 4,108,788 | 26,651 | 21 | `f3e5a0d6f2cb14d6e45aad85de7d346f90d53407e64d573a975bb0de478aa14a` |
| `games_details.csv` | 93,079,494 | 668,628 | 29 | `ea8853a475be00d546169f785534b1ed838f0b6bba09e0c54572b5753668914b` |
| `players.csv` | 265,578 | 7,228 | 4 | `447d2bc9483842b428721eeb16c2aeec61bd987f5d74fb78d746ba5fb5f54810` |
| `ranking.csv` | 15,446,241 | 210,342 | 13 | `61b082d15cd46c93bc2e16db29af8708ea0be753c26c6c6f0611fb8f8d26ba92` |
| `teams.csv` | 4,063 | 30 | 14 | `9b933f7245d6b529e6ea0046444e168fd5c3b59986597f61df7cc1ad63f1c8e9` |
| **Total** | **112,904,164** | **912,879** | — | — |

## 4. File-Level Column Specifications

### `games.csv` (26,651 rows, 21 columns)
- **Columns**: `GAME_DATE_EST`, `GAME_ID`, `GAME_STATUS_TEXT`, `HOME_TEAM_ID`, `VISITOR_TEAM_ID`, `SEASON`, `TEAM_ID_home`, `PTS_home`, `FG_PCT_home`, `FT_PCT_home`, `FG3_PCT_home`, `AST_home`, `REB_home`, `TEAM_ID_away`, `PTS_away`, `FG_PCT_away`, `FT_PCT_away`, `FG3_PCT_away`, `AST_away`, `REB_away`, `HOME_TEAM_WINS`

### `games_details.csv` (668,628 rows, 29 columns)
- **Columns**: `GAME_ID`, `TEAM_ID`, `TEAM_ABBREVIATION`, `TEAM_CITY`, `PLAYER_ID`, `PLAYER_NAME`, `NICKNAME`, `START_POSITION`, `COMMENT`, `MIN`, `FGM`, `FGA`, `FG_PCT`, `FG3M`, `FG3A`, `FG3_PCT`, `FTM`, `FTA`, `FT_PCT`, `OREB`, `DREB`, `REB`, `AST`, `STL`, `BLK`, `TO`, `PF`, `PTS`, `PLUS_MINUS`

### `players.csv` (7,228 rows, 4 columns)
- **Columns**: `PLAYER_NAME`, `TEAM_ID`, `PLAYER_ID`, `SEASON`

### `ranking.csv` (210,342 rows, 13 columns)
- **Columns**: `TEAM_ID`, `LEAGUE_ID`, `SEASON_ID`, `STANDINGSDATE`, `CONFERENCE`, `TEAM`, `G`, `W`, `L`, `W_PCT`, `HOME_RECORD`, `ROAD_RECORD`, `RETURNTOPLAY`

### `teams.csv` (30 rows, 14 columns)
- **Columns**: `LEAGUE_ID`, `TEAM_ID`, `MIN_YEAR`, `MAX_YEAR`, `ABBREVIATION`, `NICKNAME`, `YEARFOUNDED`, `CITY`, `ARENA`, `ARENACAPACITY`, `OWNER`, `GENERALMANAGER`, `HEADCOACH`, `DLEAGUEAFFILIATION`
