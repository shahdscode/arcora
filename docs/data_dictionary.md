# Arcora Canonical NBA Data Dictionary

This document provides field-level definitions, data types, keys, source mappings, nullability rules, and domain constraints for all four core entities in Arcora's canonical data model.

## Entity: `teams` (`data/processed/teams.parquet`)

- **Description**: Dimension table of NBA franchises, arenas, and organizational metadata.
- **Grain**: One row per NBA franchise (30 franchises).
- **Primary Key**: (`team_id`)
- **Source Files**: `teams.csv`

| Field Name | Logical Type | Pandas / Parquet Dtype | SQL Type | Nullable | Key | Source Mapping | Constraints | Description |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `team_id` | `integer` | `int64` | `BIGINT` | No | PK | `teams.TEAM_ID` | > 0, unique | Unique 10-digit NBA franchise identifier. |
| `league_id` | `string` | `string` | `VARCHAR(4)` | No |  | `teams.LEAGUE_ID` | == '00' | Two-digit NBA league code ('00'). |
| `abbreviation` | `string` | `string` | `CHAR(3)` | No |  | `teams.ABBREVIATION` | length == 3, unique | Standard 3-letter uppercase team code (e.g., 'BOS', 'LAL'). |
| `nickname` | `string` | `string` | `VARCHAR(64)` | No |  | `teams.NICKNAME` | — | Official franchise nickname (e.g., 'Celtics'). |
| `city` | `string` | `string` | `VARCHAR(64)` | No |  | `teams.CITY` | — | Home city or state/region of the franchise. |
| `full_name` | `string` | `string` | `VARCHAR(128)` | No |  | `Derived: CONCAT(city, ' ', nickname)` | — | Full display name of the franchise (e.g., 'Boston Celtics'). |
| `year_founded` | `integer` | `int64` | `INTEGER` | No |  | `teams.YEARFOUNDED` | >= 1946 | Year the franchise was originally established. |
| `min_year` | `integer` | `int64` | `INTEGER` | No |  | `teams.MIN_YEAR` | >= 1946 | Earliest NBA season year recorded for the franchise. |
| `max_year` | `integer` | `int64` | `INTEGER` | No |  | `teams.MAX_YEAR` | >= min_year | Latest NBA season year in the team metadata snapshot. |
| `arena` | `string` | `string` | `VARCHAR(128)` | No |  | `teams.ARENA` | — | Primary home arena name. |
| `arena_capacity` | `integer` | `Int64` | `INTEGER` | Yes |  | `teams.ARENACAPACITY` | > 0 when not null | Seating capacity of the home arena (nullable when unrecorded). |
| `owner` | `string` | `string` | `VARCHAR(128)` | No |  | `teams.OWNER` | — | Principal owner or ownership group. |
| `general_manager` | `string` | `string` | `VARCHAR(128)` | No |  | `teams.GENERALMANAGER` | — | General manager recorded in metadata snapshot. |
| `head_coach` | `string` | `string` | `VARCHAR(128)` | No |  | `teams.HEADCOACH` | — | Head coach recorded in metadata snapshot. |
| `dleague_affiliation` | `string` | `string` | `VARCHAR(128)` | No |  | `teams.DLEAGUEAFFILIATION` | — | Associated NBA G-League (formerly D-League) affiliate. |

## Entity: `players` (`data/processed/players.parquet`)

- **Description**: Master dimension table of all NBA players across 2003-2022, unified from season rosters (players.csv) and historical game box-score logs (games_details.csv).
- **Grain**: One row per unique NBA player (player_id).
- **Primary Key**: (`player_id`)
- **Source Files**: `players.csv`, `games_details.csv`

| Field Name | Logical Type | Pandas / Parquet Dtype | SQL Type | Nullable | Key | Source Mapping | Constraints | Description |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `player_id` | `integer` | `int64` | `BIGINT` | No | PK | `players.PLAYER_ID UNION games_details.PLAYER_ID` | > 0, unique | Unique official NBA player identifier. |
| `player_name` | `string` | `string` | `VARCHAR(128)` | No |  | `players.PLAYER_NAME / games_details.PLAYER_NAME` | non-empty | Standardized full name of the player. |
| `first_season` | `integer` | `int64` | `INTEGER` | No |  | `MIN(SEASON) across appearances` | 2003 <= first_season <= 2022 | First NBA season year (YYYY) the player appears in the dataset. |
| `last_season` | `integer` | `int64` | `INTEGER` | No |  | `MAX(SEASON) across appearances` | last_season >= first_season | Most recent NBA season year (YYYY) the player appears in the dataset. |
| `career_games_played` | `integer` | `int64` | `INTEGER` | No |  | `COUNT(DISTINCT game_id) WHERE did_play == True` | >= 0, <= career_games_logged | Total number of games in which the player actively entered the game. |
| `career_games_logged` | `integer` | `int64` | `INTEGER` | No |  | `COUNT(DISTINCT game_id) in player_game_stats` | >= career_games_played | Total number of game box-score entries logged (including DNP/inactive). |

## Entity: `games` (`data/processed/games.parquet`)

- **Description**: Fact table of completed historical NBA games with dates, seasons, participating teams, final scores, shooting percentages, and game outcomes.
- **Grain**: One row per completed NBA game (game_id).
- **Primary Key**: (`game_id`)
- **Source Files**: `games.csv`

| Field Name | Logical Type | Pandas / Parquet Dtype | SQL Type | Nullable | Key | Source Mapping | Constraints | Description |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `game_id` | `integer` | `int64` | `BIGINT` | No | PK | `games.GAME_ID` | > 0, unique | Unique NBA game identifier. |
| `game_date` | `date` | `datetime64[ns]` | `DATE` | No |  | `games.GAME_DATE_EST` | 2003-10-05 <= game_date <= 2022-12-22 | Calendar date of the game in Eastern Standard Time (YYYY-MM-DD). |
| `season` | `integer` | `int64` | `INTEGER` | No |  | `games.SEASON` | 2003 <= season <= 2022 | Starting calendar year of the NBA season (e.g., 2022 for 2022-23). |
| `season_label` | `string` | `string` | `CHAR(7)` | No |  | `Derived: YYYY-YY from season` | matches ^\d{4}-\d{2}$ | Formatted NBA season label (e.g., '2022-23'). |
| `game_type` | `string` | `string` | `VARCHAR(32)` | No |  | `Derived from leading digit of GAME_ID` | in {'Preseason', 'Regular Season', 'Playoffs', 'Play-In'} | Competition stage ('Preseason', 'Regular Season', 'Playoffs', 'Play-In'). |
| `game_status` | `string` | `string` | `VARCHAR(32)` | No |  | `games.GAME_STATUS_TEXT` | == 'Final' | Completion status of the game ('Final'). |
| `home_team_id` | `integer` | `int64` | `BIGINT` | No | FK (teams.team_id) | `games.HOME_TEAM_ID` | FK -> teams.team_id, != away_team_id | Franchise ID of the home team. |
| `away_team_id` | `integer` | `int64` | `BIGINT` | No | FK (teams.team_id) | `games.VISITOR_TEAM_ID` | FK -> teams.team_id, != home_team_id | Franchise ID of the away (visitor) team. |
| `home_pts` | `integer` | `int64` | `INTEGER` | No |  | `games.PTS_home` | > 0 | Final points scored by the home team. |
| `away_pts` | `integer` | `int64` | `INTEGER` | No |  | `games.PTS_away` | > 0 | Final points scored by the away team. |
| `total_pts` | `integer` | `int64` | `INTEGER` | No |  | `Derived: home_pts + away_pts` | == home_pts + away_pts | Combined total points scored by both teams. |
| `point_margin` | `integer` | `int64` | `INTEGER` | No |  | `Derived: home_pts - away_pts` | == home_pts - away_pts | Signed point differential from the home team perspective. |
| `home_win` | `boolean` | `bool` | `BOOLEAN` | No |  | `games.HOME_TEAM_WINS` | == (home_pts > away_pts) | True if the home team won the game (home_pts > away_pts). |
| `home_fg_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games.FG_PCT_home` | 0.0 <= value <= 1.0 | Home team field goal percentage (0.0 to 1.0). |
| `home_ft_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games.FT_PCT_home` | 0.0 <= value <= 1.0 | Home team free throw percentage (0.0 to 1.0). |
| `home_fg3_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games.FG3_PCT_home` | 0.0 <= value <= 1.0 | Home team 3-point field goal percentage (0.0 to 1.0). |
| `home_ast` | `integer` | `int64` | `INTEGER` | No |  | `games.AST_home` | >= 0 | Home team total assists. |
| `home_reb` | `integer` | `int64` | `INTEGER` | No |  | `games.REB_home` | >= 0 | Home team total rebounds. |
| `away_fg_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games.FG_PCT_away` | 0.0 <= value <= 1.0 | Away team field goal percentage (0.0 to 1.0). |
| `away_ft_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games.FT_PCT_away` | 0.0 <= value <= 1.0 | Away team free throw percentage (0.0 to 1.0). |
| `away_fg3_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games.FG3_PCT_away` | 0.0 <= value <= 1.0 | Away team 3-point field goal percentage (0.0 to 1.0). |
| `away_ast` | `integer` | `int64` | `INTEGER` | No |  | `games.AST_away` | >= 0 | Away team total assists. |
| `away_reb` | `integer` | `int64` | `INTEGER` | No |  | `games.REB_away` | >= 0 | Away team total rebounds. |

## Entity: `player_game_stats` (`data/processed/player_game_stats.parquet`)

- **Description**: Fact table of individual player box-score statistics for each NBA game, including playing status, starter designation, minutes played, shooting splits, rebounding, playmaking, defense, and plus/minus.
- **Grain**: One row per player per team per game (game_id, team_id, player_id).
- **Primary Key**: (`game_id`, `team_id`, `player_id`)
- **Source Files**: `games_details.csv`

| Field Name | Logical Type | Pandas / Parquet Dtype | SQL Type | Nullable | Key | Source Mapping | Constraints | Description |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `game_id` | `integer` | `int64` | `BIGINT` | No | PK, FK (games.game_id) | `games_details.GAME_ID` | FK -> games.game_id | Game identifier linking to canonical games table. |
| `team_id` | `integer` | `int64` | `BIGINT` | No | PK, FK (teams.team_id) | `games_details.TEAM_ID` | FK -> teams.team_id | Team identifier linking to canonical teams table. |
| `player_id` | `integer` | `int64` | `BIGINT` | No | PK, FK (players.player_id) | `games_details.PLAYER_ID` | FK -> players.player_id | Player identifier linking to canonical players table. |
| `player_name` | `string` | `string` | `VARCHAR(128)` | No |  | `games_details.PLAYER_NAME` | — | Player full name recorded on the game box score. |
| `team_abbreviation` | `string` | `string` | `CHAR(3)` | No |  | `games_details.TEAM_ABBREVIATION` | — | Three-letter team code on the box score. |
| `team_city` | `string` | `string` | `VARCHAR(64)` | No |  | `games_details.TEAM_CITY` | — | City name recorded on the box score. |
| `is_starter` | `boolean` | `bool` | `BOOLEAN` | No |  | `Derived: START_POSITION in {'F', 'C', 'G'}` | — | True if the player started the game. |
| `start_position` | `string` | `string` | `CHAR(1)` | Yes |  | `games_details.START_POSITION` | — | Starting position code ('F', 'C', 'G') or null for bench/DNP players. |
| `did_play` | `boolean` | `bool` | `BOOLEAN` | No |  | `Derived: PTS is not null` | — | True if the player entered the game and recorded active statistics. |
| `comment` | `string` | `string` | `VARCHAR(256)` | Yes |  | `games_details.COMMENT` | — | DNP / injury / coach's decision note when did_play is False. |
| `minutes_played` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `Parsed from games_details.MIN (MM:SS -> decimal minutes)` | >= 0.0 | Total minutes played in decimal form (0.0 for DNP). |
| `fgm` | `integer` | `int64` | `INTEGER` | No |  | `games_details.FGM` | >= 0, <= fga | Field goals made. |
| `fga` | `integer` | `int64` | `INTEGER` | No |  | `games_details.FGA` | >= 0 | Field goals attempted. |
| `fg_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games_details.FG_PCT` | 0.0 <= value <= 1.0 | Field goal percentage (0.0 to 1.0). |
| `fg3m` | `integer` | `int64` | `INTEGER` | No |  | `games_details.FG3M` | >= 0, <= fg3a | Three-point field goals made. |
| `fg3a` | `integer` | `int64` | `INTEGER` | No |  | `games_details.FG3A` | >= 0 | Three-point field goals attempted. |
| `fg3_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games_details.FG3_PCT` | 0.0 <= value <= 1.0 | Three-point field goal percentage (0.0 to 1.0). |
| `ftm` | `integer` | `int64` | `INTEGER` | No |  | `games_details.FTM` | >= 0, <= fta | Free throws made. |
| `fta` | `integer` | `int64` | `INTEGER` | No |  | `games_details.FTA` | >= 0 | Free throws attempted. |
| `ft_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `games_details.FT_PCT` | 0.0 <= value <= 1.0 | Free throw percentage (0.0 to 1.0). |
| `oreb` | `integer` | `int64` | `INTEGER` | No |  | `games_details.OREB` | >= 0 | Offensive rebounds. |
| `dreb` | `integer` | `int64` | `INTEGER` | No |  | `games_details.DREB` | >= 0 | Defensive rebounds. |
| `reb` | `integer` | `int64` | `INTEGER` | No |  | `games_details.REB` | >= 0 | Total rebounds (oreb + dreb). |
| `ast` | `integer` | `int64` | `INTEGER` | No |  | `games_details.AST` | >= 0 | Assists. |
| `stl` | `integer` | `int64` | `INTEGER` | No |  | `games_details.STL` | >= 0 | Steals. |
| `blk` | `integer` | `int64` | `INTEGER` | No |  | `games_details.BLK` | >= 0 | Blocked shots. |
| `turnovers` | `integer` | `int64` | `INTEGER` | No |  | `games_details.TO` | >= 0 | Turnovers committed (renamed from TO to avoid SQL keyword conflict). |
| `pf` | `integer` | `int64` | `INTEGER` | No |  | `games_details.PF` | >= 0 | Personal fouls committed. |
| `pts` | `integer` | `int64` | `INTEGER` | No |  | `games_details.PTS` | >= 0 | Total points scored by the player. |
| `plus_minus` | `float` | `float64` | `DOUBLE PRECISION` | Yes |  | `games_details.PLUS_MINUS` | — | On-court point differential while player was in the game. |

## Entity: `standings` (`data/processed/standings.parquet`)

- **Description**: Daily NBA team standings snapshot table tracking conference, games played, wins, losses, winning percentage, and home/road splits across seasons.
- **Grain**: One row per team per season per season_type per standings_date.
- **Primary Key**: (`team_id`, `season`, `season_type`, `standings_date`)
- **Source Files**: `ranking.csv`

| Field Name | Logical Type | Pandas / Parquet Dtype | SQL Type | Nullable | Key | Source Mapping | Constraints | Description |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| `team_id` | `integer` | `int64` | `BIGINT` | No | PK, FK (teams.team_id) | `ranking.TEAM_ID` | FK -> teams.team_id | Franchise ID linking to canonical teams table. |
| `season` | `integer` | `int64` | `INTEGER` | No | PK | `Derived from last 4 digits of ranking.SEASON_ID` | 2002 <= season <= 2022 | Starting calendar year of the NBA season (e.g., 2022 for 2022-23). |
| `season_label` | `string` | `string` | `CHAR(7)` | No |  | `Derived: YYYY-YY from season` | — | Formatted NBA season label (e.g., '2022-23'). |
| `season_type` | `string` | `string` | `VARCHAR(32)` | No | PK | `Derived from leading digit of ranking.SEASON_ID` | in {'Preseason', 'Regular Season'} | Competition stage ('Preseason' or 'Regular Season'). |
| `standings_date` | `date` | `datetime64[ns]` | `DATE` | No | PK | `ranking.STANDINGSDATE` | — | Calendar date of the daily standings snapshot (YYYY-MM-DD). |
| `conference` | `string` | `string` | `VARCHAR(16)` | No |  | `ranking.CONFERENCE` | in {'East', 'West'} | NBA conference ('East' or 'West'). |
| `team_name` | `string` | `string` | `VARCHAR(64)` | No |  | `ranking.TEAM` | — | Team city/market label recorded in standings. |
| `games_played` | `integer` | `int64` | `INTEGER` | No |  | `ranking.G` | == wins + losses | Cumulative games played as of standings_date. |
| `wins` | `integer` | `int64` | `INTEGER` | No |  | `ranking.W` | >= 0 | Cumulative wins as of standings_date. |
| `losses` | `integer` | `int64` | `INTEGER` | No |  | `ranking.L` | >= 0 | Cumulative losses as of standings_date. |
| `win_pct` | `float` | `float64` | `DOUBLE PRECISION` | No |  | `ranking.W_PCT` | 0.0 <= value <= 1.0 | Cumulative winning percentage (0.0 to 1.0). |
| `home_wins` | `integer` | `int64` | `INTEGER` | No |  | `Parsed wins from ranking.HOME_RECORD ('W-L')` | >= 0 | Cumulative home wins as of standings_date. |
| `home_losses` | `integer` | `int64` | `INTEGER` | No |  | `Parsed losses from ranking.HOME_RECORD ('W-L')` | >= 0 | Cumulative home losses as of standings_date. |
| `road_wins` | `integer` | `int64` | `INTEGER` | No |  | `Parsed wins from ranking.ROAD_RECORD ('W-L')` | >= 0 | Cumulative road wins as of standings_date. |
| `road_losses` | `integer` | `int64` | `INTEGER` | No |  | `Parsed losses from ranking.ROAD_RECORD ('W-L')` | >= 0 | Cumulative road losses as of standings_date. |
| `return_to_play` | `float` | `float64` | `DOUBLE PRECISION` | Yes |  | `ranking.RETURNTOPLAY` | — | 2019-20 Orlando Bubble restart indicator (nullable outside 2019-20). |
