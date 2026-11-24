"""Canonical NBA data model, entity definitions, and schema specifications (Issue #4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single field in a canonical entity."""

    name: str
    logical_type: str
    pandas_dtype: str
    sql_type: str
    nullable: bool
    is_primary_key: bool
    foreign_key: str | None
    source_mapping: str
    description: str
    constraints: str = ""


@dataclass(frozen=True)
class EntitySpec:
    """Canonical entity definition with primary keys, foreign keys, and fields."""

    name: str
    storage_file: str
    source_files: tuple[str, ...]
    primary_key: tuple[str, ...]
    description: str
    grain: str
    fields: tuple[FieldSpec, ...]

    @property
    def column_names(self) -> list[str]:
        """Return ordered list of field names."""
        return [f.name for f in self.fields]

    @property
    def non_nullable_columns(self) -> list[str]:
        """Return list of field names that must not contain nulls."""
        return [f.name for f in self.fields if not f.nullable]


TEAMS_ENTITY = EntitySpec(
    name="teams",
    storage_file="teams.parquet",
    source_files=("teams.csv",),
    primary_key=("team_id",),
    description="Dimension table of NBA franchises, arenas, and organizational metadata.",
    grain="One row per NBA franchise (30 franchises).",
    fields=(
        FieldSpec(
            name="team_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=True,
            foreign_key=None,
            source_mapping="teams.TEAM_ID",
            description="Unique 10-digit NBA franchise identifier.",
            constraints="> 0, unique",
        ),
        FieldSpec(
            name="league_id",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(4)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.LEAGUE_ID",
            description="Two-digit NBA league code ('00').",
            constraints="== '00'",
        ),
        FieldSpec(
            name="abbreviation",
            logical_type="string",
            pandas_dtype="string",
            sql_type="CHAR(3)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.ABBREVIATION",
            description="Standard 3-letter uppercase team code (e.g., 'BOS', 'LAL').",
            constraints="length == 3, unique",
        ),
        FieldSpec(
            name="nickname",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(64)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.NICKNAME",
            description="Official franchise nickname (e.g., 'Celtics').",
        ),
        FieldSpec(
            name="city",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(64)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.CITY",
            description="Home city or state/region of the franchise.",
        ),
        FieldSpec(
            name="full_name",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived: CONCAT(city, ' ', nickname)",
            description="Full display name of the franchise (e.g., 'Boston Celtics').",
        ),
        FieldSpec(
            name="year_founded",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.YEARFOUNDED",
            description="Year the franchise was originally established.",
            constraints=">= 1946",
        ),
        FieldSpec(
            name="min_year",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.MIN_YEAR",
            description="Earliest NBA season year recorded for the franchise.",
            constraints=">= 1946",
        ),
        FieldSpec(
            name="max_year",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.MAX_YEAR",
            description="Latest NBA season year in the team metadata snapshot.",
            constraints=">= min_year",
        ),
        FieldSpec(
            name="arena",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.ARENA",
            description="Primary home arena name.",
        ),
        FieldSpec(
            name="arena_capacity",
            logical_type="integer",
            pandas_dtype="Int64",
            sql_type="INTEGER",
            nullable=True,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.ARENACAPACITY",
            description="Seating capacity of the home arena (nullable when unrecorded).",
            constraints="> 0 when not null",
        ),
        FieldSpec(
            name="owner",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.OWNER",
            description="Principal owner or ownership group.",
        ),
        FieldSpec(
            name="general_manager",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.GENERALMANAGER",
            description="General manager recorded in metadata snapshot.",
        ),
        FieldSpec(
            name="head_coach",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.HEADCOACH",
            description="Head coach recorded in metadata snapshot.",
        ),
        FieldSpec(
            name="dleague_affiliation",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="teams.DLEAGUEAFFILIATION",
            description="Associated NBA G-League (formerly D-League) affiliate.",
        ),
    ),
)


PLAYERS_ENTITY = EntitySpec(
    name="players",
    storage_file="players.parquet",
    source_files=("players.csv", "games_details.csv"),
    primary_key=("player_id",),
    description=(
        "Master dimension table of all NBA players across 2003-2022, unified from "
        "season rosters (players.csv) and historical game box-score logs (games_details.csv)."
    ),
    grain="One row per unique NBA player (player_id).",
    fields=(
        FieldSpec(
            name="player_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=True,
            foreign_key=None,
            source_mapping="players.PLAYER_ID UNION games_details.PLAYER_ID",
            description="Unique official NBA player identifier.",
            constraints="> 0, unique",
        ),
        FieldSpec(
            name="player_name",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="players.PLAYER_NAME / games_details.PLAYER_NAME",
            description="Standardized full name of the player.",
            constraints="non-empty",
        ),
        FieldSpec(
            name="first_season",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="MIN(SEASON) across appearances",
            description="First NBA season year (YYYY) the player appears in the dataset.",
            constraints="2003 <= first_season <= 2022",
        ),
        FieldSpec(
            name="last_season",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="MAX(SEASON) across appearances",
            description="Most recent NBA season year (YYYY) the player appears in the dataset.",
            constraints="last_season >= first_season",
        ),
        FieldSpec(
            name="career_games_played",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="COUNT(DISTINCT game_id) WHERE did_play == True",
            description="Total number of games in which the player actively entered the game.",
            constraints=">= 0, <= career_games_logged",
        ),
        FieldSpec(
            name="career_games_logged",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="COUNT(DISTINCT game_id) in player_game_stats",
            description="Total number of game box-score entries logged (including DNP/inactive).",
            constraints=">= career_games_played",
        ),
    ),
)


GAMES_ENTITY = EntitySpec(
    name="games",
    storage_file="games.parquet",
    source_files=("games.csv",),
    primary_key=("game_id",),
    description=(
        "Fact table of completed historical NBA games with dates, seasons, participating "
        "teams, final scores, shooting percentages, and game outcomes."
    ),
    grain="One row per completed NBA game (game_id).",
    fields=(
        FieldSpec(
            name="game_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=True,
            foreign_key=None,
            source_mapping="games.GAME_ID",
            description="Unique NBA game identifier.",
            constraints="> 0, unique",
        ),
        FieldSpec(
            name="game_date",
            logical_type="date",
            pandas_dtype="datetime64[ns]",
            sql_type="DATE",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.GAME_DATE_EST",
            description="Calendar date of the game in Eastern Standard Time (YYYY-MM-DD).",
            constraints="2003-10-05 <= game_date <= 2022-12-22",
        ),
        FieldSpec(
            name="season",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.SEASON",
            description="Starting calendar year of the NBA season (e.g., 2022 for 2022-23).",
            constraints="2003 <= season <= 2022",
        ),
        FieldSpec(
            name="season_label",
            logical_type="string",
            pandas_dtype="string",
            sql_type="CHAR(7)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived: YYYY-YY from season",
            description="Formatted NBA season label (e.g., '2022-23').",
            constraints="matches ^\\d{4}-\\d{2}$",
        ),
        FieldSpec(
            name="game_type",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(32)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived from leading digit of GAME_ID",
            description="Competition stage ('Preseason', 'Regular Season', 'Playoffs', 'Play-In').",
            constraints="in {'Preseason', 'Regular Season', 'Playoffs', 'Play-In'}",
        ),
        FieldSpec(
            name="game_status",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(32)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.GAME_STATUS_TEXT",
            description="Completion status of the game ('Final').",
            constraints="== 'Final'",
        ),
        FieldSpec(
            name="home_team_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=False,
            foreign_key="teams.team_id",
            source_mapping="games.HOME_TEAM_ID",
            description="Franchise ID of the home team.",
            constraints="FK -> teams.team_id, != away_team_id",
        ),
        FieldSpec(
            name="away_team_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=False,
            foreign_key="teams.team_id",
            source_mapping="games.VISITOR_TEAM_ID",
            description="Franchise ID of the away (visitor) team.",
            constraints="FK -> teams.team_id, != home_team_id",
        ),
        FieldSpec(
            name="home_pts",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.PTS_home",
            description="Final points scored by the home team.",
            constraints="> 0",
        ),
        FieldSpec(
            name="away_pts",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.PTS_away",
            description="Final points scored by the away team.",
            constraints="> 0",
        ),
        FieldSpec(
            name="total_pts",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived: home_pts + away_pts",
            description="Combined total points scored by both teams.",
            constraints="== home_pts + away_pts",
        ),
        FieldSpec(
            name="point_margin",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived: home_pts - away_pts",
            description="Signed point differential from the home team perspective.",
            constraints="== home_pts - away_pts",
        ),
        FieldSpec(
            name="home_win",
            logical_type="boolean",
            pandas_dtype="bool",
            sql_type="BOOLEAN",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.HOME_TEAM_WINS",
            description="True if the home team won the game (home_pts > away_pts).",
            constraints="== (home_pts > away_pts)",
        ),
        FieldSpec(
            name="home_fg_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.FG_PCT_home",
            description="Home team field goal percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="home_ft_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.FT_PCT_home",
            description="Home team free throw percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="home_fg3_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.FG3_PCT_home",
            description="Home team 3-point field goal percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="home_ast",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.AST_home",
            description="Home team total assists.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="home_reb",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.REB_home",
            description="Home team total rebounds.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="away_fg_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.FG_PCT_away",
            description="Away team field goal percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="away_ft_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.FT_PCT_away",
            description="Away team free throw percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="away_fg3_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.FG3_PCT_away",
            description="Away team 3-point field goal percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="away_ast",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.AST_away",
            description="Away team total assists.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="away_reb",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games.REB_away",
            description="Away team total rebounds.",
            constraints=">= 0",
        ),
    ),
)


PLAYER_GAME_STATS_ENTITY = EntitySpec(
    name="player_game_stats",
    storage_file="player_game_stats.parquet",
    source_files=("games_details.csv",),
    primary_key=("game_id", "team_id", "player_id"),
    description=(
        "Fact table of individual player box-score statistics for each NBA game, "
        "including playing status, starter designation, minutes played, shooting splits, "
        "rebounding, playmaking, defense, and plus/minus."
    ),
    grain="One row per player per team per game (game_id, team_id, player_id).",
    fields=(
        FieldSpec(
            name="game_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=True,
            foreign_key="games.game_id",
            source_mapping="games_details.GAME_ID",
            description="Game identifier linking to canonical games table.",
            constraints="FK -> games.game_id",
        ),
        FieldSpec(
            name="team_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=True,
            foreign_key="teams.team_id",
            source_mapping="games_details.TEAM_ID",
            description="Team identifier linking to canonical teams table.",
            constraints="FK -> teams.team_id",
        ),
        FieldSpec(
            name="player_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=True,
            foreign_key="players.player_id",
            source_mapping="games_details.PLAYER_ID",
            description="Player identifier linking to canonical players table.",
            constraints="FK -> players.player_id",
        ),
        FieldSpec(
            name="player_name",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(128)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.PLAYER_NAME",
            description="Player full name recorded on the game box score.",
        ),
        FieldSpec(
            name="team_abbreviation",
            logical_type="string",
            pandas_dtype="string",
            sql_type="CHAR(3)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.TEAM_ABBREVIATION",
            description="Three-letter team code on the box score.",
        ),
        FieldSpec(
            name="team_city",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(64)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.TEAM_CITY",
            description="City name recorded on the box score.",
        ),
        FieldSpec(
            name="is_starter",
            logical_type="boolean",
            pandas_dtype="bool",
            sql_type="BOOLEAN",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived: START_POSITION in {'F', 'C', 'G'}",
            description="True if the player started the game.",
        ),
        FieldSpec(
            name="start_position",
            logical_type="string",
            pandas_dtype="string",
            sql_type="CHAR(1)",
            nullable=True,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.START_POSITION",
            description="Starting position code ('F', 'C', 'G') or null for bench/DNP players.",
        ),
        FieldSpec(
            name="did_play",
            logical_type="boolean",
            pandas_dtype="bool",
            sql_type="BOOLEAN",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived: PTS is not null",
            description="True if the player entered the game and recorded active statistics.",
        ),
        FieldSpec(
            name="comment",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(256)",
            nullable=True,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.COMMENT",
            description="DNP / injury / coach's decision note when did_play is False.",
        ),
        FieldSpec(
            name="minutes_played",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Parsed from games_details.MIN (MM:SS -> decimal minutes)",
            description="Total minutes played in decimal form (0.0 for DNP).",
            constraints=">= 0.0",
        ),
        FieldSpec(
            name="fgm",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FGM",
            description="Field goals made.",
            constraints=">= 0, <= fga",
        ),
        FieldSpec(
            name="fga",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FGA",
            description="Field goals attempted.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="fg_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FG_PCT",
            description="Field goal percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="fg3m",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FG3M",
            description="Three-point field goals made.",
            constraints=">= 0, <= fg3a",
        ),
        FieldSpec(
            name="fg3a",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FG3A",
            description="Three-point field goals attempted.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="fg3_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FG3_PCT",
            description="Three-point field goal percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="ftm",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FTM",
            description="Free throws made.",
            constraints=">= 0, <= fta",
        ),
        FieldSpec(
            name="fta",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FTA",
            description="Free throws attempted.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="ft_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.FT_PCT",
            description="Free throw percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="oreb",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.OREB",
            description="Offensive rebounds.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="dreb",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.DREB",
            description="Defensive rebounds.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="reb",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.REB",
            description="Total rebounds (oreb + dreb).",
            constraints=">= 0",
        ),
        FieldSpec(
            name="ast",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.AST",
            description="Assists.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="stl",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.STL",
            description="Steals.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="blk",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.BLK",
            description="Blocked shots.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="turnovers",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.TO",
            description="Turnovers committed (renamed from TO to avoid SQL keyword conflict).",
            constraints=">= 0",
        ),
        FieldSpec(
            name="pf",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.PF",
            description="Personal fouls committed.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="pts",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.PTS",
            description="Total points scored by the player.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="plus_minus",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=True,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="games_details.PLUS_MINUS",
            description="On-court point differential while player was in the game.",
        ),
    ),
)


STANDINGS_ENTITY = EntitySpec(
    name="standings",
    storage_file="standings.parquet",
    source_files=("ranking.csv",),
    primary_key=("team_id", "season", "season_type", "standings_date"),
    description=(
        "Daily NBA team standings snapshot table tracking conference, games played, wins, "
        "losses, winning percentage, and home/road splits across seasons."
    ),
    grain="One row per team per season per season_type per standings_date.",
    fields=(
        FieldSpec(
            name="team_id",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="BIGINT",
            nullable=False,
            is_primary_key=True,
            foreign_key="teams.team_id",
            source_mapping="ranking.TEAM_ID",
            description="Franchise ID linking to canonical teams table.",
            constraints="FK -> teams.team_id",
        ),
        FieldSpec(
            name="season",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=True,
            foreign_key=None,
            source_mapping="Derived from last 4 digits of ranking.SEASON_ID",
            description="Starting calendar year of the NBA season (e.g., 2022 for 2022-23).",
            constraints="2002 <= season <= 2022",
        ),
        FieldSpec(
            name="season_label",
            logical_type="string",
            pandas_dtype="string",
            sql_type="CHAR(7)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Derived: YYYY-YY from season",
            description="Formatted NBA season label (e.g., '2022-23').",
        ),
        FieldSpec(
            name="season_type",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(32)",
            nullable=False,
            is_primary_key=True,
            foreign_key=None,
            source_mapping="Derived from leading digit of ranking.SEASON_ID",
            description="Competition stage ('Preseason' or 'Regular Season').",
            constraints="in {'Preseason', 'Regular Season'}",
        ),
        FieldSpec(
            name="standings_date",
            logical_type="date",
            pandas_dtype="datetime64[ns]",
            sql_type="DATE",
            nullable=False,
            is_primary_key=True,
            foreign_key=None,
            source_mapping="ranking.STANDINGSDATE",
            description="Calendar date of the daily standings snapshot (YYYY-MM-DD).",
        ),
        FieldSpec(
            name="conference",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(16)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="ranking.CONFERENCE",
            description="NBA conference ('East' or 'West').",
            constraints="in {'East', 'West'}",
        ),
        FieldSpec(
            name="team_name",
            logical_type="string",
            pandas_dtype="string",
            sql_type="VARCHAR(64)",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="ranking.TEAM",
            description="Team city/market label recorded in standings.",
        ),
        FieldSpec(
            name="games_played",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="ranking.G",
            description="Cumulative games played as of standings_date.",
            constraints="== wins + losses",
        ),
        FieldSpec(
            name="wins",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="ranking.W",
            description="Cumulative wins as of standings_date.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="losses",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="ranking.L",
            description="Cumulative losses as of standings_date.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="win_pct",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="ranking.W_PCT",
            description="Cumulative winning percentage (0.0 to 1.0).",
            constraints="0.0 <= value <= 1.0",
        ),
        FieldSpec(
            name="home_wins",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Parsed wins from ranking.HOME_RECORD ('W-L')",
            description="Cumulative home wins as of standings_date.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="home_losses",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Parsed losses from ranking.HOME_RECORD ('W-L')",
            description="Cumulative home losses as of standings_date.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="road_wins",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Parsed wins from ranking.ROAD_RECORD ('W-L')",
            description="Cumulative road wins as of standings_date.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="road_losses",
            logical_type="integer",
            pandas_dtype="int64",
            sql_type="INTEGER",
            nullable=False,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="Parsed losses from ranking.ROAD_RECORD ('W-L')",
            description="Cumulative road losses as of standings_date.",
            constraints=">= 0",
        ),
        FieldSpec(
            name="return_to_play",
            logical_type="float",
            pandas_dtype="float64",
            sql_type="DOUBLE PRECISION",
            nullable=True,
            is_primary_key=False,
            foreign_key=None,
            source_mapping="ranking.RETURNTOPLAY",
            description="2019-20 Orlando Bubble restart indicator (nullable outside 2019-20).",
        ),
    ),
)


CANONICAL_ENTITIES: dict[str, EntitySpec] = {
    "teams": TEAMS_ENTITY,
    "players": PLAYERS_ENTITY,
    "games": GAMES_ENTITY,
    "player_game_stats": PLAYER_GAME_STATS_ENTITY,
    "standings": STANDINGS_ENTITY,
}


def generate_sql_ddl() -> str:
    """Generate ANSI/PostgreSQL-compatible DDL for the canonical NBA data model."""
    blocks: list[str] = [
        "-- Arcora Canonical NBA Database Schema (DDL)",
        "-- Generated from arcora.models.schema",
        "",
    ]
    for entity in CANONICAL_ENTITIES.values():
        col_lines: list[str] = []
        fk_lines: list[str] = []
        for field in entity.fields:
            null_clause = "NULL" if field.nullable else "NOT NULL"
            col_lines.append(f"    {field.name} {field.sql_type} {null_clause}")
            if field.foreign_key:
                ref_table, ref_col = field.foreign_key.split(".")
                fk_name = f"fk_{entity.name}_{field.name}"
                fk_lines.append(
                    f"    CONSTRAINT {fk_name} FOREIGN KEY ({field.name}) "
                    f"REFERENCES {ref_table} ({ref_col})"
                )
        pk_cols = ", ".join(entity.primary_key)
        col_lines.append(f"    CONSTRAINT pk_{entity.name} PRIMARY KEY ({pk_cols})")
        col_lines.extend(fk_lines)
        body = ",\n".join(col_lines)
        blocks.append(f"CREATE TABLE {entity.name} (\n{body}\n);\n")
    return "\n".join(blocks)


def generate_mermaid_erd() -> str:
    """Generate a Mermaid erDiagram representation of the canonical data model."""
    lines: list[str] = [
        "erDiagram",
        '    teams ||--o{ games : "hosts (home_team_id)"',
        '    teams ||--o{ games : "visits (away_team_id)"',
        '    teams ||--o{ player_game_stats : "fields (team_id)"',
        '    teams ||--o{ standings : "ranks (team_id)"',
        '    games ||--o{ player_game_stats : "includes (game_id)"',
        '    players ||--o{ player_game_stats : "records (player_id)"',
        "",
    ]
    for entity in CANONICAL_ENTITIES.values():
        lines.append(f"    {entity.name} {{")
        for field in entity.fields:
            key_markers: list[str] = []
            if field.is_primary_key:
                key_markers.append("PK")
            if field.foreign_key:
                key_markers.append("FK")
            marker_str = f" {','.join(key_markers)}" if key_markers else ""
            lines.append(f"        {field.logical_type} {field.name}{marker_str}")
        lines.append("    }")
    return "\n".join(lines)


def render_data_dictionary_markdown() -> str:
    """Render the complete Data Dictionary markdown document."""
    lines = [
        "# Arcora Canonical NBA Data Dictionary",
        "",
        "This document provides field-level definitions, data types, keys, source mappings, "
        "nullability rules, and domain constraints for all four core entities in Arcora's "
        "canonical data model.",
        "",
    ]
    for entity in CANONICAL_ENTITIES.values():
        pk_str = ", ".join(f"`{k}`" for k in entity.primary_key)
        src_str = ", ".join(f"`{s}`" for s in entity.source_files)
        lines.extend(
            [
                f"## Entity: `{entity.name}` (`data/processed/{entity.storage_file}`)",
                "",
                f"- **Description**: {entity.description}",
                f"- **Grain**: {entity.grain}",
                f"- **Primary Key**: ({pk_str})",
                f"- **Source Files**: {src_str}",
                "",
                "| Field Name | Logical Type | Pandas / Parquet Dtype | SQL Type | "
                "Nullable | Key | Source Mapping | Constraints | Description |",
                "| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |",
            ]
        )
        for f in entity.fields:
            key_badge = ""
            if f.is_primary_key and f.foreign_key:
                key_badge = f"PK, FK ({f.foreign_key})"
            elif f.is_primary_key:
                key_badge = "PK"
            elif f.foreign_key:
                key_badge = f"FK ({f.foreign_key})"
            null_str = "Yes" if f.nullable else "No"
            constraint_str = f.constraints or "—"
            lines.append(
                f"| `{f.name}` | `{f.logical_type}` | `{f.pandas_dtype}` | `{f.sql_type}` | "
                f"{null_str} | {key_badge} | `{f.source_mapping}` | {constraint_str} | "
                f"{f.description} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_canonical_model_markdown() -> str:
    """Render the Canonical Data Model architecture, ERD, and design decisions document."""
    erd_block = generate_mermaid_erd()
    sql_block = generate_sql_ddl()
    lines = [
        "# Arcora Canonical NBA Data Model & Architecture",
        "",
        "## 1. Overview",
        "",
        "Arcora's canonical NBA data model organizes historical NBA data into five clean, "
        "relational entities optimized for both analytical queries and machine learning feature "
        "engineering:",
        "",
        "1. **`teams`** (`teams.parquet`): Master franchise dimension table (30 NBA teams).",
        "2. **`players`** (`players.parquet`): Unified player dimension table (2,687 NBA players "
        "spanning 2003–2022, focused purely on player-level attributes).",
        "3. **`games`** (`games.parquet`): Completed game fact table (26,523 deduplicated games "
        "with valid final scores).",
        "4. **`player_game_stats`** (`player_game_stats.parquet`): Player-level game box-score "
        "fact table (668,338 deduplicated player-game records, capturing historical team "
        "membership per game).",
        "5. **`standings`** (`standings.parquet`): Daily conference standings snapshot table "
        "(210,313 deduplicated team-date standings records).",
        "",
        "---",
        "",
        "## 2. Entity Relationship Diagram (ERD)",
        "",
        "```mermaid",
        erd_block,
        "```",
        "",
        "---",
        "",
        "## 3. Primary Keys & Foreign Key Relationships",
        "",
        "| Entity | Primary Key | Foreign Keys | Cardinality |",
        "| :--- | :--- | :--- | :--- |",
        "| `teams` | `(team_id)` | *None* | 1 franchise : N games, standings, box scores |",
        "| `players` | `(player_id)` | *None* | 1 player : N player_game_stats |",
        "| `games` | `(game_id)` | `home_team_id` -> `teams.team_id`, "
        "`away_team_id` -> `teams.team_id` | 1 game : N player_game_stats |",
        "| `player_game_stats` | `(game_id, team_id, player_id)` | "
        "`game_id` -> `games.game_id`, `team_id` -> `teams.team_id`, "
        "`player_id` -> `players.player_id` | N:1 with games, teams, players |",
        "| `standings` | `(team_id, season, season_type, standings_date)` | "
        "`team_id` -> `teams.team_id` | N:1 with teams |",
        "",
        "---",
        "",
        "## 4. Naming Conventions & Type Standards",
        "",
        "- **`snake_case` Identifiers**: All table and column names use lowercase `snake_case` "
        "(e.g., `GAME_DATE_EST` -> `game_date`, `VISITOR_TEAM_ID` -> `away_team_id`, "
        "`PTS_home` -> `home_pts`).",
        "- **SQL Reserved Word Avoidance**: `games_details.TO` (turnovers) is renamed to "
        "`turnovers` so SQL queries do not require quoting reserved keywords.",
        "- **Strict Integer & Boolean Typing**: Score, rebound, assist, field-goal, and foul "
        "counts that were coerced to `float64` by `NaN` rows in raw CSVs are normalized to "
        "`int64`, and win/starter/active flags are stored as native `bool`.",
        "- **Derived Analytical Fields**: `games` precomputes `season_label` (`YYYY-YY`), "
        "`game_type` (`Preseason`, `Regular Season`, `Playoffs`, `Play-In`), `total_pts`, "
        "and `point_margin` (`home_pts - away_pts`).",
        "",
        "---",
        "",
        "## 5. Key Assumptions & Design Decisions",
        "",
        "1. **Unifying `players` Across `players.csv` and `games_details.csv`**:",
        "   - *Problem*: `players.csv` only contains seasons 2009–2019 (1,769 unique players), "
        "leaving 918 players from 2003–2008 and 2020–2022 without a parent record.",
        "   - *Decision*: The canonical `players` entity unions `players.csv` with distinct "
        "`(PLAYER_ID, PLAYER_NAME)` entries from `games_details.csv`, enriched with "
        "`first_season`, `last_season`, `career_games_played` (`did_play == True`), and "
        "`career_games_logged` (all box-score entries). Team membership is intentionally "
        "modeled historically in `player_game_stats` rather than a mutable `primary_team_id` "
        "on `players`.",
        "2. **Handling Unscored 2003 Preseason Games**:",
        "   - *Problem*: 99 rows in `games.csv` (all in October 2003) have `NaN` for all box-score "
        "columns and no rows in `games_details.csv`.",
        "   - *Decision*: Exclude unscored rows from canonical `games.parquet` so all records in "
        "`games` have valid, non-null scores and outcomes.",
        "3. **Handling DNP / Inactive Box-Score Rows in `player_game_stats`**:",
        "   - *Problem*: 109,690 rows in `games_details.csv` represent dressed/rostered players "
        "who did not enter the game (`MIN` and counting stats are `NaN`).",
        "   - *Decision*: Retain these rows for roster completeness, set `did_play = False`, "
        "normalize `minutes_played = 0.0` and counting stats (`pts`, `reb`, `ast`, etc.) to `0`, "
        "and preserve the DNP reason in `comment`.",
        "4. **Normalizing `MIN` (Minutes Played)**:",
        "   - *Problem*: Raw `MIN` strings mix `'MM:SS'`, integer strings `'MM'`, float-prefixed "
        "strings `'12.000000:00'`, 1,463 `'MM:60'` strings, and 12 corrupted negative strings "
        "(e.g., `'-9'`).",
        "   - *Decision*: Parse `'MM:SS'` into decimal minutes (`MM + SS / 60.0`) enforcing "
        "`0 <= SS < 60`, reset malformed or negative values to `0.0` (with explicit audit "
        "counts in `ingestion_manifest.json`), and store as `float64` in `minutes_played`.",
        "5. **Point-in-Time Leakage Prevention for `standings`**:",
        "   - *Problem*: `standings.parquet` stores daily snapshots (`standings_date`). Joining "
        "a game to end-of-season standings or same-day post-game standings would leak future "
        "outcomes into pre-game features.",
        "   - *Decision*: `standings` preserves the exact `standings_date` primary-key component "
        "so Week 2 feature pipelines can perform an as-of temporal join strictly prior to game "
        "day (`standings_date < game_date` within the same `(team_id, season, season_type)`).",
        "",
        "---",
        "",
        "## 6. Canonical SQL Schema (DDL)",
        "",
        "```sql",
        sql_block.rstrip(),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_schema_documentation(docs_dir: Path) -> dict[str, Path]:
    """Write canonical_data_model.md, data_dictionary.md, canonical_schema.sql, and erd.mmd."""
    docs_dir.mkdir(parents=True, exist_ok=True)

    model_doc = docs_dir / "canonical_data_model.md"
    model_doc.write_text(render_canonical_model_markdown(), encoding="utf-8")

    dict_doc = docs_dir / "data_dictionary.md"
    dict_doc.write_text(render_data_dictionary_markdown(), encoding="utf-8")

    sql_doc = docs_dir / "canonical_schema.sql"
    sql_doc.write_text(generate_sql_ddl(), encoding="utf-8")

    erd_doc = docs_dir / "erd.mmd"
    erd_doc.write_text(generate_mermaid_erd() + "\n", encoding="utf-8")

    return {
        "canonical_data_model": model_doc,
        "data_dictionary": dict_doc,
        "canonical_schema_sql": sql_doc,
        "erd_mermaid": erd_doc,
    }
