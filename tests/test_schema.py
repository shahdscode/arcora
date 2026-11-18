"""Tests for canonical NBA data model and schema documentation (Issue #4)."""

from __future__ import annotations

from pathlib import Path

from arcora.models.schema import (
    CANONICAL_ENTITIES,
    GAMES_ENTITY,
    PLAYER_GAME_STATS_ENTITY,
    PLAYERS_ENTITY,
    STANDINGS_ENTITY,
    TEAMS_ENTITY,
    generate_mermaid_erd,
    generate_sql_ddl,
    write_schema_documentation,
)


def test_canonical_entities_defined() -> None:
    """Verify all 5 required core entities and their primary/foreign keys are defined."""
    assert set(CANONICAL_ENTITIES.keys()) == {
        "teams",
        "players",
        "games",
        "player_game_stats",
        "standings",
    }
    assert TEAMS_ENTITY.primary_key == ("team_id",)
    assert PLAYERS_ENTITY.primary_key == ("player_id",)
    assert "primary_team_id" not in PLAYERS_ENTITY.column_names
    assert "career_games_played" in PLAYERS_ENTITY.column_names
    assert "career_games_logged" in PLAYERS_ENTITY.column_names
    assert GAMES_ENTITY.primary_key == ("game_id",)
    assert PLAYER_GAME_STATS_ENTITY.primary_key == ("game_id", "team_id", "player_id")
    assert STANDINGS_ENTITY.primary_key == (
        "team_id",
        "season",
        "season_type",
        "standings_date",
    )


def test_snake_case_naming_and_foreign_keys() -> None:
    """Ensure all canonical fields follow snake_case and foreign keys reference valid entities."""
    for entity in CANONICAL_ENTITIES.values():
        for field in entity.fields:
            assert field.name == field.name.lower()
            assert " " not in field.name
            if field.foreign_key is not None:
                target_table, target_col = field.foreign_key.split(".")
                assert target_table in CANONICAL_ENTITIES
                assert target_col in CANONICAL_ENTITIES[target_table].column_names


def test_schema_documentation_generation(tmp_path: Path) -> None:
    """Verify DDL, Mermaid ERD, data dictionary, and schema docs generate cleanly."""
    ddl = generate_sql_ddl()
    assert "CREATE TABLE teams" in ddl
    assert "CREATE TABLE players" in ddl
    assert "CREATE TABLE games" in ddl
    assert "CREATE TABLE player_game_stats" in ddl
    assert "CREATE TABLE standings" in ddl

    erd = generate_mermaid_erd()
    assert "erDiagram" in erd
    assert "player_game_stats" in erd

    artifacts = write_schema_documentation(tmp_path / "docs")
    for path in artifacts.values():
        assert path.is_file()
        assert path.stat().st_size > 0
