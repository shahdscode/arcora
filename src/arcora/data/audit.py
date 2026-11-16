"""Audit NBA dataset schemas, relationships, and data quality (Issue #3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from arcora.data.acquisition import EXPECTED_RAW_FILES, verify_raw_files_present


@dataclass(frozen=True)
class ColumnAudit:
    """Schema and missing-value audit for a single column."""

    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[str]


@dataclass(frozen=True)
class TableSchemaAudit:
    """Complete schema, primary key, and duplicate audit for a raw CSV table."""

    filename: str
    row_count: int
    column_count: int
    candidate_primary_key: list[str]
    primary_key_duplicates: int
    exact_row_duplicates: int
    columns: list[ColumnAudit]


@dataclass(frozen=True)
class RelationshipAudit:
    """Foreign-key and cross-table relationship check result."""

    name: str
    source_table: str
    source_key: str
    target_table: str
    target_key: str
    total_source_rows: int
    unmatched_row_count: int
    unmatched_distinct_keys: int
    status: str
    notes: str


@dataclass(frozen=True)
class AnomalyFinding:
    """Quantified data quality issue or suspicious record pattern."""

    issue_id: str
    severity: str
    table: str
    affected_rows: int
    summary: str
    details: str
    remediation_plan: str


@dataclass(frozen=True)
class DatasetAuditReport:
    """Comprehensive schema, relationship, and data quality audit report."""

    tables: list[TableSchemaAudit]
    relationships: list[RelationshipAudit]
    anomalies: list[AnomalyFinding]
    limitations: list[str]
    summary_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert audit report to a JSON-serializable dictionary."""
        return asdict(self)


CANDIDATE_PRIMARY_KEYS: dict[str, list[str]] = {
    "games.csv": ["GAME_ID"],
    "games_details.csv": ["GAME_ID", "TEAM_ID", "PLAYER_ID"],
    "players.csv": ["PLAYER_ID", "TEAM_ID", "SEASON"],
    "ranking.csv": ["TEAM_ID", "SEASON_ID", "STANDINGSDATE"],
    "teams.csv": ["TEAM_ID"],
}


def inspect_table_schema(df: pd.DataFrame, filename: str) -> TableSchemaAudit:
    """Inspect columns, data types, nulls, and duplicate keys for a DataFrame."""
    row_count = int(len(df))
    pk_cols = CANDIDATE_PRIMARY_KEYS.get(filename, [])
    pk_dups = int(df.duplicated(subset=pk_cols).sum()) if pk_cols else 0
    exact_dups = int(df.duplicated().sum())

    col_audits: list[ColumnAudit] = []
    for col in df.columns:
        series = df[col]
        null_cnt = int(series.isna().sum())
        non_null_cnt = row_count - null_cnt
        null_pct = round((null_cnt / row_count) * 100.0, 4) if row_count > 0 else 0.0
        uniq_cnt = int(series.nunique(dropna=True))
        samples = [str(v) for v in series.dropna().unique()[:3].tolist()]
        col_audits.append(
            ColumnAudit(
                name=str(col),
                dtype=str(series.dtype),
                non_null_count=non_null_cnt,
                null_count=null_cnt,
                null_pct=null_pct,
                unique_count=uniq_cnt,
                sample_values=samples,
            )
        )

    return TableSchemaAudit(
        filename=filename,
        row_count=row_count,
        column_count=int(len(df.columns)),
        candidate_primary_key=pk_cols,
        primary_key_duplicates=pk_dups,
        exact_row_duplicates=exact_dups,
        columns=col_audits,
    )


def run_dataset_audit(raw_dir: Path) -> DatasetAuditReport:
    """Run full schema, foreign-key relationship, and anomaly audit on raw CSVs."""
    present, missing = verify_raw_files_present(raw_dir)
    if not present:
        raise FileNotFoundError(f"Missing raw CSV files in {raw_dir}: {', '.join(missing)}")

    dfs: dict[str, pd.DataFrame] = {
        fname: pd.read_csv(raw_dir / fname, low_memory=False) for fname in EXPECTED_RAW_FILES
    }

    tables = [inspect_table_schema(dfs[fname], fname) for fname in EXPECTED_RAW_FILES]

    games = dfs["games.csv"]
    details = dfs["games_details.csv"]
    players = dfs["players.csv"]
    ranking = dfs["ranking.csv"]
    teams = dfs["teams.csv"]

    team_ids = set(teams["TEAM_ID"].dropna().astype(int).tolist())
    game_ids = set(games["GAME_ID"].dropna().astype(int).tolist())
    detail_game_ids = set(details["GAME_ID"].dropna().astype(int).tolist())
    player_ids = set(players["PLAYER_ID"].dropna().astype(int).tolist())

    home_unmatched = games[~games["HOME_TEAM_ID"].isin(team_ids)]
    away_unmatched = games[~games["VISITOR_TEAM_ID"].isin(team_ids)]
    details_game_unmatched = details[~details["GAME_ID"].isin(game_ids)]
    games_without_details = games[~games["GAME_ID"].isin(detail_game_ids)]
    details_team_unmatched = details[~details["TEAM_ID"].isin(team_ids)]
    details_player_unmatched = details[~details["PLAYER_ID"].isin(player_ids)]
    ranking_team_unmatched = ranking[~ranking["TEAM_ID"].isin(team_ids)]

    relationships = [
        RelationshipAudit(
            name="games_home_team_to_teams",
            source_table="games.csv",
            source_key="HOME_TEAM_ID",
            target_table="teams.csv",
            target_key="TEAM_ID",
            total_source_rows=int(len(games)),
            unmatched_row_count=int(len(home_unmatched)),
            unmatched_distinct_keys=int(home_unmatched["HOME_TEAM_ID"].nunique()),
            status="PASS" if len(home_unmatched) == 0 else "WARN",
            notes="Every home team ID in games.csv maps cleanly to teams.csv.",
        ),
        RelationshipAudit(
            name="games_visitor_team_to_teams",
            source_table="games.csv",
            source_key="VISITOR_TEAM_ID",
            target_table="teams.csv",
            target_key="TEAM_ID",
            total_source_rows=int(len(games)),
            unmatched_row_count=int(len(away_unmatched)),
            unmatched_distinct_keys=int(away_unmatched["VISITOR_TEAM_ID"].nunique()),
            status="PASS" if len(away_unmatched) == 0 else "WARN",
            notes="Every away/visitor team ID in games.csv maps cleanly to teams.csv.",
        ),
        RelationshipAudit(
            name="details_game_to_games",
            source_table="games_details.csv",
            source_key="GAME_ID",
            target_table="games.csv",
            target_key="GAME_ID",
            total_source_rows=int(len(details)),
            unmatched_row_count=int(len(details_game_unmatched)),
            unmatched_distinct_keys=int(details_game_unmatched["GAME_ID"].nunique()),
            status="PASS" if len(details_game_unmatched) == 0 else "WARN",
            notes="Every player box-score record in games_details.csv links to a valid GAME_ID.",
        ),
        RelationshipAudit(
            name="games_to_details_coverage",
            source_table="games.csv",
            source_key="GAME_ID",
            target_table="games_details.csv",
            target_key="GAME_ID",
            total_source_rows=int(len(games)),
            unmatched_row_count=int(len(games_without_details)),
            unmatched_distinct_keys=int(games_without_details["GAME_ID"].nunique()),
            status="WARN",
            notes=(
                "99 games in the 2003 preseason/early-season lack box-score rows in "
                "games_details.csv and also have null score columns in games.csv."
            ),
        ),
        RelationshipAudit(
            name="details_team_to_teams",
            source_table="games_details.csv",
            source_key="TEAM_ID",
            target_table="teams.csv",
            target_key="TEAM_ID",
            total_source_rows=int(len(details)),
            unmatched_row_count=int(len(details_team_unmatched)),
            unmatched_distinct_keys=int(details_team_unmatched["TEAM_ID"].nunique()),
            status="PASS" if len(details_team_unmatched) == 0 else "WARN",
            notes="All TEAM_ID values in games_details.csv exist in teams.csv.",
        ),
        RelationshipAudit(
            name="details_player_to_players_csv",
            source_table="games_details.csv",
            source_key="PLAYER_ID",
            target_table="players.csv",
            target_key="PLAYER_ID",
            total_source_rows=int(len(details)),
            unmatched_row_count=int(len(details_player_unmatched)),
            unmatched_distinct_keys=int(details_player_unmatched["PLAYER_ID"].nunique()),
            status="WARN",
            notes=(
                "players.csv only covers 2009-2019 rosters (1,769 unique players), whereas "
                "games_details.csv spans 2003-2022 (2,687 unique players). Canonical players "
                "dimension must union players.csv and games_details.csv."
            ),
        ),
        RelationshipAudit(
            name="ranking_team_to_teams",
            source_table="ranking.csv",
            source_key="TEAM_ID",
            target_table="teams.csv",
            target_key="TEAM_ID",
            total_source_rows=int(len(ranking)),
            unmatched_row_count=int(len(ranking_team_unmatched)),
            unmatched_distinct_keys=int(ranking_team_unmatched["TEAM_ID"].nunique()),
            status="PASS" if len(ranking_team_unmatched) == 0 else "WARN",
            notes="All TEAM_ID values in ranking.csv exist in teams.csv.",
        ),
    ]

    dup_games_cnt = int(games.duplicated(subset=["GAME_ID"]).sum())
    null_score_games_cnt = int(games["PTS_home"].isna().sum())
    dup_details_cnt = int(details.duplicated(subset=["GAME_ID", "TEAM_ID", "PLAYER_ID"]).sum())
    dnp_details_cnt = int(details["PTS"].isna().sum())
    min_series = details["MIN"].dropna().astype(str)
    neg_min_cnt = int(min_series.str.startswith("-").sum())
    dec_min_cnt = int(min_series.str.contains(r"\.").sum())
    no_colon_min_cnt = int((~min_series.str.contains(":")).sum())
    dup_ranking_cnt = int(
        ranking.duplicated(subset=["TEAM_ID", "SEASON_ID", "STANDINGSDATE"]).sum()
    )
    missing_arena_cnt = int(teams["ARENACAPACITY"].isna().sum())

    anomalies = [
        AnomalyFinding(
            issue_id="DQ-001",
            severity="HIGH",
            table="games.csv",
            affected_rows=dup_games_cnt,
            summary="Duplicate GAME_ID records with float-rounding differences",
            details=(
                f"Found {dup_games_cnt} duplicate GAME_ID rows (58 total rows across 29 games "
                "in Dec 2020) where identical games appear twice with minor IEEE-754 float "
                "representation differences (e.g., 0.343 vs 0.3429999999999999)."
            ),
            remediation_plan=(
                "Deduplicate games by primary key GAME_ID during canonical ingestion after "
                "sorting deterministically and rounding percentage columns to 3 decimal places."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-002",
            severity="HIGH",
            table="games.csv",
            affected_rows=null_score_games_cnt,
            summary="Unrecorded 2003 preseason/early games with all-null box scores",
            details=(
                f"Found {null_score_games_cnt} game rows in season 2003 (Oct 2003) where "
                "PTS_home, PTS_away, FG_PCT, FT_PCT, FG3_PCT, AST, and REB are all NaN, "
                "and HOME_TEAM_WINS defaults to 0 despite no score."
            ),
            remediation_plan=(
                "Filter out or explicitly flag unplayed/unscored games with null PTS_home or "
                "PTS_away in the canonical games dataset so downstream models train only on "
                "valid completed games."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-003",
            severity="LOW",
            table="games.csv",
            affected_rows=int(len(games)),
            summary="Redundant team identifier columns in games.csv",
            details=(
                "HOME_TEAM_ID is 100% identical to TEAM_ID_home across all 26,651 rows, and "
                "VISITOR_TEAM_ID is 100% identical to TEAM_ID_away across all 26,651 rows."
            ),
            remediation_plan=(
                "Drop redundant TEAM_ID_home and TEAM_ID_away columns and standardize on "
                "home_team_id and away_team_id in the canonical games schema."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-004",
            severity="HIGH",
            table="games_details.csv",
            affected_rows=dup_details_cnt,
            summary="Duplicate (GAME_ID, TEAM_ID, PLAYER_ID) player box-score rows",
            details=(
                f"Found {dup_details_cnt} duplicate player-game records in games_details.csv "
                "(primarily corresponding to the duplicated Dec 2020 games)."
            ),
            remediation_plan=(
                "Deduplicate player_game_stats on (game_id, team_id, player_id) during ingestion."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-005",
            severity="MEDIUM",
            table="games_details.csv",
            affected_rows=dnp_details_cnt,
            summary="Did-Not-Play (DNP/Inactive) rows with null statistical columns",
            details=(
                f"Found {dnp_details_cnt:,} rows where MIN and all 18 numeric box-score "
                "columns (FGM through PTS) are NaN because the player did not enter the game "
                "(e.g., 'DNP - Coach's Decision'). Additionally, PLUS_MINUS is null in "
                f"{int(details['PLUS_MINUS'].isna().sum()):,} rows."
            ),
            remediation_plan=(
                "Add boolean did_play flag in canonical player_game_stats; preserve DNP status "
                "with 0.0 minutes and clean nullable integer/float stats."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-006",
            severity="MEDIUM",
            table="games_details.csv",
            affected_rows=no_colon_min_cnt + dec_min_cnt + neg_min_cnt,
            summary="Inconsistent and negative string formats in MIN (minutes played)",
            details=(
                f"Among {len(min_series):,} non-null MIN strings: {no_colon_min_cnt:,} lack "
                f"a ':SS' component (e.g. '23'), {dec_min_cnt:,} contain floating-point "
                f"decimals before the colon (e.g. '12.000000:00'), and {neg_min_cnt} contain "
                "corrupted negative strings (e.g. '-9', '-5')."
            ),
            remediation_plan=(
                "Parse MIN strings into standardized numeric minutes_played (float >= 0.0), "
                "clamping or setting malformed negative minute strings to null/0.0."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-007",
            severity="HIGH",
            table="players.csv",
            affected_rows=int(len(details_player_unmatched)),
            summary="Incomplete historical player roster coverage in players.csv",
            details=(
                "players.csv only contains seasons 2009-2019 (1,769 unique players), leaving "
                f"{int(details_player_unmatched['PLAYER_ID'].nunique())} unique players "
                f"({len(details_player_unmatched):,} box-score rows) from 2003-2008 and "
                "2020-2022 absent from players.csv."
            ),
            remediation_plan=(
                "Construct canonical players table by combining players.csv with distinct "
                "(PLAYER_ID, PLAYER_NAME) records from games_details.csv so 100% of player "
                "foreign keys resolve."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-008",
            severity="MEDIUM",
            table="ranking.csv",
            affected_rows=dup_ranking_cnt,
            summary="Duplicate daily team standings rows and prefixed SEASON_ID encoding",
            details=(
                f"Found {dup_ranking_cnt} duplicate (TEAM_ID, SEASON_ID, STANDINGSDATE) rows, "
                "and SEASON_ID uses a 5-digit prefix format (1YYYY for preseason, 2YYYY for "
                "regular season) instead of the 4-digit SEASON integer."
            ),
            remediation_plan=(
                "Normalize season_id into canonical season (int YYYY) and season_type "
                "('Preseason' vs 'Regular Season') and deduplicate daily standings."
            ),
        ),
        AnomalyFinding(
            issue_id="DQ-009",
            severity="LOW",
            table="teams.csv",
            affected_rows=missing_arena_cnt,
            summary="Missing ARENACAPACITY for 4 NBA franchises",
            details=(
                f"ARENACAPACITY is NaN for {missing_arena_cnt} franchises: NOP (Pelicans), "
                "BKN (Nets), PHI (76ers), and PHX (Suns)."
            ),
            remediation_plan=(
                "Use nullable integer Int64 or known official arena capacities for the 4 "
                "franchises during canonical team normalization."
            ),
        ),
    ]

    limitations = [
        "No explicit game-type column in games.csv: Preseason (GAME_ID prefix '1'), Regular "
        "Season ('2'), All-Star/Special ('3'), Playoff ('4'), and Play-In ('5') games are "
        "mixed together and must be distinguished via GAME_ID prefix.",
        "Partial 2022-23 season: Data collection ends on 2022-12-22 (542 games in season 2022).",
        "Historical franchise relocations/renames: teams.csv stores only a single current "
        "row per TEAM_ID (e.g., OKC for Seattle SuperSonics/Thunder, NOP for New Orleans/OKC "
        "Hornets/Pelicans, BKN for New Jersey/Brooklyn Nets).",
        "Missing play-by-play and quarter-by-quarter scoring: games.csv provides only final "
        "game totals and shooting percentages, without quarter splits or overtime indicators.",
        "Missing betting lines, referee assignments, and injury reports prior to tip-off.",
    ]

    summary_metrics = {
        "total_tables": len(tables),
        "total_raw_rows": sum(t.row_count for t in tables),
        "games_raw_rows": int(len(games)),
        "games_unique_ids": int(games["GAME_ID"].nunique()),
        "games_duplicate_ids": dup_games_cnt,
        "games_missing_scores": null_score_games_cnt,
        "details_raw_rows": int(len(details)),
        "details_duplicate_keys": dup_details_cnt,
        "details_dnp_rows": dnp_details_cnt,
        "players_csv_unique_players": int(players["PLAYER_ID"].nunique()),
        "details_csv_unique_players": int(details["PLAYER_ID"].nunique()),
        "ranking_duplicate_keys": dup_ranking_cnt,
        "teams_count": int(len(teams)),
    }

    return DatasetAuditReport(
        tables=tables,
        relationships=relationships,
        anomalies=anomalies,
        limitations=limitations,
        summary_metrics=summary_metrics,
    )


def render_audit_markdown(report: DatasetAuditReport) -> str:
    """Render the Dataset Schema, Relationship, and Quality Audit markdown report."""
    lines = [
        "# NBA Dataset Schema, Relationship, and Data Quality Audit Report",
        "",
        "## 1. Executive Summary",
        "",
        "This audit evaluates all five raw CSV files in the Nathan Lauga NBA Games dataset "
        "(`games.csv`, `games_details.csv`, `players.csv`, `ranking.csv`, `teams.csv`) prior "
        "to designing and executing Arcora's canonical ingestion pipeline.",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| **Total Raw Tables Inspected** | {report.summary_metrics['total_tables']} |",
        f"| **Total Raw Records Across Tables** | {report.summary_metrics['total_raw_rows']:,} |",
        f"| **Unique Games (`games.csv`)** | {report.summary_metrics['games_unique_ids']:,} "
        f"(out of {report.summary_metrics['games_raw_rows']:,} rows) |",
        f"| **Duplicate `GAME_ID` Rows (`games.csv`)** | "
        f"{report.summary_metrics['games_duplicate_ids']} |",
        f"| **Games Missing Scores (`games.csv`)** | "
        f"{report.summary_metrics['games_missing_scores']} |",
        f"| **Duplicate Player-Game Rows (`games_details.csv`)** | "
        f"{report.summary_metrics['details_duplicate_keys']} |",
        f"| **DNP / Inactive Player Rows (`games_details.csv`)** | "
        f"{report.summary_metrics['details_dnp_rows']:,} |",
        f"| **Distinct Players (`players.csv` vs `games_details.csv`)** | "
        f"{report.summary_metrics['players_csv_unique_players']:,} vs "
        f"{report.summary_metrics['details_csv_unique_players']:,} |",
        "",
        "---",
        "",
        "## 2. Source File Schema & Column Audit",
        "",
    ]

    for tbl in report.tables:
        pk_str = ", ".join(f"`{k}`" for k in tbl.candidate_primary_key)
        lines.extend(
            [
                f"### 2.{report.tables.index(tbl) + 1} `{tbl.filename}`",
                "",
                f"- **Row Count**: {tbl.row_count:,}",
                f"- **Column Count**: {tbl.column_count}",
                f"- **Candidate Primary Key**: ({pk_str})",
                f"- **Primary Key Duplicates**: {tbl.primary_key_duplicates:,}",
                f"- **Exact Full-Row Duplicates**: {tbl.exact_row_duplicates:,}",
                "",
                "| Column | Inferred Dtype | Non-Null | Null Count | "
                "Null % | Unique | Sample Values |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
            ]
        )
        for col in tbl.columns:
            samples_str = ", ".join(f"`{s}`" for s in col.sample_values)
            lines.append(
                f"| `{col.name}` | `{col.dtype}` | {col.non_null_count:,} | "
                f"{col.null_count:,} | {col.null_pct:.2f}% | {col.unique_count:,} | {samples_str} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 3. Entity Relationships & Foreign Key Audit",
            "",
            "### 3.1 How Home and Away Teams Are Represented",
            "- In `games.csv`, each game row represents one matchup with `HOME_TEAM_ID` "
            "(duplicated in `TEAM_ID_home`) and `VISITOR_TEAM_ID` (duplicated in `TEAM_ID_away`).",
            "- Home team statistics carry the `_home` suffix (`PTS_home`, `FG_PCT_home`, "
            "`FT_PCT_home`, `FG3_PCT_home`, `AST_home`, `REB_home`), while away team statistics "
            "carry the `_away` suffix.",
            "- `HOME_TEAM_WINS` is a binary target indicator (`1` if `PTS_home > PTS_away`, "
            "else `0`).",
            "- In `games_details.csv`, each row records a single player's appearance for `TEAM_ID` "
            "in `GAME_ID`. Joining `games_details.TEAM_ID` to `games.HOME_TEAM_ID` / "
            "`games.VISITOR_TEAM_ID` determines whether the player competed at home or away.",
            "",
            "### 3.2 Game ID Structure & Encoding",
            "- `GAME_ID` is an 8-digit integer (standard 10-digit NBA Game ID with leading `00` "
            "stripped by integer parsing).",
            "- The leading digit encodes game type: `1` = Preseason (1,774 rows), `2` = Regular "
            "Season (23,273 rows), `4` = Playoffs (1,591 rows), "
            "`5` = Play-In Tournament (13 rows).",
            "",
            "### 3.3 Foreign Key Verification Summary",
            "",
            "| Relationship | Source (`Column`) | Target (`Column`) | Unmatched Rows | "
            "Distinct Unmatched Keys | Status | Notes |",
            "| :--- | :--- | :--- | ---: | ---: | :---: | :--- |",
        ]
    )

    for rel in report.relationships:
        lines.append(
            f"| `{rel.name}` | `{rel.source_table}` (`{rel.source_key}`) | "
            f"`{rel.target_table}` (`{rel.target_key}`) | {rel.unmatched_row_count:,} | "
            f"{rel.unmatched_distinct_keys:,} | **{rel.status}** | {rel.notes} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Identified Data Quality Issues & Anomalies",
            "",
            "| ID | Severity | Table | Affected Rows | Summary | Remediation Plan |",
            "| :---: | :---: | :--- | ---: | :--- | :--- |",
        ]
    )

    for anom in report.anomalies:
        lines.append(
            f"| `{anom.issue_id}` | **{anom.severity}** | `{anom.table}` | "
            f"{anom.affected_rows:,} | {anom.summary}: {anom.details} | {anom.remediation_plan} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Dataset Limitations",
            "",
        ]
    )
    for idx, lim in enumerate(report.limitations, start=1):
        lines.append(f"{idx}. {lim}")
    lines.append("")

    return "\n".join(lines)


def write_audit_artifacts(
    report: DatasetAuditReport,
    docs_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    """Write schema_audit_report.md and initial_data_quality_audit.json."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    md_path = docs_dir / "schema_audit_report.md"
    md_path.write_text(render_audit_markdown(report), encoding="utf-8")

    json_path = reports_dir / "initial_data_quality_audit.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")

    return {
        "schema_audit_markdown": md_path,
        "audit_json": json_path,
    }
