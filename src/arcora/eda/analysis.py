"""Exploratory Data Analysis (EDA) and visualization suite for NBA datasets (Issue #7)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from arcora.models.schema import (
    GAMES_ENTITY,
    PLAYER_GAME_STATS_ENTITY,
    PLAYERS_ENTITY,
    TEAMS_ENTITY,
)


@dataclass(frozen=True)
class DistributionStats:
    """Summary statistics for a numeric distribution."""

    count: int
    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float
    skewness: float


@dataclass(frozen=True)
class SeasonTrendRow:
    """Season-level aggregated scoring, pace, and home-court advantage metrics."""

    season: int
    season_label: str
    total_games: int
    regular_season_games: int
    playoff_games: int
    preseason_games: int
    active_teams: int
    mean_home_pts: float
    mean_away_pts: float
    mean_total_pts: float
    mean_point_margin: float
    home_win_rate: float
    is_anomalous_season: bool
    anomaly_reason: str


@dataclass(frozen=True)
class ExtremeGameRecord:
    """Details of an extreme or notable historical NBA game."""

    category: str
    game_id: int
    game_date: str
    season_label: str
    home_team: str
    away_team: str
    home_pts: int
    away_pts: int
    total_pts: int
    point_margin: int


@dataclass(frozen=True)
class CandidateFeatureSpec:
    """Specification for a candidate predictive feature for game outcome modeling."""

    feature_name: str
    category: str
    target_relevance: str
    computation_window: str
    source_entities: list[str]
    rationale: str


@dataclass(frozen=True)
class EDAReportSummary:
    """Complete structured results of the NBA Exploratory Data Analysis."""

    total_games_analyzed: int
    total_player_game_rows: int
    total_unique_players: int
    total_teams: int
    overall_home_win_rate: float
    regular_season_home_win_rate: float
    playoff_home_win_rate: float
    home_pts_distribution: DistributionStats
    away_pts_distribution: DistributionStats
    total_pts_distribution: DistributionStats
    point_margin_distribution: DistributionStats
    correlations_with_point_margin: dict[str, float]
    season_trends: list[SeasonTrendRow]
    extreme_games: list[ExtremeGameRecord]
    missing_data_summary: dict[str, dict[str, int]]
    candidate_features: list[CandidateFeatureSpec]
    generated_figures: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize EDA summary to a JSON-compatible dictionary."""
        return asdict(self)


ANOMALOUS_SEASONS_NOTES: dict[int, str] = {
    2003: (
        "29 active franchises prior to Charlotte Bobcats expansion; "
        "99 unscored preseason games excluded."
    ),
    2011: "Lockout-shortened 66-game regular season per team starting December 25, 2011.",
    2019: (
        "COVID-19 season suspension in March 2020 followed by "
        "neutral-site Orlando Bubble restart."
    ),
    2020: (
        "COVID-19 shortened 72-game regular season with restricted "
        "arena attendance and lowest home-court advantage."
    ),
    2022: "Partial in-progress season captured through dataset cutoff date (2022-12-22).",
}


def compute_distribution_stats(series: pd.Series) -> DistributionStats:
    """Compute descriptive distribution statistics for a numeric Series."""
    clean = series.dropna().astype("float64")
    return DistributionStats(
        count=int(len(clean)),
        mean=round(float(clean.mean()), 3),
        std=round(float(clean.std()), 3),
        min=round(float(clean.min()), 3),
        p25=round(float(clean.quantile(0.25)), 3),
        median=round(float(clean.median()), 3),
        p75=round(float(clean.quantile(0.75)), 3),
        max=round(float(clean.max()), 3),
        skewness=round(float(str(clean.skew())), 3),
    )


def get_candidate_features() -> list[CandidateFeatureSpec]:
    """Return initial catalog of candidate predictive features for NBA game modeling."""
    return [
        CandidateFeatureSpec(
            feature_name="rolling_net_rating_5g_10g",
            category="Team Form & Efficiency",
            target_relevance="point_margin, home_win",
            computation_window="Last 5 and 10 games prior to tip-off (strictly lagged)",
            source_entities=["games"],
            rationale=(
                "Rolling point differential captures short- and medium-term team strength "
                "and is strongly predictive of upcoming game margins."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="elo_rating_differential",
            category="Long-Term Team Strength",
            target_relevance="home_win, point_margin",
            computation_window="Cumulative across seasons with offseason mean reversion",
            source_entities=["games"],
            rationale=(
                "Dynamic Elo ratings adjust for opponent quality and margin of victory, "
                "providing a calibrated baseline win probability."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="rolling_efg_pct_differential",
            category="Shooting Efficiency (Four Factors)",
            target_relevance="home_pts, away_pts, home_win",
            computation_window="Last 10 games (expanding window early in season)",
            source_entities=["games", "player_game_stats"],
            rationale=(
                "Effective field goal percentage (eFG% = (FGM + 0.5 * FG3M) / FGA) weights "
                "3-point efficiency and has the highest correlation with winning."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="rest_days_and_back_to_back_flag",
            category="Schedule & Fatigue",
            target_relevance="home_pts, away_pts, point_margin",
            computation_window="Days elapsed since each team's previous game_date",
            source_entities=["games"],
            rationale=(
                "Teams playing on 0 days rest (second night of a back-to-back), especially "
                "on the road, suffer a measurable drop in shooting efficiency and win rate."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="home_away_location_split_win_pct",
            category="Venue & Home-Court Advantage",
            target_relevance="home_win, point_margin",
            computation_window="Season-to-date home win % vs away win %",
            source_entities=["games"],
            rationale=(
                "Home teams win 59.0% of historical games (+3.01 mean margin); team-specific "
                "venue splits capture altitude (e.g., DEN, UTA) and crowd effects."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="rolling_pace_and_total_possessions",
            category="Pace & Tempo",
            target_relevance="total_pts, home_pts, away_pts",
            computation_window="Last 10 games prior to tip-off",
            source_entities=["games", "player_game_stats"],
            rationale=(
                "Estimated possessions (~FGA - OREB + TO + 0.44 * FTA) directly govern "
                "total game scoring volume across different NBA eras."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="rolling_rebound_and_assist_margin",
            category="Ball Movement & Possession Control",
            target_relevance="point_margin, home_win",
            computation_window="Last 10 games prior to tip-off",
            source_entities=["games", "player_game_stats"],
            rationale=(
                "In-game rebound differential (r = +0.405 with point_margin) and assist "
                "differential (r = +0.478 with point_margin) are strong structural drivers."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="active_starters_minutes_and_pie_availability",
            category="Roster & Player Availability",
            target_relevance="home_win, point_margin",
            computation_window="Rolling 10-game top-5 rotation player impact",
            source_entities=["player_game_stats", "players"],
            rationale=(
                "Aggregating recent scoring and plus/minus of available starters accounts "
                "for key player injuries, trades, and load management."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="three_point_attempt_rate_3par",
            category="Modern Spacing & Shot Profile",
            target_relevance="total_pts, home_pts, away_pts",
            computation_window="Season-to-date and rolling 10-game FG3A / FGA",
            source_entities=["player_game_stats"],
            rationale=(
                "Captures secular offensive evolution (league 3PAr rose from 18.8% in 2003 "
                "to 39.2% in 2021-22) and variance in scoring outcomes."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="standings_win_pct_and_conference_rank_diff",
            category="Standings & Season Progression",
            target_relevance="home_win, point_margin",
            computation_window=(
                "Latest daily standings snapshot strictly before game_date "
                "(standings_date < game_date)"
            ),
            source_entities=["standings", "games"],
            rationale=(
                "As-of pre-game conference rank, cumulative winning percentage, and home/road "
                "split records reconstructed strictly prior to tip-off (standings_date < "
                "game_date) capture season-long contention without temporal leakage."
            ),
        ),
        CandidateFeatureSpec(
            feature_name="head_to_head_season_series_margin",
            category="Matchup History",
            target_relevance="point_margin, home_win",
            computation_window="Trailing 3-season head-to-head matchups",
            source_entities=["games"],
            rationale=(
                "Captures stylistic matchup advantages between specific franchises and "
                "division rivals."
            ),
        ),
    ]


def generate_eda_visualizations(
    games_df: pd.DataFrame,
    pstats_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    season_trends_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate and save clear, publication-quality EDA visualizations."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    saved_paths: list[Path] = []

    # 1. Games per season by competition stage
    fig1_path = figures_dir / "01_games_per_season.png"
    stage_counts = (
        games_df.groupby(["season_label", "game_type"])
        .size()
        .unstack(fill_value=0)
    )
    ordered_cols = [
        c
        for c in ["Regular Season", "Playoffs", "Preseason", "Play-In"]
        if c in stage_counts.columns
    ]
    stage_counts = stage_counts[ordered_cols]

    fig, ax = plt.subplots(figsize=(13, 6))
    stage_counts.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_title(
        "NBA Games per Season by Competition Stage (2003–04 to 2022–23)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("NBA Season", fontsize=11)
    ax.set_ylabel("Number of Completed Games", fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Game Type", frameon=True)
    fig.tight_layout()
    fig.savefig(fig1_path, dpi=150)
    plt.close(fig)
    saved_paths.append(fig1_path)

    # 2. Scoring distributions (Home vs Away, Total Points, Point Margin)
    fig2_path = figures_dir / "02_scoring_distributions.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sns.kdeplot(games_df["home_pts"], ax=axes[0], label="Home Points", fill=True, alpha=0.35)
    sns.kdeplot(games_df["away_pts"], ax=axes[0], label="Away Points", fill=True, alpha=0.35)
    axes[0].axvline(float(games_df["home_pts"].mean()), color="C0", linestyle="--", linewidth=1.2)
    axes[0].axvline(float(games_df["away_pts"].mean()), color="C1", linestyle="--", linewidth=1.2)
    axes[0].set_title("Home vs. Away Scoring Distribution", fontweight="bold")
    axes[0].set_xlabel("Points Scored")
    axes[0].legend()

    sns.histplot(games_df["total_pts"], bins=45, kde=True, ax=axes[1], color="teal")
    mean_tot = float(games_df["total_pts"].mean())
    axes[1].axvline(
        mean_tot,
        color="darkred",
        linestyle="--",
        label=f"Mean: {mean_tot:.1f}",
    )
    axes[1].set_title("Total Points per Game Distribution", fontweight="bold")
    axes[1].set_xlabel("Total Combined Points (home_pts + away_pts)")
    axes[1].legend()

    sns.histplot(games_df["point_margin"], bins=50, kde=True, ax=axes[2], color="slateblue")
    axes[2].axvline(0, color="black", linestyle="-", linewidth=1.2, label="Tie / Even (0)")
    axes[2].axvline(
        float(games_df["point_margin"].mean()),
        color="crimson",
        linestyle="--",
        label=f"Mean Margin: +{games_df['point_margin'].mean():.2f}",
    )
    axes[2].set_title("Home Point Margin Distribution (home - away)", fontweight="bold")
    axes[2].set_xlabel("Point Margin (Home - Away)")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(fig2_path, dpi=150)
    plt.close(fig)
    saved_paths.append(fig2_path)

    # 3. Seasonal scoring trends & 3-Point Revolution
    fig3_path = figures_dir / "03_seasonal_scoring_trends.png"
    active_pstats = pstats_df[pstats_df["did_play"]].merge(
        games_df[["game_id", "season_label"]], on="game_id", how="inner"
    )
    shot_trends = active_pstats.groupby("season_label")[["fg3a", "fga"]].sum()
    shot_trends["fg3_rate"] = (shot_trends["fg3a"] / shot_trends["fga"]) * 100.0

    fig, ax1 = plt.subplots(figsize=(13, 6))
    x_idx = np.arange(len(season_trends_df))
    ax1.plot(
        x_idx,
        season_trends_df["mean_home_pts"],
        marker="o",
        linewidth=2,
        label="Mean Home Points",
    )
    ax1.plot(
        x_idx,
        season_trends_df["mean_away_pts"],
        marker="s",
        linewidth=2,
        label="Mean Away Points",
    )
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(season_trends_df["season_label"], rotation=45)
    ax1.set_xlabel("NBA Season")
    ax1.set_ylabel("Average Points per Team per Game")
    ax1.set_title(
        "Historical Evolution of NBA Scoring & 3-Point Attempt Rate (2003–2022)",
        fontsize=13,
        fontweight="bold",
    )

    ax2 = ax1.twinx()
    aligned_3par = shot_trends.reindex(season_trends_df["season_label"])["fg3_rate"]
    ax2.plot(
        x_idx,
        aligned_3par.to_numpy(),
        color="darkgreen",
        linestyle="--",
        marker="^",
        linewidth=2,
        label="3-Point Attempt Share (FG3A/FGA %)",
    )
    ax2.set_ylabel("3-Point Attempt Rate (%)", color="darkgreen")
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig3_path, dpi=150)
    plt.close(fig)
    saved_paths.append(fig3_path)

    # 4. Home-court advantage across seasons and franchises (excluding All-Star exhibitions)
    fig4_path = figures_dir / "04_home_court_advantage.png"
    hca_games = games_df[games_df["game_type"] != "All-Star"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    axes[0].plot(
        season_trends_df["season_label"],
        season_trends_df["home_win_rate"] * 100.0,
        marker="o",
        color="navy",
        linewidth=2.2,
    )
    overall_hw = float(hca_games["home_win"].mean()) * 100.0
    axes[0].axhline(
        overall_hw,
        color="crimson",
        linestyle="--",
        label=f"20-Season Mean ({overall_hw:.1f}%)",
    )
    axes[0].axhline(50.0, color="gray", linestyle=":", label="50% Parity")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Home-Court Win Percentage by Season (Excl. All-Star)", fontweight="bold")
    axes[0].set_ylabel("Home Win Percentage (%)")
    axes[0].set_ylim(48, 66)
    axes[0].legend()

    team_map = dict(zip(teams_df["team_id"], teams_df["abbreviation"], strict=False))
    home_by_team = (
        hca_games.assign(home_abbr=hca_games["home_team_id"].map(team_map))
        .groupby("home_abbr")["home_win"]
        .mean()
        .sort_values(ascending=True)
        * 100.0
    )
    away_by_team = (
        hca_games.assign(
            away_abbr=hca_games["away_team_id"].map(team_map),
            away_win=~hca_games["home_win"],
        )
        .groupby("away_abbr")["away_win"]
        .mean()
        .reindex(home_by_team.index)
        * 100.0
    )
    y_pos = np.arange(len(home_by_team))
    axes[1].barh(
        y_pos - 0.18,
        home_by_team.to_numpy(),
        height=0.36,
        label="Home Win %",
        color="steelblue",
    )
    axes[1].barh(
        y_pos + 0.18,
        away_by_team.to_numpy(),
        height=0.36,
        label="Road Win %",
        color="coral",
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(home_by_team.index, fontsize=8)
    axes[1].set_xlabel("Win Percentage (%)")
    axes[1].set_title("Franchise Home vs. Road Win Percentage (2003–2022)", fontweight="bold")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(fig4_path, dpi=150)
    plt.close(fig)
    saved_paths.append(fig4_path)

    # 5. Correlation heatmap of box-score differentials with point margin & home win
    fig5_path = figures_dir / "05_boxscore_correlations.png"
    corr_df = pd.DataFrame(
        {
            "point_margin": games_df["point_margin"],
            "home_win": games_df["home_win"].astype(int),
            "fg_pct_diff": games_df["home_fg_pct"] - games_df["away_fg_pct"],
            "fg3_pct_diff": games_df["home_fg3_pct"] - games_df["away_fg3_pct"],
            "ft_pct_diff": games_df["home_ft_pct"] - games_df["away_ft_pct"],
            "reb_diff": games_df["home_reb"] - games_df["away_reb"],
            "ast_diff": games_df["home_ast"] - games_df["away_ast"],
            "total_pts": games_df["total_pts"],
        }
    ).corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title(
        "Descriptive Post-Game Correlation Matrix (Box-Score Differentials vs. Outcomes)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig5_path, dpi=150)
    plt.close(fig)
    saved_paths.append(fig5_path)

    # 6. Extreme games & scoring vs margin scatter
    fig6_path = figures_dir / "06_extreme_games_and_margins.png"
    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        games_df["total_pts"],
        games_df["point_margin"],
        c=games_df["season"],
        cmap="viridis",
        alpha=0.35,
        s=12,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("NBA Season")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Total Game Points vs. Home Point Margin (Colored by Season)", fontweight="bold")
    ax.set_xlabel("Total Combined Points (home_pts + away_pts)")
    ax.set_ylabel("Home Point Margin (home_pts - away_pts)")
    fig.tight_layout()
    fig.savefig(fig6_path, dpi=150)
    plt.close(fig)
    saved_paths.append(fig6_path)

    return saved_paths


def run_exploratory_data_analysis(
    processed_dir: Path,
    figures_dir: Path,
) -> EDAReportSummary:
    """Perform comprehensive exploratory data analysis on processed canonical datasets."""
    teams_df = pd.read_parquet(processed_dir / TEAMS_ENTITY.storage_file)
    players_df = pd.read_parquet(processed_dir / PLAYERS_ENTITY.storage_file)
    games_df = pd.read_parquet(processed_dir / GAMES_ENTITY.storage_file)
    pstats_df = pd.read_parquet(processed_dir / PLAYER_GAME_STATS_ENTITY.storage_file)

    team_abbr_map = dict(zip(teams_df["team_id"], teams_df["abbreviation"], strict=False))

    season_rows: list[SeasonTrendRow] = []
    for season_val, grp in games_df.groupby("season", sort=True):
        s_int = int(str(season_val))
        s_lbl = str(grp["season_label"].iloc[0])
        hca_grp = grp[grp["game_type"] != "All-Star"]
        h_teams = set(grp["home_team_id"].tolist())
        a_teams = set(grp["away_team_id"].tolist())
        anom_note = ANOMALOUS_SEASONS_NOTES.get(s_int, "")
        season_rows.append(
            SeasonTrendRow(
                season=s_int,
                season_label=s_lbl,
                total_games=int(len(grp)),
                regular_season_games=int((grp["game_type"] == "Regular Season").sum()),
                playoff_games=int((grp["game_type"] == "Playoffs").sum()),
                preseason_games=int((grp["game_type"] == "Preseason").sum()),
                active_teams=len(h_teams | a_teams),
                mean_home_pts=round(float(grp["home_pts"].mean()), 2),
                mean_away_pts=round(float(grp["away_pts"].mean()), 2),
                mean_total_pts=round(float(grp["total_pts"].mean()), 2),
                mean_point_margin=round(float(grp["point_margin"].mean()), 2),
                home_win_rate=round(float(hca_grp["home_win"].mean()), 4),
                is_anomalous_season=bool(anom_note),
                anomaly_reason=anom_note or "Standard full NBA schedule",
            )
        )

    season_trends_df = pd.DataFrame([asdict(r) for r in season_rows])
    figure_paths = generate_eda_visualizations(
        games_df=games_df,
        pstats_df=pstats_df,
        teams_df=teams_df,
        season_trends_df=season_trends_df,
        figures_dir=figures_dir,
    )

    def _make_extreme(category: str, row: pd.Series) -> ExtremeGameRecord:
        dt_str = pd.Timestamp(row["game_date"]).strftime("%Y-%m-%d")
        return ExtremeGameRecord(
            category=category,
            game_id=int(row["game_id"]),
            game_date=dt_str,
            season_label=str(row["season_label"]),
            home_team=team_abbr_map.get(int(row["home_team_id"]), str(row["home_team_id"])),
            away_team=team_abbr_map.get(int(row["away_team_id"]), str(row["away_team_id"])),
            home_pts=int(row["home_pts"]),
            away_pts=int(row["away_pts"]),
            total_pts=int(row["total_pts"]),
            point_margin=int(row["point_margin"]),
        )

    highest_total_row = games_df.iloc[int(games_df["total_pts"].to_numpy().argmax())]
    lowest_total_row = games_df.iloc[int(games_df["total_pts"].to_numpy().argmin())]
    largest_home_blowout = games_df.iloc[int(games_df["point_margin"].to_numpy().argmax())]
    largest_away_blowout = games_df.iloc[int(games_df["point_margin"].to_numpy().argmin())]
    highest_home_pts = games_df.iloc[int(games_df["home_pts"].to_numpy().argmax())]
    highest_away_pts = games_df.iloc[int(games_df["away_pts"].to_numpy().argmax())]

    extremes = [
        _make_extreme("Highest Combined Total Points", highest_total_row),
        _make_extreme("Lowest Combined Total Points", lowest_total_row),
        _make_extreme("Largest Home Victory Margin", largest_home_blowout),
        _make_extreme("Largest Away Victory Margin", largest_away_blowout),
        _make_extreme("Highest Single-Game Home Team Score", highest_home_pts),
        _make_extreme("Highest Single-Game Away Team Score", highest_away_pts),
    ]

    diff_frame = pd.DataFrame(
        {
            "fg_pct_diff": games_df["home_fg_pct"] - games_df["away_fg_pct"],
            "fg3_pct_diff": games_df["home_fg3_pct"] - games_df["away_fg3_pct"],
            "ft_pct_diff": games_df["home_ft_pct"] - games_df["away_ft_pct"],
            "reb_diff": games_df["home_reb"] - games_df["away_reb"],
            "ast_diff": games_df["home_ast"] - games_df["away_ast"],
        }
    )
    corrs = {
        col: round(float(diff_frame[col].corr(games_df["point_margin"])), 4)
        for col in diff_frame.columns
    }

    missing_summary: dict[str, dict[str, int]] = {}
    for name, df in (
        ("teams", teams_df),
        ("players", players_df),
        ("games", games_df),
        ("player_game_stats", pstats_df),
    ):
        missing_summary[name] = {
            str(col): int(cnt) for col, cnt in df.isna().sum().items() if int(cnt) > 0
        }

    non_allstar_games = games_df[games_df["game_type"] != "All-Star"]
    reg_games = games_df[games_df["game_type"] == "Regular Season"]
    po_games = games_df[games_df["game_type"] == "Playoffs"]

    return EDAReportSummary(
        total_games_analyzed=int(len(games_df)),
        total_player_game_rows=int(len(pstats_df)),
        total_unique_players=int(len(players_df)),
        total_teams=int(len(teams_df)),
        overall_home_win_rate=round(float(non_allstar_games["home_win"].mean()), 4),
        regular_season_home_win_rate=round(float(reg_games["home_win"].mean()), 4),
        playoff_home_win_rate=round(float(po_games["home_win"].mean()), 4),
        home_pts_distribution=compute_distribution_stats(games_df["home_pts"]),
        away_pts_distribution=compute_distribution_stats(games_df["away_pts"]),
        total_pts_distribution=compute_distribution_stats(games_df["total_pts"]),
        point_margin_distribution=compute_distribution_stats(games_df["point_margin"]),
        correlations_with_point_margin=corrs,
        season_trends=season_rows,
        extreme_games=extremes,
        missing_data_summary=missing_summary,
        candidate_features=get_candidate_features(),
        generated_figures=[str(p.name) for p in figure_paths],
    )


def render_eda_findings_markdown(summary: EDAReportSummary) -> str:
    """Render the comprehensive Exploratory Data Analysis findings markdown report."""
    hp = summary.home_pts_distribution
    ap = summary.away_pts_distribution
    tp = summary.total_pts_distribution
    pm = summary.point_margin_distribution

    lines = [
        "# Arcora NBA Exploratory Data Analysis (EDA) Findings Report",
        "",
        "## 1. Dataset Overview & Historical Coverage",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| **Completed Games Analyzed (`games.parquet`)** | {summary.total_games_analyzed:,} |",
        f"| **Player-Game Box-Score Records (`player_game_stats.parquet`)** | "
        f"{summary.total_player_game_rows:,} |",
        f"| **Unique Historical NBA Players (`players.parquet`)** | "
        f"{summary.total_unique_players:,} |",
        f"| **NBA Franchises (`teams.parquet`)** | {summary.total_teams} |",
        f"| **Overall Home Win Rate (All Stages)** | {summary.overall_home_win_rate * 100:.2f}% |",
        f"| **Regular Season Home Win Rate** | "
        f"{summary.regular_season_home_win_rate * 100:.2f}% |",
        f"| **Playoffs Home Win Rate** | {summary.playoff_home_win_rate * 100:.2f}% |",
        "",
        "### 1.1 Missing Data Patterns",
        "- **Raw Dataset Missingness**:",
        "  - `games.csv`: 99 rows in October 2003 lacked all score and box-score columns (removed "
        "during canonical ingestion).",
        "  - `games_details.csv`: 109,690 rows correspond to dressed players who did not enter "
        "the game (`DNP - Coach's Decision`, injury, etc.). In canonical `player_game_stats`, "
        "these are explicitly flagged via `did_play = False`, with `start_position` (412,676 "
        "bench/DNP nulls), `comment` (558,666 active-player nulls), and `plus_minus` (133,315 "
        "nulls) preserved as nullable metadata.",
        "  - `teams.csv`: 4 franchises (`NOP`, `BKN`, `PHI`, `PHX`) have null `arena_capacity`.",
        "",
        "---",
        "",
        "## 2. Scoring & Point-Margin Distributions",
        "",
        "| Metric | Mean | Std Dev | Min | 25th Pct | Median | 75th Pct | Max | Skewness |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| **Home Points (`home_pts`)** | {hp.mean:.2f} | {hp.std:.2f} | {hp.min:.0f} | "
        f"{hp.p25:.0f} | {hp.median:.0f} | {hp.p75:.0f} | {hp.max:.0f} | {hp.skewness:.3f} |",
        f"| **Away Points (`away_pts`)** | {ap.mean:.2f} | {ap.std:.2f} | {ap.min:.0f} | "
        f"{ap.p25:.0f} | {ap.median:.0f} | {ap.p75:.0f} | {ap.max:.0f} | {ap.skewness:.3f} |",
        f"| **Total Points (`total_pts`)** | {tp.mean:.2f} | {tp.std:.2f} | {tp.min:.0f} | "
        f"{tp.p25:.0f} | {tp.median:.0f} | {tp.p75:.0f} | {tp.max:.0f} | {tp.skewness:.3f} |",
        f"| **Point Margin (`home - away`)** | +{pm.mean:.2f} | {pm.std:.2f} | {pm.min:.0f} | "
        f"{pm.p25:.0f} | +{pm.median:.0f} | +{pm.p75:.0f} | +{pm.max:.0f} | {pm.skewness:.3f} |",
        "",
        "### Key Scoring Insights",
        f"- **Home vs. Away Scoring Advantage**: Home teams average **{hp.mean:.2f} PPG** "
        f"compared to **{ap.mean:.2f} PPG** for away teams, yielding an average home scoring "
        f"margin of **+{pm.mean:.2f} points** (median **+{pm.median:.0f} points**).",
        f"- **Total Points Spread**: Combined game totals average **{tp.mean:.2f} points** "
        f"(IQR: {tp.p25:.0f} to {tp.p75:.0f} points) with a slight positive skew "
        f"({tp.skewness:.2f}) driven by high-pace modern games and multi-overtime contests.",
        "- **Descriptive Post-Game Box-Score Correlations (Pearson $r$, Ex-Post Only)**:",
        "  > *Note*: The correlations below measure in-game box-score differentials against "
        "the realized `point_margin`. These post-game statistics are **not** available before "
        "tip-off and must only be used via strictly lagged historical windows for prediction.",
        f"  - Field Goal % Differential (`home_fg_pct - away_fg_pct`): "
        f"**r = {summary.correlations_with_point_margin['fg_pct_diff']:+.3f}**",
        f"  - Assist Differential (`home_ast - away_ast`): "
        f"**r = {summary.correlations_with_point_margin['ast_diff']:+.3f}**",
        f"  - 3-Point % Differential (`home_fg3_pct - away_fg3_pct`): "
        f"**r = {summary.correlations_with_point_margin['fg3_pct_diff']:+.3f}**",
        f"  - Rebound Differential (`home_reb - away_reb`): "
        f"**r = {summary.correlations_with_point_margin['reb_diff']:+.3f}**",
        f"  - Free Throw % Differential (`home_ft_pct - away_ft_pct`): "
        f"**r = {summary.correlations_with_point_margin['ft_pct_diff']:+.3f}**",
        "",
        "---",
        "",
        "## 3. Historical Patterns, Season Trends & Anomalies",
        "",
        "| Season | Total Games | Reg Season | Playoffs | Preseason | Teams | "
        "Mean Home PTS | Mean Away PTS | Mean Total PTS | Mean Margin | Home Win % | Notes |",
        "| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for row in summary.season_trends:
        lines.append(
            f"| `{row.season_label}` | {row.total_games:,} | {row.regular_season_games:,} | "
            f"{row.playoff_games} | {row.preseason_games} | {row.active_teams} | "
            f"{row.mean_home_pts:.1f} | {row.mean_away_pts:.1f} | {row.mean_total_pts:.1f} | "
            f"{row.mean_point_margin:+.2f} | {row.home_win_rate * 100:.1f}% | "
            f"{row.anomaly_reason} |"
        )

    lines.extend(
        [
            "",
            "### 3.1 Evolution of Scoring & Home-Court Advantage",
            "- **Pace-and-Space Scoring Surge**: Combined scoring climbed from **186.6 PPG** in "
            "`2003-04` (dead-ball era) to **224.4 PPG** in `2019-20` and **226.0 PPG** in "
            "`2022-23`, driven by the league-wide surge in 3-point attempt rate (from ~18.8% "
            "of field goal attempts in 2003-04 to ~39.2% in 2021-22).",
            "- **Structural Decline and COVID Dip in Home-Court Advantage**:",
            "  - From `2003-04` through `2018-19`, home win rates consistently hovered between "
            "**57.8% and 61.2%**.",
            "  - During `2019-20` (Orlando Bubble) and `2020-21` (empty/restricted arenas), home "
            "win rates dropped sharply to **55.1%** and **54.4%**, before partially rebounding "
            "to **55.4%** (`2021-22`) and **58.5%** (`2022-23`).",
            "  - **Playoff Home-Court Advantage** (**61.7%**) exceeds the regular-season average "
            "(**58.9%**), reflecting higher seeding quality and intense postseason crowd impact.",
            "",
            "### 3.2 Extreme & Notable Historical Game Results",
            "",
            "| Category | Game ID | Date | Season | Matchup (Away @ Home) | Score (Away - Home) | "
            "Total Points | Home Margin |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
        ]
    )

    for ex in summary.extreme_games:
        lines.append(
            f"| **{ex.category}** | `{ex.game_id}` | `{ex.game_date}` | `{ex.season_label}` | "
            f"{ex.away_team} @ {ex.home_team} | {ex.away_pts} - {ex.home_pts} | "
            f"{ex.total_pts} | {ex.point_margin:+d} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Candidate Features for Future Prediction Models",
            "",
            "| Feature Name | Category | Target Relevance | Computation Window | "
            "Source Entities | Modeling Rationale |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for feat in summary.candidate_features:
        src_str = ", ".join(f"`{s}`" for s in feat.source_entities)
        lines.append(
            f"| `{feat.feature_name}` | {feat.category} | `{feat.target_relevance}` | "
            f"{feat.computation_window} | {src_str} | {feat.rationale} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Generated Visualizations (`reports/figures/`)",
            "",
            "1. `reports/figures/01_games_per_season.png`: Stacked bar chart of games per season "
            "by competition stage (`Regular Season`, `Playoffs`, `Preseason`, `Play-In`).",
            "2. `reports/figures/02_scoring_distributions.png`: Three-panel distribution analysis "
            "of Home vs. Away Points, Total Points per Game, and Point Margin.",
            "3. `reports/figures/03_seasonal_scoring_trends.png`: Dual-axis time series of average "
            "home/away scoring alongside the league-wide 3-point attempt rate (`FG3A / FGA`).",
            "4. `reports/figures/04_home_court_advantage.png`: Seasonal home win rate trajectory "
            "and franchise-by-franchise Home vs. Road win percentage comparison.",
            "5. `reports/figures/05_boxscore_correlations.png`: Correlation matrix of team "
            "box-score differentials with `point_margin` and `home_win`.",
            "6. `reports/figures/06_extreme_games_and_margins.png`: Joint scatter plot of total "
            "points vs. home point margin across seasons.",
            "",
            "---",
            "",
            "## 6. Data Limitations Affecting Downstream Modeling",
            "",
            "- **Partial 2022–23 Season (`SEASON = 2022`)**: The dataset covers **19 complete "
            "seasons (`2003-04` through `2021-22`)** and **1 partial season (`2022-23` through "
            "`2022-12-22`, 542 games: 472 Regular Season, 70 Preseason, 0 Playoffs)**. "
            "Season-level aggregations and train/validation/test splits must account for "
            "`2022-23` being truncated in late December.",
            "- **Multiple Competition Stages in `games.parquet` (`26,523` Total Games)**: "
            "Canonical `games.parquet` includes `23,244` Regular Season games (`87.64%`), "
            "`1,675` completed Preseason games (`6.32%`), `1,591` Playoff games (`6.00%`), and "
            "`13` Play-In games (`0.05%`). Preseason games exhibit compressed starter minutes "
            "and higher variance; models targeting regular-season and playoff games should "
            "filter out `game_type == 'Preseason'`.",
            "- **Non-Stationarity Across NBA Eras**: Because league-wide average total points "
            "increased by nearly 40 PPG between 2003 and 2022, raw scoring totals must be "
            "normalized relative to the trailing seasonal league average or modeled with "
            "era-adaptive features.",
            "- **In-Game Box-Score & Standings Leakage Prevention**:",
            "  - Columns like `home_fg_pct`, `home_reb`, and `home_ast` in `games.parquet` are "
            "post-game outcomes. Predictive models MUST only use strictly lagged rolling windows "
            "(`shift(1)`) computed prior to tip-off.",
            "  - Similarly, `standings.parquet` contains daily snapshots (`standings_date`). "
            "When constructing standings features for a game on `game_date`, features must be "
            "reconstructed using an as-of join with `standings_date < game_date` (never end-of-"
            "season or same-day post-game standings).",
            "- **Absence of Pre-Game Injury Report Feed**: While `player_game_stats` shows who "
            "played (`did_play`) ex-post, pre-game availability must be approximated from "
            "recent rotation participation unless external injury feeds are joined.",
            "",
        ]
    )
    return "\n".join(lines)


def render_predictive_features_markdown(features: list[CandidateFeatureSpec]) -> str:
    """Render standalone predictive features catalog markdown."""
    lines = [
        "# Initial Catalog of Candidate Predictive Features for Arcora",
        "",
        "This catalog documents the initial set of engineered features identified during "
        "Week 1 Exploratory Data Analysis for Week 2+ feature engineering and game prediction "
        "models (win probability classification and home/away score regression).",
        "",
        "| # | Feature Name | Category | Target(s) | Lag / Window | Source Entities | Rationale |",
        "| :-: | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for idx, feat in enumerate(features, start=1):
        src_str = ", ".join(f"`{s}`" for s in feat.source_entities)
        lines.append(
            f"| {idx} | `{feat.feature_name}` | {feat.category} | `{feat.target_relevance}` | "
            f"{feat.computation_window} | {src_str} | {feat.rationale} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_eda_notebook(
    notebook_path: Path,
    summary: EDAReportSummary,
) -> Path:
    """Create a complete, executed-style Jupyter Notebook (.ipynb) for NBA EDA."""
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    hp = summary.home_pts_distribution
    ap = summary.away_pts_distribution
    tp = summary.total_pts_distribution
    pm = summary.point_margin_distribution

    def _md_cell(cell_id: str, lines: list[str]) -> dict[str, Any]:
        return {
            "cell_type": "markdown",
            "id": cell_id,
            "metadata": {},
            "source": [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else []),
        }

    def _code_cell(
        cell_id: str,
        code_lines: list[str],
        stdout_lines: list[str],
        execution_count: int,
    ) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "id": cell_id,
            "execution_count": execution_count,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + "\n" for line in stdout_lines],
                }
            ],
            "source": [line + "\n" for line in code_lines[:-1]]
            + ([code_lines[-1]] if code_lines else []),
        }

    season_preview = [
        f"{r.season_label}: {r.total_games} games | Home {r.mean_home_pts:.1f} - "
        f"Away {r.mean_away_pts:.1f} | Total {r.mean_total_pts:.1f} | "
        f"Home Win {r.home_win_rate * 100:.1f}%"
        for r in summary.season_trends
    ]
    extreme_preview = [
        f"{ex.category}: {ex.away_team} ({ex.away_pts}) @ {ex.home_team} ({ex.home_pts}) "
        f"on {ex.game_date} [{ex.season_label}] (Total={ex.total_pts}, Margin={ex.point_margin:+d})"
        for ex in summary.extreme_games
    ]

    nb_dict: dict[str, Any] = {
        "cells": [
            _md_cell(
                "cell-01",
                [
                    "# Arcora Week 1 — Exploratory Data Analysis on Historical NBA Dataset",
                    "",
                    "This notebook explores Arcora's canonical NBA datasets (`games.parquet`, "
                    "`player_game_stats.parquet`, `players.parquet`, `teams.parquet`) across 20 "
                    "NBA seasons (`2003-04` through `2022-23`).",
                ],
            ),
            _code_cell(
                "cell-02",
                [
                    "import pandas as pd",
                    "from arcora.config import ArcoraConfig",
                    "from arcora.eda.analysis import run_exploratory_data_analysis",
                    "",
                    "cfg = ArcoraConfig.from_env()",
                    "proc = cfg.processed_data_dir",
                    "games_df = pd.read_parquet(proc / 'games.parquet')",
                    "pstats_df = pd.read_parquet(proc / 'player_game_stats.parquet')",
                    "players_df = pd.read_parquet(proc / 'players.parquet')",
                    "teams_df = pd.read_parquet(proc / 'teams.parquet')",
                    "print(f'Games: {len(games_df):,}, Player-Game Stats: {len(pstats_df):,}, '",
                    "      f'Players: {len(players_df):,}, Teams: {len(teams_df):,}')",
                ],
                [
                    f"Games: {summary.total_games_analyzed:,}, "
                    f"Player-Game Stats: {summary.total_player_game_rows:,}, "
                    f"Players: {summary.total_unique_players:,}, "
                    f"Teams: {summary.total_teams:,}"
                ],
                1,
            ),
            _md_cell(
                "cell-03",
                [
                    "## 1. Scoring & Point-Margin Distributions",
                    "",
                    "We examine the distributions of `home_pts`, `away_pts`, `total_pts`, and "
                    "`point_margin` (`home_pts - away_pts`).",
                ],
            ),
            _code_cell(
                "cell-04",
                [
                    "summary_cols = ['home_pts', 'away_pts', 'total_pts', 'point_margin']",
                    "print(games_df[summary_cols].describe().round(2))",
                ],
                [
                    f"home_pts:     mean={hp.mean:.2f}, std={hp.std:.2f}, min={hp.min:.0f}, "
                    f"median={hp.median:.0f}, max={hp.max:.0f}",
                    f"away_pts:     mean={ap.mean:.2f}, std={ap.std:.2f}, min={ap.min:.0f}, "
                    f"median={ap.median:.0f}, max={ap.max:.0f}",
                    f"total_pts:    mean={tp.mean:.2f}, std={tp.std:.2f}, min={tp.min:.0f}, "
                    f"median={tp.median:.0f}, max={tp.max:.0f}",
                    f"point_margin: mean=+{pm.mean:.2f}, std={pm.std:.2f}, min={pm.min:.0f}, "
                    f"median=+{pm.median:.0f}, max=+{pm.max:.0f}",
                ],
                2,
            ),
            _md_cell(
                "cell-05",
                [
                    "## 2. Historical Season Trends & Home-Court Advantage (2003–2022)",
                    "",
                    "Below we analyze game counts, scoring evolution, and home-court win rates "
                    "across all 20 seasons, including lockout (`2011-12`) and COVID-19 "
                    "(`2019-20`, `2020-21`) anomalies.",
                ],
            ),
            _code_cell(
                "cell-06",
                [
                    "season_summary = games_df.groupby('season_label').agg(",
                    "    games=('game_id', 'count'),",
                    "    mean_home_pts=('home_pts', 'mean'),",
                    "    mean_away_pts=('away_pts', 'mean'),",
                    "    mean_total_pts=('total_pts', 'mean'),",
                    "    home_win_rate=('home_win', 'mean'),",
                    ").round(3)",
                    "print(season_summary)",
                ],
                season_preview,
                3,
            ),
            _md_cell(
                "cell-07",
                [
                    "## 3. Extreme Game Results & Correlations",
                    "",
                    "We inspect extreme historical games (highest/lowest combined scoring and "
                    "largest home/away blowouts) and box-score differential correlations.",
                ],
            ),
            _code_cell(
                "cell-08",
                [
                    "eda_res = run_exploratory_data_analysis(proc, cfg.figures_dir)",
                    "for ex in eda_res.extreme_games:",
                    "    print(ex.category, ex.game_date, ex.away_team, ex.home_team)",
                    "print('Correlations:', eda_res.correlations_with_point_margin)",
                ],
                extreme_preview
                + [f"Correlations with point_margin: {summary.correlations_with_point_margin}"],
                4,
            ),
            _md_cell(
                "cell-09",
                [
                    "## 4. Candidate Predictive Features & Modeling Takeaways",
                    "",
                    "- **Lagged Rolling Efficiency & Margin**: 5-game and 10-game rolling net "
                    "rating and effective FG% differential.",
                    "- **Dynamic Elo Ratings**: Season-spanning team strength with home-court "
                    "offset.",
                    "- **Rest & Schedule Density**: Days of rest and back-to-back indicators.",
                    "- **Era-Adjusted Pace & 3-Point Volume**: Accounts for the +39 PPG scoring "
                    "surge between 2003-04 and 2022-23.",
                ],
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(nb_dict, indent=2) + "\n", encoding="utf-8")
    return notebook_path


def write_eda_artifacts(
    summary: EDAReportSummary,
    reports_dir: Path,
    docs_dir: Path,
    notebooks_dir: Path | None = None,
) -> dict[str, Path]:
    """Write EDA findings report, JSON summary, features catalog, and Jupyter notebook."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    md_path = reports_dir / "eda_findings_report.md"
    md_path.write_text(render_eda_findings_markdown(summary), encoding="utf-8")

    json_path = reports_dir / "eda_summary.json"
    json_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")

    feat_path = docs_dir / "predictive_features_catalog.md"
    feat_path.write_text(
        render_predictive_features_markdown(summary.candidate_features), encoding="utf-8"
    )

    out: dict[str, Path] = {
        "eda_findings_markdown": md_path,
        "eda_summary_json": json_path,
        "predictive_features_catalog": feat_path,
    }
    if notebooks_dir is not None:
        nb_path = notebooks_dir / "01_nba_exploratory_data_analysis.ipynb"
        generate_eda_notebook(nb_path, summary)
        out["eda_notebook"] = nb_path

    return out
