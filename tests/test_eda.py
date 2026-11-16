"""Tests for NBA Exploratory Data Analysis and visualization generation (Issue #7)."""

from __future__ import annotations

from pathlib import Path

from arcora.config import ArcoraConfig
from arcora.eda.analysis import run_exploratory_data_analysis, write_eda_artifacts


def test_run_exploratory_data_analysis_and_visualizations(tmp_path: Path) -> None:
    """Verify EDA computes distributions, trends, extreme games, and saves all 6 figures."""
    cfg = ArcoraConfig.from_env()
    fig_dir = tmp_path / "figures"
    rep_dir = tmp_path / "reports"
    docs_dir = tmp_path / "docs"

    summary = run_exploratory_data_analysis(
        processed_dir=cfg.processed_data_dir,
        figures_dir=fig_dir,
    )

    assert summary.total_games_analyzed == 26523
    assert summary.total_teams == 30
    assert summary.total_unique_players == 2687
    assert 0.55 < summary.overall_home_win_rate < 0.65
    assert summary.home_pts_distribution.mean > summary.away_pts_distribution.mean
    assert summary.point_margin_distribution.mean > 0
    assert len(summary.season_trends) == 20
    assert len(summary.extreme_games) == 6
    assert len(summary.candidate_features) >= 10
    assert len(summary.generated_figures) == 6

    for fname in summary.generated_figures:
        fpath = fig_dir / fname
        assert fpath.is_file()
        assert fpath.stat().st_size > 0

    artifacts = write_eda_artifacts(
        summary,
        reports_dir=rep_dir,
        docs_dir=docs_dir,
        notebooks_dir=tmp_path / "notebooks",
    )
    assert "eda_notebook" in artifacts
    for path in artifacts.values():
        assert path.is_file()
        assert path.stat().st_size > 0
