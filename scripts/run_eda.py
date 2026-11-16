"""CLI script to execute Exploratory Data Analysis and generate notebooks/figures (Issue #7)."""

from __future__ import annotations

import sys

from arcora.config import ArcoraConfig
from arcora.eda.analysis import run_exploratory_data_analysis, write_eda_artifacts


def main() -> int:
    """Run Exploratory Data Analysis, generate visualizations, and write reports."""
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()

    summary = run_exploratory_data_analysis(
        processed_dir=cfg.processed_data_dir,
        figures_dir=cfg.figures_dir,
    )
    artifacts = write_eda_artifacts(
        summary=summary,
        reports_dir=cfg.reports_dir,
        docs_dir=cfg.docs_dir,
        notebooks_dir=cfg.project_root / "notebooks",
    )

    print("NBA Exploratory Data Analysis Complete:")
    print(f"  Games analyzed:       {summary.total_games_analyzed:,}")
    print(f"  Overall Home Win %:   {summary.overall_home_win_rate * 100:.2f}%")
    print(f"  Mean Home Points:     {summary.home_pts_distribution.mean:.2f}")
    print(f"  Mean Away Points:     {summary.away_pts_distribution.mean:.2f}")
    print(f"  Figures generated:    {len(summary.generated_figures)}")
    for key, path in artifacts.items():
        print(f"  Wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
