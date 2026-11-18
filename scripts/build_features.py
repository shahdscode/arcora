"""CLI script to execute the Week 2 NBA ML feature engineering pipeline (Issue #15)."""

from __future__ import annotations

import sys

from arcora.config import ArcoraConfig
from arcora.features.pipeline import run_feature_pipeline


def main() -> int:
    """Execute the end-to-end feature engineering pipeline and generate reports."""
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()

    _, summary = run_feature_pipeline(
        processed_dir=cfg.processed_data_dir,
        features_dir=cfg.features_data_dir,
        docs_dir=cfg.docs_dir,
        reports_dir=cfg.reports_dir,
        logs_dir=cfg.logs_dir,
    )

    status = "PASSED" if summary.leakage_checks_passed else "FAILED"
    print(f"Arcora Week 2 ML Feature Pipeline Completed ({status}):")
    print(f"  Output dataset:       {summary.output_parquet_path}")
    print(f"  Eligible games:       {summary.total_games:,}")
    print(f"  Metadata columns:     {summary.total_metadata_columns}")
    print(f"  Input features (X):   {summary.total_feature_columns}")
    print(f"  Target columns (y):   {summary.total_target_columns}")
    print(f"  Duplicate game_ids:   {summary.duplicate_game_ids}")
    passed_cnt = sum(1 for r in summary.leakage_results if r.passed)
    total_cnt = len(summary.leakage_results)
    print(f"  Unexpected nulls:     {summary.unexpected_missing_feature_count}")
    print(f"  Leakage checks:       {passed_cnt}/{total_cnt} passed")
    return 0 if summary.leakage_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
