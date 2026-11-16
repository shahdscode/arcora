"""CLI script to execute automated NBA data quality checks (Issue #6)."""

from __future__ import annotations

import sys

from arcora.config import ArcoraConfig
from arcora.validation.checks import run_validation_suite, write_validation_artifacts


def main() -> int:
    """Run automated data quality checks on processed Parquet datasets."""
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()

    summary = run_validation_suite(cfg.processed_data_dir)
    artifacts = write_validation_artifacts(summary, cfg.reports_dir)

    status = "PASSED" if summary.overall_passed else "FAILED"
    print(f"NBA Data Quality Validation Suite: {status}")
    print(f"  Checks passed: {summary.passed_checks} / {summary.total_checks}")
    for key, path in artifacts.items():
        print(f"  Wrote {key}: {path}")

    return 0 if summary.overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
