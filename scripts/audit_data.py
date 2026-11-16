"""CLI script to execute the raw NBA dataset schema and quality audit (Issue #3)."""

from __future__ import annotations

import sys

from arcora.config import ArcoraConfig
from arcora.data.audit import run_dataset_audit, write_audit_artifacts


def main() -> int:
    """Execute the raw dataset audit and write markdown and JSON reports."""
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()

    report = run_dataset_audit(cfg.raw_data_dir)
    artifacts = write_audit_artifacts(
        report=report,
        docs_dir=cfg.docs_dir,
        reports_dir=cfg.reports_dir,
    )

    print("NBA Dataset Schema & Quality Audit Complete:")
    print(f"  Tables inspected:     {report.summary_metrics['total_tables']}")
    print(f"  Relationships tested: {len(report.relationships)}")
    print(f"  Anomalies identified: {len(report.anomalies)}")
    for key, path in artifacts.items():
        print(f"  Wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
