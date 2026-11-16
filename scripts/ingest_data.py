"""CLI script to execute the reproducible NBA data ingestion pipeline (Issue #5)."""

from __future__ import annotations

import sys

from arcora.config import ArcoraConfig
from arcora.data.ingestion import run_ingestion_pipeline


def main() -> int:
    """Execute the raw-to-Parquet ingestion pipeline."""
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()

    result = run_ingestion_pipeline(
        raw_dir=cfg.raw_data_dir,
        processed_dir=cfg.processed_data_dir,
        logs_dir=cfg.logs_dir,
        reports_dir=cfg.reports_dir,
    )

    print("NBA Canonical Data Ingestion Pipeline Completed:")
    for name, st in result.entities.items():
        handled = st.malformed_or_missing_handled
        print(
            f"  {name:18s}: {st.raw_rows_read:8,d} raw -> {st.rows_written:8,d} clean rows "
            f"(dups removed: {st.duplicates_removed:,}, handled: {handled:,}) "
            f"-> {st.output_file}"
        )
    print(f"  Ingestion log:      {result.log_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
