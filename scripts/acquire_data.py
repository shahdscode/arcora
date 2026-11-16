"""CLI script to acquire, verify, and document the raw NBA dataset (Issue #2)."""

from __future__ import annotations

import sys

from arcora.config import ArcoraConfig
from arcora.data.acquisition import (
    download_raw_dataset,
    inspect_raw_dataset,
    verify_raw_checksums,
    write_acquisition_artifacts,
)


def main() -> int:
    """Acquire (if missing), verify, and document the historical NBA dataset."""
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()

    download_raw_dataset(
        raw_dir=cfg.raw_data_dir,
        download_url=cfg.kaggle_download_api_url,
        force=False,
    )

    checksums_ok, mismatches = verify_raw_checksums(cfg.raw_data_dir)
    if not checksums_ok:
        print(f"ERROR: Raw file checksum mismatch detected: {mismatches}", file=sys.stderr)
        return 1

    report = inspect_raw_dataset(cfg.raw_data_dir)
    artifacts = write_acquisition_artifacts(
        report=report,
        raw_dir=cfg.raw_data_dir,
        docs_dir=cfg.docs_dir,
    )

    min_s = min(report.seasons_available)
    max_s = max(report.seasons_available)
    print("Historical NBA Dataset Verified & Documented:")
    print(f"  Total files:   {len(report.files)}")
    print(f"  Total bytes:   {report.total_size_bytes:,}")
    print(f"  Total rows:    {report.total_rows_across_files:,}")
    print(f"  Date range:    {report.overall_min_game_date} -> {report.overall_max_game_date}")
    print(f"  Seasons:       {min_s} - {max_s} ({report.season_count} seasons)")
    for key, path in artifacts.items():
        print(f"  Wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
