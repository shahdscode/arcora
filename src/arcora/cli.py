"""Command-line interface for Arcora."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from arcora import __version__
from arcora.config import ArcoraConfig
from arcora.data.acquisition import (
    download_raw_dataset,
    inspect_raw_dataset,
    verify_raw_checksums,
    write_acquisition_artifacts,
)
from arcora.data.audit import run_dataset_audit, write_audit_artifacts
from arcora.data.ingestion import run_ingestion_pipeline
from arcora.eda.analysis import run_exploratory_data_analysis, write_eda_artifacts
from arcora.models.schema import write_schema_documentation
from arcora.validation.checks import run_validation_suite, write_validation_artifacts


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Arcora CLI."""
    parser = argparse.ArgumentParser(
        prog="arcora",
        description="Arcora NBA Analytics and Game Prediction Platform CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"arcora {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("info", help="Display current environment and path configuration.")
    subparsers.add_parser("acquire", help="Acquire, verify, and document the raw NBA dataset.")
    subparsers.add_parser(
        "audit", help="Audit raw NBA dataset schemas, relationships, and quality."
    )
    subparsers.add_parser("schema", help="Generate canonical data model documentation and DDL.")
    subparsers.add_parser("ingest", help="Run canonical raw-to-Parquet ingestion pipeline.")
    subparsers.add_parser("validate", help="Run automated data quality validation checks.")
    subparsers.add_parser("eda", help="Run exploratory data analysis and generate visualizations.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the Arcora CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = ArcoraConfig.from_env()

    if args.command in (None, "info"):
        print(f"Arcora v{__version__} ({cfg.env})")
        print(f"  Project root:       {cfg.project_root}")
        print(f"  Raw data dir:       {cfg.raw_data_dir}")
        print(f"  Processed data dir: {cfg.processed_data_dir}")
        print(f"  Reports dir:        {cfg.reports_dir}")
        print(f"  Docs dir:           {cfg.docs_dir}")
        print(f"  Logs dir:           {cfg.logs_dir}")
        return 0

    cfg.ensure_directories()

    if args.command == "acquire":
        download_raw_dataset(cfg.raw_data_dir, cfg.kaggle_download_api_url, force=False)
        ok, mismatches = verify_raw_checksums(cfg.raw_data_dir)
        if not ok:
            print(f"Checksum mismatch: {mismatches}", file=sys.stderr)
            return 1
        acq_report = inspect_raw_dataset(cfg.raw_data_dir)
        write_acquisition_artifacts(acq_report, cfg.raw_data_dir, cfg.docs_dir)
        return 0

    if args.command == "audit":
        audit_report = run_dataset_audit(cfg.raw_data_dir)
        write_audit_artifacts(audit_report, cfg.docs_dir, cfg.reports_dir)
        return 0

    if args.command == "schema":
        write_schema_documentation(cfg.docs_dir)
        return 0

    if args.command == "ingest":
        run_ingestion_pipeline(
            cfg.raw_data_dir, cfg.processed_data_dir, cfg.logs_dir, cfg.reports_dir
        )
        return 0

    if args.command == "validate":
        val_summary = run_validation_suite(cfg.processed_data_dir)
        write_validation_artifacts(val_summary, cfg.reports_dir)
        return 0 if val_summary.overall_passed else 1

    if args.command == "eda":
        eda_summary = run_exploratory_data_analysis(cfg.processed_data_dir, cfg.figures_dir)
        write_eda_artifacts(eda_summary, cfg.reports_dir, cfg.docs_dir)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
