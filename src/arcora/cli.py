"""Command-line interface for Arcora."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from arcora import __version__
from arcora.config import ArcoraConfig


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the Arcora CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in (None, "info"):
        cfg = ArcoraConfig.from_env()
        print(f"Arcora v{__version__} ({cfg.env})")
        print(f"  Project root:       {cfg.project_root}")
        print(f"  Raw data dir:       {cfg.raw_data_dir}")
        print(f"  Processed data dir: {cfg.processed_data_dir}")
        print(f"  Reports dir:        {cfg.reports_dir}")
        print(f"  Docs dir:           {cfg.docs_dir}")
        print(f"  Logs dir:           {cfg.logs_dir}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
