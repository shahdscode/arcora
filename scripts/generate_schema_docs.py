"""CLI script to generate canonical NBA data model documentation and DDL (Issue #4)."""

from __future__ import annotations

import sys

from arcora.config import ArcoraConfig
from arcora.models.schema import CANONICAL_ENTITIES, write_schema_documentation


def main() -> int:
    """Generate canonical data model documentation, ERD, SQL DDL, and data dictionary."""
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()

    artifacts = write_schema_documentation(cfg.docs_dir)
    print("Arcora Canonical Data Model Documentation Generated:")
    for name, entity in CANONICAL_ENTITIES.items():
        print(f"  Entity '{name}': {len(entity.fields)} fields, PK={entity.primary_key}")
    for key, path in artifacts.items():
        print(f"  Wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
