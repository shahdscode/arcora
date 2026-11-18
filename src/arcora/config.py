"""Configuration and path management for Arcora."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    env_root = os.environ.get("ARCORA_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ArcoraConfig:
    """Immutable configuration settings resolved from environment variables."""

    env: str
    log_level: str
    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    features_data_dir: Path
    reports_dir: Path
    figures_dir: Path
    docs_dir: Path
    logs_dir: Path
    kaggle_dataset_slug: str
    kaggle_dataset_url: str
    kaggle_download_api_url: str

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> ArcoraConfig:
        """Load configuration from environment variables with sensible defaults."""
        root = (project_root or get_project_root()).resolve()

        def _resolve_dir(env_key: str, default_rel: str) -> Path:
            raw_val = os.environ.get(env_key, default_rel)
            path = Path(raw_val)
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        reports_dir = _resolve_dir("ARCORA_REPORTS_DIR", "reports")
        return cls(
            env=os.environ.get("ARCORA_ENV", "development"),
            log_level=os.environ.get("ARCORA_LOG_LEVEL", "INFO").upper(),
            project_root=root,
            raw_data_dir=_resolve_dir("ARCORA_RAW_DATA_DIR", "data/raw"),
            processed_data_dir=_resolve_dir("ARCORA_PROCESSED_DATA_DIR", "data/processed"),
            features_data_dir=_resolve_dir("ARCORA_FEATURES_DATA_DIR", "data/features"),
            reports_dir=reports_dir,
            figures_dir=reports_dir / "figures",
            docs_dir=_resolve_dir("ARCORA_DOCS_DIR", "docs"),
            logs_dir=_resolve_dir("ARCORA_LOGS_DIR", "logs"),
            kaggle_dataset_slug=os.environ.get("KAGGLE_DATASET_SLUG", "nathanlauga/nba-games"),
            kaggle_dataset_url=os.environ.get(
                "KAGGLE_DATASET_URL",
                "https://www.kaggle.com/datasets/nathanlauga/nba-games",
            ),
            kaggle_download_api_url=os.environ.get(
                "KAGGLE_DOWNLOAD_API_URL",
                "https://www.kaggle.com/api/v1/datasets/download/nathanlauga/nba-games",
            ),
        )

    def ensure_directories(self) -> None:
        """Create required project directories if they do not exist."""
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.features_data_dir,
            self.reports_dir,
            self.figures_dir,
            self.docs_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
