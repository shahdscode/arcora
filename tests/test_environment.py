"""Tests for project setup, configuration, and environment structure (Issue #1)."""

from __future__ import annotations

from pathlib import Path

from arcora import __version__
from arcora.cli import main as cli_main
from arcora.config import ArcoraConfig, get_project_root


def test_version_defined() -> None:
    """Ensure package version is properly exposed."""
    assert __version__ == "0.1.0"


def test_project_root_structure() -> None:
    """Verify core project files and directory layout exist."""
    root = get_project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "requirements.txt").is_file()
    assert (root / ".gitignore").is_file()
    assert (root / ".env.example").is_file()
    assert (root / "README.md").is_file()
    assert (root / "src" / "arcora").is_dir()


def test_gitignore_excludes_sensitive_and_generated_files() -> None:
    """Verify .gitignore excludes .env, virtual environments, and caches."""
    root = get_project_root()
    gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", ".venv/", "__pycache__/", "kaggle.json"):
        assert pattern in gitignore_text


def test_config_from_env(tmp_path: Path, monkeypatch: object) -> None:
    """Verify ArcoraConfig resolves directories and creates them cleanly."""
    cfg = ArcoraConfig.from_env(project_root=tmp_path)
    cfg.ensure_directories()

    assert cfg.project_root == tmp_path.resolve()
    assert cfg.raw_data_dir == (tmp_path / "data" / "raw").resolve()
    assert cfg.processed_data_dir == (tmp_path / "data" / "processed").resolve()
    assert cfg.splits_dir == (tmp_path / "data" / "splits").resolve()
    assert cfg.models_dir == (tmp_path / "artifacts" / "models").resolve()
    assert cfg.reports_dir == (tmp_path / "reports").resolve()
    assert cfg.figures_dir == (tmp_path / "reports" / "figures").resolve()
    assert cfg.docs_dir == (tmp_path / "docs").resolve()
    assert cfg.logs_dir == (tmp_path / "logs").resolve()

    assert cfg.raw_data_dir.is_dir()
    assert cfg.processed_data_dir.is_dir()
    assert cfg.reports_dir.is_dir()
    assert cfg.figures_dir.is_dir()
    assert cfg.docs_dir.is_dir()
    assert cfg.logs_dir.is_dir()


def test_cli_info_command(capsys: object) -> None:
    """Verify CLI info subcommand runs and exits with code 0."""
    exit_code = cli_main(["info"])
    assert exit_code == 0
