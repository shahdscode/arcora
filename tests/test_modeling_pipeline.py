"""Tests for the unified Week 3 modeling pipeline and CLI (Issue #24)."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arcora.cli import main as cli_main
from arcora.models.artifact_validation import (
    check_training_reproducibility,
    validate_saved_artifacts,
)
from arcora.models.artifacts import list_model_artifacts
from arcora.models.data import ModelingData
from arcora.models.pipeline import (
    ModelingPipelineConfig,
    ModelingPipelineSummary,
    build_arg_parser,
    run_modeling_pipeline,
)
from tests.conftest import make_synthetic_feature_dataset

FAMILIES = ("baseline", "linear", "random_forest", "xgboost")


def _config(root: Path, **overrides: object) -> ModelingPipelineConfig:
    features = root / "features" / "nba_game_features.parquet"
    if not features.is_file():
        features.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_feature_dataset().to_parquet(features, index=False)
    config = ModelingPipelineConfig(
        features_path=features,
        manifest_path=root / "splits" / "split_manifest.json",
        reports_dir=root / "reports",
        models_dir=root / "artifacts",
        docs_dir=root / "docs",
        logs_dir=root / "logs",
        teams_path=None,
        families=FAMILIES,
        run_leakage_audit=False,  # synthetic features do not satisfy Week 2 value bounds
        reproducibility_targets=("home_win",),
    )
    return replace(config, **overrides)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def pipeline_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, ModelingPipelineSummary]:
    root = tmp_path_factory.mktemp("pipeline")
    return root, run_modeling_pipeline(_config(root))


def test_pipeline_runs_every_family_and_passes(
    pipeline_run: tuple[Path, ModelingPipelineSummary],
) -> None:
    _, summary = pipeline_run
    assert summary.passed
    assert summary.families == list(FAMILIES)
    assert summary.n_models == 15 + 13 + 5 + 5
    assert summary.split_sizes == {"train": 480, "validation": 60, "test": 30, "holdout": 30}
    assert all(r.passed for r in summary.split_checks)
    assert len(summary.artifact_checks) == summary.n_models
    assert all(c.passed for c in summary.artifact_checks)
    assert summary.reproducibility_checks and summary.reproducible
    assert not summary.locked_test_evaluated


def test_pipeline_writes_all_deliverables(
    pipeline_run: tuple[Path, ModelingPipelineSummary],
) -> None:
    root, _ = pipeline_run
    reports = root / "reports"
    for path in (
        reports / "week-3-modeling-report.md",
        reports / "model_comparison.csv",
        reports / "model_comparison.json",
        reports / "model_evaluation_report.md",
        reports / "models" / "pipeline_summary.json",
        root / "splits" / "split_manifest.json",
        root / "docs" / "modeling_strategy.md",
        root / "logs" / "modeling_pipeline.log",
        root / "artifacts" / "feature_list.json",
    ):
        assert path.is_file(), path
    for family in FAMILIES:
        assert (reports / "models" / f"{family}_results.json").is_file()
        assert list_model_artifacts(root / "artifacts", family)

    report = (reports / "week-3-modeling-report.md").read_text()
    for heading in (
        "## 1. Dataset and Feature Summary",
        "## 2. Chronological Splits",
        "## 3. Models Implemented",
        "## 4. Evaluation Methodology",
        "## 5. Regression Model Comparison",
        "## 6. Classification Model Comparison",
        "## 7. Best-Performing Model per Task",
        "## 8. Baseline Comparisons",
        "## 9. Error Analysis",
        "## 10. Quality and Reproducibility Checks",
        "## 11. Locked Evaluation Status",
        "## 12. Limitations",
        "## 13. Recommended Candidates for Week 4",
    ):
        assert heading in report
    assert "were **not evaluated**" in report
    summary_json = json.loads((reports / "models" / "pipeline_summary.json").read_text())
    assert summary_json["passed"] is True


def test_evaluate_only_reuses_artifacts_and_reproduces_metrics(
    pipeline_run: tuple[Path, ModelingPipelineSummary],
) -> None:
    root, _ = pipeline_run
    before = pd.read_csv(root / "reports" / "model_comparison.csv")
    summary = run_modeling_pipeline(
        _config(root, evaluate_only=True, reuse_manifest=True, write_docs=False)
    )
    after = pd.read_csv(root / "reports" / "model_comparison.csv")
    assert summary.mode == "evaluate_only"
    assert summary.passed
    assert summary.reproducibility_checks  # carried forward from the training run
    assert before[["model_id", "split"]].equals(after[["model_id", "split"]])
    for metric in ("mae", "rmse", "log_loss", "brier"):
        assert np.allclose(before[metric], after[metric], equal_nan=True, atol=1e-12)


def test_reused_manifest_detects_changed_dataset(
    pipeline_run: tuple[Path, ModelingPipelineSummary], tmp_path: Path
) -> None:
    root, _ = pipeline_run
    changed = make_synthetic_feature_dataset().iloc[1:]
    features = tmp_path / "nba_game_features.parquet"
    changed.to_parquet(features, index=False)
    config = _config(
        root,
        features_path=features,
        reuse_manifest=True,
        families=("baseline",),
        reports_dir=tmp_path / "reports",
        models_dir=tmp_path / "artifacts",
    )
    with pytest.raises(ValueError, match="manifest"):
        run_modeling_pipeline(config)


def test_locked_test_evaluation_is_explicit_and_logged(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        families=("baseline", "linear"),
        evaluate_locked_test=True,
        check_reproducibility=False,
    )
    summary = run_modeling_pipeline(config)
    assert summary.locked_test_evaluated
    results = json.loads((tmp_path / "reports" / "models" / "locked_test_results.json").read_text())
    assert {r["split"] for r in results["results"]} == {"test"}
    log_lines = (tmp_path / "reports" / "models" / "locked_evaluation_log.jsonl").read_text()
    assert len(log_lines.strip().splitlines()) == 1
    report = (tmp_path / "reports" / "week-3-modeling-report.md").read_text()
    assert "test split was evaluated in this run" in report


def test_leakage_audit_failure_blocks_training(tmp_path: Path) -> None:
    config = _config(tmp_path, run_leakage_audit=True, families=("baseline",))
    with pytest.raises(RuntimeError, match="leakage audit failed"):
        run_modeling_pipeline(config)


def test_unknown_family_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown model families"):
        run_modeling_pipeline(_config(tmp_path, families=("neural_net",)))


def test_artifact_validation_flags_corrupted_artifact(
    pipeline_run: tuple[Path, ModelingPipelineSummary],
    synthetic_modeling_data: ModelingData,
    tmp_path: Path,
) -> None:
    root, _ = pipeline_run
    copy = tmp_path / "artifacts"
    shutil.copytree(root / "artifacts", copy)
    target = list_model_artifacts(copy, "baseline")[0]
    target.write_bytes(target.read_bytes()[:-10] + b"corrupted!")
    checks = validate_saved_artifacts(copy, synthetic_modeling_data, ["baseline"])
    failed = [c for c in checks if not c.passed]
    assert len(failed) == 1
    assert failed[0].error is not None and "Checksum" in failed[0].error


def test_training_reproducibility_check(synthetic_modeling_data: ModelingData) -> None:
    checks = check_training_reproducibility(synthetic_modeling_data, ["baseline", "linear"])
    assert checks and all(c.reproducible for c in checks)


def test_cli_parser_options_and_dispatch() -> None:
    args = build_arg_parser().parse_args(
        ["--families", "baseline", "xgboost", "--targets", "home_win", "--evaluate-only"]
    )
    assert args.families == ["baseline", "xgboost"]
    assert args.targets == ["home_win"]
    assert args.evaluate_only and not args.evaluate_locked_test
    defaults = build_arg_parser().parse_args([])
    assert defaults.families == list(FAMILIES)
    with pytest.raises(SystemExit) as excinfo:
        cli_main(["train", "--help"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--families", "neural_net"])


def _digest_tree(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in patterns
        for p in sorted(root.rglob(pattern))
    }


def test_reports_only_rerenders_without_touching_models(
    pipeline_run: tuple[Path, ModelingPipelineSummary],
) -> None:
    root, _ = pipeline_run
    protected = (
        _digest_tree(root / "artifacts", ("*.joblib", "*.parquet", "*.json"))
        | _digest_tree(root / "reports" / "models", ("*_results.json", "*_results.csv"))
        | _digest_tree(root / "splits", ("*.json",))
    )
    before_csv = pd.read_csv(root / "reports" / "model_comparison.csv")

    summary = run_modeling_pipeline(_config(root, reports_only=True))

    after = (
        _digest_tree(root / "artifacts", ("*.joblib", "*.parquet", "*.json"))
        | _digest_tree(root / "reports" / "models", ("*_results.json", "*_results.csv"))
        | _digest_tree(root / "splits", ("*.json",))
    )
    assert after == protected
    assert summary.mode == "reports_only"
    assert summary.passed and summary.reproducibility_checks  # carried forward
    after_csv = pd.read_csv(root / "reports" / "model_comparison.csv")
    for metric in ("mae", "log_loss", "delta_vs_baseline", "relative_loss_reduction_pct"):
        assert np.allclose(before_csv[metric], after_csv[metric], equal_nan=True)
    report = (root / "reports" / "week-3-modeling-report.md").read_text()
    assert "Reports-only run" in report
    assert "### Regression Error Summary (Validation)" in report


def test_reports_only_rejects_retraining_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reports_only"):
        run_modeling_pipeline(_config(tmp_path, reports_only=True, evaluate_only=True))
