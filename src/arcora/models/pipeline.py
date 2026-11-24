"""Unified Week 3 modeling pipeline and command-line interface (Issue #24).

Steps:

1. Load the Week 2 feature dataset.
2. Validate the feature schema and run the Week 2 leakage audit (LEAK-001 to LEAK-007).
3. Build chronological splits (and verify or write the split manifest).
4. Select the 242 approved pregame features.
5-8. Train baselines, linear/logistic, Random Forest, and XGBoost models (or reload them).
9. Evaluate every model on the training and validation splits.
10. Save artifacts, metrics, and predictions; validate artifacts and reproducibility.
11. Generate the comparison report and ``reports/week-3-modeling-report.md``.

The primary 2021-22 test set and the partial 2022-23 holdout are never touched unless
``--evaluate-locked-test`` is passed; such a run is recorded in an append-only log.

Usage::

    python scripts/train_models.py
    python scripts/train_models.py --families baseline linear --targets home_win point_margin
    python scripts/train_models.py --evaluate-only --reuse-manifest
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arcora.config import ArcoraConfig
from arcora.evaluation.comparison import best_models, comparison_table, rank_models
from arcora.evaluation.report import generate_evaluation_report, load_all_results
from arcora.features.leakage import LeakageCheckResult, run_leakage_audit_suite
from arcora.models.artifact_validation import (
    ArtifactCheck,
    ReproducibilityCheck,
    check_training_reproducibility,
    validate_saved_artifacts,
)
from arcora.models.artifacts import artifact_paths, load_model_artifact
from arcora.models.base import evaluate_trained_model, to_jsonable, utc_timestamp, write_results
from arcora.models.data import ModelingData, load_feature_dataset, prepare_modeling_data
from arcora.models.experiment import (
    evaluate_saved_family,
    get_model_families,
    run_model_family,
)
from arcora.models.modeling_strategy import MODEL_TARGETS, write_modeling_strategy_documentation
from arcora.models.splits import SPLIT_MANIFEST_FILENAME, TEST_SPLIT, VALIDATION_SPLIT
from arcora.models.week3_report import write_week3_report

FEATURES_FILENAME: str = "nba_game_features.parquet"
WEEK3_REPORT_FILENAME: str = "week-3-modeling-report.md"
LOCKED_LOG_FILENAME: str = "locked_evaluation_log.jsonl"
DEFAULT_REPRODUCIBILITY_TARGETS: tuple[str, ...] = ("home_win",)


@dataclass
class ModelingPipelineConfig:
    """Resolved configuration for one pipeline run."""

    features_path: Path
    manifest_path: Path
    reports_dir: Path
    models_dir: Path
    docs_dir: Path
    logs_dir: Path
    teams_path: Path | None
    families: tuple[str, ...]
    targets: tuple[str, ...] = MODEL_TARGETS
    reuse_manifest: bool = False
    evaluate_only: bool = False
    validate_artifacts: bool = True
    check_reproducibility: bool = True
    reproducibility_targets: tuple[str, ...] = DEFAULT_REPRODUCIBILITY_TARGETS
    evaluate_locked_test: bool = False
    write_docs: bool = True
    run_leakage_audit: bool = True

    @property
    def results_dir(self) -> Path:
        """Per-family results directory (``<reports>/models``)."""
        return self.reports_dir / "models"

    @property
    def figures_dir(self) -> Path:
        """Model figures directory (``<reports>/figures/models``)."""
        return self.reports_dir / "figures" / "models"

    @classmethod
    def from_arcora_config(cls, cfg: ArcoraConfig, **overrides: Any) -> ModelingPipelineConfig:
        """Build a pipeline configuration from project paths plus overrides."""
        defaults: dict[str, Any] = {
            "features_path": cfg.features_data_dir / FEATURES_FILENAME,
            "manifest_path": cfg.splits_dir / SPLIT_MANIFEST_FILENAME,
            "reports_dir": cfg.reports_dir,
            "models_dir": cfg.models_dir,
            "docs_dir": cfg.docs_dir,
            "logs_dir": cfg.logs_dir,
            "teams_path": cfg.processed_data_dir / "teams.parquet",
            "families": tuple(get_model_families()),
        }
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**defaults)


@dataclass
class ModelingPipelineSummary:
    """Everything the pipeline produced and verified."""

    started_at: str
    finished_at: str = ""
    duration_seconds: float = 0.0
    mode: str = "train"
    families: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    n_games: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)
    dataset_fingerprint: str = ""
    leakage_checks: list[LeakageCheckResult] = field(default_factory=list)
    split_checks: list[LeakageCheckResult] = field(default_factory=list)
    n_models: int = 0
    family_seconds: dict[str, float] = field(default_factory=dict)
    artifact_checks: list[ArtifactCheck] = field(default_factory=list)
    reproducibility_checks: list[ReproducibilityCheck] = field(default_factory=list)
    locked_test_evaluated: bool = False
    written_files: dict[str, str] = field(default_factory=dict)

    @property
    def leakage_passed(self) -> bool:
        """All leakage and split-integrity checks passed."""
        return all(r.passed for r in (*self.leakage_checks, *self.split_checks))

    @property
    def artifacts_passed(self) -> bool:
        """All artifact checks passed (vacuously true when skipped)."""
        return all(c.passed for c in self.artifact_checks)

    @property
    def reproducible(self) -> bool:
        """All reproducibility checks passed (vacuously true when skipped)."""
        return all(c.reproducible for c in self.reproducibility_checks)

    @property
    def passed(self) -> bool:
        """Overall pipeline status."""
        return self.leakage_passed and self.artifacts_passed and self.reproducible

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload: dict[str, Any] = to_jsonable(
            {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_seconds": self.duration_seconds,
                "mode": self.mode,
                "families": self.families,
                "targets": self.targets,
                "n_games": self.n_games,
                "split_sizes": self.split_sizes,
                "dataset_fingerprint": self.dataset_fingerprint,
                "leakage_checks": [r.to_dict() for r in self.leakage_checks],
                "split_checks": [r.to_dict() for r in self.split_checks],
                "n_models": self.n_models,
                "family_seconds": self.family_seconds,
                "artifact_checks": [c.to_dict() for c in self.artifact_checks],
                "reproducibility_checks": [c.to_dict() for c in self.reproducibility_checks],
                "locked_test_evaluated": self.locked_test_evaluated,
                "leakage_passed": self.leakage_passed,
                "artifacts_passed": self.artifacts_passed,
                "reproducible": self.reproducible,
                "passed": self.passed,
                "written_files": self.written_files,
            }
        )
        return payload


def configure_modeling_logger(log_path: Path) -> logging.Logger:
    """Return a logger writing to ``logs/modeling_pipeline.log`` and the console."""
    logger = logging.getLogger("arcora.modeling")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def run_locked_test_evaluation(
    data: ModelingData, results_dir: Path, models_dir: Path, logger: logging.Logger
) -> Path:
    """Evaluate the validation-selected best model and best baseline per target on the test set.

    This is a one-off locked evaluation: model choice is fixed by validation results before
    the test split is scored, and every run is appended to an audit log.
    """
    ranked = rank_models(comparison_table(load_all_results(results_dir)), VALIDATION_SPLIT)
    chosen = best_models(ranked)
    baselines = (
        ranked[ranked["model_family"] == "baseline"].sort_values("rank").groupby("target").head(1)
    )
    test = data.get(TEST_SPLIT)
    results = []
    for row in [*chosen.to_dict("records"), *baselines.to_dict("records")]:
        joblib_path, _ = artifact_paths(models_dir, str(row["model_family"]), str(row["model_id"]))
        model = load_model_artifact(joblib_path)
        result, _ = evaluate_trained_model(model, test, data)
        results.append(result)
    path = write_results(results, results_dir / "locked_test_results.json")
    log_entry = {
        "evaluated_at": utc_timestamp(),
        "split": TEST_SPLIT,
        "split_game_id_sha256": data.split_manifest["splits"][TEST_SPLIT]["game_id_sha256"],
        "models": sorted({r.model_id for r in results}),
        "selection_basis": "validation primary metric",
    }
    with (results_dir / LOCKED_LOG_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(log_entry) + "\n")
    logger.warning(
        "LOCKED TEST SPLIT EVALUATED (%d models). Do not tune against these results.", len(results)
    )
    return path


def run_modeling_pipeline(config: ModelingPipelineConfig) -> ModelingPipelineSummary:
    """Execute the full Week 3 modeling workflow."""
    start = time.perf_counter()
    summary = ModelingPipelineSummary(started_at=utc_timestamp())
    summary.mode = "evaluate_only" if config.evaluate_only else "train"
    summary.families = list(config.families)
    summary.targets = list(config.targets)
    logger = configure_modeling_logger(config.logs_dir / "modeling_pipeline.log")
    registry = get_model_families()
    unknown = [f for f in config.families if f not in registry]
    if unknown:
        raise ValueError(f"Unknown model families {unknown}. Available: {list(registry)}")
    logger.info("Starting Arcora Week 3 modeling pipeline (%s)", summary.mode)

    # 1-2. Load, validate schema, and audit leakage.
    dataset_df = load_feature_dataset(config.features_path)
    summary.n_games = len(dataset_df)
    if config.run_leakage_audit:
        summary.leakage_checks = run_leakage_audit_suite(dataset_df)
    else:
        logger.warning("Week 2 leakage audit skipped by configuration")
    logger.info(
        "Loaded %d games; leakage checks %d/%d passed",
        len(dataset_df),
        sum(r.passed for r in summary.leakage_checks),
        len(summary.leakage_checks),
    )
    if not all(r.passed for r in summary.leakage_checks):
        raise RuntimeError("Week 2 leakage audit failed; refusing to train models")

    # 3-4. Chronological splits and approved features.
    data = prepare_modeling_data(
        dataset_df,
        source_dataset=f"data/features/{config.features_path.name}",
        manifest_path=config.manifest_path,
        reuse_manifest=config.reuse_manifest,
    )
    summary.split_checks = data.split_checks
    summary.split_sizes = {name: split.n_games for name, split in data.splits.items()}
    summary.dataset_fingerprint = data.dataset_fingerprint
    logger.info("Chronological splits: %s", summary.split_sizes)
    if config.write_docs:
        path = write_modeling_strategy_documentation(config.docs_dir, data.split_manifest)
        summary.written_files["modeling_strategy"] = path.as_posix()

    # 5-10. Train (or reload) and evaluate each family.
    for family in config.families:
        family_start = time.perf_counter()
        if config.evaluate_only:
            output = evaluate_saved_family(
                family, data, config.results_dir, config.models_dir, config.targets
            )
        else:
            output = run_model_family(
                family, data, config.results_dir, config.models_dir, config.targets
            )
        summary.family_seconds[family] = time.perf_counter() - family_start
        summary.n_models += len(output.models)
        for key, value in output.written_files.items():
            if not key.startswith("artifact:"):
                summary.written_files[f"{family}:{key}"] = value.as_posix()
        logger.info(
            "Family %-14s %d models in %.1fs",
            family,
            len(output.models),
            summary.family_seconds[family],
        )

    if config.validate_artifacts:
        summary.artifact_checks = validate_saved_artifacts(
            config.models_dir, data, list(config.families)
        )
        logger.info(
            "Artifact validation: %d/%d passed",
            sum(c.passed for c in summary.artifact_checks),
            len(summary.artifact_checks),
        )
    if config.check_reproducibility and not config.evaluate_only:
        targets = [t for t in config.reproducibility_targets if t in config.targets]
        summary.reproducibility_checks = check_training_reproducibility(
            data, list(config.families), targets or list(config.targets[:1])
        )
        logger.info(
            "Reproducibility: %d/%d retrained models identical",
            sum(c.reproducible for c in summary.reproducibility_checks),
            len(summary.reproducibility_checks),
        )

    # 11. Reports.
    evaluation_files = generate_evaluation_report(
        data,
        config.results_dir,
        config.models_dir,
        config.reports_dir,
        config.figures_dir,
        config.teams_path,
        families=list(config.families),
    )
    summary.written_files.update({k: v.as_posix() for k, v in evaluation_files.items()})

    if config.evaluate_locked_test:
        path = run_locked_test_evaluation(data, config.results_dir, config.models_dir, logger)
        summary.locked_test_evaluated = True
        summary.written_files["locked_test_results"] = path.as_posix()

    summary.finished_at = utc_timestamp()
    summary.duration_seconds = time.perf_counter() - start

    report_path = write_week3_report(summary, data, config)
    summary.written_files["week3_report"] = report_path.as_posix()
    summary_path = config.results_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Modeling pipeline finished in %.1fs (passed=%s)", summary.duration_seconds, summary.passed
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the Week 3 training CLI parser."""
    families = list(get_model_families())
    parser = argparse.ArgumentParser(
        prog="train_models",
        description="Train, evaluate, and report Arcora's Week 3 NBA prediction models.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=families,
        default=families,
        help="Model families to run (default: all, in order).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=list(MODEL_TARGETS),
        default=list(MODEL_TARGETS),
        help="Prediction targets (default: all five).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Reports directory (default: reports/). Results go to <dir>/models.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: artifacts/models).",
    )
    parser.add_argument(
        "--features-path", type=Path, default=None, help="Feature dataset Parquet path."
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=None, help="Split manifest JSON path."
    )
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="Verify splits against the existing manifest instead of rewriting it.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Reload saved artifacts and re-run evaluation and reporting without training.",
    )
    parser.add_argument(
        "--skip-artifact-validation", action="store_true", help="Skip artifact reload checks."
    )
    parser.add_argument(
        "--skip-reproducibility-check",
        action="store_true",
        help="Skip retraining models twice to verify identical predictions.",
    )
    parser.add_argument(
        "--evaluate-locked-test",
        action="store_true",
        help="ONE-OFF locked evaluation of validation-selected models on the 2021-22 test set. "
        "The run is logged; never tune against its results.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns 0 when every check passed."""
    args = build_arg_parser().parse_args(argv)
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()
    config = ModelingPipelineConfig.from_arcora_config(
        cfg,
        features_path=args.features_path,
        manifest_path=args.manifest_path,
        reports_dir=args.output_dir,
        models_dir=args.models_dir,
        families=tuple(args.families),
        targets=tuple(t for t in MODEL_TARGETS if t in args.targets),
        reuse_manifest=args.reuse_manifest,
        evaluate_only=args.evaluate_only,
        validate_artifacts=not args.skip_artifact_validation,
        check_reproducibility=not args.skip_reproducibility_check,
        evaluate_locked_test=args.evaluate_locked_test,
    )
    summary = run_modeling_pipeline(config)
    status = "PASSED" if summary.passed else "FAILED"
    print(
        f"Arcora Week 3 Modeling Pipeline Completed ({status}) in {summary.duration_seconds:.0f}s"
    )
    print(f"  Mode:               {summary.mode}")
    print(f"  Games / splits:     {summary.n_games:,} / {summary.split_sizes}")
    print(f"  Families:           {', '.join(summary.families)}")
    print(f"  Models evaluated:   {summary.n_models}")
    print(
        f"  Leakage checks:     "
        f"{sum(r.passed for r in (*summary.leakage_checks, *summary.split_checks))}/"
        f"{len(summary.leakage_checks) + len(summary.split_checks)} passed"
    )
    if summary.artifact_checks:
        print(
            f"  Artifact checks:    {sum(c.passed for c in summary.artifact_checks)}/"
            f"{len(summary.artifact_checks)} passed"
        )
    if summary.reproducibility_checks:
        print(
            f"  Reproducibility:    {sum(c.reproducible for c in summary.reproducibility_checks)}/"
            f"{len(summary.reproducibility_checks)} identical"
        )
    print(f"  Locked test used:   {'YES (logged)' if summary.locked_test_evaluated else 'no'}")
    print(f"  Report:             {summary.written_files.get('week3_report')}")
    return 0 if summary.passed else 1
