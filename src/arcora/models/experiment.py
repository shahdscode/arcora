"""Train, evaluate, persist, and report one model family at a time.

Every family exposes a trainer ``(ModelingData, targets) -> list[TrainedModel]``; this module
runs it, scores each model on the training and validation splits with the shared metric
implementation, saves artifacts and predictions, and writes machine-readable and Markdown
results. The locked test and partial-season holdout splits are refused unless explicitly
allowed.

Usage::

    python -m arcora.models.experiment --family baseline
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from arcora.config import ArcoraConfig
from arcora.models.artifacts import (
    list_model_artifacts,
    load_model_artifact,
    save_model_artifact,
    write_feature_list,
)
from arcora.models.base import (
    ModelResult,
    TrainedModel,
    evaluate_trained_model,
    load_results,
    results_to_frame,
    write_results,
)
from arcora.models.data import ModelingData, load_modeling_data
from arcora.models.modeling_strategy import (
    CLASSIFICATION_TASK,
    METRIC_SPECS_BY_NAME,
    MODEL_TARGETS,
    PRIMARY_METRIC,
    REGRESSION_TASK,
    task_for_target,
)
from arcora.models.splits import SPLIT_DEFINITIONS_BY_NAME, TRAIN_SPLIT, VALIDATION_SPLIT

FamilyTrainer = Callable[[ModelingData, Sequence[str]], list[TrainedModel]]
ReportSections = Callable[[list[TrainedModel], ModelingData], list[str]]
PostRunHook = Callable[[list[TrainedModel], ModelingData, Path], dict[str, Path]]
DEFAULT_EVAL_SPLITS: tuple[str, ...] = (TRAIN_SPLIT, VALIDATION_SPLIT)
BASELINE_FAMILY: str = "baseline"


@dataclass(frozen=True)
class ModelFamily:
    """A registered model family."""

    name: str
    display_name: str
    description: str
    trainer: FamilyTrainer
    report_sections: ReportSections | None = None
    post_run: PostRunHook | None = None


def get_model_families() -> dict[str, ModelFamily]:
    """Return the registry of available model families (in training order)."""
    from arcora.models import baselines, linear_models, random_forest, xgboost_models

    return {
        baselines.MODEL_FAMILY: ModelFamily(
            name=baselines.MODEL_FAMILY,
            display_name="Naive Baselines",
            description=baselines.FAMILY_DESCRIPTION,
            trainer=baselines.train_baseline_models,
        ),
        linear_models.MODEL_FAMILY: ModelFamily(
            name=linear_models.MODEL_FAMILY,
            display_name="Linear and Logistic Regression",
            description=linear_models.FAMILY_DESCRIPTION,
            trainer=linear_models.train_linear_models,
            report_sections=linear_models.linear_report_sections,
        ),
        random_forest.MODEL_FAMILY: ModelFamily(
            name=random_forest.MODEL_FAMILY,
            display_name="Random Forest",
            description=random_forest.FAMILY_DESCRIPTION,
            trainer=random_forest.train_random_forest_models,
            report_sections=random_forest.random_forest_report_sections,
            post_run=random_forest.random_forest_post_run,
        ),
        xgboost_models.MODEL_FAMILY: ModelFamily(
            name=xgboost_models.MODEL_FAMILY,
            display_name="XGBoost",
            description=xgboost_models.FAMILY_DESCRIPTION,
            trainer=xgboost_models.train_xgboost_models,
            report_sections=xgboost_models.xgboost_report_sections,
            post_run=xgboost_models.xgboost_post_run,
        ),
    }


@dataclass
class FamilyRunOutput:
    """Everything produced by one family run."""

    family: str
    models: list[TrainedModel]
    results: list[ModelResult]
    predictions: pd.DataFrame
    written_files: dict[str, Path]


def check_evaluation_splits(splits: Sequence[str], allow_locked: bool = False) -> None:
    """Refuse locked splits (primary test / partial holdout) unless explicitly allowed."""
    for name in splits:
        if name not in SPLIT_DEFINITIONS_BY_NAME:
            raise ValueError(f"Unknown split '{name}'")
        if SPLIT_DEFINITIONS_BY_NAME[name].locked and not allow_locked:
            raise PermissionError(
                f"Split '{name}' is locked for final evaluation and cannot be used during "
                "model development. Pass allow_locked=True only for a documented locked run."
            )


def evaluate_models(
    models: Sequence[TrainedModel],
    data: ModelingData,
    splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
    allow_locked: bool = False,
) -> tuple[list[ModelResult], pd.DataFrame]:
    """Evaluate models on the given splits; return results and long-format predictions."""
    check_evaluation_splits(splits, allow_locked)
    results: list[ModelResult] = []
    frames: list[pd.DataFrame] = []
    for split_name in splits:
        split = data.get(split_name)
        for model in models:
            result, preds = evaluate_trained_model(model, split, data)
            results.append(result)
            frames.append(
                pd.DataFrame(
                    {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "model_family": model.model_family,
                        "target": model.target,
                        "task": model.task,
                        "split": split_name,
                        "game_id": split.metadata["game_id"].to_numpy(),
                        "y_true": split.target(model.target).to_numpy(dtype="float64"),
                        "y_pred": preds,
                    }
                )
            )
    predictions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return results, predictions


def _fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _best_baseline(
    baseline_results: Sequence[ModelResult], target: str, split: str
) -> ModelResult | None:
    task = task_for_target(target)
    metric = PRIMARY_METRIC[task]
    candidates = [
        r
        for r in baseline_results
        if r.target == target and r.split == split and r.metrics.get(metric) is not None
    ]
    if not candidates:
        return None
    sign = -1.0 if METRIC_SPECS_BY_NAME[metric].greater_is_better else 1.0
    return min(candidates, key=lambda r: sign * float(r.metrics[metric] or 0.0))


def render_family_report(
    family: ModelFamily,
    results: Sequence[ModelResult],
    baseline_results: Sequence[ModelResult] = (),
    extra_sections: Sequence[str] = (),
) -> str:
    """Render a Markdown summary of a family's train and validation results."""
    by_key: dict[tuple[str, str], dict[str, ModelResult]] = {}
    for r in results:
        by_key.setdefault((r.model_name, r.target), {})[r.split] = r

    lines = [
        f"# Arcora Week 3 — {family.display_name} Results",
        "",
        family.description,
        "",
        "All models are fitted on the **training split (2003-04 to 2018-19)** and compared on "
        "the **validation split (2019-20 to 2020-21)**. The primary 2021-22 test set and the "
        "partial 2022-23 holdout are locked and were not evaluated.",
        "",
    ]
    show_delta = family.name != BASELINE_FAMILY and bool(baseline_results)

    lines.extend(["## Regression (validation)", ""])
    header = "| Model | Target | MAE | RMSE | R² | Train MAE |"
    sep = "| :--- | :--- | ---: | ---: | ---: | ---: |"
    if show_delta:
        header += " Δ MAE vs best baseline (model − baseline) |"
        sep += " ---: |"
    lines.extend([header, sep])
    for (name, target), per_split in by_key.items():
        if task_for_target(target) != REGRESSION_TASK or VALIDATION_SPLIT not in per_split:
            continue
        val = per_split[VALIDATION_SPLIT].metrics
        train = per_split.get(TRAIN_SPLIT)
        row = (
            f"| `{name}` | `{target}` | {_fmt(val['mae'], 3)} | {_fmt(val['rmse'], 3)} | "
            f"{_fmt(val['r2'])} | {_fmt(train.metrics['mae'] if train else None, 3)} |"
        )
        if show_delta:
            best = _best_baseline(baseline_results, target, VALIDATION_SPLIT)
            delta = (
                None
                if best is None or val["mae"] is None or best.metrics["mae"] is None
                else val["mae"] - best.metrics["mae"]
            )
            row += f" {_fmt(delta, 3)} |"
        lines.append(row)

    lines.extend(["", "## Classification — `home_win` (validation)", ""])
    header = (
        "| Model | Accuracy | Balanced acc. | Precision | Recall | ROC-AUC | Log loss | Brier |"
    )
    sep = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    if show_delta:
        header += " Δ log loss vs best baseline (model − baseline) |"
        sep += " ---: |"
    lines.extend([header, sep])
    for (name, target), per_split in by_key.items():
        if task_for_target(target) != CLASSIFICATION_TASK or VALIDATION_SPLIT not in per_split:
            continue
        m = per_split[VALIDATION_SPLIT].metrics
        row = (
            f"| `{name}` | {_fmt(m['accuracy'])} | {_fmt(m['balanced_accuracy'])} | "
            f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['roc_auc'])} | "
            f"{_fmt(m['log_loss'])} | {_fmt(m['brier'])} |"
        )
        if show_delta:
            best = _best_baseline(baseline_results, target, VALIDATION_SPLIT)
            delta = (
                None
                if best is None or m["log_loss"] is None or best.metrics["log_loss"] is None
                else m["log_loss"] - best.metrics["log_loss"]
            )
            row += f" {_fmt(delta)} |"
        lines.append(row)

    undefined = sorted(
        {
            f"`{r.model_name}` / `{r.target}` ({r.split}): {', '.join(r.undefined_metrics)}"
            for r in results
            if r.undefined_metrics
        }
    )
    if undefined:
        lines.extend(["", "### Undefined metrics", ""])
        lines.extend(f"- {u}" for u in undefined)

    lines.extend(
        [
            "",
            "## Hyperparameters and training configuration",
            "",
            "| Model | Target | Parameters |",
            "| :--- | :--- | :--- |",
        ]
    )
    for (name, target), per_split in by_key.items():
        any_result = next(iter(per_split.values()))
        params = ", ".join(f"`{k}={v}`" for k, v in sorted(any_result.params.items()))
        lines.append(f"| `{name}` | `{target}` | {params} |")

    if extra_sections:
        lines.extend(["", *extra_sections])

    sample = results[0].reproducibility if results else {}
    if sample:
        lines.extend(
            [
                "",
                "## Reproducibility",
                "",
                f"- Dataset fingerprint: `{sample['dataset_fingerprint']}`",
                f"- Training split `game_id` SHA-256: `{sample['train_game_id_sha256']}`",
                f"- Feature fingerprint ({sample['n_features']} features): "
                f"`{sample['feature_fingerprint']}`",
                "- Library versions: "
                + ", ".join(f"{k} {v}" for k, v in sample["library_versions"].items() if v),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_cross_family_comparison(
    results: Sequence[ModelResult], reference_results: Sequence[ModelResult]
) -> list[str]:
    """Markdown table: best validation model of every family for each target."""
    combined = [r for r in (*reference_results, *results) if r.split == VALIDATION_SPLIT]
    if not combined:
        return []
    lines = [
        "## Comparison with every trained family (validation)",
        "",
        "Best model per family on the primary metric (MAE for regression, log loss for "
        "`home_win`), evaluated on the identical validation split.",
        "",
        "| Target | Metric | " + " | ".join(f"`{f}`" for f in _families_in(combined)) + " |",
        "| :--- | :--- | " + " | ".join("---:" for _ in _families_in(combined)) + " |",
    ]
    for target in MODEL_TARGETS:
        metric = PRIMARY_METRIC[task_for_target(target)]
        cells = []
        for fam in _families_in(combined):
            best = _best_baseline(
                [r for r in combined if r.model_family == fam], target, VALIDATION_SPLIT
            )
            cells.append(
                "—" if best is None else f"{_fmt(best.metrics[metric])} (`{best.model_name}`)"
            )
        if any(c != "—" for c in cells):
            lines.append(f"| `{target}` | {metric} | " + " | ".join(cells) + " |")
    return lines


def _families_in(results: Sequence[ModelResult]) -> list[str]:
    order = list(get_model_families())
    present = {r.model_family for r in results}
    return [f for f in order if f in present] + sorted(present - set(order))


def load_reference_results(results_dir: Path, exclude_family: str) -> list[ModelResult]:
    """Load persisted results of every other family found in ``results_dir``."""
    references: list[ModelResult] = []
    for family in get_model_families():
        path = results_dir / f"{family}_results.json"
        if family != exclude_family and path.is_file():
            references.extend(load_results(path))
    return references


def load_baseline_results(results_dir: Path) -> list[ModelResult]:
    """Load persisted baseline results, if they exist."""
    path = results_dir / f"{BASELINE_FAMILY}_results.json"
    return load_results(path) if path.is_file() else []


def run_model_family(
    family_name: str,
    data: ModelingData,
    results_dir: Path,
    models_dir: Path | None = None,
    targets: Sequence[str] = MODEL_TARGETS,
    eval_splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
    allow_locked: bool = False,
    extra_report_sections: ReportSections | None = None,
    run_post_hooks: bool = True,
) -> FamilyRunOutput:
    """Train, evaluate, persist, and report one model family."""
    families = get_model_families()
    if family_name not in families:
        raise KeyError(f"Unknown model family '{family_name}'. Available: {list(families)}")
    family = families[family_name]
    for t in targets:
        task_for_target(t)

    models = family.trainer(data, targets)
    return _evaluate_persist_and_report(
        family,
        models,
        data,
        results_dir,
        models_dir,
        eval_splits,
        allow_locked,
        extra_report_sections,
        run_post_hooks,
        save_artifacts=True,
    )


def evaluate_saved_family(
    family_name: str,
    data: ModelingData,
    results_dir: Path,
    models_dir: Path,
    targets: Sequence[str] = MODEL_TARGETS,
    eval_splits: Sequence[str] = DEFAULT_EVAL_SPLITS,
    allow_locked: bool = False,
    run_post_hooks: bool = True,
) -> FamilyRunOutput:
    """Re-evaluate previously saved artifacts of one family without retraining."""
    families = get_model_families()
    if family_name not in families:
        raise KeyError(f"Unknown model family '{family_name}'. Available: {list(families)}")
    paths = list_model_artifacts(models_dir, family_name)
    models = [load_model_artifact(p) for p in paths]
    models = [m for m in models if m.target in targets]
    if not models:
        raise FileNotFoundError(
            f"No saved '{family_name}' artifacts for targets {list(targets)} in {models_dir}"
        )
    for model in models:
        if model.feature_names != data.feature_names:
            raise ValueError(f"Artifact {model.model_id} was trained on a different feature list")
    order = {t: i for i, t in enumerate(MODEL_TARGETS)}
    models.sort(key=lambda m: (order[m.target], m.model_name))
    return _evaluate_persist_and_report(
        families[family_name],
        models,
        data,
        results_dir,
        models_dir,
        eval_splits,
        allow_locked,
        None,
        run_post_hooks,
        save_artifacts=False,
    )


def _evaluate_persist_and_report(
    family: ModelFamily,
    models: list[TrainedModel],
    data: ModelingData,
    results_dir: Path,
    models_dir: Path | None,
    eval_splits: Sequence[str],
    allow_locked: bool,
    extra_report_sections: ReportSections | None,
    run_post_hooks: bool,
    save_artifacts: bool,
) -> FamilyRunOutput:
    family_name = family.name
    results, predictions = evaluate_models(models, data, eval_splits, allow_locked)

    written: dict[str, Path] = {}
    if models_dir is not None:
        written["feature_list"] = write_feature_list(data.feature_names, models_dir)
        for model in models if save_artifacts else []:
            joblib_path, _ = save_model_artifact(
                model,
                models_dir,
                extra_metadata={
                    "dataset_fingerprint": data.dataset_fingerprint,
                    "train_game_id_sha256": data.split_manifest["splits"][TRAIN_SPLIT][
                        "game_id_sha256"
                    ],
                    "validation_metrics": next(
                        (
                            r.metrics
                            for r in results
                            if r.model_id == model.model_id and r.split == VALIDATION_SPLIT
                        ),
                        None,
                    ),
                },
            )
            written[f"artifact:{model.model_id}"] = joblib_path
        pred_path = models_dir / family_name / "predictions.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(pred_path, index=False)
        written["predictions"] = pred_path

    results_dir.mkdir(parents=True, exist_ok=True)
    written["results_json"] = write_results(results, results_dir / f"{family_name}_results.json")
    csv_path = results_dir / f"{family_name}_results.csv"
    results_to_frame(results).to_csv(csv_path, index=False)
    written["results_csv"] = csv_path

    baseline_results = [] if family_name == BASELINE_FAMILY else load_baseline_results(results_dir)
    section_builder = extra_report_sections or family.report_sections
    sections = section_builder(models, data) if section_builder else []
    references = load_reference_results(results_dir, exclude_family=family_name)
    if family_name != BASELINE_FAMILY and references:
        sections = [*render_cross_family_comparison(results, references), "", *sections]
    report_path = results_dir / f"{family_name}_report.md"
    report_path.write_text(
        render_family_report(family, results, baseline_results, sections), encoding="utf-8"
    )
    written["report"] = report_path
    if family.post_run is not None and run_post_hooks:
        written.update(family.post_run(models, data, results_dir))
    return FamilyRunOutput(family_name, models, results, predictions, written)


def build_parser() -> argparse.ArgumentParser:
    """Build the single-family experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Train and evaluate one Arcora model family.")
    parser.add_argument("--family", required=True, help="Model family name.")
    parser.add_argument("--targets", nargs="+", default=list(MODEL_TARGETS))
    parser.add_argument("--no-artifacts", action="store_true", help="Skip saving artifacts.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one model family end to end with project-default paths."""
    args = build_parser().parse_args(argv)
    cfg = ArcoraConfig.from_env()
    cfg.ensure_directories()
    data = load_modeling_data(
        cfg.features_data_dir / "nba_game_features.parquet",
        manifest_path=cfg.splits_dir / "split_manifest.json",
        reuse_manifest=(cfg.splits_dir / "split_manifest.json").is_file(),
    )
    output = run_model_family(
        args.family,
        data,
        results_dir=cfg.reports_dir / "models",
        models_dir=None if args.no_artifacts else cfg.models_dir,
        targets=args.targets,
    )
    for key, path in output.written_files.items():
        if not key.startswith("artifact:"):
            print(f"  {key}: {path}")
    print(f"Trained {len(output.models)} models; {len(output.results)} evaluations written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
