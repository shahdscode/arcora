"""Render ``reports/week-3-modeling-report.md`` from pipeline outputs (Issue #24).

Every number in the report is read from the persisted results, comparison JSON, split
manifest, and pipeline summary of the current run, so the report regenerates reproducibly.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from arcora.evaluation.comparison import best_per_family, comparison_table, rank_models
from arcora.evaluation.report import load_all_results
from arcora.features.strategy import FEATURE_SPECIFICATIONS, METADATA_COLUMNS
from arcora.models.data import ModelingData
from arcora.models.modeling_strategy import (
    APPROVED_MISSING_INDICATORS,
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGETS,
    REGRESSION_TARGETS,
)
from arcora.models.splits import TRAIN_SPLIT, VALIDATION_SPLIT

if TYPE_CHECKING:
    from arcora.models.pipeline import ModelingPipelineConfig, ModelingPipelineSummary

FAMILY_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "baseline": (
        "Naive baselines",
        "`src/arcora/models/baselines.py`",
        "Mean / median `DummyRegressor`, most-frequent / prior `DummyClassifier`, and a "
        "team-strength historical reference with league-level fallbacks.",
    ),
    "linear": (
        "Linear & logistic regression",
        "`src/arcora/models/linear_models.py`",
        "OLS, Ridge, Lasso, and L2 logistic regression in median-imputation + scaling "
        "pipelines; penalties chosen on validation.",
    ),
    "random_forest": (
        "Random Forest",
        "`src/arcora/models/random_forest.py`",
        "300 trees, depth 14, 25-game leaves, median imputation; fixed configuration.",
    ),
    "xgboost": (
        "XGBoost",
        "`src/arcora/models/xgboost_models.py`",
        "Histogram gradient boosting, depth 4, learning rate 0.03, native missing values, "
        "early stopping on validation.",
    ),
}


def _f(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _delta(row: Any) -> str:
    """Absolute difference ``model - best baseline`` in metric units (negative = better)."""
    value = row.get("delta_vs_baseline")
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.4f}"


def _pct(row: Any) -> str:
    """Relative loss reduction ``(baseline - model) / baseline`` in percent (positive = better)."""
    value = row.get("relative_loss_reduction_pct")
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.2f}%"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _ranking_table(ranked: pd.DataFrame, target: str) -> list[str]:
    sub = ranked[ranked["target"] == target].sort_values("rank")
    if target in REGRESSION_TARGETS:
        lines = [
            "| Rank | Model | Family | MAE | RMSE | R² | `delta_vs_baseline` | "
            "`relative_loss_reduction_pct` |",
            "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in sub.to_dict("records"):
            lines.append(
                f"| {r['rank']} | `{r['model_name']}` | {r['model_family']} | {_f(r['mae'])} | "
                f"{_f(r['rmse'])} | {_f(r['r2'], 4)} | {_delta(r)} | {_pct(r)} |"
            )
        return lines
    lines = [
        "| Rank | Model | Family | Log loss | Brier | ROC-AUC | Accuracy | Balanced acc. | "
        "`delta_vs_baseline` | `relative_loss_reduction_pct` |",
        "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sub.to_dict("records"):
        lines.append(
            f"| {r['rank']} | `{r['model_name']}` | {r['model_family']} | {_f(r['log_loss'], 4)} | "
            f"{_f(r['brier'], 4)} | {_f(r['roc_auc'], 4)} | {_f(r['accuracy'], 4)} | "
            f"{_f(r['balanced_accuracy'], 4)} | {_delta(r)} | {_pct(r)} |"
        )
    return lines


def _top_permutation_groups(results_dir: Path, family: str, target: str, n: int = 3) -> str:
    path = results_dir / f"{family}_permutation_importance.csv"
    if not path.is_file():
        return ""
    frame = pd.read_csv(path)
    sub = frame[frame["target"] == target].sort_values("mean_increase", ascending=False).head(n)
    return ", ".join(
        f"{r['category']} (+{float(r['mean_increase']):.4f} {r['metric']})"
        for r in sub.to_dict("records")
    )


def render_week3_report(
    summary: ModelingPipelineSummary, data: ModelingData, config: ModelingPipelineConfig
) -> str:
    """Render the Week 3 modeling report."""
    results = load_all_results(config.results_dir, list(config.families))
    table = comparison_table(results)
    ranked = rank_models(table, VALIDATION_SPLIT)
    family_best = best_per_family(ranked)
    comparison = _load_json(config.reports_dir / "model_comparison.json")
    manifest = data.split_manifest
    train_table = table[table["split"] == TRAIN_SPLIT].set_index("model_id")

    category_counts = Counter(f.category for f in FEATURE_SPECIFICATIONS)
    lines: list[str] = [
        "# Arcora Week 3 — Baseline Modeling Report",
        "",
        "This report documents Arcora's first NBA game prediction models: naive baselines, "
        "linear and logistic regression, Random Forest, and XGBoost, trained and compared under "
        "one chronological protocol. It is generated by the unified modeling pipeline:",
        "",
        "```bash",
        "python scripts/train_models.py",
        "```",
        "",
        f"- **Run**: {summary.started_at} → {summary.finished_at} "
        f"({summary.duration_seconds:.0f}s, mode `{summary.mode}`)",
        *(
            [
                "- **Reports-only run**: tables and figures were re-rendered from the persisted "
                "results and artifacts of the last training run; no model was retrained or "
                "re-evaluated, and reproducibility results are carried forward from that run.",
            ]
            if summary.mode == "reports_only"
            else [
                "- **Evaluate-only run**: saved artifacts were reloaded and re-scored without "
                "retraining; reproducibility results are carried forward from the last training "
                "run.",
            ]
            if summary.mode == "evaluate_only"
            else []
        ),
        f"- **Models evaluated**: {summary.n_models} across {len(summary.families)} families "
        f"and {len(summary.targets)} targets",
        f"- **Overall status**: {'PASSED' if summary.passed else 'FAILED'} (leakage, split, "
        "artifact, and reproducibility checks)",
        "",
        "---",
        "",
        "## 1. Dataset and Feature Summary",
        "",
        f"- **Source**: `{manifest['source_dataset']}` — {summary.n_games:,} eligible "
        "non-preseason games (Regular Season, Play-In, Playoffs), 2003-04 to 2022-23 (partial).",
        f"- **Model inputs**: {len(MODEL_FEATURE_COLUMNS)} approved pregame features, including "
        f"{len(APPROVED_MISSING_INDICATORS)} missing-value indicators. Metadata "
        f"({len(METADATA_COLUMNS)} columns), identifiers, targets, and post-game columns are "
        "excluded.",
        f"- **Targets**: {', '.join(f'`{t}`' for t in MODEL_TARGETS)} (four regression targets, "
        "one binary classification target).",
        f"- **Dataset fingerprint**: `{summary.dataset_fingerprint}`",
        "",
        "| Feature category | Features |",
        "| :--- | ---: |",
    ]
    for category, count in category_counts.most_common():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## 2. Chronological Splits",
            "",
            "| Split | Seasons | Games | Dates | Role |",
            "| :--- | :--- | ---: | :--- | :--- |",
        ]
    )
    for name in manifest["split_order"]:
        s = manifest["splits"][name]
        seasons = (
            s["season_labels"][0]
            if len(s["season_labels"]) == 1
            else f"{s['season_labels'][0]} to {s['season_labels'][-1]}"
        )
        lines.append(
            f"| `{name}` | {seasons} | {s['n_games']:,} | {s['date_min']} → {s['date_max']} | "
            f"{s['role']} |"
        )
    passed_splits = sum(r.passed for r in summary.split_checks)
    lines.extend(
        [
            "",
            "Split sizes differ from the illustrative figures in the Week 2 report "
            "(20,762 / 2,282 / 1,327); the season definitions are unchanged and the counts above "
            "are computed from the dataset. The cause is documented in "
            "[`docs/modeling_strategy.md`](../docs/modeling_strategy.md#split-count-reconciliation-week-2-plan-vs-dataset).",
            "",
            f"Split integrity checks: **{passed_splits}/{len(summary.split_checks)} passed** (no "
            "game in multiple splits, strict date ordering, same-day games never cross a "
            "boundary, partial 2022-23 season isolated). Full definitions: "
            "[`docs/modeling_strategy.md`](../docs/modeling_strategy.md).",
            "",
            "## 3. Models Implemented",
            "",
            "| Family | Module | Description |",
            "| :--- | :--- | :--- |",
        ]
    )
    for family in summary.families:
        title, module, desc = FAMILY_DESCRIPTIONS.get(family, (family, "", ""))
        lines.append(f"| {title} (`{family}`) | {module} | {desc} |")
    lines.extend(
        [
            "",
            "Details and hyperparameters: [`docs/model_families.md`](../docs/model_families.md).",
            "",
            "## 4. Evaluation Methodology",
            "",
            "- Every model is fitted on the training split only; imputers, scalers, and baseline "
            "statistics included.",
            "- Model selection (Ridge/Lasso penalties, logistic `C`, XGBoost early stopping) uses "
            "the validation split only, so tuned models' validation scores are mildly optimistic.",
            "- Primary metrics: **MAE** for regression and **log loss** for `home_win`; RMSE, R², "
            "accuracy, balanced accuracy, precision, recall, ROC-AUC, and Brier score are also "
            "reported. All metrics come from one implementation "
            "(`src/arcora/evaluation/metrics.py`).",
            "- **Reading the baseline comparisons.** `delta_vs_baseline` is the absolute "
            "difference `model − baseline` in the metric's own units (points of MAE, or nats of "
            "log loss); negative is better. `relative_loss_reduction_pct` is "
            "`(baseline − model) / baseline × 100`, where the denominator is the best baseline's "
            "MAE or log loss on validation; positive is better. It is a percentage reduction in "
            "loss, not a gain in accuracy points. The same names are used in "
            "`model_comparison.csv`, `model_comparison.json`, and every report table. The best "
            "baseline is the lowest-loss baseline per target, which is always a non-degenerate "
            "model; `most_frequent_baseline` (0/1 probabilities) can never be the probability "
            "benchmark.",
            "- Comparisons are refused unless every result shares the same dataset, split, and "
            "feature fingerprints. A paired bootstrap over validation games tests whether the best "
            "model's improvement over the best baseline is meaningful.",
            "",
            "## 5. Regression Model Comparison (validation)",
            "",
        ]
    )
    for target in REGRESSION_TARGETS:
        lines.extend([f"### `{target}`", "", *_ranking_table(ranked, target), ""])

    lines.extend(
        [
            "## 6. Classification Model Comparison — `home_win` (validation)",
            "",
            *_ranking_table(ranked, "home_win"),
            "",
            "## 7. Best-Performing Model per Task",
            "",
            "| Target | Best model | Family | Validation score | Best baseline | "
            "`delta_vs_baseline` | `relative_loss_reduction_pct` | Bootstrap 95% CI (Δ loss) | "
            "Significant |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :---: |",
        ]
    )
    paired = {p["target"]: p for p in comparison.get("paired_bootstrap_vs_baseline", [])}
    best = ranked[ranked["rank"] == 1]
    for r in best.to_dict("records"):
        p = paired.get(r["target"])
        ci = f"[{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]" if p else "—"
        sig = ("Yes" if p["significant"] else "No") if p else "—"
        lines.append(
            f"| `{r['target']}` | `{r['model_name']}` | {r['model_family']} | "
            f"{_f(r['primary_score'], 4)} {r['primary_metric']} | "
            f"{_f(r['best_baseline_score'], 4)} | {_delta(r)} | {_pct(r)} | "
            f"{ci} | {sig} |"
        )

    lines.extend(
        [
            "",
            "### Relative loss reduction vs best baseline (chart)",
            "",
            "![relative_loss_reduction_pct by family and target]"
            "(figures/models/model_loss_reduction_vs_baseline.png)",
            "",
            "Best model of each family (primary metric, validation):",
            "",
            "| Target | " + " | ".join(f"`{f}`" for f in summary.families) + " |",
            "| :--- | " + " | ".join("---:" for _ in summary.families) + " |",
        ]
    )
    for target in summary.targets:
        cells = []
        for family in summary.families:
            row = family_best[
                (family_best["target"] == target) & (family_best["model_family"] == family)
            ]
            cells.append("—" if row.empty else _f(row["primary_score"].iloc[0], 4))
        lines.append(f"| `{target}` | " + " | ".join(cells) + " |")

    baseline_rows = ranked[ranked["model_family"] == "baseline"].sort_values(["target", "rank"])
    lines.extend(
        [
            "",
            "## 8. Baseline Comparisons",
            "",
            "| Target | Baseline | Validation score | Training score |",
            "| :--- | :--- | ---: | ---: |",
        ]
    )
    for r in baseline_rows.to_dict("records"):
        train_score = (
            train_table.loc[r["model_id"], "primary_score"]
            if r["model_id"] in train_table.index
            else None
        )
        lines.append(
            f"| `{r['target']}` | `{r['model_name']}` | {_f(r['primary_score'], 4)} "
            f"{r['primary_metric']} | {_f(train_score, 4)} |"
        )
    const = ranked[
        (ranked["model_name"] == "mean_baseline") & (ranked["target"].isin(REGRESSION_TARGETS))
    ]
    min_r2 = const["r2"].min() if not const.empty else None
    best_baselines = baseline_rows.groupby("target").head(1)
    reference_wins = int((best_baselines["model_name"] == "historical_reference").sum())
    n_targets = len(best_baselines)
    lines.extend(
        [
            "",
            "- Constant (mean/median) baselines are anchored to the 2003-19 scoring level; "
            "league scoring rose in 2019-21, so their validation R² is negative "
            f"(as low as {_f(min_r2)}).",
            "- The **historical reference** (season-to-date team averages plus a home-court "
            f"offset) is the strongest baseline for {reference_wins} of {n_targets} targets.",
            "- `most_frequent_baseline` outputs degenerate 0/1 probabilities, so its log loss is "
            "very large by construction; `prior_baseline` is the meaningful probability baseline.",
            "",
            "## 9. Error Analysis",
            "",
            "Full breakdowns (by season, home team, away team, game type; largest errors; "
            "confusion matrix; calibration): "
            "[`model_evaluation_report.md`](model_evaluation_report.md).",
        ]
    )
    residuals = comparison.get("regression_residuals", {})
    if residuals:
        lines.extend(
            [
                "### Regression Error Summary (Validation)",
                "",
                "Residual = predicted − actual for the best model of each regression target.",
                "",
                "| Target (best model) | Bias (mean residual) | Residual std | Within ±5 | "
                "Within ±10 |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
        )
        for target, res in residuals.items():
            lines.append(
                f"| `{target}` | {res['mean_residual']:+.2f} | {res['std_residual']:.2f} | "
                f"{100 * res['share_within_5']:.1f}% | {100 * res['share_within_10']:.1f}% |"
            )
    confusion = comparison.get("classification_confusion")
    if confusion:
        lines.extend(
            [
                "",
                f"`home_win` confusion (best model, validation): TP {confusion['true_positive']}, "
                f"FP {confusion['false_positive']}, TN {confusion['true_negative']}, "
                f"FN {confusion['false_negative']}.",
            ]
        )
    lines.extend(
        [
            "",
            "![Error by season](figures/models/error_by_season.png)",
            "",
            "Key observations:",
            "",
            "- Errors jump at the validation boundary. Part of this is ordinary out-of-sample "
            "error, but 2019-20 and 2020-21 were also atypical: a shortened season with bubble "
            "playoffs, then a season with limited crowds. Home advantage and scoring levels "
            "shifted away from the training era.",
            "- Score-level targets (`home_pts`, `away_pts`, `total_pts`) are dominated by the "
            "scoring-environment level; models improve only modestly on the historical "
            "reference. Relative targets (`point_margin`, `home_win`) benefit most from the "
            "matchup features.",
            "- Classifiers over-predict home wins in validation (more false positives than false "
            "negatives) because the training-era home-win rate is higher than in 2019-21.",
        ]
    )
    rf_groups = _top_permutation_groups(config.results_dir, "random_forest", "home_win")
    if rf_groups:
        lines.append(
            f"- Grouped permutation importance (Random Forest, `home_win`): {rf_groups}. "
            "Importance is exploratory, not causal."
        )

    lines.extend(
        [
            "",
            "## 10. Quality and Reproducibility Checks",
            "",
            "| Check | Result |",
            "| :--- | :--- |",
            f"| Week 2 leakage audit (LEAK-001–LEAK-007) | "
            f"{sum(r.passed for r in summary.leakage_checks)}/{len(summary.leakage_checks)} "
            "passed |",
            f"| Chronological split checks (SPLIT-001–SPLIT-007) | "
            f"{passed_splits}/{len(summary.split_checks)} passed |",
        ]
    )
    if summary.artifact_checks:
        lines.append(
            "| Artifact reload, feature order, prediction shape, valid probabilities, missing "
            "values, column-order invariance, match with cached predictions | "
            f"{sum(c.passed for c in summary.artifact_checks)}/{len(summary.artifact_checks)} "
            "artifacts passed |"
        )
    if summary.reproducibility_checks:
        max_diff = max(c.max_abs_difference for c in summary.reproducibility_checks)
        lines.append(
            "| Repeated training with identical configuration | "
            f"{sum(c.reproducible for c in summary.reproducibility_checks)}/"
            f"{len(summary.reproducibility_checks)} identical "
            f"(max abs. difference = {max_diff:.2e}) |"
        )
    lines.extend(
        [
            "| Test suite, Ruff, Mypy | `pytest`, `ruff check src tests scripts`, `mypy src` |",
            "",
            "## 11. Locked Evaluation Status",
            "",
        ]
    )
    if summary.locked_test_evaluated:
        locked = _load_json(config.results_dir / "locked_test_results.json")
        lines.extend(
            [
                "> **The primary 2021-22 test split was evaluated in this run** as a locked, "
                "one-off evaluation of validation-selected models (logged in "
                "`reports/models/locked_evaluation_log.jsonl`). These results must not be used "
                "for tuning.",
                "",
                "| Model | Target | Test score |",
                "| :--- | :--- | ---: |",
            ]
        )
        for r in locked.get("results", []):
            metric = "log_loss" if r["task"] == "classification" else "mae"
            lines.append(
                f"| `{r['model_name']}` | `{r['target']}` | "
                f"{_f(r['metrics'][metric], 4)} {metric} |"
            )
    else:
        lines.append(
            "The primary 2021-22 test split and the partial 2022-23 holdout were **not "
            "evaluated**. All model selection in this report used validation data only; both "
            "locked splits remain available for an unbiased final assessment."
        )

    lines.extend(
        [
            "",
            "## 12. Limitations",
            "",
            "- **Distribution shift**: the validation seasons are unusual (COVID-shortened season, "
            "bubble playoffs, limited crowds), so validation rankings may not transfer fully to "
            "normal seasons.",
            "- **Optimistic tuned scores**: penalties and boosting rounds were selected on "
            "validation; differences between top models are often within bootstrap noise.",
            "- **No player-level availability**: injuries, rest decisions, and lineup changes are "
            "not represented, which caps achievable accuracy for single games.",
            "- **Era-level features**: absolute scoring features carry the training era's scoring "
            "level; there is no explicit league-environment or recency weighting.",
            "- **Independent targets**: home, away, total, and margin models are trained "
            "separately, so their predictions are not guaranteed to form a consistent scoreline.",
            "- **Limited tuning**: Random Forest and XGBoost use single baseline configurations.",
            "",
            "## 13. Recommended Candidates for Week 4",
            "",
            "1. **XGBoost and Random Forest for `point_margin` and `home_win`** — the strongest "
            "relative-outcome models; tune them with chronological (rolling-origin) validation.",
            "2. **Regularized linear models for score-level targets** — Ridge/Lasso currently lead "
            "`home_pts`, `away_pts`, and `total_pts`; keep them as strong, interpretable "
            "references.",
            "3. **Probability calibration** — fit Platt or isotonic calibration on validation "
            "predictions for the `home_win` classifiers and track Brier score / reliability.",
            "4. **Drift-aware features** — add league-average-to-date scoring, season-relative "
            "(z-scored) team metrics, and recency-weighted training to reduce era bias.",
            "5. **Consistent scoreline modeling** — predict `point_margin` and `total_pts` jointly "
            "and derive home/away scores, or use multi-output models.",
            "6. **Final locked evaluation** — after Week 4 selection, evaluate the chosen models "
            "once on the 2021-22 test split and report the partial 2022-23 holdout separately.",
            "",
        ]
    )
    return "\n".join(lines)


def write_week3_report(
    summary: ModelingPipelineSummary, data: ModelingData, config: ModelingPipelineConfig
) -> Path:
    """Write ``<reports>/week-3-modeling-report.md``."""
    path = config.reports_dir / "week-3-modeling-report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_week3_report(summary, data, config), encoding="utf-8")
    return path
