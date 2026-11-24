"""Unified model comparison and evaluation report generator (Issue #23).

Inputs are the persisted ``reports/models/<family>_results.json`` files and the
``artifacts/models/<family>/predictions.parquet`` caches written by every family run. Outputs:

* ``reports/model_comparison.csv`` — every (model, target, split) with all metrics.
* ``reports/model_comparison.json`` — rankings, best models, paired bootstrap tests, metadata.
* ``reports/model_evaluation_report.md`` — human-readable comparison and error analysis.
* ``reports/figures/models/*.png`` — comparison, actual-vs-predicted, residual, calibration,
  and per-season error plots.

Only development splits (train / validation) are analyzed; the locked splits are refused.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arcora.evaluation.comparison import (
    BASELINE_FAMILY,
    PairedComparison,
    best_models,
    best_per_family,
    comparison_table,
    paired_bootstrap,
    rank_models,
)
from arcora.evaluation.error_analysis import (
    attach_metadata,
    classification_error_analysis,
    errors_by_season_across_splits,
    guard_locked_split,
    load_team_abbreviations,
    regression_error_analysis,
)
from arcora.models.base import ModelResult, load_results, to_jsonable, utc_timestamp
from arcora.models.data import ModelingData
from arcora.models.modeling_strategy import (
    CLASSIFICATION_TARGETS,
    MODEL_TARGETS,
    REGRESSION_TARGETS,
)
from arcora.models.splits import SPLIT_DEFINITIONS_BY_NAME, TRAIN_SPLIT, VALIDATION_SPLIT

DEFAULT_FAMILY_ORDER: tuple[str, ...] = ("baseline", "linear", "random_forest", "xgboost")

# Validated categorical palette (light mode), assigned to families in fixed order.
FAMILY_COLORS: dict[str, str] = {
    "baseline": "#2a78d6",
    "linear": "#eb6834",
    "random_forest": "#1baf7a",
    "xgboost": "#eda100",
}
TEXT_SECONDARY = "#5f5e5a"
GRID_COLOR = "#e4e2dc"


def load_all_results(
    results_dir: Path, families: Sequence[str] = DEFAULT_FAMILY_ORDER
) -> list[ModelResult]:
    """Load every persisted family results file that exists."""
    results: list[ModelResult] = []
    for family in families:
        path = results_dir / f"{family}_results.json"
        if path.is_file():
            results.extend(load_results(path))
    if not results:
        raise FileNotFoundError(f"No model results found in {results_dir}")
    return results


def load_all_predictions(
    models_dir: Path, families: Sequence[str] = DEFAULT_FAMILY_ORDER
) -> pd.DataFrame:
    """Concatenate cached long-format predictions for every family."""
    frames = [
        pd.read_parquet(models_dir / family / "predictions.parquet")
        for family in families
        if (models_dir / family / "predictions.parquet").is_file()
    ]
    if not frames:
        raise FileNotFoundError(f"No prediction caches found in {models_dir}")
    return pd.concat(frames, ignore_index=True)


def _style(ax: Any) -> None:
    ax.grid(axis="both", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    return path


def plot_loss_reduction_vs_baseline(family_best: pd.DataFrame, path: Path) -> Path:
    """Grouped bars: relative loss reduction vs the best naive baseline (positive = better)."""
    families = [f for f in DEFAULT_FAMILY_ORDER if f != BASELINE_FAMILY]
    families = [f for f in families if f in set(family_best["model_family"])]
    targets = [t for t in MODEL_TARGETS if t in set(family_best["target"])]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.8 / max(len(families), 1)
    x = np.arange(len(targets))
    for i, family in enumerate(families):
        sub = family_best[family_best["model_family"] == family].set_index("target")
        values = [float(sub["relative_loss_reduction_pct"].get(t, np.nan)) for t in targets]
        ax.bar(
            x + (i - (len(families) - 1) / 2) * width,
            values,
            width=width * 0.92,
            color=FAMILY_COLORS.get(family, "#6250d6"),
            label=family,
        )
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1)
    ax.set_xticks(x, [f"{t}\n({'log loss' if t == 'home_win' else 'MAE'})" for t in targets])
    ax.set_ylabel("relative_loss_reduction_pct (%)", color=TEXT_SECONDARY)
    ax.set_title(
        "Validation: (baseline loss − model loss) / baseline loss  (higher is better)",
        fontsize=11,
    )
    ax.legend(title="Model family", frameon=False)
    _style(ax)
    return _save(fig, path)


def plot_actual_vs_predicted(frame: pd.DataFrame, best: pd.DataFrame, path: Path) -> Path:
    """2×2 hexbin grid of actual vs predicted for the best model per regression target."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, target in zip(axes.ravel(), REGRESSION_TARGETS, strict=True):
        row = best[best["target"] == target]
        if row.empty:
            ax.set_visible(False)
            continue
        model_id = str(row["model_id"].iloc[0])
        sub = frame[(frame["model_id"] == model_id) & (frame["split"] == VALIDATION_SPLIT)]
        ax.hexbin(sub["y_true"], sub["y_pred"], gridsize=35, cmap="Blues", mincnt=1)
        lo = float(min(sub["y_true"].min(), sub["y_pred"].min()))
        hi = float(max(sub["y_true"].max(), sub["y_pred"].max()))
        ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.set_title(f"{target} — {row['model_name'].iloc[0]}", fontsize=10)
        ax.set_xlabel("Actual", color=TEXT_SECONDARY)
        ax.set_ylabel("Predicted", color=TEXT_SECONDARY)
        _style(ax)
    fig.suptitle("Actual vs predicted (validation, best model per target)")
    return _save(fig, path)


def plot_residual_distributions(frame: pd.DataFrame, best: pd.DataFrame, path: Path) -> Path:
    """Residual histograms for the best model per regression target."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, target in zip(axes.ravel(), REGRESSION_TARGETS, strict=True):
        row = best[best["target"] == target]
        if row.empty:
            ax.set_visible(False)
            continue
        model_id = str(row["model_id"].iloc[0])
        sub = frame[(frame["model_id"] == model_id) & (frame["split"] == VALIDATION_SPLIT)]
        residual = sub["y_pred"] - sub["y_true"]
        ax.hist(residual, bins=40, color=FAMILY_COLORS["baseline"], edgecolor="white")
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.axvline(float(residual.mean()), color="#e34948", linewidth=2, label="mean residual")
        ax.set_title(f"{target} — {row['model_name'].iloc[0]}", fontsize=10)
        ax.set_xlabel("Predicted − actual", color=TEXT_SECONDARY)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.suptitle("Residual distributions (validation)")
    return _save(fig, path)


def plot_calibration(frame: pd.DataFrame, family_best: pd.DataFrame, path: Path) -> Path:
    """Reliability curves for the best home-win classifier of each family."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle="--", linewidth=1, label="perfect")
    clf = family_best[family_best["target"].isin(CLASSIFICATION_TARGETS)]
    bins = np.linspace(0, 1, 11).tolist()
    for row in clf.itertuples():
        sub = frame[(frame["model_id"] == row.model_id) & (frame["split"] == VALIDATION_SPLIT)]
        if sub["y_pred"].nunique() <= 2:
            continue  # degenerate constant / 0-1 probabilities carry no calibration curve
        binned = pd.cut(sub["y_pred"], bins=bins, include_lowest=True)
        grouped = sub.groupby(binned, observed=True).agg(p=("y_pred", "mean"), o=("y_true", "mean"))
        ax.plot(
            grouped["p"],
            grouped["o"],
            marker="o",
            markersize=8,
            linewidth=2,
            color=FAMILY_COLORS.get(str(row.model_family), "#6250d6"),
            label=f"{row.model_family}: {row.model_name}",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted home-win probability", color=TEXT_SECONDARY)
    ax.set_ylabel("Observed home-win rate", color=TEXT_SECONDARY)
    ax.set_title("Calibration — home_win (validation)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _style(ax)
    return _save(fig, path)


def plot_error_by_season(season_errors: dict[str, pd.DataFrame], path: Path) -> Path:
    """Per-season primary-metric error across train and validation seasons."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (label, metric, frame) in zip(
        axes,
        [
            ("point_margin", "mae", season_errors.get("point_margin")),
            ("home_win", "log_loss", season_errors.get("home_win")),
        ],
        strict=True,
    ):
        if frame is None or frame.empty:
            ax.set_visible(False)
            continue
        ax.plot(
            frame["season_label"],
            frame[metric],
            marker="o",
            markersize=8,
            linewidth=2,
            color=FAMILY_COLORS["baseline"],
        )
        train_labels = set(SPLIT_DEFINITIONS_BY_NAME[TRAIN_SPLIT].season_labels)
        boundary = float(frame["season_label"].isin(train_labels).sum()) - 0.5
        ax.axvline(boundary, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.text(
            boundary + 0.1,
            float(frame[metric].max()),
            "validation →",
            color=TEXT_SECONDARY,
            fontsize=8,
        )
        ax.set_title(f"{label}: {metric} by season (best model)", fontsize=10)
        ax.tick_params(axis="x", rotation=60)
        _style(ax)
    return _save(fig, path)


def _md_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 3) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| "
        + " | ".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else ":---" for c in columns)
        + " |",
    ]
    for record in frame.loc[:, list(columns)].to_dict("records"):
        cells = []
        for c in columns:
            v = record[c]
            if isinstance(v, float | np.floating):
                cells.append("—" if np.isnan(v) else f"{float(v):.{digits}f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.{digits}f}"


def _baseline_bootstrap_record(comparison: PairedComparison) -> dict[str, Any]:
    """Bootstrap result vs the best baseline, with the point estimate as ``delta_vs_baseline``.

    The mean per-game loss difference equals ``model loss - baseline loss`` exactly (the mean of
    per-game absolute errors is MAE and the mean of per-game log losses is log loss).
    """
    record = comparison.to_dict()
    record["delta_vs_baseline"] = record.pop("mean_difference")
    record["baseline_model"] = record.pop("model_b")
    record["model"] = record.pop("model_a")
    return record


def render_evaluation_markdown(
    ranked: pd.DataFrame,
    family_best: pd.DataFrame,
    paired: Sequence[PairedComparison],
    regression_analyses: dict[str, dict[str, Any]],
    classification_analysis: dict[str, Any] | None,
    season_errors: dict[str, pd.DataFrame],
    figures: dict[str, str],
    split: str,
) -> str:
    """Render ``model_evaluation_report.md``."""
    lines = [
        "# Arcora Week 3 — Model Evaluation and Comparison",
        "",
        f"All models are compared on the **{split}** split (2019-20 to 2020-21) with the shared "
        "metric implementation in `src/arcora/evaluation/metrics.py`. Every result carries the "
        "same dataset, split, and feature fingerprints (verified before comparison). The locked "
        "2021-22 test set and partial 2022-23 holdout are not analyzed here.",
        "",
        "## Best model per target",
        "",
        "`delta_vs_baseline` = model loss − best baseline loss, in metric units (negative is "
        "better). `relative_loss_reduction_pct` = (baseline loss − model loss) / baseline loss "
        "× 100, with the best baseline's MAE or log loss as denominator (positive is better).",
        "",
        "| Target | Metric | Best model | Family | Score | Best baseline | `delta_vs_baseline` | "
        "`relative_loss_reduction_pct` |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in best_models(ranked).to_dict("records"):
        lines.append(
            f"| `{row['target']}` | {row['primary_metric']} | `{row['model_name']}` | "
            f"{row['model_family']} | {_fmt(row['primary_score'])} | "
            f"{_fmt(row['best_baseline_score'])} | {float(row['delta_vs_baseline']):+.4f} | "
            f"{float(row['relative_loss_reduction_pct']):+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Paired bootstrap: best model vs best naive baseline",
            "",
            "`delta_vs_baseline` is the mean per-game loss difference (best model − baseline; "
            "negative is better). It equals the difference in MAE or log loss shown above. The "
            "95% confidence interval comes from a paired bootstrap over validation games "
            "(2,000 resamples); *Significant* means the interval excludes zero.",
            "",
            "| Target | Metric | Best model | Baseline | `delta_vs_baseline` | "
            "95% CI (paired bootstrap) | Significant |",
            "| :--- | :--- | :--- | :--- | ---: | :--- | :---: |",
        ]
    )
    for p in paired:
        lines.append(
            f"| `{p.target}` | {p.metric} | `{p.model_a}` | `{p.model_b}` | "
            f"{p.mean_difference:+.4f} | [{p.ci_low:+.4f}, {p.ci_high:+.4f}] | "
            f"{'Yes' if p.significant else 'No'} |"
        )

    if "loss_reduction" in figures:
        lines.extend(
            [
                "",
                "## Relative loss reduction vs best baseline (chart)",
                "",
                f"![relative_loss_reduction_pct by family and target]({figures['loss_reduction']})",
            ]
        )
    lines.extend(["", "## Regression comparison (validation)", ""])
    reg = ranked[ranked["task"] == "regression"].copy()
    lines.extend(
        _md_table(
            reg,
            [
                "target",
                "rank",
                "model_name",
                "model_family",
                "mae",
                "rmse",
                "r2",
                "delta_vs_baseline",
                "relative_loss_reduction_pct",
            ],
        )
    )
    lines.extend(["", "## Classification comparison — `home_win` (validation)", ""])
    clf = ranked[ranked["task"] == "classification"].copy()
    lines.extend(
        _md_table(
            clf,
            [
                "rank",
                "model_name",
                "model_family",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "roc_auc",
                "log_loss",
                "brier",
                "delta_vs_baseline",
                "relative_loss_reduction_pct",
            ],
            4,
        )
    )
    lines.extend(
        [
            "",
            "## Best model per family",
            "",
            *_md_table(
                family_best,
                ["target", "model_family", "model_name", "primary_metric", "primary_score", "rank"],
                4,
            ),
            "",
            "## Error analysis — regression (validation)",
            "",
        ]
    )
    for key in ("actual_vs_predicted", "residuals"):
        if key in figures:
            lines.extend([f"![{key}]({figures[key]})", ""])
    for target, analysis in regression_analyses.items():
        res = analysis["residuals"]
        lines.extend(
            [
                f"### `{target}` — `{analysis['model_id']}`",
                "",
                f"- Mean residual (bias): **{res['mean_residual']:+.2f}**, residual std "
                f"{res['std_residual']:.2f}; 90% of residuals in "
                f"[{res['p05']:+.1f}, {res['p95']:+.1f}].",
                f"- Within ±5 points: {100 * res['share_within_5']:.1f}%; within ±10: "
                f"{100 * res['share_within_10']:.1f}%.",
                "",
                "By season:",
                "",
                *_md_table(
                    analysis["by_season"], ["season_label", "n_games", "mae", "rmse", "bias"]
                ),
                "",
                "By game type:",
                "",
                *_md_table(
                    analysis["by_game_type"], ["game_type", "n_games", "mae", "rmse", "bias"]
                ),
                "",
                "Home teams with the highest / lowest MAE:",
                "",
            ]
        )
        by_home = analysis["by_home_team"].sort_values("mae", ascending=False)
        extremes = pd.concat([by_home.head(3), by_home.tail(3)])
        lines.extend(_md_table(extremes, ["home_team", "n_games", "mae", "bias"]))
        by_away = analysis["by_away_team"].sort_values("mae", ascending=False)
        lines.extend(["", "Away teams with the highest / lowest MAE:", ""])
        lines.extend(
            _md_table(
                pd.concat([by_away.head(3), by_away.tail(3)]),
                ["away_team", "n_games", "mae", "bias"],
            )
        )
        lines.extend(["", "Largest errors:", ""])
        lines.extend(
            _md_table(
                analysis["largest_errors"].head(5),
                [
                    "game_date",
                    "game_type",
                    "home_team",
                    "away_team",
                    "y_true",
                    "y_pred",
                    "residual",
                ],
                1,
            )
        )
        lines.append("")

    if classification_analysis is not None:
        ca = classification_analysis
        cm = ca["confusion"]
        lines.extend(
            [
                f"## Error analysis — classification (`{ca['model_id']}`, validation)",
                "",
                "| | Predicted home win | Predicted away win |",
                "| :--- | ---: | ---: |",
                f"| **Actual home win** | {cm['true_positive']} (TP) | "
                f"{cm['false_negative']} (FN) |",
                f"| **Actual away win** | {cm['false_positive']} (FP) | "
                f"{cm['true_negative']} (TN) |",
                "",
                f"False positives (predicted home win, away team won): **{cm['false_positive']}**; "
                f"false negatives: **{cm['false_negative']}**. "
                + (
                    "The model leans toward predicting home wins (more false positives than "
                    "false negatives); compare the observed and predicted home-win rates by "
                    "season below."
                    if cm["false_positive"] > cm["false_negative"]
                    else "Errors are not skewed toward predicting home wins."
                ),
                "",
            ]
        )
        if "calibration" in figures:
            lines.extend([f"![Calibration]({figures['calibration']})", ""])
        lines.extend(["By season:", ""])
        lines.extend(
            _md_table(
                ca["by_season"],
                [
                    "season_label",
                    "n_games",
                    "home_win_rate",
                    "mean_predicted_prob",
                    "accuracy",
                    "log_loss",
                    "brier",
                ],
            )
        )
        lines.extend(["", "By game type:", ""])
        lines.extend(
            _md_table(
                ca["by_game_type"],
                [
                    "game_type",
                    "n_games",
                    "home_win_rate",
                    "mean_predicted_prob",
                    "accuracy",
                    "log_loss",
                ],
            )
        )
        lines.extend(["", "Most confident false positives:", ""])
        lines.extend(
            _md_table(
                ca["most_confident_false_positives"].head(5),
                ["game_date", "game_type", "home_team", "away_team", "y_pred"],
            )
        )
        lines.append("")

    if season_errors:
        lines.extend(
            [
                "## Error by season across development splits",
                "",
                "Per-season error of the best model over training and validation seasons. Training "
                "seasons are in-sample, so their error is optimistic; the jump at the validation "
                "boundary reflects both out-of-sample error and the 2019-21 scoring and "
                "home-advantage shift (shortened 2019-20 season, bubble playoffs, limited crowds "
                "in 2020-21).",
                "",
            ]
        )
        if "error_by_season" in figures:
            lines.extend([f"![Error by season]({figures['error_by_season']})", ""])

    lines.extend(
        [
            "## Reproducibility and rules",
            "",
            "- One metric implementation for every model (`compute_metrics`); undefined metrics "
            "are stored as `null` and listed per result.",
            "- `validate_comparability()` refuses to compare results scored on different datasets, "
            "splits, or feature lists.",
            "- Error analysis is restricted to development splits; locked splits raise "
            "`PermissionError` unless explicitly allowed for a documented final evaluation.",
            f"- Generated: {utc_timestamp()}.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_evaluation_report(
    data: ModelingData,
    results_dir: Path,
    models_dir: Path,
    reports_dir: Path,
    figures_dir: Path | None = None,
    teams_path: Path | None = None,
    split: str = VALIDATION_SPLIT,
    families: Sequence[str] = DEFAULT_FAMILY_ORDER,
    n_bootstrap: int = 2000,
) -> dict[str, Path]:
    """Build comparison tables, error analysis, figures, and the evaluation report."""
    guard_locked_split(split)
    figures_dir = figures_dir or reports_dir / "figures" / "models"
    results = load_all_results(results_dir, families)
    table = comparison_table(results)
    ranked = rank_models(table, split)
    family_best = best_per_family(ranked)
    best = best_models(ranked)

    predictions = load_all_predictions(models_dir, families)
    frame = attach_metadata(predictions, data, load_team_abbreviations(teams_path))

    paired: list[PairedComparison] = []
    for row in best.to_dict("records"):
        target = str(row["target"])
        base = ranked[(ranked["target"] == target) & (ranked["model_family"] == BASELINE_FAMILY)]
        if base.empty or row["model_family"] == BASELINE_FAMILY:
            continue
        paired.append(
            paired_bootstrap(
                predictions,
                str(row["model_id"]),
                str(base.sort_values("rank")["model_id"].iloc[0]),
                target,
                split,
                n_bootstrap=n_bootstrap,
            )
        )

    regression_analyses = {
        str(row["target"]): regression_error_analysis(frame, str(row["model_id"]), split)
        for row in best.to_dict("records")
        if row["target"] in REGRESSION_TARGETS
    }
    clf_rows = best[best["target"].isin(CLASSIFICATION_TARGETS)]
    classification_analysis = (
        classification_error_analysis(frame, str(clf_rows["model_id"].iloc[0]), split)
        if not clf_rows.empty
        else None
    )
    season_errors = {
        str(row["target"]): errors_by_season_across_splits(
            frame, str(row["model_id"]), (TRAIN_SPLIT, VALIDATION_SPLIT)
        )
        for row in best.to_dict("records")
        if row["target"] in ("point_margin", "home_win")
    }

    written: dict[str, Path] = {}
    figure_paths = {
        "loss_reduction": plot_loss_reduction_vs_baseline(
            family_best, figures_dir / "model_loss_reduction_vs_baseline.png"
        ),
        "actual_vs_predicted": plot_actual_vs_predicted(
            frame, best, figures_dir / "actual_vs_predicted.png"
        ),
        "residuals": plot_residual_distributions(
            frame, best, figures_dir / "residual_distributions.png"
        ),
        "calibration": plot_calibration(
            frame, family_best, figures_dir / "calibration_home_win.png"
        ),
        "error_by_season": plot_error_by_season(season_errors, figures_dir / "error_by_season.png"),
    }
    figure_refs = {}
    for key, path in figure_paths.items():
        written[f"figure:{key}"] = path
        try:
            figure_refs[key] = path.relative_to(reports_dir).as_posix()
        except ValueError:
            figure_refs[key] = path.as_posix()

    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "model_comparison.csv"
    baseline_cols = ["best_baseline_score", "delta_vs_baseline", "relative_loss_reduction_pct"]
    csv_frame = table.merge(
        ranked.loc[:, ["model_id", "split", *baseline_cols]], on=["model_id", "split"], how="left"
    )
    csv_frame.drop(columns=["dataset_fingerprint", "feature_fingerprint"]).to_csv(
        csv_path, index=False
    )
    written["comparison_csv"] = csv_path

    sample = results[0].reproducibility
    payload = {
        "generated_at": utc_timestamp(),
        "evaluation_split": split,
        "dataset_fingerprint": sample["dataset_fingerprint"],
        "feature_fingerprint": sample["feature_fingerprint"],
        "split_game_id_sha256": data.split_manifest["splits"][split]["game_id_sha256"],
        "n_models": int(table["model_id"].nunique()),
        "families": sorted(table["model_family"].unique().tolist()),
        "ranking": ranked.drop(
            columns=["dataset_fingerprint", "feature_fingerprint", "split_game_id_sha256"]
        ).to_dict("records"),
        "best_models": best.loc[
            :, ["target", "model_id", "model_family", "primary_metric", "primary_score"]
        ].to_dict("records"),
        "paired_bootstrap_vs_baseline": [_baseline_bootstrap_record(p) for p in paired],
        "regression_residuals": {t: a["residuals"] for t, a in regression_analyses.items()},
        "classification_confusion": (
            classification_analysis["confusion"] if classification_analysis else None
        ),
    }
    json_path = reports_dir / "model_comparison.json"
    json_path.write_text(json.dumps(to_jsonable(payload), indent=2, default=str) + "\n", "utf-8")
    written["comparison_json"] = json_path

    md_path = reports_dir / "model_evaluation_report.md"
    md_path.write_text(
        render_evaluation_markdown(
            ranked,
            family_best,
            paired,
            regression_analyses,
            classification_analysis,
            season_errors,
            figure_refs,
            split,
        ),
        encoding="utf-8",
    )
    written["evaluation_report"] = md_path
    return written
