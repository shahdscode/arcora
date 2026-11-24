"""Exploratory feature-importance summaries for tree-based models (Issues #21, #22).

Two complementary views are produced:

* **Model-native importance** (impurity decrease for Random Forest, gain for XGBoost), computed
  from the training fit, per feature and aggregated by Week 2 feature category.
* **Grouped permutation importance** on the validation split: every feature in a category is
  permuted together and the increase in the primary metric (MAE or log loss) is recorded.
  Permuting whole groups avoids the dilution that single-feature permutation suffers when
  home, away, and difference features are strongly correlated.

Importance is descriptive of what a model relies on, not evidence of causation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arcora.evaluation.metrics import compute_metrics
from arcora.features.strategy import FEATURE_CATEGORIES, FEATURE_SPECIFICATIONS
from arcora.models.base import TrainedModel
from arcora.models.data import SplitData
from arcora.models.modeling_strategy import PRIMARY_METRIC, RANDOM_SEED

CATEGORY_BY_FEATURE: dict[str, str] = {f.name: f.category for f in FEATURE_SPECIFICATIONS}


def _final_estimator(model: TrainedModel) -> Any:
    est = model.estimator
    return est.named_steps["model"] if hasattr(est, "named_steps") else est


def native_importance_frame(model: TrainedModel) -> pd.DataFrame:
    """Return the model's native per-feature importance, normalized to sum to 1."""
    values = np.asarray(_final_estimator(model).feature_importances_, dtype="float64")
    total = values.sum()
    normalized = values / total if total > 0 else values
    frame = pd.DataFrame(
        {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "target": model.target,
            "feature": list(model.feature_names),
            "category": [CATEGORY_BY_FEATURE.get(f, "Other") for f in model.feature_names],
            "importance": normalized,
        }
    )
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


def category_importance(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-feature importance by feature category."""
    grouped = (
        frame.groupby(["model_id", "model_name", "target", "category"], as_index=False)
        .agg(importance=("importance", "sum"), n_features=("feature", "size"))
        .sort_values(["model_id", "importance"], ascending=[True, False])
    )
    return grouped.reset_index(drop=True)


def grouped_permutation_importance(
    model: TrainedModel,
    split: SplitData,
    n_repeats: int = 5,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Increase in the primary metric when each feature category is permuted together."""
    metric = PRIMARY_METRIC[model.task]
    y = split.target(model.target).to_numpy(dtype="float64")
    baseline = compute_metrics(model.task, y, model.predict(split.X))[metric]
    rng = np.random.default_rng(random_state)
    rows = []
    for category in FEATURE_CATEGORIES:
        cols = [f for f in model.feature_names if CATEGORY_BY_FEATURE.get(f) == category]
        if not cols:
            continue
        deltas = []
        for _ in range(n_repeats):
            permuted = split.X.copy()
            order = rng.permutation(len(permuted))
            permuted[cols] = permuted[cols].to_numpy()[order]
            score = compute_metrics(model.task, y, model.predict(permuted))[metric]
            if score is not None and baseline is not None:
                deltas.append(score - baseline)
        rows.append(
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "target": model.target,
                "category": category,
                "n_features": len(cols),
                "metric": metric,
                "baseline_score": baseline,
                "mean_increase": float(np.mean(deltas)) if deltas else None,
                "std_increase": float(np.std(deltas)) if deltas else None,
                "n_repeats": n_repeats,
                "split": split.name,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["model_id", "mean_increase"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot_category_importance(categories: pd.DataFrame, title: str, path: Path) -> Path:
    """Plot category-level native importance for each model as grouped horizontal bars."""
    pivot = categories.pivot_table(
        index="category", columns="target", values="importance", aggfunc="sum"
    ).fillna(0.0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot.barh(ax=ax, width=0.8)
    ax.set_xlabel("Share of total importance")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(title="Target", loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def render_importance_markdown(
    title: str,
    importance_label: str,
    features: pd.DataFrame,
    categories: pd.DataFrame,
    permutation: pd.DataFrame,
    highlight_models: Sequence[str],
    figure_relpath: str | None = None,
    top_n: int = 15,
) -> str:
    """Render the exploratory feature-importance report."""
    lines = [
        f"# {title}",
        "",
        "> **Exploratory analysis.** Importance shows what the fitted models rely on; it is not "
        "evidence of causation. Many features are strongly correlated (home, away, and "
        "difference variants), so importance is shared across related columns.",
        "",
        f"- **{importance_label}** is computed from the training fit and normalized to sum to 1 "
        "per model.",
        "- **Grouped permutation importance** permutes every feature in a category together on "
        "the validation split and reports the mean increase in the primary metric (MAE for "
        "regression, log loss for classification) over repeated shuffles.",
        "",
    ]
    if figure_relpath:
        lines.extend([f"![Category importance]({figure_relpath})", ""])

    lines.extend(["## Category share of native importance", ""])
    pivot = categories.pivot_table(
        index="category", columns="target", values="importance", aggfunc="sum"
    ).fillna(0.0)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    targets = list(pivot.columns)
    lines.append("| Category | " + " | ".join(f"`{t}`" for t in targets) + " |")
    lines.append("| :--- | " + " | ".join("---:" for _ in targets) + " |")
    for category in pivot.index:
        values = pivot.loc[category]
        lines.append(f"| {category} | " + " | ".join(f"{values[t]:.3f}" for t in targets) + " |")

    lines.extend(
        [
            "",
            "## Grouped permutation importance (validation)",
            "",
            "| Model | Category | Features | Metric | Mean increase | Std |",
            "| :--- | :--- | ---: | :--- | ---: | ---: |",
        ]
    )
    for perm in permutation.to_dict("records"):
        mean_inc, std_inc = perm["mean_increase"], perm["std_increase"]
        mean = "—" if mean_inc is None else f"{float(mean_inc):+.4f}"
        std = "—" if std_inc is None else f"{float(std_inc):.4f}"
        lines.append(
            f"| `{perm['model_id']}` | {perm['category']} | {perm['n_features']} | "
            f"{perm['metric']} | {mean} | {std} |"
        )

    for model_id in highlight_models:
        top = features[features["model_id"] == model_id].head(top_n)
        if top.empty:
            continue
        lines.extend(
            [
                "",
                f"## Top {top_n} features — `{model_id}`",
                "",
                "| Rank | Feature | Category | Importance |",
                "| ---: | :--- | :--- | ---: |",
            ]
        )
        for feat in top.itertuples():
            lines.append(
                f"| {feat.rank} | `{feat.feature}` | {feat.category} | {feat.importance:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_importance_report(
    models: Sequence[TrainedModel],
    validation: SplitData,
    family: str,
    title: str,
    importance_label: str,
    results_dir: Path,
    figures_dir: Path,
    n_repeats: int = 5,
) -> dict[str, Path]:
    """Compute and write feature-importance CSVs, a figure, and a Markdown report."""
    features = pd.concat([native_importance_frame(m) for m in models], ignore_index=True)
    categories = category_importance(features)
    permutation = pd.concat(
        [grouped_permutation_importance(m, validation, n_repeats) for m in models],
        ignore_index=True,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    feat_path = results_dir / f"{family}_feature_importance.csv"
    features.to_csv(feat_path, index=False)
    perm_path = results_dir / f"{family}_permutation_importance.csv"
    permutation.to_csv(perm_path, index=False)

    fig_path = plot_category_importance(
        categories,
        f"{title}: category share of {importance_label.lower()}",
        figures_dir / f"{family}_category_importance.png",
    )
    try:
        fig_rel = fig_path.relative_to(results_dir.parent).as_posix()
        fig_rel = f"../{fig_rel}"
    except ValueError:
        fig_rel = fig_path.as_posix()

    highlight = [m.model_id for m in models if m.target in ("point_margin", "home_win")]
    md_path = results_dir / f"{family}_feature_importance.md"
    md_path.write_text(
        render_importance_markdown(
            title, importance_label, features, categories, permutation, highlight, fig_rel
        ),
        encoding="utf-8",
    )
    return {
        "feature_importance_csv": feat_path,
        "permutation_importance_csv": perm_path,
        "feature_importance_figure": fig_path,
        "feature_importance_report": md_path,
    }
