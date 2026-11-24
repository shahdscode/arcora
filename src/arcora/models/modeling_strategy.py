"""Model targets, input-feature rules, metric definitions, and reproducibility (Issue #18).

Everything here is derived from the Week 2 feature strategy in
:mod:`arcora.features.strategy`; feature, metadata, target, and forbidden-column definitions
are imported rather than redefined so the model input allowlist can never drift from the
feature catalog.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from arcora.features.strategy import (
    FEATURE_COLUMN_NAMES,
    FEATURE_SPECIFICATIONS,
    FORBIDDEN_POSTGAME_COLUMNS,
    METADATA_COLUMNS,
    PREDICTION_TARGETS,
    TARGET_COLUMNS,
)

RANDOM_SEED: int = 42
CLASSIFICATION_THRESHOLD: float = 0.5
PROBABILITY_CLIP_EPSILON: float = 1e-15

REGRESSION_TASK: str = "regression"
CLASSIFICATION_TASK: str = "classification"

REGRESSION_TARGETS: tuple[str, ...] = ("home_pts", "away_pts", "total_pts", "point_margin")
CLASSIFICATION_TARGETS: tuple[str, ...] = ("home_win",)
MODEL_TARGETS: tuple[str, ...] = REGRESSION_TARGETS + CLASSIFICATION_TARGETS

IDENTIFIER_COLUMNS: tuple[str, ...] = ("game_id", "home_team_id", "away_team_id")
EXCLUDED_INPUT_COLUMNS: frozenset[str] = frozenset(
    set(METADATA_COLUMNS)
    | set(TARGET_COLUMNS)
    | set(FORBIDDEN_POSTGAME_COLUMNS)
    | set(IDENTIFIER_COLUMNS)
)
MODEL_FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMN_NAMES

APPROVED_MISSING_INDICATORS: tuple[str, ...] = tuple(
    f.name for f in FEATURE_SPECIFICATIONS if "insufficient" in f.name or "season_opener" in f.name
)


@dataclass(frozen=True)
class ModelTargetSpec:
    """A prediction target together with its learning task."""

    name: str
    task: str
    formula: str
    description: str
    derived_from_scores: bool


_TARGET_SPEC_BY_NAME = {t.name: t for t in PREDICTION_TARGETS}

MODEL_TARGET_SPECS: tuple[ModelTargetSpec, ...] = tuple(
    ModelTargetSpec(
        name=name,
        task=CLASSIFICATION_TASK if name in CLASSIFICATION_TARGETS else REGRESSION_TASK,
        formula=_TARGET_SPEC_BY_NAME[name].formula,
        description=_TARGET_SPEC_BY_NAME[name].description,
        derived_from_scores=name in ("total_pts", "point_margin", "home_win"),
    )
    for name in MODEL_TARGETS
)
MODEL_TARGET_SPECS_BY_NAME: dict[str, ModelTargetSpec] = {t.name: t for t in MODEL_TARGET_SPECS}


def task_for_target(target: str) -> str:
    """Return the learning task (``regression`` or ``classification``) for a target."""
    if target not in MODEL_TARGET_SPECS_BY_NAME:
        raise KeyError(f"Unknown model target '{target}'. Expected one of {MODEL_TARGETS}.")
    return MODEL_TARGET_SPECS_BY_NAME[target].task


def derive_targets(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the five model targets from final scores and verify stored target columns.

    ``total_pts``, ``point_margin`` and ``home_win`` are recomputed from ``home_pts`` and
    ``away_pts``. When the dataset already stores those columns they must agree exactly,
    otherwise a ValueError is raised. ``home_win`` is returned as ``int64`` (0/1).
    """
    for col in ("home_pts", "away_pts"):
        if col not in dataset_df.columns:
            raise ValueError(f"Cannot derive targets: missing final-score column '{col}'")
        if dataset_df[col].isna().any():
            raise ValueError(f"Final-score column '{col}' contains missing values")

    home = dataset_df["home_pts"].astype("int64")
    away = dataset_df["away_pts"].astype("int64")
    derived = pd.DataFrame(
        {
            "home_pts": home,
            "away_pts": away,
            "total_pts": home + away,
            "point_margin": home - away,
            "home_win": (home > away).astype("int64"),
        },
        index=dataset_df.index,
    )
    for col in ("total_pts", "point_margin", "home_win"):
        if col in dataset_df.columns:
            stored = dataset_df[col].astype("int64")
            mismatches = int((stored != derived[col]).sum())
            if mismatches:
                raise ValueError(
                    f"Stored target '{col}' disagrees with final scores in {mismatches} rows"
                )
    return derived[list(MODEL_TARGETS)]


def validate_feature_schema(dataset_df: pd.DataFrame) -> None:
    """Verify a feature dataset contains every metadata, feature, and target column.

    Also confirms every approved feature is numeric or boolean and no forbidden post-game
    column is present. Raises ValueError describing every problem found.
    """
    problems: list[str] = []
    columns = set(dataset_df.columns)
    for label, required in (
        ("metadata", METADATA_COLUMNS),
        ("feature", MODEL_FEATURE_COLUMNS),
        ("target", TARGET_COLUMNS),
    ):
        missing = [c for c in required if c not in columns]
        if missing:
            problems.append(f"missing {label} columns: {missing[:10]} ({len(missing)} total)")

    forbidden = sorted(columns & set(FORBIDDEN_POSTGAME_COLUMNS))
    if forbidden:
        problems.append(f"forbidden post-game columns present: {forbidden}")

    known = set(METADATA_COLUMNS) | set(MODEL_FEATURE_COLUMNS) | set(TARGET_COLUMNS)
    unexpected = sorted(columns - known - set(FORBIDDEN_POSTGAME_COLUMNS))
    if unexpected:
        problems.append(f"unexpected columns not in the feature catalog: {unexpected[:10]}")

    non_numeric = [
        c
        for c in MODEL_FEATURE_COLUMNS
        if c in columns
        and not pd.api.types.is_numeric_dtype(dataset_df[c])
        and not pd.api.types.is_bool_dtype(dataset_df[c])
    ]
    if non_numeric:
        problems.append(f"non-numeric feature columns: {non_numeric}")

    if problems:
        raise ValueError("Feature dataset schema validation failed: " + "; ".join(problems))


def assert_feature_allowlist(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Verify a feature list is exactly the approved pregame allowlist, in canonical order."""
    names = tuple(feature_names)
    leaked = sorted(set(names) & EXCLUDED_INPUT_COLUMNS)
    if leaked:
        raise ValueError(f"Excluded columns found in model inputs: {leaked}")
    unknown = sorted(set(names) - set(MODEL_FEATURE_COLUMNS))
    if unknown:
        raise ValueError(f"Unapproved columns found in model inputs: {unknown}")
    if names != MODEL_FEATURE_COLUMNS:
        missing = [c for c in MODEL_FEATURE_COLUMNS if c not in names]
        raise ValueError(
            "Model inputs must match the approved feature allowlist in canonical order; "
            f"missing={missing[:10]}"
        )
    return names


def select_model_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Return the model input matrix ``X`` containing only the 242 approved pregame features.

    Columns follow the canonical Week 2 feature order and are cast to ``float64`` (boolean
    missing-value indicators become 0.0/1.0) so every estimator receives an identical matrix.
    Metadata, identifiers, targets, and post-game columns are never included.
    """
    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in dataset_df.columns]
    if missing:
        raise ValueError(f"Dataset is missing approved feature columns: {missing[:10]}")
    assert_feature_allowlist(MODEL_FEATURE_COLUMNS)
    return dataset_df.loc[:, list(MODEL_FEATURE_COLUMNS)].astype("float64")


def feature_list_fingerprint(feature_names: Iterable[str]) -> str:
    """Return an order-sensitive SHA-256 fingerprint of a feature list."""
    return hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetricSpec:
    """Definition of an evaluation metric."""

    name: str
    display_name: str
    task: str
    greater_is_better: bool
    input_type: str
    formula: str
    undefined_handling: str


REGRESSION_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="mae",
        display_name="MAE",
        task=REGRESSION_TASK,
        greater_is_better=False,
        input_type="point prediction",
        formula="mean(|y - y_hat|)",
        undefined_handling="Undefined (null) only for empty inputs.",
    ),
    MetricSpec(
        name="rmse",
        display_name="RMSE",
        task=REGRESSION_TASK,
        greater_is_better=False,
        input_type="point prediction",
        formula="sqrt(mean((y - y_hat)^2))",
        undefined_handling="Undefined (null) only for empty inputs.",
    ),
    MetricSpec(
        name="r2",
        display_name="R²",
        task=REGRESSION_TASK,
        greater_is_better=True,
        input_type="point prediction",
        formula="1 - SS_res / SS_tot",
        undefined_handling="Null when fewer than 2 rows or y is constant (SS_tot = 0).",
    ),
)

CLASSIFICATION_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="accuracy",
        display_name="Accuracy",
        task=CLASSIFICATION_TASK,
        greater_is_better=True,
        input_type="label at threshold 0.5",
        formula="mean(y == y_hat)",
        undefined_handling="Undefined (null) only for empty inputs.",
    ),
    MetricSpec(
        name="balanced_accuracy",
        display_name="Balanced accuracy",
        task=CLASSIFICATION_TASK,
        greater_is_better=True,
        input_type="label at threshold 0.5",
        formula="(TPR + TNR) / 2",
        undefined_handling="Null when y contains a single class.",
    ),
    MetricSpec(
        name="precision",
        display_name="Precision",
        task=CLASSIFICATION_TASK,
        greater_is_better=True,
        input_type="label at threshold 0.5",
        formula="TP / (TP + FP)",
        undefined_handling="Null when the model predicts no positives (TP + FP = 0).",
    ),
    MetricSpec(
        name="recall",
        display_name="Recall",
        task=CLASSIFICATION_TASK,
        greater_is_better=True,
        input_type="label at threshold 0.5",
        formula="TP / (TP + FN)",
        undefined_handling="Null when y contains no positives (TP + FN = 0).",
    ),
    MetricSpec(
        name="roc_auc",
        display_name="ROC-AUC",
        task=CLASSIFICATION_TASK,
        greater_is_better=True,
        input_type="probability",
        formula="P(score(pos) > score(neg))",
        undefined_handling="Null when y contains a single class.",
    ),
    MetricSpec(
        name="log_loss",
        display_name="Log loss",
        task=CLASSIFICATION_TASK,
        greater_is_better=False,
        input_type="probability",
        formula="-mean(y log p + (1 - y) log(1 - p)), p clipped to [1e-15, 1 - 1e-15]",
        undefined_handling="Always defined (labels fixed to {0, 1}).",
    ),
    MetricSpec(
        name="brier",
        display_name="Brier score",
        task=CLASSIFICATION_TASK,
        greater_is_better=False,
        input_type="probability",
        formula="mean((p - y)^2)",
        undefined_handling="Always defined for non-empty inputs.",
    ),
)

METRIC_SPECS: tuple[MetricSpec, ...] = REGRESSION_METRICS + CLASSIFICATION_METRICS
METRIC_SPECS_BY_NAME: dict[str, MetricSpec] = {m.name: m for m in METRIC_SPECS}
PRIMARY_METRIC: dict[str, str] = {REGRESSION_TASK: "mae", CLASSIFICATION_TASK: "log_loss"}

REPRODUCIBILITY_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    (
        "Fixed random seeds",
        f"Every stochastic estimator uses `random_state={RANDOM_SEED}` (`RANDOM_SEED`).",
    ),
    (
        "Consistent feature selection",
        "`select_model_features()` returns the 242-feature allowlist in canonical order; "
        "its SHA-256 fingerprint is stored with every model.",
    ),
    (
        "Persisted split manifest",
        "`data/splits/split_manifest.json` stores per-split counts, date ranges, and "
        "`game_id` fingerprints; reused manifests are verified before training.",
    ),
    (
        "Recorded model configurations",
        "Hyperparameters, seeds, and library versions are saved alongside each artifact.",
    ),
    (
        "Versioned model artifacts",
        "Artifacts are written under `artifacts/models/<family>/` with a JSON metadata "
        "sidecar recording the dataset fingerprint, feature fingerprint, and versions.",
    ),
    (
        "Consistent evaluation datasets",
        "All model comparisons use the same validation split, verified by its "
        "`game_id` fingerprint.",
    ),
)


# Split sizes quoted in the Week 2 feature quality report (commit 96933df). They were written
# into the report text as an illustrative example and were never computed from the dataset.
WEEK2_DOCUMENTED_SPLIT_COUNTS: dict[str, int] = {
    "train": 20_762,
    "validation": 2_282,
    "test": 1_327,
    "holdout": 477,
}


def render_split_count_reconciliation(manifest: Mapping[str, Any]) -> list[str]:
    """Markdown section reconciling Week 2 documented split sizes with actual counts."""
    splits: Mapping[str, Any] = manifest["splits"]
    lines = [
        "### Split-count reconciliation (Week 2 plan vs dataset)",
        "",
        "The split **definitions** (season ranges) are unchanged from the Week 2 plan. Only the "
        "game counts quoted in the Week 2 feature quality report differ from the dataset:",
        "",
        "| Split | Seasons (unchanged) | Week 2 documented | Counted in dataset | Difference |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ]
    documented_total = actual_total = 0
    for name in manifest["split_order"]:
        s = splits[name]
        documented = WEEK2_DOCUMENTED_SPLIT_COUNTS.get(name)
        actual = int(s["n_games"])
        actual_total += actual
        documented_total += documented or 0
        seasons = (
            s["season_labels"][0]
            if len(s["season_labels"]) == 1
            else f"{s['season_labels'][0]} to {s['season_labels'][-1]}"
        )
        diff = "—" if documented is None else f"{actual - documented:+,}"
        doc = "—" if documented is None else f"{documented:,}"
        lines.append(f"| `{name}` | {seasons} | {doc} | {actual:,} | {diff} |")
    lines.extend(
        [
            f"| **Total** | 2003-04 to 2022-23 | **{documented_total:,}** | "
            f"**{actual_total:,}** | **{actual_total - documented_total:+,}** |",
            "",
            "**Cause.** The Week 2 figures were written into the feature quality report text "
            "(commit `96933df`) as an illustrative example and were never computed from the data. "
            "Only their total was checked; the per-split values do not correspond to any season "
            "grouping of the dataset. Every data-derived definition gives the counts above:",
            "",
            "- the feature dataset's `season` column;",
            "- the raw `games.csv` `SEASON` column (identical season for every game);",
            "- season inferred from `game_date` with an August or October cutoff;",
            "- de-duplicated, non-preseason raw `games.csv` rows.",
            "",
            "**Same-day grouping is not the cause.** No calendar date contains games from two "
            "different seasons, so grouping by date cannot move games between season blocks "
            "(check `SPLIT-005` below, and `test_no_calendar_date_spans_two_seasons`).",
            "",
            "**Resolution.** The season definitions were kept exactly as planned; the counts "
            "computed from the dataset are authoritative and are verified by tests "
            "(`EXPECTED_SPLIT_GAME_COUNTS`). The Week 2 report text was corrected to these values.",
            "",
            "Games per season:",
            "",
            "| Split | Season | Games |",
            "| :--- | :--- | ---: |",
        ]
    )
    for name in manifest["split_order"]:
        for label, count in splits[name].get("season_game_counts", {}).items():
            lines.append(f"| `{name}` | {label} | {int(count):,} |")
    return lines


def render_modeling_strategy_markdown(manifest: Mapping[str, Any]) -> str:
    """Render ``docs/modeling_strategy.md`` from the strategy definitions and split manifest."""
    splits: Mapping[str, Any] = manifest["splits"]
    lines = [
        "# Arcora Week 3 — Modeling Strategy",
        "",
        "This document defines how Arcora trains and evaluates its NBA game prediction models: "
        "the chronological dataset splits, prediction targets, model-input rules, evaluation "
        "metrics, and reproducibility requirements. It is generated from "
        "`src/arcora/models/modeling_strategy.py` and `src/arcora/models/splits.py`.",
        "",
        "---",
        "",
        "## 1. Source Dataset",
        "",
        f"- **Dataset**: `{manifest['source_dataset']}`",
        f"- **Games**: {manifest['total_games']:,} eligible non-preseason games "
        "(Regular Season, Play-In, Playoffs; preseason was excluded in Week 2).",
        f"- **Approved pregame features**: {len(MODEL_FEATURE_COLUMNS)} "
        "(from `arcora.features.strategy.FEATURE_COLUMN_NAMES`).",
        f"- **Metadata columns**: {len(METADATA_COLUMNS)} — "
        + ", ".join(f"`{c}`" for c in METADATA_COLUMNS),
        f"- **Target columns**: {len(TARGET_COLUMNS)} — "
        + ", ".join(f"`{c}`" for c in TARGET_COLUMNS),
        f"- **Dataset fingerprint (SHA-256)**: `{manifest['dataset_fingerprint']}`",
        "",
        "---",
        "",
        "## 2. Chronological Splits",
        "",
        "Splits are contiguous blocks of NBA seasons. Games are never randomly sampled, so all "
        "games from the same calendar date belong to the same split and every training date "
        "precedes every validation date, which precedes every test date.",
        "",
        "| Split | Seasons | Games | Dates | Role | Locked |",
        "| :--- | :--- | ---: | :--- | :--- | :---: |",
    ]
    for name in manifest["split_order"]:
        s = splits[name]
        seasons = (
            s["season_labels"][0]
            if len(s["season_labels"]) == 1
            else f"{s['season_labels'][0]} to {s['season_labels'][-1]}"
        )
        lines.append(
            f"| `{name}` | {seasons} | {s['n_games']:,} | {s['date_min']} → {s['date_max']} | "
            f"{s['role']} | {'Yes' if s['locked'] else 'No'} |"
        )
    lines.extend(
        [
            "",
            "Game-type composition per split:",
            "",
            "| Split | Regular Season | Play-In | Playoffs |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for name in manifest["split_order"]:
        gt = splits[name]["game_type_counts"]
        lines.append(
            f"| `{name}` | {gt.get('Regular Season', 0):,} | {gt.get('Play-In', 0):,} | "
            f"{gt.get('Playoffs', 0):,} |"
        )
    lines.extend(
        [
            "",
            *render_split_count_reconciliation(manifest),
            "",
            "### Isolation rules",
            "",
            "- **Training** fits every estimator, imputer, scaler, and baseline statistic.",
            "- **Validation** is the only split used for model comparison, early stopping, and "
            "hyperparameter selection.",
            "- **Primary test (2021-22)** is locked. It is not evaluated unless explicitly "
            "requested; any evaluation must be documented and must not feed back into tuning.",
            "- **Partial 2022-23 holdout** is locked, isolated from model selection, and "
            "reported separately; it is never aggregated with complete seasons.",
            "",
            "### Integrity checks",
            "",
            "| Check | Description | Result | Details |",
            "| :--- | :--- | :---: | :--- |",
        ]
    )
    for check in manifest["integrity_checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"| `{check['check_id']}` | {check['check_name']} | {status} | {check['details']} |"
        )
    lines.extend(
        [
            "",
            "The persisted manifest (`data/splits/split_manifest.json`) records a SHA-256 "
            "fingerprint of each split's `game_id` set. Reusing the manifest re-derives the "
            "splits and fails if any fingerprint differs.",
            "",
            "---",
            "",
            "## 3. Prediction Targets",
            "",
            "| Target | Task | Formula | Derived from final scores | Description |",
            "| :--- | :--- | :--- | :---: | :--- |",
        ]
    )
    for t in MODEL_TARGET_SPECS:
        lines.append(
            f"| `{t.name}` | {t.task} | `{t.formula}` | "
            f"{'Yes' if t.derived_from_scores else 'No'} | {t.description} |"
        )
    lines.extend(
        [
            "",
            "`derive_targets()` recomputes `total_pts`, `point_margin`, and `home_win` from "
            "`home_pts` / `away_pts` and raises an error if the stored columns disagree.",
            "",
            "---",
            "",
            "## 4. Model Input Rules",
            "",
            f"1. Model inputs are exactly the {len(MODEL_FEATURE_COLUMNS)} approved pregame "
            "features, in canonical Week 2 order (`select_model_features()`).",
            "2. All metadata columns are excluded from model inputs.",
            "3. Every target column is excluded from model inputs.",
            "4. Post-game box-score columns (`FORBIDDEN_POSTGAME_COLUMNS`) are excluded.",
            "5. Identifiers (`game_id`, `home_team_id`, `away_team_id`) are never features.",
            f"6. The {len(APPROVED_MISSING_INDICATORS)} approved missing-value indicators are "
            "kept as inputs: " + ", ".join(f"`{c}`" for c in APPROVED_MISSING_INDICATORS) + ".",
            "7. Boolean features are cast to `float64` (0/1); early-season `NaN` values are "
            "preserved in `X` and handled by each model family (training-fitted imputation for "
            "linear and Random Forest models, native missing-value handling for XGBoost).",
            "",
            "`assert_feature_allowlist()` rejects any feature list that contains an excluded "
            "column, an unapproved column, or a different column order.",
            "",
            "---",
            "",
            "## 5. Evaluation Metrics",
            "",
            "| Metric | Task | Better | Input | Formula | Undefined handling |",
            "| :--- | :--- | :---: | :--- | :--- | :--- |",
        ]
    )
    for m in METRIC_SPECS:
        better = "Higher" if m.greater_is_better else "Lower"
        lines.append(
            f"| {m.display_name} (`{m.name}`) | {m.task} | {better} | {m.input_type} | "
            f"`{m.formula}` | {m.undefined_handling} |"
        )
    lines.extend(
        [
            "",
            f"- **Primary selection metrics**: `{PRIMARY_METRIC[REGRESSION_TASK]}` for regression "
            f"and `{PRIMARY_METRIC[CLASSIFICATION_TASK]}` for classification (probability "
            "quality matters for win-probability forecasts).",
            f"- **Classification threshold**: labels are `p >= {CLASSIFICATION_THRESHOLD}`.",
            "- **Undefined metrics** are stored as `null` in JSON outputs (never as 0) and the "
            "metric name is listed under `undefined_metrics` so comparisons skip it explicitly.",
            "- Metric implementations live in `src/arcora/evaluation/metrics.py` and are shared "
            "by every model family.",
            "",
            "---",
            "",
            "## 6. Reproducibility Requirements",
            "",
            "| Requirement | Implementation |",
            "| :--- | :--- |",
        ]
    )
    for req, impl in REPRODUCIBILITY_REQUIREMENTS:
        lines.append(f"| {req} | {impl} |")
    lines.append("")
    return "\n".join(lines)


def write_modeling_strategy_documentation(docs_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``docs/modeling_strategy.md``."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / "modeling_strategy.md"
    path.write_text(render_modeling_strategy_markdown(manifest), encoding="utf-8")
    return path
