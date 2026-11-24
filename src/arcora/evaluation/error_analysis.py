"""Error analysis for regression and classification models (Issue #23).

All functions operate on long-format prediction frames (one row per model × game) joined with
game metadata. Error analysis is a development tool: running it on the locked test split or the
partial-season holdout is refused unless explicitly allowed, and results from those splits must
never guide tuning.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from arcora.evaluation.metrics import classification_metrics, regression_metrics
from arcora.models.data import ModelingData
from arcora.models.modeling_strategy import CLASSIFICATION_THRESHOLD
from arcora.models.splits import SPLIT_DEFINITIONS_BY_NAME

METADATA_FOR_ANALYSIS: tuple[str, ...] = (
    "game_id",
    "game_date",
    "season",
    "season_label",
    "game_type",
    "home_team_id",
    "away_team_id",
)


def guard_locked_split(split: str, allow_locked: bool = False) -> None:
    """Refuse error analysis on locked splits unless explicitly allowed."""
    definition = SPLIT_DEFINITIONS_BY_NAME.get(split)
    if definition is None:
        raise ValueError(f"Unknown split '{split}'")
    if definition.locked and not allow_locked:
        raise PermissionError(
            f"Error analysis on locked split '{split}' is disabled; its results must not guide "
            "tuning. Pass allow_locked=True only for a documented final evaluation."
        )


def load_team_abbreviations(teams_path: Path | None) -> dict[int, str]:
    """Return ``{team_id: abbreviation}`` from ``teams.parquet`` when available."""
    if teams_path is None or not teams_path.is_file():
        return {}
    teams = pd.read_parquet(teams_path, columns=["team_id", "abbreviation"])
    return {int(t): str(a) for t, a in zip(teams["team_id"], teams["abbreviation"], strict=True)}


def attach_metadata(
    predictions: pd.DataFrame,
    data: ModelingData,
    team_abbreviations: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Join prediction rows with game metadata (season, game type, teams, date)."""
    meta = pd.concat(
        [split.metadata.loc[:, list(METADATA_FOR_ANALYSIS)] for split in data.splits.values()],
        ignore_index=True,
    )
    merged = predictions.merge(meta, on="game_id", how="left", validate="many_to_one")
    if merged["season"].isna().any():
        raise ValueError("Some predictions reference game_ids missing from the dataset")
    abbreviations = team_abbreviations or {}
    for side in ("home", "away"):
        ids = merged[f"{side}_team_id"].astype("int64")
        merged[f"{side}_team"] = [abbreviations.get(int(i), str(i)) for i in ids]
    merged["season_label"] = merged["season_label"].astype("string")
    merged["game_type"] = merged["game_type"].astype("string")
    return merged


def _select(frame: pd.DataFrame, model_id: str, split: str) -> pd.DataFrame:
    sub = frame[(frame["model_id"] == model_id) & (frame["split"] == split)]
    if sub.empty:
        raise ValueError(f"No predictions for model '{model_id}' on split '{split}'")
    return sub.copy()


def residual_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Distribution of residuals (``predicted - actual``)."""
    residual = frame["y_pred"] - frame["y_true"]
    q = residual.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean_residual": float(residual.mean()),
        "std_residual": float(residual.std(ddof=1)),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "median": float(q.loc[0.5]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "share_within_5": float((residual.abs() <= 5).mean()),
        "share_within_10": float((residual.abs() <= 10).mean()),
    }


def regression_errors_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """MAE, RMSE, bias, and game count per group."""
    rows = []
    for key, group in frame.groupby(column, sort=True):
        m = regression_metrics(group["y_true"], group["y_pred"])
        rows.append(
            {
                column: key,
                "n_games": len(group),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias": float((group["y_pred"] - group["y_true"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def largest_errors(frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Games with the largest absolute prediction error."""
    out = frame.assign(
        residual=frame["y_pred"] - frame["y_true"],
        abs_error=(frame["y_pred"] - frame["y_true"]).abs(),
    )
    cols = [
        "game_id",
        "game_date",
        "season_label",
        "game_type",
        "home_team",
        "away_team",
        "y_true",
        "y_pred",
        "residual",
        "abs_error",
    ]
    return out.sort_values("abs_error", ascending=False).head(n).loc[:, cols].reset_index(drop=True)


def regression_error_analysis(
    frame: pd.DataFrame,
    model_id: str,
    split: str,
    allow_locked: bool = False,
    n_largest: int = 10,
) -> dict[str, Any]:
    """Full regression error analysis for one model on one split."""
    guard_locked_split(split, allow_locked)
    sub = _select(frame, model_id, split)
    return {
        "model_id": model_id,
        "split": split,
        "target": str(sub["target"].iloc[0]),
        "n_games": len(sub),
        "metrics": regression_metrics(sub["y_true"], sub["y_pred"]),
        "residuals": residual_summary(sub),
        "by_season": regression_errors_by(sub, "season_label"),
        "by_game_type": regression_errors_by(sub, "game_type"),
        "by_home_team": regression_errors_by(sub, "home_team"),
        "by_away_team": regression_errors_by(sub, "away_team"),
        "largest_errors": largest_errors(sub, n_largest),
    }


def confusion_counts(
    y_true: Any, y_prob: Any, threshold: float = CLASSIFICATION_THRESHOLD
) -> dict[str, int]:
    """True/false positive/negative counts (positive class = home win)."""
    true = np.asarray(y_true).astype("int64")
    pred = (np.asarray(y_prob, dtype="float64") >= threshold).astype("int64")
    return {
        "true_positive": int(((pred == 1) & (true == 1)).sum()),
        "false_positive": int(((pred == 1) & (true == 0)).sum()),
        "true_negative": int(((pred == 0) & (true == 0)).sum()),
        "false_negative": int(((pred == 0) & (true == 1)).sum()),
    }


def classification_metrics_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Classification metrics and confusion counts per group (undefined metrics -> null)."""
    rows = []
    for key, group in frame.groupby(column, sort=True):
        m = classification_metrics(group["y_true"], group["y_pred"])
        rows.append(
            {
                column: key,
                "n_games": len(group),
                "home_win_rate": float(group["y_true"].mean()),
                "mean_predicted_prob": float(group["y_pred"].mean()),
                **m,
                **confusion_counts(group["y_true"], group["y_pred"]),
            }
        )
    return pd.DataFrame(rows)


def calibration_table(frame: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Observed home-win rate per predicted-probability bin."""
    bins = np.linspace(0.0, 1.0, n_bins + 1).tolist()
    binned = pd.cut(frame["y_pred"], bins=bins, include_lowest=True)
    grouped = frame.groupby(binned, observed=True).agg(
        n_games=("y_true", "size"),
        mean_predicted=("y_pred", "mean"),
        observed_rate=("y_true", "mean"),
    )
    return grouped.reset_index(names="probability_bin").assign(
        probability_bin=lambda d: d["probability_bin"].astype(str)
    )


def classification_error_analysis(
    frame: pd.DataFrame,
    model_id: str,
    split: str,
    allow_locked: bool = False,
    n_examples: int = 10,
) -> dict[str, Any]:
    """Full classification error analysis for one model on one split."""
    guard_locked_split(split, allow_locked)
    sub = _select(frame, model_id, split)
    sub["predicted_label"] = (sub["y_pred"] >= CLASSIFICATION_THRESHOLD).astype("int64")
    cols = [
        "game_id",
        "game_date",
        "season_label",
        "game_type",
        "home_team",
        "away_team",
        "y_pred",
    ]
    false_pos = sub[(sub["predicted_label"] == 1) & (sub["y_true"] == 0)]
    false_neg = sub[(sub["predicted_label"] == 0) & (sub["y_true"] == 1)]
    return {
        "model_id": model_id,
        "split": split,
        "n_games": len(sub),
        "metrics": classification_metrics(sub["y_true"], sub["y_pred"]),
        "confusion": confusion_counts(sub["y_true"], sub["y_pred"]),
        "by_season": classification_metrics_by(sub, "season_label"),
        "by_game_type": classification_metrics_by(sub, "game_type"),
        "calibration": calibration_table(sub),
        "most_confident_false_positives": false_pos.sort_values("y_pred", ascending=False)
        .head(n_examples)
        .loc[:, cols]
        .reset_index(drop=True),
        "most_confident_false_negatives": false_neg.sort_values("y_pred", ascending=True)
        .head(n_examples)
        .loc[:, cols]
        .reset_index(drop=True),
    }


def errors_by_season_across_splits(
    frame: pd.DataFrame, model_id: str, splits: Sequence[str]
) -> pd.DataFrame:
    """Primary-metric error per season for one model across development splits.

    Useful for spotting scoring-environment drift between training and validation seasons.
    """
    for split in splits:
        guard_locked_split(split)
    sub = frame[(frame["model_id"] == model_id) & (frame["split"].isin(splits))]
    task = str(sub["task"].iloc[0])
    if task == "classification":
        out = classification_metrics_by(sub, "season_label")
        return out.loc[:, ["season_label", "n_games", "log_loss", "accuracy", "brier"]]
    return regression_errors_by(sub, "season_label")
