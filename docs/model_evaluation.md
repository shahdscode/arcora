# Arcora Week 3 — Model Evaluation Framework

The evaluation framework (`src/arcora/evaluation/`) scores, compares, and diagnoses every model
with one implementation, so results from different model families are directly comparable.

| Module | Responsibility |
| :--- | :--- |
| `metrics.py` | Regression (MAE, RMSE, R²) and classification (accuracy, balanced accuracy, precision, recall, ROC-AUC, log loss, Brier) metrics with safe handling of undefined cases |
| `comparison.py` | Comparability checks, comparison tables, rankings, baseline comparisons, paired bootstrap tests |
| `error_analysis.py` | Residuals, per-season / per-team / per-game-type errors, largest errors, confusion matrices, calibration |
| `report.py` | Writes `reports/model_comparison.{csv,json}`, `reports/model_evaluation_report.md`, and `reports/figures/models/*.png` |

## Metrics

All models call `compute_metrics(task, y_true, y_pred)`; classification inputs are home-win
probabilities and labels use `p >= 0.5`. Undefined metrics are returned as `None` (JSON `null`),
never as 0, and are listed in each result's `undefined_metrics`:

| Metric | Undefined when |
| :--- | :--- |
| R² | fewer than 2 games or a constant target |
| Balanced accuracy, ROC-AUC | the evaluated games contain only one class |
| Precision | the model predicts no home wins |
| Recall | no home wins occurred |
| Log loss | never (probabilities clipped to `[1e-15, 1 − 1e-15]`, labels fixed to `{0, 1}`) |

## Comparison rules

- `validate_comparability()` raises `ComparabilityError` if results for the same split differ in
  dataset fingerprint, split `game_id` fingerprint, feature fingerprint, or game count. Metrics
  from different evaluation periods can therefore never be mixed in one ranking.
- `rank_models()` ranks every model per target on the primary metric (MAE for regression, log loss
  for `home_win`) and compares each model with the best (lowest-loss) naive baseline using two
  quantities, reported identically in every table, chart, CSV, and JSON output:

  | Quantity | Formula | Units | Better |
  | :--- | :--- | :--- | :--- |
  | `delta_vs_baseline` (Absolute Δ) | model loss − baseline loss | metric units (MAE points, log loss) | negative |
  | `relative_loss_reduction_pct` | (baseline loss − model loss) / baseline loss × 100 | % of the baseline's loss | positive |

  The relative figure is a percentage reduction in loss, not percentage points of accuracy.
- `paired_bootstrap()` resamples games to put a 95% interval on the per-game loss difference
  between two models scored on identical games, indicating whether an improvement is meaningful
  rather than noise.

## Error analysis

Regression: actual-vs-predicted plots, residual distribution and bias, MAE by season, by home team,
by away team, and by game type, and the largest individual errors. Classification: confusion
matrix, false positives / negatives (most confident examples), metrics by season and game type,
and calibration curves.

Error analysis runs only on development splits. `guard_locked_split()` raises `PermissionError` for
the 2021-22 test split and the partial 2022-23 holdout unless `allow_locked=True` is passed for a
documented final evaluation; locked-split diagnostics must never guide tuning.

## Generating the report

The report reads persisted family results and prediction caches, so it can be regenerated without
retraining:

```python
from arcora.config import ArcoraConfig
from arcora.evaluation.report import generate_evaluation_report
from arcora.models.data import load_modeling_data

cfg = ArcoraConfig.from_env()
data = load_modeling_data(
    cfg.features_data_dir / "nba_game_features.parquet",
    cfg.splits_dir / "split_manifest.json",
    reuse_manifest=True,
)
generate_evaluation_report(
    data,
    results_dir=cfg.reports_dir / "models",
    models_dir=cfg.models_dir,
    reports_dir=cfg.reports_dir,
    teams_path=cfg.processed_data_dir / "teams.parquet",
)
```
