# Arcora Week 3 — Model Families

This document describes every model family Arcora trains in Week 3, how each is configured, and
which design decisions protect the chronological evaluation protocol defined in
[`modeling_strategy.md`](modeling_strategy.md).

Shared conventions for every family:

- **Inputs**: the 242 approved pregame features from `select_model_features()`, in canonical order,
  cast to `float64`. Metadata, identifiers, targets, and post-game columns are never inputs.
- **Fitting**: every estimator, preprocessing step, and learned statistic is fitted on the
  **training split only** (2003-04 to 2018-19, 20,734 games).
- **Selection**: models are compared on the **validation split** (2019-20 to 2020-21, 2,314 games).
  The 2021-22 test set and the partial 2022-23 holdout are locked; `run_model_family()` refuses to
  evaluate them unless `allow_locked=True` is passed explicitly.
- **Outputs**: each model is wrapped in a `TrainedModel` (`src/arcora/models/base.py`) that
  reorders inputs to the training feature order before predicting. Regression models return point
  predictions; classification models return the home-win probability.
- **Results**: every evaluation is stored as a `ModelResult` (model, family, target, task, split,
  metrics, undefined metrics, parameters, training metadata, reproducibility fingerprints, and a
  UTC timestamp) in `reports/models/<family>_results.json` and `.csv`.
- **Artifacts**: `artifacts/models/<family>/<model_name>__<target>.joblib` plus a `.json` metadata
  sidecar (parameters, feature fingerprint, library versions, SHA-256 of the artifact). The ordered
  feature list is written once to `artifacts/models/feature_list.json`. Binary artifacts and
  prediction caches are git-ignored; metadata sidecars are tracked.

Run a single family with:

```bash
python -m arcora.models.experiment --family <family>
```

---

## 1. Naive Baselines (`baseline`, Issue #19)

Module: `src/arcora/models/baselines.py`

| Model | Targets | Method | Learned from training |
| :--- | :--- | :--- | :--- |
| `mean_baseline` | `home_pts`, `away_pts`, `total_pts`, `point_margin` | `DummyRegressor(strategy="mean")` | Target mean |
| `median_baseline` | `home_pts`, `away_pts`, `total_pts`, `point_margin` | `DummyRegressor(strategy="median")` | Target median |
| `most_frequent_baseline` | `home_win` | `DummyClassifier(strategy="most_frequent")` | Majority class (home win) |
| `prior_baseline` | `home_win` | `DummyClassifier(strategy="prior")` | Home-win rate as probability |
| `historical_reference` | all five | Team-strength reference (below) | League means, home/away offsets, margin σ |

### Historical reference

A transparent, non-ML reference built from season-to-date team averages:

- Expected home score = mean(`home_hist_pts_avg`, `away_hist_pts_allowed_avg`) + home offset.
- Expected away score = mean(`away_hist_pts_avg`, `home_hist_pts_allowed_avg`) + away offset.
- `total_pts` = home + away, `point_margin` = home − away (a consistent scoreline).
- P(home win) = Φ(expected margin / σ), where σ is the training standard deviation of the margin
  residual.

The home and away offsets capture home-court advantage (team averages mix home and road games) and
are the mean training residuals. When a team has no prior games in the season the league-level
training references are used instead: **average home score**, **average away score**, and the
**historical home-win rate**.

### Notes

- `most_frequent_baseline` outputs degenerate 0/1 probabilities, so its log loss is very large by
  construction; `prior_baseline` is the meaningful probability baseline.
- League scoring rose sharply between the training era and the 2019-21 validation seasons. Constant
  baselines therefore have **negative validation R²** — they are anchored to the 2003-19 scoring
  level. Models that use season-to-date features adapt to the current scoring environment and are
  far less affected.

---

## 2. Linear and Logistic Regression (`linear`, Issue #20)

Module: `src/arcora/models/linear_models.py`

Every model is a scikit-learn `Pipeline`:

```text
SimpleImputer(strategy="median", keep_empty_features=True) -> StandardScaler -> estimator
```

- Imputation medians and scaling statistics are learned **from the training split only** and are
  saved inside the artifact, so inference never rebuilds preprocessing manually.
- `keep_empty_features=True` keeps the output width equal to the 242 inputs, so feature ordering
  is stable.
- The approved missing-value indicators (`*_insufficient_*`, `*_is_season_opener`) are kept as
  inputs; imputation only fills the numeric `NaN` values they describe.

| Model | Targets | Regularization | Selection |
| :--- | :--- | :--- | :--- |
| `linear_regression` | 4 regression targets | None (OLS) | — |
| `ridge_regression` | 4 regression targets | L2, `alpha ∈ {1, 10, 100, 1e3, 1e4, 3e4, 1e5, 3e5}` | Lowest validation MAE |
| `lasso_regression` | 4 regression targets | L1, `alpha ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}` | Lowest validation MAE |
| `logistic_regression` | `home_win` | L2, `C ∈ {1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.1}` | Lowest validation log loss |

Each grid candidate is fitted on the training split and scored on validation; the full grid is
stored in `training_metadata.validation_selection`. Because validation is used to pick the
penalty, validation scores of tuned models are slightly optimistic — the locked test split exists
to measure that.

### Observations

- Strong shrinkage is preferred for `home_pts` and `total_pts` (`alpha ≈ 3e4`). The training era
  (2003-19) scored less than the validation seasons, and heavy regularization reduces reliance on
  era-specific level effects.
- **Class imbalance**: the training home-win rate is ~0.60 and the validation rate ~0.55 (the
  2020-21 season was played with limited or no crowds). `class_weight="balanced"` raises balanced
  accuracy slightly but worsens log loss and Brier score, so the unweighted model is kept for
  probability quality. The comparison is stored in
  `training_metadata.class_imbalance_investigation`.
- Standardized coefficients are reported for interpretability only: home, away, and difference
  features are strongly collinear, so individual coefficients are not causal effects.
