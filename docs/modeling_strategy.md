# Arcora Week 3 — Modeling Strategy

This document defines how Arcora trains and evaluates its NBA game prediction models: the chronological dataset splits, prediction targets, model-input rules, evaluation metrics, and reproducibility requirements. It is generated from `src/arcora/models/modeling_strategy.py` and `src/arcora/models/splits.py`.

---

## 1. Source Dataset

- **Dataset**: `data/features/nba_game_features.parquet`
- **Games**: 24,848 eligible non-preseason games (Regular Season, Play-In, Playoffs; preseason was excluded in Week 2).
- **Approved pregame features**: 242 (from `arcora.features.strategy.FEATURE_COLUMN_NAMES`).
- **Metadata columns**: 8 — `game_id`, `game_date`, `season`, `season_label`, `game_type`, `is_partial_season`, `home_team_id`, `away_team_id`
- **Target columns**: 5 — `home_pts`, `away_pts`, `point_margin`, `total_pts`, `home_win`
- **Dataset fingerprint (SHA-256)**: `eebe4ed42461191c0073cf1817916da55887a0ebab333dbf84781bed2c1d8ff6`

---

## 2. Chronological Splits

Splits are contiguous blocks of NBA seasons. Games are never randomly sampled, so all games from the same calendar date belong to the same split and every training date precedes every validation date, which precedes every test date.

| Split | Seasons | Games | Dates | Role | Locked |
| :--- | :--- | ---: | :--- | :--- | :---: |
| `train` | 2003-04 to 2018-19 | 20,734 | 2003-10-28 → 2019-06-13 | Model fitting | No |
| `validation` | 2019-20 to 2020-21 | 2,314 | 2019-10-22 → 2021-07-20 | Model comparison, early stopping, and hyperparameter selection | No |
| `test` | 2021-22 | 1,323 | 2021-10-19 → 2022-06-16 | Primary locked evaluation | Yes |
| `holdout` | 2022-23 | 477 | 2022-10-18 → 2022-12-22 | Secondary out-of-time partial-season holdout | Yes |

Game-type composition per split:

| Split | Regular Season | Play-In | Playoffs |
| :--- | ---: | ---: | ---: |
| `train` | 19,398 | 0 | 1,336 |
| `validation` | 2,139 | 7 | 168 |
| `test` | 1,230 | 6 | 87 |
| `holdout` | 477 | 0 | 0 |

> **Count correction.** The Week 2 feature quality report previously listed approximate split sizes of 20,762 / 2,282 / 1,327 games. Counting the season blocks directly in `nba_game_features.parquet` gives 20,734 / 2,314 / 1,323 (the 24,371-game total for complete seasons is unchanged). The counts above are computed from the dataset and are the authoritative values.

### Isolation rules

- **Training** fits every estimator, imputer, scaler, and baseline statistic.
- **Validation** is the only split used for model comparison, early stopping, and hyperparameter selection.
- **Primary test (2021-22)** is locked. It is not evaluated unless explicitly requested; any evaluation must be documented and must not feed back into tuning.
- **Partial 2022-23 holdout** is locked, isolated from model selection, and reported separately; it is never aggregated with complete seasons.

### Integrity checks

| Check | Description | Result | Details |
| :--- | :--- | :---: | :--- |
| `SPLIT-001` | All Splits Present and Non-Empty | PASS | All splits contain games. |
| `SPLIT-002` | No Duplicate Games Within a Split | PASS | 0 duplicated game_id rows found within splits. |
| `SPLIT-003` | No Game Appears in Multiple Splits | PASS | 0 game_ids assigned to more than one split. |
| `SPLIT-004` | Strict Chronological Ordering Across Splits | PASS | Every split's latest game_date precedes the next split's earliest game_date. |
| `SPLIT-005` | Same-Day Games Share a Single Split | PASS | 0 calendar dates have games in more than one split. |
| `SPLIT-006` | Split Membership Matches Season Definitions | PASS | 0 rows fall outside their split's season definition. |
| `SPLIT-007` | Partial-Season Holdout Isolation | PASS | 0 partial-season rows outside the holdout; 0 complete-season rows inside the holdout. |

The persisted manifest (`data/splits/split_manifest.json`) records a SHA-256 fingerprint of each split's `game_id` set. Reusing the manifest re-derives the splits and fails if any fingerprint differs.

---

## 3. Prediction Targets

| Target | Task | Formula | Derived from final scores | Description |
| :--- | :--- | :--- | :---: | :--- |
| `home_pts` | regression | `games.home_pts` | No | Final points scored by the home team. |
| `away_pts` | regression | `games.away_pts` | No | Final points scored by the away team. |
| `total_pts` | regression | `home_pts + away_pts` | Yes | Combined final points scored by both teams. |
| `point_margin` | regression | `home_pts - away_pts` | Yes | Final signed point differential from the home team perspective. |
| `home_win` | classification | `home_pts > away_pts` | Yes | Binary outcome indicator (True if home team wins). |

`derive_targets()` recomputes `total_pts`, `point_margin`, and `home_win` from `home_pts` / `away_pts` and raises an error if the stored columns disagree.

---

## 4. Model Input Rules

1. Model inputs are exactly the 242 approved pregame features, in canonical Week 2 order (`select_model_features()`).
2. All metadata columns are excluded from model inputs.
3. Every target column is excluded from model inputs.
4. Post-game box-score columns (`FORBIDDEN_POSTGAME_COLUMNS`) are excluded.
5. Identifiers (`game_id`, `home_team_id`, `away_team_id`) are never features.
6. The 8 approved missing-value indicators are kept as inputs: `home_hist_insufficient_history`, `away_hist_insufficient_history`, `home_rolling_insufficient_5`, `away_rolling_insufficient_5`, `home_rolling_insufficient_10`, `away_rolling_insufficient_10`, `home_is_season_opener`, `away_is_season_opener`.
7. Boolean features are cast to `float64` (0/1); early-season `NaN` values are preserved in `X` and handled by each model family (training-fitted imputation for linear and Random Forest models, native missing-value handling for XGBoost).

`assert_feature_allowlist()` rejects any feature list that contains an excluded column, an unapproved column, or a different column order.

---

## 5. Evaluation Metrics

| Metric | Task | Better | Input | Formula | Undefined handling |
| :--- | :--- | :---: | :--- | :--- | :--- |
| MAE (`mae`) | regression | Lower | point prediction | `mean(|y - y_hat|)` | Undefined (null) only for empty inputs. |
| RMSE (`rmse`) | regression | Lower | point prediction | `sqrt(mean((y - y_hat)^2))` | Undefined (null) only for empty inputs. |
| R² (`r2`) | regression | Higher | point prediction | `1 - SS_res / SS_tot` | Null when fewer than 2 rows or y is constant (SS_tot = 0). |
| Accuracy (`accuracy`) | classification | Higher | label at threshold 0.5 | `mean(y == y_hat)` | Undefined (null) only for empty inputs. |
| Balanced accuracy (`balanced_accuracy`) | classification | Higher | label at threshold 0.5 | `(TPR + TNR) / 2` | Null when y contains a single class. |
| Precision (`precision`) | classification | Higher | label at threshold 0.5 | `TP / (TP + FP)` | Null when the model predicts no positives (TP + FP = 0). |
| Recall (`recall`) | classification | Higher | label at threshold 0.5 | `TP / (TP + FN)` | Null when y contains no positives (TP + FN = 0). |
| ROC-AUC (`roc_auc`) | classification | Higher | probability | `P(score(pos) > score(neg))` | Null when y contains a single class. |
| Log loss (`log_loss`) | classification | Lower | probability | `-mean(y log p + (1 - y) log(1 - p)), p clipped to [1e-15, 1 - 1e-15]` | Always defined (labels fixed to {0, 1}). |
| Brier score (`brier`) | classification | Lower | probability | `mean((p - y)^2)` | Always defined for non-empty inputs. |

- **Primary selection metrics**: `mae` for regression and `log_loss` for classification (probability quality matters for win-probability forecasts).
- **Classification threshold**: labels are `p >= 0.5`.
- **Undefined metrics** are stored as `null` in JSON outputs (never as 0) and the metric name is listed under `undefined_metrics` so comparisons skip it explicitly.
- Metric implementations live in `src/arcora/evaluation/metrics.py` and are shared by every model family.

---

## 6. Reproducibility Requirements

| Requirement | Implementation |
| :--- | :--- |
| Fixed random seeds | Every stochastic estimator uses `random_state=42` (`RANDOM_SEED`). |
| Consistent feature selection | `select_model_features()` returns the 242-feature allowlist in canonical order; its SHA-256 fingerprint is stored with every model. |
| Persisted split manifest | `data/splits/split_manifest.json` stores per-split counts, date ranges, and `game_id` fingerprints; reused manifests are verified before training. |
| Recorded model configurations | Hyperparameters, seeds, and library versions are saved alongside each artifact. |
| Versioned model artifacts | Artifacts are written under `artifacts/models/<family>/` with a JSON metadata sidecar recording the dataset fingerprint, feature fingerprint, and versions. |
| Consistent evaluation datasets | All model comparisons use the same validation split, verified by its `game_id` fingerprint. |
