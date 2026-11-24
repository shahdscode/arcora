# Arcora Week 3 — XGBoost Results

XGBoost gradient-boosted trees (`tree_method='hist'`) with native missing-value handling. Models are fitted on the training split; early stopping (100 rounds of patience) monitors the chronological validation split to choose the number of boosting rounds. One baseline configuration is used: learning rate 0.03, depth 4, row subsample 0.8, column subsample 0.6.

All models are fitted on the **training split (2003-04 to 2018-19)** and compared on the **validation split (2019-20 to 2020-21)**. The primary 2021-22 test set and the partial 2022-23 holdout are locked and were not evaluated.

Baseline comparison columns: `delta_vs_baseline` = model loss − best baseline loss on validation, in metric units (negative is better); `relative_loss_reduction_pct` = (baseline loss − model loss) / baseline loss × 100 (positive is better).

## Regression (validation)

| Model | Target | MAE | RMSE | R² | Train MAE | `delta_vs_baseline` | `relative_loss_reduction_pct` |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `xgboost_regressor` | `home_pts` | 9.489 | 12.016 | 0.0845 | 8.253 | -0.0459 | +0.48% |
| `xgboost_regressor` | `away_pts` | 9.401 | 11.842 | 0.0860 | 8.183 | -0.0277 | +0.29% |
| `xgboost_regressor` | `total_pts` | 15.413 | 19.565 | 0.0611 | 13.671 | +0.0419 | -0.27% |
| `xgboost_regressor` | `point_margin` | 10.755 | 13.649 | 0.1317 | 9.213 | -0.2100 | +1.92% |

## Classification — `home_win` (validation)

| Model | Accuracy | Balanced acc. | Precision | Recall | ROC-AUC | Log loss | Brier | `delta_vs_baseline` | `relative_loss_reduction_pct` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `xgboost_classifier` | 0.6396 | 0.6255 | 0.6406 | 0.7763 | 0.6824 | 0.6402 | 0.2243 | -0.0114 | +1.75% |

## Hyperparameters and training configuration

| Model | Target | Parameters |
| :--- | :--- | :--- |
| `xgboost_regressor` | `home_pts` | `colsample_bytree=0.6`, `early_stopping_rounds=100`, `estimator=XGBRegressor`, `eval_metric=mae`, `gamma=0.0`, `importance_type=gain`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=10`, `n_estimators=3000`, `n_jobs=-1`, `objective=reg:squarederror`, `random_state=42`, `reg_alpha=0.0`, `reg_lambda=1.0`, `subsample=0.8`, `tree_method=hist` |
| `xgboost_regressor` | `away_pts` | `colsample_bytree=0.6`, `early_stopping_rounds=100`, `estimator=XGBRegressor`, `eval_metric=mae`, `gamma=0.0`, `importance_type=gain`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=10`, `n_estimators=3000`, `n_jobs=-1`, `objective=reg:squarederror`, `random_state=42`, `reg_alpha=0.0`, `reg_lambda=1.0`, `subsample=0.8`, `tree_method=hist` |
| `xgboost_regressor` | `total_pts` | `colsample_bytree=0.6`, `early_stopping_rounds=100`, `estimator=XGBRegressor`, `eval_metric=mae`, `gamma=0.0`, `importance_type=gain`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=10`, `n_estimators=3000`, `n_jobs=-1`, `objective=reg:squarederror`, `random_state=42`, `reg_alpha=0.0`, `reg_lambda=1.0`, `subsample=0.8`, `tree_method=hist` |
| `xgboost_regressor` | `point_margin` | `colsample_bytree=0.6`, `early_stopping_rounds=100`, `estimator=XGBRegressor`, `eval_metric=mae`, `gamma=0.0`, `importance_type=gain`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=10`, `n_estimators=3000`, `n_jobs=-1`, `objective=reg:squarederror`, `random_state=42`, `reg_alpha=0.0`, `reg_lambda=1.0`, `subsample=0.8`, `tree_method=hist` |
| `xgboost_classifier` | `home_win` | `colsample_bytree=0.6`, `early_stopping_rounds=100`, `estimator=XGBClassifier`, `eval_metric=logloss`, `gamma=0.0`, `importance_type=gain`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=10`, `n_estimators=3000`, `n_jobs=-1`, `objective=binary:logistic`, `random_state=42`, `reg_alpha=0.0`, `reg_lambda=1.0`, `subsample=0.8`, `tree_method=hist` |

## Comparison with every trained family (validation)

Best model per family on the primary metric (MAE for regression, log loss for `home_win`), evaluated on the identical validation split.

| Target | Metric | `baseline` | `linear` | `random_forest` | `xgboost` |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `home_pts` | mae | 9.5347 (`historical_reference`) | 9.4469 (`ridge_regression`) | 9.4782 (`random_forest_regressor`) | 9.4889 (`xgboost_regressor`) |
| `away_pts` | mae | 9.4288 (`historical_reference`) | 9.3794 (`lasso_regression`) | 9.3942 (`random_forest_regressor`) | 9.4011 (`xgboost_regressor`) |
| `total_pts` | mae | 15.3710 (`historical_reference`) | 15.2369 (`ridge_regression`) | 15.3250 (`random_forest_regressor`) | 15.4129 (`xgboost_regressor`) |
| `point_margin` | mae | 10.9651 (`historical_reference`) | 10.7766 (`lasso_regression`) | 10.7479 (`random_forest_regressor`) | 10.7550 (`xgboost_regressor`) |
| `home_win` | log_loss | 0.6516 (`historical_reference`) | 0.6428 (`logistic_regression`) | 0.6393 (`random_forest_classifier`) | 0.6402 (`xgboost_classifier`) |

## Early stopping (validation split)

| Model | Target | Metric | Best iteration | Trees used | Rounds trained | Best validation score | Fit seconds |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `xgboost_regressor` | `home_pts` | mae | 172 | 173 | 273 | 9.4889 | 1.4 |
| `xgboost_regressor` | `away_pts` | mae | 239 | 240 | 340 | 9.4011 | 1.7 |
| `xgboost_regressor` | `total_pts` | mae | 175 | 176 | 276 | 15.4129 | 1.4 |
| `xgboost_regressor` | `point_margin` | mae | 134 | 135 | 235 | 10.7550 | 1.2 |
| `xgboost_classifier` | `home_win` | logloss | 206 | 207 | 307 | 0.6402 | 1.7 |

Early stopping uses only the validation split, so validation metrics for XGBoost are mildly optimistic; the locked 2021-22 test split remains untouched for an unbiased final estimate.

Feature importance (exploratory): see [`xgboost_feature_importance.md`](xgboost_feature_importance.md).

## Reproducibility

- Dataset fingerprint: `eebe4ed42461191c0073cf1817916da55887a0ebab333dbf84781bed2c1d8ff6`
- Training split `game_id` SHA-256: `059f065459f80db7190607d5c73c02d7a4f8df5febceecf4051ad464efc72740`
- Feature fingerprint (242 features): `e4d280bc07f09447968d467bf315c1e7587ae78980bd33be58af76da91c84f8c`
- Library versions: python 3.12.16, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, joblib 1.6.0
