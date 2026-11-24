# Arcora Week 3 — Random Forest Results

Random Forest ensembles (training-fitted median imputation → forest) for every target. One fixed, conservative configuration is used — 300 trees, maximum depth 14, at least 25 games per leaf — to capture nonlinear effects and interactions without an expensive hyperparameter search. Seeds and hyperparameters are recorded with each model.

All models are fitted on the **training split (2003-04 to 2018-19)** and compared on the **validation split (2019-20 to 2020-21)**. The primary 2021-22 test set and the partial 2022-23 holdout are locked and were not evaluated.

## Regression (validation)

| Model | Target | MAE | RMSE | R² | Train MAE | Δ MAE vs best baseline |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `random_forest_regressor` | `home_pts` | 9.478 | 12.010 | 0.0854 | 7.278 | -0.057 |
| `random_forest_regressor` | `away_pts` | 9.394 | 11.814 | 0.0902 | 7.316 | -0.035 |
| `random_forest_regressor` | `total_pts` | 15.325 | 19.475 | 0.0697 | 12.060 | -0.046 |
| `random_forest_regressor` | `point_margin` | 10.748 | 13.631 | 0.1339 | 8.161 | -0.217 |

## Classification — `home_win` (validation)

| Model | Accuracy | Balanced acc. | Precision | Recall | ROC-AUC | Log loss | Brier | Δ log loss vs best baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `random_forest_classifier` | 0.6340 | 0.6163 | 0.6290 | 0.8055 | 0.6824 | 0.6393 | 0.2242 | -0.0123 |

## Hyperparameters and training configuration

| Model | Target | Parameters |
| :--- | :--- | :--- |
| `random_forest_regressor` | `home_pts` | `bootstrap=True`, `criterion=squared_error`, `estimator=RandomForestRegressor`, `max_depth=14`, `max_features=0.33`, `min_samples_leaf=25`, `n_estimators=300`, `n_jobs=-1`, `preprocessing=SimpleImputer(median)`, `random_state=42` |
| `random_forest_regressor` | `away_pts` | `bootstrap=True`, `criterion=squared_error`, `estimator=RandomForestRegressor`, `max_depth=14`, `max_features=0.33`, `min_samples_leaf=25`, `n_estimators=300`, `n_jobs=-1`, `preprocessing=SimpleImputer(median)`, `random_state=42` |
| `random_forest_regressor` | `total_pts` | `bootstrap=True`, `criterion=squared_error`, `estimator=RandomForestRegressor`, `max_depth=14`, `max_features=0.33`, `min_samples_leaf=25`, `n_estimators=300`, `n_jobs=-1`, `preprocessing=SimpleImputer(median)`, `random_state=42` |
| `random_forest_regressor` | `point_margin` | `bootstrap=True`, `criterion=squared_error`, `estimator=RandomForestRegressor`, `max_depth=14`, `max_features=0.33`, `min_samples_leaf=25`, `n_estimators=300`, `n_jobs=-1`, `preprocessing=SimpleImputer(median)`, `random_state=42` |
| `random_forest_classifier` | `home_win` | `bootstrap=True`, `class_weight=None`, `criterion=gini`, `estimator=RandomForestClassifier`, `max_depth=14`, `max_features=sqrt`, `min_samples_leaf=25`, `n_estimators=300`, `n_jobs=-1`, `preprocessing=SimpleImputer(median)`, `random_state=42` |

## Comparison with every trained family (validation)

Best model per family on the primary metric (MAE for regression, log loss for `home_win`), evaluated on the identical validation split.

| Target | Metric | `baseline` | `linear` | `random_forest` | `xgboost` |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `home_pts` | mae | 9.5347 (`historical_reference`) | 9.4469 (`ridge_regression`) | 9.4782 (`random_forest_regressor`) | 9.4889 (`xgboost_regressor`) |
| `away_pts` | mae | 9.4288 (`historical_reference`) | 9.3794 (`lasso_regression`) | 9.3942 (`random_forest_regressor`) | 9.4011 (`xgboost_regressor`) |
| `total_pts` | mae | 15.3710 (`historical_reference`) | 15.2369 (`ridge_regression`) | 15.3250 (`random_forest_regressor`) | 15.4129 (`xgboost_regressor`) |
| `point_margin` | mae | 10.9651 (`historical_reference`) | 10.7766 (`lasso_regression`) | 10.7479 (`random_forest_regressor`) | 10.7550 (`xgboost_regressor`) |
| `home_win` | log_loss | 0.6516 (`historical_reference`) | 0.6428 (`logistic_regression`) | 0.6393 (`random_forest_classifier`) | 0.6402 (`xgboost_classifier`) |

## Training statistics

| Model | Target | Fit seconds | Mean tree depth | Mean leaves per tree |
| :--- | :--- | ---: | ---: | ---: |
| `random_forest_regressor` | `home_pts` | 11.1 | 14.0 | 386 |
| `random_forest_regressor` | `away_pts` | 11.2 | 14.0 | 384 |
| `random_forest_regressor` | `total_pts` | 11.1 | 14.0 | 386 |
| `random_forest_regressor` | `point_margin` | 12.6 | 14.0 | 346 |
| `random_forest_classifier` | `home_win` | 2.9 | 14.0 | 363 |

Feature importance (exploratory): see [`random_forest_feature_importance.md`](random_forest_feature_importance.md).

## Reproducibility

- Dataset fingerprint: `eebe4ed42461191c0073cf1817916da55887a0ebab333dbf84781bed2c1d8ff6`
- Training split `game_id` SHA-256: `059f065459f80db7190607d5c73c02d7a4f8df5febceecf4051ad464efc72740`
- Feature fingerprint (242 features): `e4d280bc07f09447968d467bf315c1e7587ae78980bd33be58af76da91c84f8c`
- Library versions: python 3.12.16, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, joblib 1.6.0
