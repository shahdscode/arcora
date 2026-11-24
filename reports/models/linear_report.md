# Arcora Week 3 — Linear and Logistic Regression Results

Interpretable linear models in scikit-learn pipelines (median imputation → standardization → estimator). Ordinary least squares, Ridge (L2), and Lasso (L1) regression are trained for every regression target; L2-regularized logistic regression predicts home-win probabilities. Imputers and scalers are fitted on the training split only.

All models are fitted on the **training split (2003-04 to 2018-19)** and compared on the **validation split (2019-20 to 2020-21)**. The primary 2021-22 test set and the partial 2022-23 holdout are locked and were not evaluated.

## Regression (validation)

| Model | Target | MAE | RMSE | R² | Train MAE | Δ MAE vs best baseline |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `linear_regression` | `home_pts` | 9.470 | 11.995 | 0.0876 | 8.536 | -0.065 |
| `ridge_regression` | `home_pts` | 9.447 | 11.965 | 0.0922 | 8.639 | -0.088 |
| `lasso_regression` | `home_pts` | 9.459 | 11.980 | 0.0900 | 8.587 | -0.076 |
| `linear_regression` | `away_pts` | 9.446 | 11.844 | 0.0856 | 8.590 | 0.017 |
| `ridge_regression` | `away_pts` | 9.383 | 11.756 | 0.0991 | 8.643 | -0.045 |
| `lasso_regression` | `away_pts` | 9.379 | 11.744 | 0.1010 | 8.643 | -0.049 |
| `linear_regression` | `total_pts` | 15.434 | 19.519 | 0.0654 | 14.155 | 0.063 |
| `ridge_regression` | `total_pts` | 15.237 | 19.342 | 0.0823 | 14.336 | -0.134 |
| `lasso_regression` | `total_pts` | 15.281 | 19.387 | 0.0781 | 14.381 | -0.090 |
| `linear_regression` | `point_margin` | 10.837 | 13.686 | 0.1269 | 9.443 | -0.129 |
| `ridge_regression` | `point_margin` | 10.818 | 13.681 | 0.1275 | 9.458 | -0.147 |
| `lasso_regression` | `point_margin` | 10.777 | 13.670 | 0.1290 | 9.503 | -0.188 |

## Classification — `home_win` (validation)

| Model | Accuracy | Balanced acc. | Precision | Recall | ROC-AUC | Log loss | Brier | Δ log loss vs best baseline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `logistic_regression` | 0.6283 | 0.6136 | 0.6309 | 0.7715 | 0.6787 | 0.6428 | 0.2255 | -0.0088 |

## Hyperparameters and training configuration

| Model | Target | Parameters |
| :--- | :--- | :--- |
| `linear_regression` | `home_pts` | `estimator=LinearRegression`, `fit_intercept=True`, `preprocessing=SimpleImputer(median)+StandardScaler` |
| `ridge_regression` | `home_pts` | `alpha=30000.0`, `alpha_grid=[1.0, 10.0, 100.0, 1000.0, 10000.0, 30000.0, 100000.0, 300000.0]`, `estimator=Ridge`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `lasso_regression` | `home_pts` | `alpha=0.1`, `alpha_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]`, `estimator=Lasso`, `max_iter=20000`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `linear_regression` | `away_pts` | `estimator=LinearRegression`, `fit_intercept=True`, `preprocessing=SimpleImputer(median)+StandardScaler` |
| `ridge_regression` | `away_pts` | `alpha=10000.0`, `alpha_grid=[1.0, 10.0, 100.0, 1000.0, 10000.0, 30000.0, 100000.0, 300000.0]`, `estimator=Ridge`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `lasso_regression` | `away_pts` | `alpha=0.1`, `alpha_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]`, `estimator=Lasso`, `max_iter=20000`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `linear_regression` | `total_pts` | `estimator=LinearRegression`, `fit_intercept=True`, `preprocessing=SimpleImputer(median)+StandardScaler` |
| `ridge_regression` | `total_pts` | `alpha=30000.0`, `alpha_grid=[1.0, 10.0, 100.0, 1000.0, 10000.0, 30000.0, 100000.0, 300000.0]`, `estimator=Ridge`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `lasso_regression` | `total_pts` | `alpha=1.0`, `alpha_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]`, `estimator=Lasso`, `max_iter=20000`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `linear_regression` | `point_margin` | `estimator=LinearRegression`, `fit_intercept=True`, `preprocessing=SimpleImputer(median)+StandardScaler` |
| `ridge_regression` | `point_margin` | `alpha=1000.0`, `alpha_grid=[1.0, 10.0, 100.0, 1000.0, 10000.0, 30000.0, 100000.0, 300000.0]`, `estimator=Ridge`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `lasso_regression` | `point_margin` | `alpha=0.1`, `alpha_grid=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]`, `estimator=Lasso`, `max_iter=20000`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42` |
| `logistic_regression` | `home_win` | `C=0.001`, `C_grid=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1]`, `class_weight=None`, `estimator=LogisticRegression`, `max_iter=5000`, `penalty=l2`, `preprocessing=SimpleImputer(median)+StandardScaler`, `random_state=42`, `solver=lbfgs` |

## Comparison with every trained family (validation)

Best model per family on the primary metric (MAE for regression, log loss for `home_win`), evaluated on the identical validation split.

| Target | Metric | `baseline` | `linear` | `random_forest` | `xgboost` |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `home_pts` | mae | 9.5347 (`historical_reference`) | 9.4469 (`ridge_regression`) | 9.4782 (`random_forest_regressor`) | 9.4889 (`xgboost_regressor`) |
| `away_pts` | mae | 9.4288 (`historical_reference`) | 9.3794 (`lasso_regression`) | 9.3942 (`random_forest_regressor`) | 9.4011 (`xgboost_regressor`) |
| `total_pts` | mae | 15.3710 (`historical_reference`) | 15.2369 (`ridge_regression`) | 15.3250 (`random_forest_regressor`) | 15.4129 (`xgboost_regressor`) |
| `point_margin` | mae | 10.9651 (`historical_reference`) | 10.7766 (`lasso_regression`) | 10.7479 (`random_forest_regressor`) | 10.7550 (`xgboost_regressor`) |
| `home_win` | log_loss | 0.6516 (`historical_reference`) | 0.6428 (`logistic_regression`) | 0.6393 (`random_forest_classifier`) | 0.6402 (`xgboost_classifier`) |

## Regularization selection (validation)

| Model | Target | Parameter | Selected | Validation score | Grid |
| :--- | :--- | :--- | ---: | ---: | :--- |
| `ridge_regression` | `home_pts` | `alpha` | 30000.0 | 9.4469 (mae) | 1.0→9.4732, 10.0→9.4722, 100.0→9.4726, 1000.0→9.4685, 10000.0→9.4571, 30000.0→9.4469, 100000.0→9.4631, 300000.0→9.8037 |
| `lasso_regression` | `home_pts` | `alpha` | 0.1 | 9.4592 (mae) | 0.01→9.4777, 0.03→9.4707, 0.1→9.4592, 0.3→9.4666, 1.0→9.4998, 3.0→10.4691 |
| `ridge_regression` | `away_pts` | `alpha` | 10000.0 | 9.3834 (mae) | 1.0→9.4366, 10.0→9.4286, 100.0→9.4166, 1000.0→9.4058, 10000.0→9.3834, 30000.0→9.3843, 100000.0→9.4588, 300000.0→9.9886 |
| `lasso_regression` | `away_pts` | `alpha` | 0.1 | 9.3794 (mae) | 0.01→9.4027, 0.03→9.3855, 0.1→9.3794, 0.3→9.3913, 1.0→9.4683, 3.0→10.8326 |
| `ridge_regression` | `total_pts` | `alpha` | 30000.0 | 15.2369 (mae) | 1.0→15.4278, 10.0→15.4193, 100.0→15.3975, 1000.0→15.3599, 10000.0→15.2831, 30000.0→15.2369, 100000.0→15.3030, 300000.0→16.2768 |
| `lasso_regression` | `total_pts` | `alpha` | 1.0 | 15.2806 (mae) | 0.01→15.3867, 0.03→15.3525, 0.1→15.3217, 0.3→15.3008, 1.0→15.2806, 3.0→15.7967 |
| `ridge_regression` | `point_margin` | `alpha` | 1000.0 | 10.8180 (mae) | 1.0→10.8311, 10.0→10.8254, 100.0→10.8216, 1000.0→10.8180, 10000.0→10.8320, 30000.0→10.8489, 100000.0→10.8883, 300000.0→10.9775 |
| `lasso_regression` | `point_margin` | `alpha` | 0.1 | 10.7766 (mae) | 0.01→10.8125, 0.03→10.8011, 0.1→10.7766, 0.3→10.8163, 1.0→10.9148, 3.0→11.2104 |
| `logistic_regression` | `home_win` | `C` | 0.001 | 0.6428 (log_loss) | 0.0001→0.6434, 0.0003→0.6429, 0.001→0.6428, 0.003→0.6431, 0.01→0.6435, 0.1→0.6442 |

## Class imbalance investigation (`home_win`)

Training home-win rate: **0.598**; validation home-win rate: **0.547**.

| Variant | Accuracy | Balanced acc. | Recall | ROC-AUC | Log loss | Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| class_weight=None | 0.6283 | 0.6136 | 0.7715 | 0.6787 | 0.6428 | 0.2255 |
| class_weight='balanced' | 0.6253 | 0.6291 | 0.5889 | 0.6788 | 0.6502 | 0.2286 |

Decision: Keep class_weight=None: the ~60/40 split is mild and balancing distorts probability calibration (higher log loss / Brier) for a small balanced-accuracy gain.

## Top standardized coefficients — `ridge_regression` / `point_margin`

Coefficients are on the standardized feature scale. Many features are highly collinear (e.g. home, away, and difference features), so individual coefficients are not causal effects.

| Feature | Category | Coefficient |
| :--- | :--- | ---: |
| `home_standings_losses` | Standings and Season Context | -1.4527 |
| `home_standings_wins` | Standings and Season Context | +1.4392 |
| `away_standings_losses` | Standings and Season Context | +1.0358 |
| `away_standings_wins` | Standings and Season Context | -1.0001 |
| `away_is_back_to_back` | Rest and Schedule-Related Features | +0.7484 |
| `venue_margin_avg_diff` | Home-Away Matchup Differences | +0.6550 |
| `h2h_home_team_margin_avg` | Head-to-Head Historical Performance | +0.5429 |
| `home_is_back_to_back` | Rest and Schedule-Related Features | -0.5344 |
| `away_rolling_fg_pct_10` | Rolling Offensive and Defensive Statistics | +0.4908 |
| `rolling_fg_pct_10_diff` | Home-Away Matchup Differences | -0.4519 |

## Top standardized coefficients — `logistic_regression` / `home_win`

Coefficients are on the standardized feature scale. Many features are highly collinear (e.g. home, away, and difference features), so individual coefficients are not causal effects.

| Feature | Category | Coefficient |
| :--- | :--- | ---: |
| `home_standings_losses` | Standings and Season Context | -0.1375 |
| `home_standings_wins` | Standings and Season Context | +0.1285 |
| `away_standings_losses` | Standings and Season Context | +0.0968 |
| `away_standings_wins` | Standings and Season Context | -0.0798 |
| `away_is_back_to_back` | Rest and Schedule-Related Features | +0.0776 |
| `h2h_home_team_margin_avg` | Head-to-Head Historical Performance | +0.0689 |
| `venue_margin_avg_diff` | Home-Away Matchup Differences | +0.0547 |
| `home_is_back_to_back` | Rest and Schedule-Related Features | -0.0474 |
| `away_hist_fg_pct` | Historical Team Performance | +0.0440 |
| `away_rolling_fg_pct_10` | Rolling Offensive and Defensive Statistics | +0.0434 |

## Reproducibility

- Dataset fingerprint: `eebe4ed42461191c0073cf1817916da55887a0ebab333dbf84781bed2c1d8ff6`
- Training split `game_id` SHA-256: `059f065459f80db7190607d5c73c02d7a4f8df5febceecf4051ad464efc72740`
- Feature fingerprint (242 features): `e4d280bc07f09447968d467bf315c1e7587ae78980bd33be58af76da91c84f8c`
- Library versions: python 3.12.16, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, joblib 1.6.0
