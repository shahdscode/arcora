# Arcora Week 3 — Naive Baselines Results

Naive reference models learned exclusively from the 2003-04 to 2018-19 training split. Regression baselines predict a constant (training mean or median); classification baselines predict the majority class or the training home-win rate. The historical reference combines each team's season-to-date scoring averages with a training-learned home-court offset and falls back to league training means when a team has no history.

All models are fitted on the **training split (2003-04 to 2018-19)** and compared on the **validation split (2019-20 to 2020-21)**. The primary 2021-22 test set and the partial 2022-23 holdout are locked and were not evaluated.

## Regression (validation)

| Model | Target | MAE | RMSE | R² | Train MAE |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `mean_baseline` | `home_pts` | 13.298 | 16.510 | -0.7285 | 10.101 |
| `median_baseline` | `home_pts` | 13.232 | 16.443 | -0.7144 | 10.101 |
| `historical_reference` | `home_pts` | 9.535 | 12.067 | 0.0766 | 8.835 |
| `mean_baseline` | `away_pts` | 14.238 | 17.372 | -0.9671 | 10.140 |
| `median_baseline` | `away_pts` | 14.809 | 17.961 | -1.1027 | 10.138 |
| `historical_reference` | `away_pts` | 9.429 | 11.887 | 0.0789 | 8.852 |
| `mean_baseline` | `total_pts` | 25.213 | 30.530 | -1.2862 | 17.268 |
| `median_baseline` | `total_pts` | 25.756 | 31.074 | -1.3685 | 17.246 |
| `historical_reference` | `total_pts` | 15.371 | 19.512 | 0.0662 | 14.674 |
| `mean_baseline` | `point_margin` | 11.679 | 14.720 | -0.0100 | 10.540 |
| `median_baseline` | `point_margin` | 11.732 | 14.841 | -0.0267 | 10.507 |
| `historical_reference` | `point_margin` | 10.965 | 13.898 | 0.0997 | 9.755 |

## Classification — `home_win` (validation)

| Model | Accuracy | Balanced acc. | Precision | Recall | ROC-AUC | Log loss | Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `most_frequent_baseline` | 0.5467 | 0.5000 | 0.5467 | 1.0000 | 0.5000 | 15.6577 | 0.4533 |
| `prior_baseline` | 0.5467 | 0.5000 | 0.5467 | 1.0000 | 0.5000 | 0.6943 | 0.2505 |
| `historical_reference` | 0.6175 | 0.5902 | 0.6025 | 0.8830 | 0.6744 | 0.6516 | 0.2297 |

## Hyperparameters and training configuration

| Model | Target | Parameters |
| :--- | :--- | :--- |
| `mean_baseline` | `home_pts` | `estimator=DummyRegressor`, `strategy=mean` |
| `median_baseline` | `home_pts` | `estimator=DummyRegressor`, `strategy=median` |
| `historical_reference` | `home_pts` | `estimator=HistoricalReferenceModel`, `features_used=['home_hist_pts_avg', 'home_hist_pts_allowed_avg', 'away_hist_pts_avg', 'away_hist_pts_allowed_avg']` |
| `mean_baseline` | `away_pts` | `estimator=DummyRegressor`, `strategy=mean` |
| `median_baseline` | `away_pts` | `estimator=DummyRegressor`, `strategy=median` |
| `historical_reference` | `away_pts` | `estimator=HistoricalReferenceModel`, `features_used=['home_hist_pts_avg', 'home_hist_pts_allowed_avg', 'away_hist_pts_avg', 'away_hist_pts_allowed_avg']` |
| `mean_baseline` | `total_pts` | `estimator=DummyRegressor`, `strategy=mean` |
| `median_baseline` | `total_pts` | `estimator=DummyRegressor`, `strategy=median` |
| `historical_reference` | `total_pts` | `estimator=HistoricalReferenceModel`, `features_used=['home_hist_pts_avg', 'home_hist_pts_allowed_avg', 'away_hist_pts_avg', 'away_hist_pts_allowed_avg']` |
| `mean_baseline` | `point_margin` | `estimator=DummyRegressor`, `strategy=mean` |
| `median_baseline` | `point_margin` | `estimator=DummyRegressor`, `strategy=median` |
| `historical_reference` | `point_margin` | `estimator=HistoricalReferenceModel`, `features_used=['home_hist_pts_avg', 'home_hist_pts_allowed_avg', 'away_hist_pts_avg', 'away_hist_pts_allowed_avg']` |
| `most_frequent_baseline` | `home_win` | `estimator=DummyClassifier`, `strategy=most_frequent` |
| `prior_baseline` | `home_win` | `estimator=DummyClassifier`, `strategy=prior` |
| `historical_reference` | `home_win` | `estimator=HistoricalReferenceModel`, `features_used=['home_hist_pts_avg', 'home_hist_pts_allowed_avg', 'away_hist_pts_avg', 'away_hist_pts_allowed_avg']` |

## Reproducibility

- Dataset fingerprint: `eebe4ed42461191c0073cf1817916da55887a0ebab333dbf84781bed2c1d8ff6`
- Training split `game_id` SHA-256: `059f065459f80db7190607d5c73c02d7a4f8df5febceecf4051ad464efc72740`
- Feature fingerprint (242 features): `e4d280bc07f09447968d467bf315c1e7587ae78980bd33be58af76da91c84f8c`
- Library versions: python 3.12.16, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, xgboost 3.4.1, joblib 1.6.0
