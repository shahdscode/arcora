"""Shared pytest fixtures for modeling tests (Week 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arcora.features.strategy import FEATURE_COLUMN_NAMES
from arcora.models.data import ModelingData, prepare_modeling_data
from arcora.models.modeling_strategy import APPROVED_MISSING_INDICATORS

SIGNAL_FEATURE = "home_hist_net_rtg"
REFERENCE_FEATURES = (
    "home_hist_pts_avg",
    "home_hist_pts_allowed_avg",
    "away_hist_pts_avg",
    "away_hist_pts_allowed_avg",
)


def make_synthetic_feature_dataset(games_per_season: int = 30, seed: int = 0) -> pd.DataFrame:
    """Build a small feature dataset with the real schema and a learnable signal.

    Seasons 2003-2022 are covered so every chronological split is populated; the 2022 season
    is flagged as partial. ``home_hist_net_rtg`` drives the point margin so models can learn.
    """
    rng = np.random.default_rng(seed)
    rows = []
    game_id = 1
    for season in range(2003, 2023):
        for i in range(games_per_season):
            rows.append(
                {
                    "game_id": game_id,
                    "game_date": pd.Timestamp(f"{season}-11-01") + pd.Timedelta(days=i // 3),
                    "season": season,
                    "season_label": f"{season}-{(season + 1) % 100:02d}",
                    "game_type": "Playoffs" if i == games_per_season - 1 else "Regular Season",
                    "is_partial_season": season == 2022,
                    "home_team_id": 10 + (i % 4),
                    "away_team_id": 20 + (i % 4),
                }
            )
            game_id += 1
    df = pd.DataFrame(rows)
    n = len(df)
    feats = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMN_NAMES))), columns=list(FEATURE_COLUMN_NAMES)
    )
    for col in REFERENCE_FEATURES:
        feats[col] = rng.normal(105.0, 4.0, n)
    for col in APPROVED_MISSING_INDICATORS:
        feats[col] = rng.integers(0, 2, n).astype(bool)
    # Early-season style missingness on a few features.
    feats.loc[feats.index[::17], ["home_hist_pts_avg", "away_rolling_pts_avg_5"]] = np.nan
    df = pd.concat([df, feats], axis=1)

    signal = df[SIGNAL_FEATURE].to_numpy()
    df["home_pts"] = np.round(108 + 4 * signal + rng.normal(0, 6, n)).astype("int64")
    df["away_pts"] = np.round(105 - 4 * signal + rng.normal(0, 6, n)).astype("int64")
    ties = df["home_pts"] == df["away_pts"]
    df.loc[ties, "home_pts"] += 1
    df["point_margin"] = df["home_pts"] - df["away_pts"]
    df["total_pts"] = df["home_pts"] + df["away_pts"]
    df["home_win"] = df["home_pts"] > df["away_pts"]
    return df


@pytest.fixture(scope="session")
def synthetic_feature_dataset() -> pd.DataFrame:
    """Session-scoped synthetic feature dataset."""
    return make_synthetic_feature_dataset()


@pytest.fixture(scope="session")
def synthetic_modeling_data(synthetic_feature_dataset: pd.DataFrame) -> ModelingData:
    """Session-scoped chronological ModelingData built from the synthetic dataset."""
    return prepare_modeling_data(synthetic_feature_dataset, source_dataset="synthetic.parquet")
