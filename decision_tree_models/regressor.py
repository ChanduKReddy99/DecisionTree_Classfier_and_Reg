from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class RegressionMetrics:
    r2: float
    mean_squared_error: float


def load_regression_data(
    path: str | Path,
    target_column: str = "price",
) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    return data.drop(columns=target_column), data[target_column]


def scale_features(
    train_features: pd.DataFrame | np.ndarray,
    test_features: pd.DataFrame | np.ndarray,
) -> tuple[StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    return scaler, train_scaled, test_scaled


def train_regressor(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    max_depth: int | None = None,
    random_state: int = 42,
) -> DecisionTreeRegressor:
    regressor = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
    )
    return regressor.fit(features, target)


def evaluate_regressor(
    regressor: DecisionTreeRegressor,
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
) -> RegressionMetrics:
    predictions = regressor.predict(features)
    return RegressionMetrics(
        r2=float(r2_score(target, predictions)),
        mean_squared_error=float(mean_squared_error(target, predictions)),
    )


def adjusted_r2_score(
    r2: float,
    sample_count: int,
    feature_count: int,
) -> float:
    if sample_count <= feature_count + 1:
        raise ValueError("sample_count must be greater than feature_count + 1")
    return 1 - (1 - r2) * (sample_count - 1) / (
        sample_count - feature_count - 1
    )
