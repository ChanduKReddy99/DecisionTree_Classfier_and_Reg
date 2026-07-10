from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from decision_tree_models.regressor import (
    adjusted_r2_score,
    evaluate_regressor,
    load_regression_data,
    scale_features,
    train_regressor,
)


DATA_PATH = Path(__file__).parents[1] / "data" / "cars_preprocessed_data.csv"


def test_regressor_workflow_scales_training_data_and_returns_metrics() -> None:
    features, target = load_regression_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    regressor = train_regressor(X_train_scaled, y_train, max_depth=9)
    metrics = evaluate_regressor(regressor, X_test_scaled, y_test)

    assert features.shape == (205, 15)
    np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0))
    assert np.isfinite(metrics.r2)
    assert metrics.mean_squared_error >= 0


def test_adjusted_r2_score_matches_the_notebook_formula() -> None:
    assert adjusted_r2_score(0.8, sample_count=100, feature_count=4) == pytest.approx(
        0.791578947368421
    )


def test_adjusted_r2_score_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        adjusted_r2_score(0.8, sample_count=5, feature_count=4)
