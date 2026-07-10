from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_PARAM_DISTRIBUTIONS = {
    "max_depth": list(range(3, 7)),
    "min_samples_split": list(range(2, 11)),
    "min_samples_leaf": list(range(1, 5)),
}


def load_dataset(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
):
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


def build_refined_param_grid(best_params: Mapping[str, int]) -> dict[str, list[int]]:
    return {
        "max_depth": _centered_range(best_params["max_depth"], radius=2, minimum=1),
        "min_samples_split": _centered_range(
            best_params["min_samples_split"],
            radius=2,
            minimum=2,
        ),
        "min_samples_leaf": _centered_range(
            best_params["min_samples_leaf"],
            radius=1,
            minimum=1,
        ),
    }


def tune_decision_tree(
    estimator: BaseEstimator,
    features,
    target,
    scoring: str,
) -> tuple[RandomizedSearchCV, GridSearchCV]:
    random_search = RandomizedSearchCV(
        estimator,
        param_distributions=DEFAULT_PARAM_DISTRIBUTIONS,
        n_iter=10,
        cv=5,
        scoring=scoring,
        random_state=42,
    )
    random_search.fit(features, target)

    grid_search = GridSearchCV(
        estimator,
        param_grid=build_refined_param_grid(random_search.best_params_),
        cv=5,
        scoring=scoring,
        verbose=1,
    )
    grid_search.fit(features, target)
    return random_search, grid_search


def plot_actual_vs_predicted(actual, predicted) -> None:
    minimum = min(min(actual), min(predicted))
    maximum = max(max(actual), max(predicted))

    plt.figure()
    plt.scatter(actual, predicted, color="blue", alpha=0.5, label="Actual & Predicted")
    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="red",
        linestyle="--",
        label="Regression Line",
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs. Predicted Values")
    plt.legend()
    plt.show()


def _centered_range(value: int, radius: int, minimum: int) -> list[int]:
    start = max(minimum, value - radius)
    return list(range(start, value + radius + 1))
