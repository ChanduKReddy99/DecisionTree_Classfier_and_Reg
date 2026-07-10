from decision_tree_models.tuning import (
    default_parameter_distributions,
    refined_parameter_grid,
)


def test_default_parameter_distributions_match_notebook_search_ranges() -> None:
    assert default_parameter_distributions() == {
        "max_depth": [3, 4, 5, 6],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "min_samples_leaf": [1, 2, 3, 4],
    }


def test_refined_parameter_grid_is_centered_on_best_parameters() -> None:
    assert refined_parameter_grid(
        {
            "max_depth": 5,
            "min_samples_split": 7,
            "min_samples_leaf": 3,
        }
    ) == {
        "max_depth": [3, 4, 5, 6, 7],
        "min_samples_split": [5, 6, 7, 8, 9],
        "min_samples_leaf": [1, 2, 3, 4, 5],
    }


def test_refined_parameter_grid_keeps_values_valid() -> None:
    assert refined_parameter_grid(
        {
            "max_depth": 1,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
        }
    ) == {
        "max_depth": [1, 2, 3],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2, 3],
    }
