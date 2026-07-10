import unittest
from unittest.mock import patch

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from decision_tree_utils import (
    build_refined_param_grid,
    load_dataset,
    plot_actual_vs_predicted,
    split_train_test,
    tune_decision_tree,
)


class DecisionTreeUtilsTest(unittest.TestCase):
    def test_load_dataset_uses_project_data_directory(self):
        iris = load_dataset("iris.csv")

        self.assertEqual(iris.shape, (150, 5))
        self.assertIn("species", iris.columns)

    def test_split_train_test_uses_reproducible_defaults(self):
        features = pd.DataFrame({"feature": range(8)})
        target = pd.Series(range(8))

        first_split = split_train_test(features, target)
        second_split = split_train_test(features, target)

        for first, second in zip(first_split, second_split):
            self.assertTrue(first.equals(second))

    def test_build_refined_param_grid_centers_valid_ranges(self):
        grid = build_refined_param_grid(
            {
                "max_depth": 2,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
            }
        )

        self.assertEqual(grid["max_depth"], [1, 2, 3, 4])
        self.assertEqual(grid["min_samples_split"], [2, 3, 4])
        self.assertEqual(grid["min_samples_leaf"], [1, 2])

    def test_tune_decision_tree_returns_fitted_searches(self):
        features, target = make_classification(
            n_samples=60,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )

        random_search, grid_search = tune_decision_tree(
            DecisionTreeClassifier(random_state=42),
            features,
            target,
            scoring="accuracy",
        )

        self.assertIsNotNone(random_search.best_estimator_)
        self.assertIsNotNone(grid_search.best_estimator_)

    @patch("decision_tree_utils.plt.show")
    def test_plot_actual_vs_predicted_displays_plot(self, show):
        plot_actual_vs_predicted([1, 2, 3], [1.1, 1.9, 3.2])

        show.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
