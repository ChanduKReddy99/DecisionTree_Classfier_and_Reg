from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from decision_tree_models.classifier import (
    encode_labels,
    evaluate_classifier,
    load_classification_data,
    train_classifier,
)


DATA_PATH = Path(__file__).parents[1] / "data" / "iris.csv"


def test_classifier_workflow_uses_all_iris_features_and_classes() -> None:
    features, target = load_classification_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    encoder, y_train_encoded, y_test_encoded = encode_labels(y_train, y_test)
    classifier = train_classifier(X_train, y_train_encoded, max_depth=4)
    metrics = evaluate_classifier(classifier, X_test, y_test_encoded)

    assert features.shape == (150, 4)
    assert encoder.classes_.tolist() == ["setosa", "versicolor", "virginica"]
    assert metrics.accuracy >= 0.9
    np.testing.assert_array_equal(
        metrics.confusion_matrix.sum(axis=1),
        np.bincount(y_test_encoded),
    )
    assert all(
        class_name in metrics.classification_report
        for class_name in ("0", "1", "2")
    )
