from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str


def load_classification_data(
    path: str | Path,
    target_column: str = "species",
) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    return data.drop(columns=target_column), data[target_column]


def encode_labels(
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[LabelEncoder, np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    return encoder, y_train_encoded, y_test_encoded


def train_classifier(
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
    max_depth: int | None = None,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    return classifier.fit(features, target)


def evaluate_classifier(
    classifier: DecisionTreeClassifier,
    features: pd.DataFrame | np.ndarray,
    target: pd.Series | np.ndarray,
) -> ClassificationMetrics:
    predictions = classifier.predict(features)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(target, predictions)),
        confusion_matrix=confusion_matrix(target, predictions),
        classification_report=classification_report(
            target,
            predictions,
            zero_division=0,
        ),
    )
