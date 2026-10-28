"""
Model Training Module
=====================
Defines scikit-learn candidate pipelines and executes model training on training folds.
"""

import logging
from typing import Dict

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tradepilot.config import DEFAULT_MODEL_CONFIG

logger = logging.getLogger(__name__)


def get_candidate_pipelines(random_seed: int = DEFAULT_MODEL_CONFIG.random_seed) -> Dict[str, Pipeline]:
    """
    Construct candidate model pipelines using scikit-learn Pipeline.

    Parameters:
        random_seed: Random state for deterministic model fitting.

    Returns:
        Dict[str, Pipeline]: Dictionary of candidate pipelines.
    """
    pipelines = {
        "Majority_Baseline": Pipeline([
            ("classifier", DummyClassifier(strategy="most_frequent"))
        ]),
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_seed,
            )),
        ]),
        "Random_Forest": Pipeline([
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=random_seed,
                n_jobs=-1,
            )),
        ]),
    }
    return pipelines


def train_candidates(
    pipelines: Dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Pipeline]:
    """
    Fit each candidate pipeline on the provided training set.

    Parameters:
        pipelines: Dictionary of candidate pipelines.
        X_train: Training feature matrix.
        y_train: Training target labels.

    Returns:
        Dict[str, Pipeline]: Dictionary of fitted pipelines.
    """
    fitted = {}
    for name, pipe in pipelines.items():
        logger.info("Fitting candidate pipeline: %s...", name)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        logger.info("Successfully fitted: %s", name)
    return fitted
