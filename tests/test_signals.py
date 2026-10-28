"""
Tests for Probability-to-Class Mapping and Signal Generation
============================================================
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from tradepilot.backtesting.signals import SignalGenerator


class MockClassifier(BaseEstimator, ClassifierMixin):
    """Mock classifier allowing arbitrary class order and fixed probabilities."""
    def __init__(self, classes, prob_matrix):
        self.classes_ = np.array(classes)
        self.prob_matrix = prob_matrix

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.prob_matrix


def test_signals_class_order_independence():
    # Test with inverted class order: ['UP', 'NEUTRAL', 'DOWN'] vs standard ['DOWN', 'NEUTRAL', 'UP']
    # Observation 0: High UP probability (0.70) -> Should signal BUY (1)
    # Observation 1: High DOWN probability (0.60) -> Should signal EXIT (-1)
    # Observation 2: Unconfident / NEUTRAL (0.33 each) -> Should signal HOLD (0)
    classes_inverted = ["UP", "NEUTRAL", "DOWN"]
    probs = np.array([
        [0.70, 0.20, 0.10],
        [0.10, 0.30, 0.60],
        [0.34, 0.33, 0.33],
    ])

    pipeline = Pipeline([("classifier", MockClassifier(classes_inverted, probs))])

    gen = SignalGenerator(
        pipeline=pipeline,
        feature_cols=["f1"],
        entry_threshold=0.40,
    )

    df_test = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "symbol": ["AAPL"] * 3,
        "f1": [1.0, 2.0, 3.0],
    })

    signals = gen.generate_signals(df_test)

    assert signals.loc[0, "signal"] == 1
    assert signals.loc[1, "signal"] == -1
    assert signals.loc[2, "signal"] == 0
    assert pytest.approx(signals.loc[0, "prob_up"]) == 0.70
    assert pytest.approx(signals.loc[1, "prob_down"]) == 0.60
