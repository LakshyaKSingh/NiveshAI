"""
Tests for Pipeline Reproducibility and Serialization
====================================================
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from tradepilot.modeling.evaluate_model import evaluate_classifier
from tradepilot.modeling.train_model import get_candidate_pipelines, train_candidates


def test_model_training_reproducibility():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        f"feat_{i}": np.random.randn(n) for i in range(5)
    })
    y = pd.Series(np.random.choice(["DOWN", "NEUTRAL", "UP"], size=n))

    # Fit run 1
    pipes1 = get_candidate_pipelines(random_seed=42)
    fitted1 = train_candidates(pipes1, X, y)
    preds1 = fitted1["Random_Forest"].predict(X)

    # Fit run 2
    pipes2 = get_candidate_pipelines(random_seed=42)
    fitted2 = train_candidates(pipes2, X, y)
    preds2 = fitted2["Random_Forest"].predict(X)

    # Predictions must match exactly
    assert (preds1 == preds2).all()


def test_model_serialization_integrity():
    np.random.seed(42)
    n = 100
    X = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})
    y = pd.Series(np.random.choice(["DOWN", "NEUTRAL", "UP"], size=n))

    pipes = get_candidate_pipelines(random_seed=42)
    fitted = train_candidates(pipes, X, y)
    rf = fitted["Random_Forest"]

    original_preds = rf.predict(X)
    original_probs = rf.predict_proba(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_model_path = Path(tmpdir) / "test_model.joblib"
        joblib.dump(rf, tmp_model_path)

        reloaded = joblib.load(tmp_model_path)
        reloaded_preds = reloaded.predict(X)
        reloaded_probs = reloaded.predict_proba(X)

        assert (original_preds == reloaded_preds).all()
        assert np.allclose(original_probs, reloaded_probs)
