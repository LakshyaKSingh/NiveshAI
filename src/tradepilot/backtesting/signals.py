"""
Signal Generation Module
========================
Converts model probability estimates into explicit trading signals.
Maps model classes dynamically without assuming fixed class order.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from tradepilot.config import DEFAULT_BACKTEST_CONFIG, DEFAULT_DATA_CONFIG

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Transforms trained classifier probabilities into actionable long/exit signals.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        feature_cols: Optional[List[str]] = None,
        entry_threshold: float = DEFAULT_BACKTEST_CONFIG.entry_threshold,
    ):
        """
        Initialize the SignalGenerator.

        Parameters:
            pipeline: Fitted scikit-learn Pipeline with predict_proba.
            feature_cols: List of feature names expected by the model.
            entry_threshold: Minimum predicted probability for 'UP' to enter.
        """
        self.pipeline = pipeline
        self.feature_cols = feature_cols or DEFAULT_DATA_CONFIG.feature_cols
        self.entry_threshold = entry_threshold

        classifier = pipeline.named_steps.get("classifier", pipeline)
        if not hasattr(classifier, "predict_proba"):
            raise AttributeError("Classifier in pipeline does not support predict_proba.")

        # Dynamically inspect class ordering
        self.classes_ = list(classifier.classes_)
        logger.info("Classifier classes detected dynamically: %s", self.classes_)

        if "UP" not in self.classes_ or "DOWN" not in self.classes_:
            raise ValueError(f"Required classes 'UP' and 'DOWN' not found in model classes: {self.classes_}")

        self.up_idx = self.classes_.index("UP")
        self.down_idx = self.classes_.index("DOWN")
        self.neutral_idx = self.classes_.index("NEUTRAL") if "NEUTRAL" in self.classes_ else None

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals on out-of-sample data using strictly backward-looking features.

        Parameters:
            df: DataFrame containing symbol, timestamp, and feature columns.

        Returns:
            pd.DataFrame: DataFrame with columns:
                ['timestamp', 'symbol', 'prob_up', 'prob_neutral', 'prob_down', 'signal']
                where signal == 1 (BUY/ENTER), -1 (EXIT), 0 (HOLD/NO_ACTION).
        """
        missing = [f for f in self.feature_cols if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features for signal generation: {missing}")

        X = df[self.feature_cols]
        probs = self.pipeline.predict_proba(X)

        prob_up = probs[:, self.up_idx]
        prob_down = probs[:, self.down_idx]
        prob_neutral = probs[:, self.neutral_idx] if self.neutral_idx is not None else np.zeros(len(df))

        # Long-only rule:
        # Buy if P(UP) >= entry_threshold AND P(UP) > P(DOWN)
        # Exit if P(DOWN) >= entry_threshold AND P(DOWN) > P(UP)
        signals = np.zeros(len(df), dtype=int)
        buy_condition = (prob_up >= self.entry_threshold) & (prob_up > prob_down)
        exit_condition = (prob_down >= self.entry_threshold) & (prob_down > prob_up)

        signals[buy_condition] = 1
        signals[exit_condition] = -1

        result_df = pd.DataFrame({
            "timestamp": df["timestamp"].reset_index(drop=True),
            "symbol": df["symbol"].reset_index(drop=True),
            "prob_up": prob_up,
            "prob_neutral": prob_neutral,
            "prob_down": prob_down,
            "signal": signals,
        })

        logger.info(
            "Signals generated for %d bars: %d BUY, %d EXIT, %d HOLD",
            len(result_df),
            (signals == 1).sum(),
            (signals == -1).sum(),
            (signals == 0).sum(),
        )
        return result_df
