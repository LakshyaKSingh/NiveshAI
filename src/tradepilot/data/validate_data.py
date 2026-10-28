"""
Data Validation Module
======================
Validates schema integrity, financial logical constraints, duplicate records,
missing values, and prevents lookahead target leakage.
"""

import logging
from typing import List, Optional, Set

import pandas as pd

from tradepilot.config import DEFAULT_DATA_CONFIG

logger = logging.getLogger(__name__)


def validate_ohlcv_schema(df: pd.DataFrame) -> None:
    """
    Validate raw OHLCV dataset structure and physical price logic.

    Raises:
        ValueError: If columns are missing, duplicates exist, or physical bounds are violated.
    """
    required_cols = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw OHLCV missing required columns: {sorted(missing)}")

    # Duplicate check
    duplicates = df.duplicated(subset=["symbol", "timestamp"]).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate (symbol, timestamp) records in raw OHLCV.")

    # Non-positive prices or negative volume
    invalid_prices = (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
    if invalid_prices.any():
        raise ValueError(f"Found {invalid_prices.sum()} rows with non-positive price values.")

    negative_vol = df["volume"] < 0
    if negative_vol.any():
        raise ValueError(f"Found {negative_vol.sum()} rows with negative volume.")

    # Logical OHLC bounds
    high_violations = (df["high"] < df["open"]) | (df["high"] < df["close"])
    if high_violations.any():
        raise ValueError(f"Logical OHLC violation: High < Open or High < Close in {high_violations.sum()} rows.")

    low_violations = (df["low"] > df["open"]) | (df["low"] > df["close"])
    if low_violations.any():
        raise ValueError(f"Logical OHLC violation: Low > Open or Low > Close in {low_violations.sum()} rows.")

    low_high_violations = df["low"] > df["high"]
    if low_high_violations.any():
        raise ValueError(f"Logical OHLC violation: Low > High in {low_high_violations.sum()} rows.")

    logger.info("Raw OHLCV data passed all physical integrity and schema checks.")


def validate_features_dataset(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    prohibited_cols: Optional[Set[str]] = None,
    expected_classes: Optional[List[str]] = None,
) -> None:
    """
    Validate processed feature dataset for model consumption.

    Raises:
        ValueError: If features are missing, leakage columns are included,
                    NaNs are detected, or target classes do not align.
    """
    features = feature_cols or DEFAULT_DATA_CONFIG.feature_cols
    target = target_col or DEFAULT_DATA_CONFIG.target_col
    prohibited = prohibited_cols or DEFAULT_DATA_CONFIG.prohibited_cols
    classes = expected_classes or DEFAULT_DATA_CONFIG.target_classes

    # 1. Feature existence check
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required feature columns in dataset: {missing_feats}")

    # 2. Target leakage check: ensure prohibited columns are not in input features
    for f in features:
        if f in prohibited:
            raise ValueError(f"Target leakage violation: Prohibited column '{f}' found in model features.")

    # 3. Target existence
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # 4. Check future_return_5d exclusion
    if "future_return_5d" in df.columns and "future_return_5d" in features:
        raise ValueError("Target leakage violation: 'future_return_5d' must never be in feature inputs.")

    # 5. Missing values
    nan_features = df[features].isna().sum()
    if nan_features.any():
        failing = nan_features[nan_features > 0].to_dict()
        raise ValueError(f"NaN values detected in feature columns: {failing}")

    nan_target = df[target].isna().sum()
    if nan_target > 0:
        raise ValueError(f"Detected {nan_target} NaN values in target column '{target}'.")

    # 6. Duplicates
    if "symbol" in df.columns and "timestamp" in df.columns:
        dups = df.duplicated(subset=["symbol", "timestamp"]).sum()
        if dups > 0:
            raise ValueError(f"Detected {dups} duplicate (symbol, timestamp) records in feature dataset.")

    # 7. Target classes alignment
    actual_classes = sorted(df[target].unique())
    exp_classes = sorted(classes)
    if actual_classes != exp_classes:
        raise ValueError(f"Target classes {actual_classes} do not match expected {exp_classes}.")

    logger.info("Processed feature dataset passed schema, leakage, and integrity checks.")
