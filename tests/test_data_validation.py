"""
Tests for Data Schema and Validation Rules
==========================================
"""

import pandas as pd
import pytest

from tradepilot.data.validate_data import validate_features_dataset, validate_ohlcv_schema


@pytest.fixture
def sample_valid_ohlcv():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame({
        "symbol": ["AAPL"] * 10,
        "timestamp": dates,
        "open": [150.0 + i for i in range(10)],
        "high": [155.0 + i for i in range(10)],
        "low": [148.0 + i for i in range(10)],
        "close": [152.0 + i for i in range(10)],
        "volume": [1000000 + i * 1000 for i in range(10)],
    })


def test_validate_ohlcv_schema_passes_valid(sample_valid_ohlcv):
    # Should not raise
    validate_ohlcv_schema(sample_valid_ohlcv)


def test_validate_ohlcv_detects_duplicates(sample_valid_ohlcv):
    df_dup = pd.concat([sample_valid_ohlcv, sample_valid_ohlcv.iloc[0:1]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_ohlcv_schema(df_dup)


def test_validate_ohlcv_detects_negative_price(sample_valid_ohlcv):
    df_bad = sample_valid_ohlcv.copy()
    df_bad.loc[0, "close"] = -10.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_ohlcv_schema(df_bad)


def test_validate_ohlcv_detects_high_low_violation(sample_valid_ohlcv):
    df_bad = sample_valid_ohlcv.copy()
    # High < Open triggers high violation
    df_bad.loc[0, "high"] = 140.0
    with pytest.raises(ValueError, match="Logical OHLC violation"):
        validate_ohlcv_schema(df_bad)

    # Low > High specifically: open=160, close=160, high=150, low=155
    df_bad2 = sample_valid_ohlcv.copy()
    df_bad2.loc[0, "open"] = 160.0
    df_bad2.loc[0, "close"] = 160.0
    df_bad2.loc[0, "high"] = 150.0
    df_bad2.loc[0, "low"] = 155.0
    with pytest.raises(ValueError, match="Logical OHLC violation"):
        validate_ohlcv_schema(df_bad2)


def test_validate_features_dataset_detects_target_leakage():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": pd.date_range("2023-01-01", periods=1, tz="UTC"),
        "return_1d": [0.01],
        "future_return_5d": [0.05],  # Prohibited in features!
        "target": ["UP"],
    })
    with pytest.raises(ValueError, match="Target leakage violation"):
        validate_features_dataset(
            df=df,
            feature_cols=["return_1d", "future_return_5d"],
            target_col="target",
            prohibited_cols={"future_return_5d", "target"},
        )


def test_validate_features_dataset_detects_nan_values():
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "timestamp": pd.date_range("2023-01-01", periods=1, tz="UTC"),
        "return_1d": [float("nan")],
        "target": ["UP"],
    })
    with pytest.raises(ValueError, match="NaN values detected"):
        validate_features_dataset(
            df=df,
            feature_cols=["return_1d"],
            target_col="target",
            prohibited_cols={"target"},
        )
