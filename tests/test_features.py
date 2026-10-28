"""
Tests for Feature Engineering and Target Generation
===================================================
"""

import numpy as np
import pandas as pd
import pytest

from tradepilot.features.build_features import (
    calculate_macd,
    calculate_rsi,
    compute_technical_features,
    generate_features_and_targets,
)


def test_rsi_bounds():
    prices = pd.Series([100.0 + (i % 5) * 2.0 for i in range(50)])
    rsi = calculate_rsi(prices, period=14)
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all()
    assert (valid_rsi <= 100.0).all()


def test_macd_calculation():
    prices = pd.Series([100.0 + i * 0.5 for i in range(60)])
    macd_line, signal_line = calculate_macd(prices, fast=12, slow=26, signal=9)
    assert len(macd_line) == len(prices)
    assert len(signal_line) == len(prices)
    # Steady upward drift implies fast EMA > slow EMA, hence MACD > 0
    assert macd_line.iloc[-1] > 0.0


def test_features_independent_per_symbol():
    # Verify symbols do not bleed across groups
    dates = pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC")
    df = pd.DataFrame({
        "symbol": ["A"] * 20 + ["B"] * 20,
        "timestamp": list(dates) + list(dates),
        "open": [10.0] * 20 + [100.0] * 20,
        "high": [12.0] * 20 + [102.0] * 20,
        "low": [9.0] * 20 + [99.0] * 20,
        "close": [11.0 + (i * 0.5) for i in range(20)] + [101.0 + (i * 2.0) for i in range(20)],
        "volume": [1000] * 40,
    })
    feat_df = compute_technical_features(df)
    # The first row of symbol B must have NaN for return_1d because it is the first bar of B
    first_b_idx = feat_df[feat_df["symbol"] == "B"].index[0]
    assert pd.isna(feat_df.loc[first_b_idx, "return_1d"])


def test_target_classification_logic():
    dates = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
    # Upward ramp
    df_raw = pd.DataFrame({
        "symbol": ["TEST"] * 30,
        "timestamp": dates,
        "open": [100.0 + i for i in range(30)],
        "high": [102.0 + i for i in range(30)],
        "low": [99.0 + i for i in range(30)],
        "close": [100.0 * (1.02 ** i) for i in range(30)],
        "volume": [50000] * 30,
    })
    clean_df = generate_features_and_targets(df_raw, horizon=5, threshold=0.01)
    # Since compounding at 2% daily, 5-day return is (1.02)^5 - 1 ~ 10.4% > 1%, target should be 'UP'
    assert (clean_df["target"] == "UP").all()
