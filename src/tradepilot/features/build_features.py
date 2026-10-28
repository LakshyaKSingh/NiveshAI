"""
Feature Engineering Module
==========================
Generates technical indicators and forward-looking prediction targets per symbol
without forward lookahead leakage in features.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from tradepilot.config import DEFAULT_DATA_CONFIG

logger = logging.getLogger(__name__)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI) using standard rolling window.

    Parameters:
        series: Closing price series.
        period: Lookback window (default 14).

    Returns:
        pd.Series: Calculated RSI values in [0, 100].
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Moving Average Convergence Divergence (MACD) and signal line.

    Parameters:
        series: Closing price series.
        fast: Fast EMA span (default 12).
        slow: Slow EMA span (default 26).
        signal: Signal line EMA span (default 9).

    Returns:
        Tuple[pd.Series, pd.Series]: (MACD line, Signal line).
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute strictly backward-looking technical indicators grouped by symbol.

    Parameters:
        df: Raw OHLCV DataFrame sorted by (symbol, timestamp).

    Returns:
        pd.DataFrame: DataFrame with engineered technical features added.
    """
    df_out = df.copy()

    # Returns
    df_out["return_1d"] = df_out.groupby("symbol")["close"].pct_change(1)
    df_out["return_5d"] = df_out.groupby("symbol")["close"].pct_change(5)
    df_out["return_10d"] = df_out.groupby("symbol")["close"].pct_change(10)

    # Volatility & Volume
    df_out["volatility_10d"] = df_out.groupby("symbol")["return_1d"].transform(
        lambda x: x.rolling(10).std()
    )
    df_out["volume_change"] = df_out.groupby("symbol")["volume"].pct_change(1)

    # RSI
    df_out["RSI_14"] = df_out.groupby("symbol")["close"].transform(
        lambda x: calculate_rsi(x, period=14)
    )

    # MACD & Signal line: compute robustly per group preserving the original index
    macd_series_list = []
    signal_series_list = []

    for _, group in df_out.groupby("symbol", sort=False):
        m, s = calculate_macd(group["close"])
        macd_series_list.append(m)
        signal_series_list.append(s)

    df_out["MACD"] = pd.concat(macd_series_list).sort_index()
    df_out["MACD_signal"] = pd.concat(signal_series_list).sort_index()

    return df_out


def generate_features_and_targets(
    df_raw: pd.DataFrame,
    horizon: int = DEFAULT_DATA_CONFIG.prediction_horizon_days,
    threshold: float = DEFAULT_DATA_CONFIG.target_threshold,
) -> pd.DataFrame:
    """
    Generate complete technical features and future-return target classification.

    Parameters:
        df_raw: Raw OHLCV DataFrame.
        horizon: Prediction horizon in trading days (default 5).
        threshold: Absolute return threshold for UP/DOWN classes (default 0.01).

    Returns:
        pd.DataFrame: Clean dataset with NaNs dropped.
    """
    logger.info("Computing technical features across %d symbols...", df_raw["symbol"].nunique())
    df_feat = compute_technical_features(df_raw)

    # Target calculation: forward 5-day return
    df_feat["future_return_5d"] = (
        df_feat.groupby("symbol")["close"].shift(-horizon) / df_feat["close"] - 1.0
    )

    def assign_target(ret: float) -> str:
        if pd.isna(ret):
            return np.nan
        if ret > threshold:
            return "UP"
        if ret < -threshold:
            return "DOWN"
        return "NEUTRAL"

    df_feat["target"] = df_feat["future_return_5d"].apply(assign_target)

    # Drop rolling warm-up NaNs and terminal shift NaNs
    df_clean = df_feat.dropna().reset_index(drop=True)
    logger.info("Generated feature dataset: %d rows remaining after dropna", len(df_clean))
    return df_clean
