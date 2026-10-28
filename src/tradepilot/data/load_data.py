"""
Data Loading Module
===================
Handles loading raw OHLCV and processed feature datasets with timezone normalization
and chronological sorting.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from tradepilot.config import DEFAULT_DATA_CONFIG

logger = logging.getLogger(__name__)


def load_raw_ohlcv(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw historical OHLCV data from disk.

    Parameters:
        path: Path to the raw OHLCV CSV file. Defaults to config path.

    Returns:
        pd.DataFrame: Loaded and chronologically sorted OHLCV data.
    """
    file_path = path or DEFAULT_DATA_CONFIG.raw_ohlcv_path
    if not file_path.exists():
        raise FileNotFoundError(f"Raw OHLCV data file not found at: {file_path.resolve()}")

    logger.info("Loading raw OHLCV data from %s", file_path)
    df = pd.read_csv(file_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "symbol" in df.columns and "timestamp" in df.columns:
        df = df.sort_values(by=["symbol", "timestamp"]).reset_index(drop=True)

    logger.info("Loaded raw OHLCV: %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


def load_processed_features(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load processed, feature-engineered, and labeled dataset from disk.

    Parameters:
        path: Path to the processed features CSV file. Defaults to config path.

    Returns:
        pd.DataFrame: Cleanly parsed and chronologically sorted dataset.
    """
    file_path = path or DEFAULT_DATA_CONFIG.processed_features_path
    if not file_path.exists():
        raise FileNotFoundError(f"Processed feature data file not found at: {file_path.resolve()}")

    logger.info("Loading processed feature data from %s", file_path)
    df = pd.read_csv(file_path)

    if "timestamp" not in df.columns:
        raise ValueError("Processed dataset must contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "symbol" in df.columns:
        df = df.sort_values(by=["symbol", "timestamp"]).reset_index(drop=True)

    logger.info("Loaded processed features: %d rows x %d cols, %d symbols", len(df), df.shape[1], df["symbol"].nunique())
    return df
