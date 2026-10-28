"""
Data Splitting Module
=====================
Chronological train/validation/test splitting with boundary purging to prevent
target-horizon leakage across splits.
"""

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from tradepilot.config import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_CONFIG

logger = logging.getLogger(__name__)


def chronological_split(
    df: pd.DataFrame,
    train_end_year: int = DEFAULT_MODEL_CONFIG.train_end_year,
    val_year: int = DEFAULT_MODEL_CONFIG.val_year,
    test_year: int = DEFAULT_MODEL_CONFIG.test_year,
    purge_days: int = DEFAULT_DATA_CONFIG.purge_boundary_days,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split the dataset chronologically per symbol with boundary purging.

    Parameters:
        df: Processed DataFrame with 'timestamp' and 'symbol'.
        train_end_year: Final year of training (e.g. 2023).
        val_year: Year for validation (e.g. 2024).
        test_year: Year for final test evaluation (e.g. 2025).
        purge_days: Number of observations per symbol purged at train/val ends.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            (train_df, val_df, test_df, split_info_metadata).
    """
    logger.info(
        "Splitting data chronologically: Train <= %d, Val == %d, Test == %d (Purge: %d days/sym)",
        train_end_year,
        val_year,
        test_year,
        purge_days,
    )

    df_sorted = df.sort_values(by=["symbol", "timestamp"]).reset_index(drop=True)

    train_mask = df_sorted["timestamp"].dt.year <= train_end_year
    val_mask = df_sorted["timestamp"].dt.year == val_year
    test_mask = df_sorted["timestamp"].dt.year == test_year

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    purged_train_count = 0
    purged_val_count = 0

    for sym, group in df_sorted.groupby("symbol", sort=False):
        g_train = group.loc[train_mask[group.index]]
        g_val = group.loc[val_mask[group.index]]
        g_test = group.loc[test_mask[group.index]]

        if len(g_train) <= purge_days or len(g_val) <= purge_days or len(g_test) == 0:
            raise ValueError(
                f"Insufficient data for symbol '{sym}' (Train: {len(g_train)}, Val: {len(g_val)}, Test: {len(g_test)})."
            )

        # Purge boundary rows from Train
        g_train_clean = g_train.iloc[:-purge_days]
        purged_train_count += purge_days
        train_indices.extend(g_train_clean.index)

        # Purge boundary rows from Validation
        g_val_clean = g_val.iloc[:-purge_days]
        purged_val_count += purge_days
        val_indices.extend(g_val_clean.index)

        # Test set
        test_indices.extend(g_test.index)

    train_df = df_sorted.loc[train_indices].copy()
    val_df = df_sorted.loc[val_indices].copy()
    test_df = df_sorted.loc[test_indices].copy()

    # Verify boundary separation
    train_max_date = train_df["timestamp"].max()
    val_min_date = val_df["timestamp"].min()
    val_max_date = val_df["timestamp"].max()
    test_min_date = test_df["timestamp"].min()

    if train_max_date >= val_min_date:
        raise ValueError(
            f"Boundary Leakage: Train max date ({train_max_date}) >= Val min date ({val_min_date})"
        )
    if val_max_date >= test_min_date:
        raise ValueError(
            f"Boundary Leakage: Val max date ({val_max_date}) >= Test min date ({test_min_date})"
        )

    split_info = {
        "train_rows": len(train_df),
        "train_purged_rows": purged_train_count,
        "train_start": train_df["timestamp"].min().strftime("%Y-%m-%d"),
        "train_end": train_max_date.strftime("%Y-%m-%d"),
        "val_rows": len(val_df),
        "val_purged_rows": purged_val_count,
        "val_start": val_min_date.strftime("%Y-%m-%d"),
        "val_end": val_max_date.strftime("%Y-%m-%d"),
        "test_rows": len(test_df),
        "test_start": test_min_date.strftime("%Y-%m-%d"),
        "test_end": test_df["timestamp"].max().strftime("%Y-%m-%d"),
        "purge_days_per_symbol": purge_days,
    }

    logger.info(
        "Split complete: Train=%d (purged %d), Val=%d (purged %d), Test=%d",
        len(train_df),
        purged_train_count,
        len(val_df),
        purged_val_count,
        len(test_df),
    )

    return train_df, val_df, test_df, split_info
