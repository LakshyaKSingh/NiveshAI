"""
Tests for Chronological Splitting and Boundary Purging
======================================================
"""

import pandas as pd
import pytest

from tradepilot.modeling.split_data import chronological_split


def test_chronological_split_and_purge():
    # 2 symbols across 2023, 2024, 2025
    # 20 observations per year per symbol
    records = []
    for sym in ["SYM1", "SYM2"]:
        for year in [2023, 2024, 2025]:
            dates = pd.date_range(f"{year}-01-10", periods=20, freq="D", tz="UTC")
            for d in dates:
                records.append({
                    "symbol": sym,
                    "timestamp": d,
                    "feat1": 1.0,
                    "target": "UP",
                })
    df = pd.DataFrame(records)

    train_df, val_df, test_df, split_info = chronological_split(
        df=df,
        train_end_year=2023,
        val_year=2024,
        test_year=2025,
        purge_days=5,
    )

    # 20 obs in 2023 - 5 purged = 15 obs per symbol = 30 train rows
    assert len(train_df) == 30
    assert split_info["train_purged_rows"] == 10  # 5 * 2 symbols

    # 20 obs in 2024 - 5 purged = 15 obs per symbol = 30 val rows
    assert len(val_df) == 30
    assert split_info["val_purged_rows"] == 10

    # 20 obs in 2025, unpurged = 40 test rows
    assert len(test_df) == 40

    # Boundary separation check
    assert train_df["timestamp"].max() < val_df["timestamp"].min()
    assert val_df["timestamp"].max() < test_df["timestamp"].min()
