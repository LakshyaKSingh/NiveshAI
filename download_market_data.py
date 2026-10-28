
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

# Load API credentials
load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    raise ValueError("Missing Alpaca API credentials in .env")

client = StockHistoricalDataClient(api_key, secret_key)

# Start with a small stock universe
symbols = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO",
    "ORCL", "CRM", "ADBE", "CSCO", "IBM",
    "QCOM", "TXN", "INTC", "AMAT", "MU",
    "NOW", "INTU", "PLTR", "ANET", "PANW",

    # Consumer and retail
    "AMZN", "WMT", "COST", "HD", "MCD",
    "NKE", "SBUX", "TGT", "LOW", "KO",

    # Communication and media
    "GOOGL", "META", "NFLX", "DIS", "TMUS",

    # Finance
    "JPM", "BAC", "GS", "MS", "V",
    "MA", "AXP", "BLK", "C", "WFC",

    # Healthcare
    "LLY", "JNJ", "ABBV", "MRK", "UNH"
]

# Historical training period
start_date = datetime(2021, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

request = StockBarsRequest(
    symbol_or_symbols=symbols,
    timeframe=TimeFrame.Day,
    start=start_date,
    end=end_date,
    feed=DataFeed.IEX,
    adjustment=Adjustment.ALL
)

print("Downloading historical market data...")

bars = client.get_stock_bars(request)
df = bars.df.reset_index()

# Standardize column names
df.columns = [str(col).lower() for col in df.columns]

# Keep the raw columns needed for the initial ML pipeline
columns = [
    "symbol", "timestamp",
    "open", "high", "low", "close", "volume"
]

df = df[columns].copy()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df = df.sort_values(["symbol", "timestamp"])

# Basic data-quality checks
df = df.drop_duplicates(subset=["symbol", "timestamp"])
df = df.dropna(subset=["open", "high", "low", "close", "volume"])

# Save the raw dataset
os.makedirs("data/raw", exist_ok=True)

output_path = "data/raw/stock_ohlcv_2021_2025.csv"
df.to_csv(output_path, index=False)

print(f"Saved dataset: {output_path}")
print(f"Total rows: {len(df):,}")
print(f"Stocks downloaded: {df['symbol'].nunique()}")
print(df.head())
print(df.groupby("symbol").size())
