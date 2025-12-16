"""Utilities for loading and preparing COMEX gold futures data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/Gold_Spot_historical_data.csv")


_REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def load_price_data(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Return cleaned OHLCV prices without derived columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find price file at {path}. Download the Kaggle CSV into data/."
        )

    raw = pd.read_csv(path)

    column_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = raw.rename(columns=column_map)

    missing = [col for col in _REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    prices = df[_REQUIRED_COLUMNS].copy()
    prices["date"] = pd.to_datetime(prices["date"], utc=True)
    numeric_cols = [c for c in prices.columns if c != "date"]
    prices[numeric_cols] = prices[numeric_cols].apply(pd.to_numeric, errors="coerce")
    prices = prices.sort_values("date").reset_index(drop=True)

    prices["volume"] = prices["volume"].fillna(0)

    return prices


def build_feature_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of prices with derived columns (daily returns, indicators)."""
    features = prices.copy()
    features["daily_return"] = features["close"].pct_change().fillna(0.0)
    return features
