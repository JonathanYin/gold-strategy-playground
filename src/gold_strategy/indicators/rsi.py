"""Relative Strength Index calculation."""
from __future__ import annotations

import pandas as pd


def relative_strength_index(series: pd.Series, window: int = 14) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be > 1")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(0.0)
    return rsi
