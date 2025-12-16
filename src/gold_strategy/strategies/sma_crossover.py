"""SMA crossover strategy utilities."""
from __future__ import annotations

import pandas as pd

from gold_strategy.indicators.sma import simple_moving_average


def generate_sma_crossover_signals(
    features: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return enriched feature frame and raw signals (1 long, 0 flat)."""
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")

    enriched = features.copy()
    short_col = f"sma_{short_window}"
    long_col = f"sma_{long_window}"

    enriched[short_col] = simple_moving_average(enriched["close"], short_window)
    enriched[long_col] = simple_moving_average(enriched["close"], long_window)

    valid = enriched[short_col].notna() & enriched[long_col].notna()
    signals = (enriched[short_col] > enriched[long_col]).astype(int)
    signals = signals.where(valid, 0)
    if "date" in enriched.columns:
        signals.index = pd.to_datetime(enriched["date"], utc=True)
        signals.index.name = "date"

    enriched["signal"] = signals.values

    return enriched, signals
