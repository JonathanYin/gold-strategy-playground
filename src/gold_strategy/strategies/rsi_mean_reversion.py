"""RSI mean reversion strategy."""
from __future__ import annotations

import pandas as pd

from gold_strategy.indicators.rsi import relative_strength_index


def generate_rsi_mean_reversion_signals(
    features: pd.DataFrame,
    window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> tuple[pd.DataFrame, pd.Series]:
    if oversold >= overbought:
        raise ValueError("oversold threshold must be below overbought")

    enriched = features.copy()
    rsi_col = f"rsi_{window}"
    enriched[rsi_col] = relative_strength_index(enriched["close"], window)

    signals = pd.Series(0.0, index=enriched.index, dtype=float)
    in_position = False
    for idx, row in enriched.iterrows():
        if row[rsi_col] <= oversold:
            in_position = True
        elif row[rsi_col] >= overbought:
            in_position = False
        signals.loc[idx] = 1.0 if in_position else 0.0

    if "date" in enriched.columns:
        signals.index = pd.to_datetime(enriched["date"], utc=True)
        signals.index.name = "date"

    enriched["signal"] = signals.values
    return enriched, signals
