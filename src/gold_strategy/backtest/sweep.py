"""Parameter sweep helpers for SMA strategies."""
from __future__ import annotations

from itertools import product
from typing import Iterable, Sequence

import pandas as pd

from gold_strategy.backtest.engine import run_backtest
from gold_strategy.strategies.sma_crossover import generate_sma_crossover_signals


def _unique_sorted(values: Iterable[int]) -> Sequence[int]:
    uniq = sorted({int(v) for v in values})
    return [v for v in uniq if v > 0]


def run_sma_parameter_sweep(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    short_windows: Iterable[int],
    long_windows: Iterable[int],
    *,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Evaluate SMA crossover strategy over a parameter grid."""
    short_list = _unique_sorted(short_windows)
    long_list = _unique_sorted(long_windows)

    records: list[dict[str, float | int]] = []
    for short, long in product(short_list, long_list):
        if short >= long:
            continue
        enriched, signals = generate_sma_crossover_signals(features, short, long)
        result = run_backtest(
            prices,
            enriched,
            signals,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            initial_capital=initial_capital,
        )
        row = {"short_window": short, "long_window": long}
        row.update(result.metrics)
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["short_window", "long_window"])

    return pd.DataFrame.from_records(records)
