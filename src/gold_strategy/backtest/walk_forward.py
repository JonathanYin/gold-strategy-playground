"""Walk-forward evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gold_strategy.backtest.engine import BacktestResult, run_backtest
from gold_strategy.strategies.rsi_mean_reversion import generate_rsi_mean_reversion_signals
from gold_strategy.strategies.sma_crossover import generate_sma_crossover_signals


@dataclass
class WalkForwardResult:
    train: BacktestResult
    test: BacktestResult


def run_walk_forward(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    parameters: dict,
    *,
    train_end: pd.Timestamp,
    strategy: str,
    transaction_cost_bps: float,
    slippage_bps: float,
    initial_capital: float,
) -> WalkForwardResult:
    train_mask = features["date"] <= train_end
    train_prices = prices.loc[train_mask]
    train_features = features.loc[train_mask]

    test_prices = prices.loc[~train_mask]
    test_features = features.loc[~train_mask]

    if train_prices.empty or test_prices.empty:
        raise ValueError("Train or test segment is empty. Adjust the cutoff date.")

    if strategy == "sma":
        enriched_train, train_signals = generate_sma_crossover_signals(
            train_features,
            short_window=parameters["short_window"],
            long_window=parameters["long_window"],
        )
        enriched_test, test_signals = generate_sma_crossover_signals(
            test_features,
            short_window=parameters["short_window"],
            long_window=parameters["long_window"],
        )
    else:
        enriched_train, train_signals = generate_rsi_mean_reversion_signals(
            train_features,
            window=parameters["window"],
            oversold=parameters["oversold"],
            overbought=parameters["overbought"],
        )
        enriched_test, test_signals = generate_rsi_mean_reversion_signals(
            test_features,
            window=parameters["window"],
            oversold=parameters["oversold"],
            overbought=parameters["overbought"],
        )

    train_result = run_backtest(
        train_prices,
        enriched_train,
        train_signals,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
    )
    test_result = run_backtest(
        test_prices,
        enriched_test,
        test_signals,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
    )
    return WalkForwardResult(train=train_result, test=test_result)
