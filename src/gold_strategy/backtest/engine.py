"""Backtest engine for long/cash strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from gold_strategy.backtest.metrics import summarize_metrics


@dataclass
class BacktestResult:
    prices: pd.DataFrame
    features: pd.DataFrame
    signals: pd.Series
    positions: pd.Series
    turnover: pd.Series
    strategy_returns: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    metrics: Dict[str, float]


def _ensure_datetime_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.index.name == "date":
        return frame
    if "date" in frame.columns:
        frame = frame.set_index("date")
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index()
    frame.index.name = "date"
    return frame


def run_backtest(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    signals: pd.Series,
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Execute backtest with t+1 position application and trade-only costs."""
    price_frame = _ensure_datetime_index(prices.copy())
    feature_frame = _ensure_datetime_index(features.copy())

    aligned_signals = signals.copy()
    if isinstance(aligned_signals.index, pd.DatetimeIndex):
        if aligned_signals.index.tz is None:
            aligned_signals.index = aligned_signals.index.tz_localize("UTC")
        else:
            aligned_signals.index = aligned_signals.index.tz_convert("UTC")
    aligned_signals = aligned_signals.reindex(price_frame.index).fillna(0.0)
    aligned_signals.name = "signal"

    positions = aligned_signals.shift(1).fillna(0.0)
    positions.name = "position"

    returns = price_frame["close"].pct_change().fillna(0.0)

    turnover = positions.diff().abs().fillna(positions.abs())
    turnover.name = "turnover"

    total_cost_bps = transaction_cost_bps + slippage_bps
    costs = turnover * (total_cost_bps / 10_000)

    strategy_returns = positions * returns - costs
    strategy_returns.name = "strategy_return"

    equity_curve = (1 + strategy_returns).cumprod() * initial_capital
    equity_curve.name = "equity"

    normalized_equity = equity_curve / initial_capital
    drawdown = normalized_equity / normalized_equity.cummax() - 1
    drawdown.name = "drawdown"

    metrics = summarize_metrics(strategy_returns, normalized_equity, drawdown)

    return BacktestResult(
        prices=price_frame,
        features=feature_frame,
        signals=aligned_signals,
        positions=positions,
        turnover=turnover,
        strategy_returns=strategy_returns,
        equity_curve=equity_curve,
        drawdown=drawdown,
        metrics=metrics,
    )
