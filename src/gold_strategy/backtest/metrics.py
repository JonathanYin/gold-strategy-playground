"""Performance metrics helpers."""
from __future__ import annotations

import math
from typing import Dict

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    return float(equity_curve.iloc[-1] - 1.0)


def cagr(equity_curve: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if equity_curve.empty:
        return 0.0
    total_periods = len(equity_curve)
    years = total_periods / periods_per_year
    if years <= 0:
        return 0.0
    ending_value = equity_curve.iloc[-1]
    if ending_value <= 0:
        return 0.0
    return float(ending_value ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    if returns.empty:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    vol = excess.std(ddof=0)
    if vol == 0:
        return 0.0
    return float(excess.mean() / vol * math.sqrt(periods_per_year))


def max_drawdown(drawdown: pd.Series) -> float:
    if drawdown.empty:
        return 0.0
    return float(drawdown.min())


def summarize_metrics(strategy_returns: pd.Series, equity_curve: pd.Series, drawdown: pd.Series) -> Dict[str, float]:
    return {
        "total_return": total_return(equity_curve),
        "cagr": cagr(equity_curve),
        "volatility": annualized_volatility(strategy_returns),
        "max_drawdown": max_drawdown(drawdown),
        "sharpe": sharpe_ratio(strategy_returns),
    }
