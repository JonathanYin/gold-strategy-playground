import pandas as pd

from gold_strategy.backtest.walk_forward import run_walk_forward
from gold_strategy.data.loaders import build_feature_frame


def make_prices():
    dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "open": range(10),
            "high": range(10),
            "low": range(10),
            "close": range(10),
            "volume": 0,
        }
    )


def test_run_walk_forward_raises_when_segments_empty():
    prices = make_prices()
    features = build_feature_frame(prices)
    cutoff = pd.Timestamp("2025-01-01", tz="UTC")
    try:
        run_walk_forward(
            prices,
            features,
            parameters={"short_window": 2, "long_window": 3},
            train_end=cutoff,
            strategy="sma",
            transaction_cost_bps=0,
            slippage_bps=0,
            initial_capital=1.0,
        )
    except ValueError as exc:
        assert "Train or test segment is empty" in str(exc)
    else:
        assert False, "Expected ValueError"


def test_run_walk_forward_returns_metrics_for_both_splits():
    prices = make_prices()
    features = build_feature_frame(prices)
    cutoff = pd.Timestamp("2020-01-05", tz="UTC")
    result = run_walk_forward(
        prices,
        features,
        parameters={"short_window": 2, "long_window": 3},
        train_end=cutoff,
        strategy="sma",
        transaction_cost_bps=0,
        slippage_bps=0,
        initial_capital=1.0,
    )
    assert result.train.metrics
    assert result.test.metrics
