import pandas as pd

from gold_strategy.backtest.sweep import run_sma_parameter_sweep
from gold_strategy.data.loaders import build_feature_frame


def make_prices():
    dates = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    base = pd.DataFrame(
        {
            "date": dates,
            "open": range(100, 106),
            "high": range(101, 107),
            "low": range(99, 105),
            "close": [100, 101, 102, 103, 104, 105],
            "volume": 0,
        }
    )
    return base


def test_run_sma_parameter_sweep_skips_invalid_pairs():
    prices = make_prices()
    features = build_feature_frame(prices)

    df = run_sma_parameter_sweep(
        prices,
        features,
        short_windows=[2, 3],
        long_windows=[3, 4],
    )

    assert not df.empty
    assert set(df["short_window"]) == {2, 3}
    assert set(df["long_window"]) == {3, 4}
    invalid = df[df["short_window"] >= df["long_window"]]
    assert invalid.empty
    assert {"total_return", "cagr", "volatility", "max_drawdown", "sharpe"}.issubset(df.columns)
