import pandas as pd
import pandas.testing as pdt

from gold_strategy.backtest.engine import run_backtest


def make_prices():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 102, 101, 103, 104],
            "volume": 0,
        }
    )


def test_positions_shift_and_costs_on_trades():
    prices = make_prices()
    features = prices.copy()
    signals = pd.Series([0, 1, 1, 0, 0], index=prices["date"])

    result = run_backtest(
        prices,
        features,
        signals,
        transaction_cost_bps=10,
        slippage_bps=0,
        initial_capital=1.0,
    )

    expected_positions = signals.shift(1).fillna(0.0)
    expected_positions.index = expected_positions.index.tz_localize("UTC")
    pdt.assert_series_equal(result.positions, expected_positions, check_names=False)

    turnover = expected_positions.diff().abs().fillna(expected_positions.abs())
    pdt.assert_series_equal(result.turnover, turnover, check_names=False)

    # Costs only occur on turnover days
    costs = turnover * (10 / 10_000)
    price_returns = result.prices["close"].pct_change().fillna(0.0)
    raw_returns = expected_positions * price_returns
    strategy_returns = raw_returns - costs
    pdt.assert_series_equal(result.strategy_returns, strategy_returns, check_names=False)

    assert result.equity_curve.iloc[0] == 1.0
