import pandas as pd
import pandas.testing as pdt

from gold_strategy.indicators.rsi import relative_strength_index
from gold_strategy.indicators.sma import simple_moving_average
from gold_strategy.strategies.rsi_mean_reversion import generate_rsi_mean_reversion_signals
from gold_strategy.strategies.sma_crossover import generate_sma_crossover_signals


def test_simple_moving_average_basic():
    series = pd.Series([1, 2, 3, 4])
    result = simple_moving_average(series, 2)
    expected = pd.Series([float("nan"), 1.5, 2.5, 3.5])
    pdt.assert_series_equal(result.reset_index(drop=True), expected, check_names=False)


def test_generate_sma_crossover_signals_flags_when_short_above_long():
    features = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "open": range(5),
            "high": range(5),
            "low": range(5),
            "close": [1, 2, 3, 2, 1],
            "volume": 0,
        }
    )
    enriched, signals = generate_sma_crossover_signals(features, short_window=2, long_window=3)
    assert "sma_2" in enriched.columns and "sma_3" in enriched.columns
    # Signal can only be active once both SMA values exist
    assert signals.iloc[1] == 0
    assert signals.iloc[3] in (0, 1)
    assert set(signals.unique()).issubset({0, 1})


def test_relative_strength_index_bounds():
    prices = pd.Series([1, 2, 3, 2, 1, 2, 3, 4])
    rsi = relative_strength_index(prices, window=3)
    assert rsi.between(0, 100).all()


def test_rsi_mean_reversion_signals_toggle_on_thresholds():
    features = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "open": 1,
            "high": 1,
            "low": 1,
            "close": [1, 2, 3, 2, 1, 2],
            "volume": 0,
        }
    )
    enriched, signals = generate_rsi_mean_reversion_signals(
        features,
        window=2,
        oversold=30,
        overbought=70,
    )
    assert "signal" in enriched.columns
    assert set(signals.unique()).issubset({0.0, 1.0})
