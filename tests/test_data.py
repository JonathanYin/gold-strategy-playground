from pathlib import Path

import pandas as pd

from gold_strategy.data.loaders import build_feature_frame, load_price_data


def test_load_price_data(tmp_path: Path):
    csv = tmp_path / "gold.csv"
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [None, 10, None],
        }
    )
    df.to_csv(csv, index=False)

    prices = load_price_data(csv)

    assert list(prices.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert prices["volume"].tolist() == [0, 10, 0]
    assert prices["date"].is_monotonic_increasing


def test_build_feature_frame_does_not_mutate_source():
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [0, 0, 0],
        }
    )
    features = build_feature_frame(prices)

    assert "daily_return" in features.columns
    assert "daily_return" not in prices.columns
    assert features.loc[0, "daily_return"] == 0
