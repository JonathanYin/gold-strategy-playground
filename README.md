# Gold Strategy Playground

Interactive Streamlit app for exploring simple quantitative strategies on COMEX gold futures.

## Getting started

1. Install Python 3.11 (e.g., `brew install python@3.11`).

2. Create a virtual environment and install dependencies with the provided constraints:

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -e '.[dev]' -c constraints.txt
    ```

3. Download `Gold_Spot_historical_data.csv` from the Kaggle dataset [Gold Historical Data daily updated](https://www.kaggle.com/datasets/isaaclopgu/gold-historical-data-daily-updated) and place it in `data/`.
4. Run tests to confirm the setup:

    ```bash
    python -m pytest
    ```

5. Launch the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Project layout

```bash
.
├── app.py                # Streamlit UI
├── data/                 # CSV input (ignored in git)
├── notebooks/            # Scratch exploration
├── src/gold_strategy/
│   ├── data/             # Raw/feature loaders
│   ├── indicators/       # SMA, RSI, etc.
│   ├── strategies/       # Strategy signal logic
│   └── backtest/         # Backtest engine + metrics
└── tests/
```

## MVP scope

-   Load OHLCV data locally (no Kaggle API).
-   Compute SMAs (default 20/50) and daily returns.
-   SMA crossover and RSI mean-reversion strategies with proper signal timing (positions applied t+1).
-   Backtest with transaction/slippage costs applied on trades only.
-   Candlestick + SMA overlay, equity and drawdown charts.
-   Summary metrics: total return, CAGR, annualized volatility, max drawdown, Sharpe-lite.
-   Parameter sweep tab for SMA short/long ranges with heatmap visualization.
-   Walk-forward evaluation tab to compare train/test metrics for a chosen cutoff date.

aimed at education, not investment advice.

## Methodology notes

-   Volume remains as reported (missing values filled with `0`) to avoid fabricating activity.
-   Strategy signals are generated on close `t` and executed the following day (`t+1`).
-   Transaction/slippage costs are applied when positions change using turnover = `abs(position.diff())`.
-   Parameter sweep respects the active date range and re-runs the SMA crossover for each valid short/long pair.
-   RSI mean reversion goes long when RSI falls below the oversold threshold and exits when it rises above overbought.
-   Walk-forward evaluation reuses the chosen strategy parameters on train/test splits to highlight robustness gaps.
