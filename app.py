"""Streamlit UI for the Gold Strategy Playground."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gold_strategy.backtest.engine import run_backtest
from gold_strategy.data.loaders import build_feature_frame, load_price_data
from gold_strategy.strategies.sma_crossover import generate_sma_crossover_signals

st.set_page_config(page_title="Gold Strategy Playground", layout="wide")


@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = load_price_data()
    features = build_feature_frame(prices)
    return prices, features


def _filter_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].reset_index(drop=True)


def plot_candles(features: pd.DataFrame, short_col: str, long_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=features["date"],
            open=features["open"],
            high=features["high"],
            low=features["low"],
            close=features["close"],
            name="Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=features["date"],
            y=features[short_col],
            name=short_col.upper(),
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=features["date"],
            y=features[long_col],
            name=long_col.upper(),
            line=dict(width=2),
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=20))
    return fig


def plot_equity(result) -> go.Figure:
    equity = result.equity_curve.copy()
    equity.index = equity.index.tz_convert(None)
    fig = go.Figure(
        data=go.Scatter(x=equity.index, y=equity.values, line=dict(width=2), name="Equity"),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=20))
    return fig


def plot_drawdown(result) -> go.Figure:
    drawdown = result.drawdown.copy()
    drawdown.index = drawdown.index.tz_convert(None)
    fig = go.Figure(
        data=go.Scatter(x=drawdown.index, y=drawdown.values, fill="tozeroy", name="Drawdown"),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=20), yaxis=dict(tickformat=".0%"))
    return fig


try:
    prices, base_features = get_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.title("Gold Strategy Playground")
st.caption("Educational tool for backtesting simple gold futures strategies")

min_date = prices["date"].min().date()
max_date = prices["date"].max().date()

with st.sidebar:
    st.header("Parameters")
    short_window = st.number_input("Short SMA", min_value=5, max_value=120, value=20, step=1)
    long_window = st.number_input("Long SMA", min_value=10, max_value=200, value=50, step=1)
    if long_window <= short_window:
        st.warning("Long window must be > short window")

    date_range = st.date_input(
        "Date range",
        (min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    start_date, end_date = date_range if isinstance(date_range, tuple) else (date_range, date_range)
    start_ts = pd.Timestamp(start_date).tz_localize("UTC")
    end_ts = pd.Timestamp(end_date).tz_localize("UTC")

    transaction_cost = st.slider("Transaction cost (bps)", 0.0, 50.0, 5.0, 0.5)
    slippage_cost = st.slider("Slippage (bps)", 0.0, 50.0, 0.0, 0.5)
    initial_capital = st.number_input("Initial capital", min_value=1.0, value=1.0, step=1.0)
    run_clicked = st.button("Run Backtest", type="primary")

if "auto_run" not in st.session_state:
    st.session_state.auto_run = True
if run_clicked:
    st.session_state.auto_run = True

if not st.session_state.auto_run:
    st.info("Adjust parameters and click 'Run Backtest' to see results.")
    st.stop()

filtered_prices = _filter_range(prices, start_ts, end_ts)
filtered_features = _filter_range(base_features, start_ts, end_ts)

if filtered_prices.empty:
    st.warning("No data for selected range")
    st.stop()

enriched_features, signals = generate_sma_crossover_signals(
    filtered_features,
    short_window=short_window,
    long_window=long_window,
)

result = run_backtest(
    filtered_prices,
    enriched_features,
    signals,
    transaction_cost_bps=transaction_cost,
    slippage_bps=slippage_cost,
    initial_capital=initial_capital,
)

metrics = result.metrics

st.subheader("Performance metrics")
metric_cols = st.columns(5)
metric_cols[0].metric("Total return", f"{metrics['total_return']:.2%}")
metric_cols[1].metric("CAGR", f"{metrics['cagr']:.2%}")
metric_cols[2].metric("Volatility", f"{metrics['volatility']:.2%}")
metric_cols[3].metric("Max drawdown", f"{metrics['max_drawdown']:.2%}")
metric_cols[4].metric("Sharpe", f"{metrics['sharpe']:.2f}")

chart_tab, equity_tab, drawdown_tab = st.tabs(["Price & SMAs", "Equity curve", "Drawdown"])

with chart_tab:
    short_col = f"sma_{short_window}"
    long_col = f"sma_{long_window}"
    st.plotly_chart(plot_candles(enriched_features, short_col, long_col), use_container_width=True)

with equity_tab:
    st.plotly_chart(plot_equity(result), use_container_width=True)

with drawdown_tab:
    st.plotly_chart(plot_drawdown(result), use_container_width=True)

st.caption(
    "Results use t+1 execution on daily closes with transaction/slippage costs applied only on trades."
)
