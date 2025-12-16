"""Streamlit UI for the Gold Strategy Playground."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gold_strategy.backtest.engine import run_backtest
from gold_strategy.backtest.sweep import run_sma_parameter_sweep
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


def plot_sweep_heatmap(df: pd.DataFrame, metric: str) -> go.Figure:
    pivot = df.pivot(index="long_window", columns="short_window", values=metric)
    if pivot.empty:
        return go.Figure()
    fig = px.imshow(
        pivot,
        labels=dict(x="Short SMA", y="Long SMA", color=metric.replace("_", " ").title()),
        aspect="auto",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=20))
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

chart_tab, equity_tab, drawdown_tab, sweep_tab = st.tabs(
    ["Price & SMAs", "Equity curve", "Drawdown", "Parameter sweep"]
)

with chart_tab:
    short_col = f"sma_{short_window}"
    long_col = f"sma_{long_window}"
    st.plotly_chart(plot_candles(enriched_features, short_col, long_col), use_container_width=True)

with equity_tab:
    st.plotly_chart(plot_equity(result), use_container_width=True)

with drawdown_tab:
    st.plotly_chart(plot_drawdown(result), use_container_width=True)

with sweep_tab:
    st.subheader("SMA parameter sweep")
    st.write("Explore how different SMA pairs performed over the selected date range.")
    with st.form("sweep_form"):
        short_min = st.number_input("Short SMA min", min_value=5, max_value=200, value=10)
        short_max = st.number_input("Short SMA max", min_value=6, max_value=250, value=30)
        short_step = st.number_input("Short step", min_value=1, max_value=50, value=5)
        long_min = st.number_input("Long SMA min", min_value=20, max_value=250, value=40)
        long_max = st.number_input("Long SMA max", min_value=21, max_value=300, value=120)
        long_step = st.number_input("Long step", min_value=1, max_value=100, value=10)
        metric_choice = st.selectbox(
            "Metric",
            ["cagr", "total_return", "sharpe", "volatility", "max_drawdown"],
            index=0,
            format_func=lambda x: x.replace("_", " ").title(),
        )
        sweep_submit = st.form_submit_button("Run sweep")

    def _build_range(min_val: int, max_val: int, step: int) -> list[int]:
        if max_val < min_val:
            return []
        return list(range(min_val, max_val + 1, step))

    if sweep_submit:
        short_values = _build_range(int(short_min), int(short_max), int(short_step))
        long_values = _build_range(int(long_min), int(long_max), int(long_step))
        combos = [(s, l) for s in short_values for l in long_values if s < l]
        if not combos:
            st.warning("No valid short/long pairs in the provided ranges.")
        else:
            if len(combos) > 400:
                st.warning(
                    f"Large grid detected ({len(combos)} combos). This may take a while to compute."
                )
            with st.spinner("Running parameter sweep..."):
                sweep_df = run_sma_parameter_sweep(
                    filtered_prices,
                    filtered_features,
                    short_values,
                    long_values,
                    transaction_cost_bps=transaction_cost,
                    slippage_bps=slippage_cost,
                    initial_capital=initial_capital,
                )
            st.session_state["sweep_results"] = sweep_df
            st.session_state["sweep_metric"] = metric_choice

    sweep_data = st.session_state.get("sweep_results")
    if sweep_data is not None:
        if sweep_data.empty:
            st.info("Sweep returned no valid results. Try expanding the date range or windows.")
        else:
            metric_name = st.session_state.get("sweep_metric", "cagr")
            precision = 4 if metric_name == "sharpe" else 2
            st.dataframe(
                sweep_data[["short_window", "long_window", metric_name]].round(precision),
                use_container_width=True,
            )
            st.plotly_chart(plot_sweep_heatmap(sweep_data, metric_name), use_container_width=True)
    else:
        st.info("Submit a sweep to visualize the grid search results.")

st.caption(
    "Results use t+1 execution on daily closes with transaction/slippage costs applied only on trades."
)
