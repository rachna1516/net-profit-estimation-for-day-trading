"""Streamlit UI for exploring the data and running the ARIMA / SARIMA / LSTM models.

Run with:  streamlit run app.py
"""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import LSTM_implementation as lstm
from arima_model import TARGET_COLUMN, evaluate_forecast, run_arima
from data_eda import add_indicators
from data_utils import load_data
from sarima_model import bars_per_day, run_sarima

st.set_page_config(page_title="Day Trading Forecaster", layout="wide")

METRIC_LABELS = {
    "mae": ("MAE", "${:.3f}"),
    "rmse": ("RMSE", "${:.3f}"),
    "r2": ("R²", "{:.3f}"),
    "direction_accuracy": ("Direction accuracy", "{:.1%}"),
}


@st.cache_data(show_spinner=False)
def cached_load(source):
    """``source`` is a path, or the raw bytes of an uploaded CSV."""
    return load_data(io.BytesIO(source) if isinstance(source, bytes) else source)


def show_metrics(metrics: dict) -> None:
    cols = st.columns(len(METRIC_LABELS))
    for col, (key, (label, fmt)) in zip(cols, METRIC_LABELS.items()):
        col.metric(label, fmt.format(metrics[key]))


def show_forecast_chart(index, actual, predicted) -> None:
    chart = pd.DataFrame({"Actual": actual, "Predicted": predicted}, index=index)
    st.line_chart(chart, color=["#1f77b4", "#ff7f0e"])


def store_result(name: str, metrics: dict, index, actual, predicted, params: str) -> None:
    st.session_state.setdefault("results", {})[name] = {
        "metrics": metrics, "index": index, "actual": actual, "predicted": predicted, "params": params,
    }


def show_stored(name: str) -> None:
    result = st.session_state.get("results", {}).get(name)
    if result:
        st.caption(f"Last run: {result['params']}")
        show_metrics(result["metrics"])
        show_forecast_chart(result["index"], result["actual"], result["predicted"])


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Data")
uploaded = st.sidebar.file_uploader(
    "Upload OHLCV CSV", type="csv",
    help="Columns: Datetime, Open, High, Low, Close, Adj Close, Volume",
)
try:
    df = cached_load(uploaded.getvalue() if uploaded is not None else "data.csv")
except (ValueError, FileNotFoundError) as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

if uploaded is not None and st.session_state.get("source") != uploaded.name:
    st.session_state["results"] = {}  # results from another dataset no longer apply
st.session_state["source"] = uploaded.name if uploaded is not None else "data.csv"

train_frac = st.sidebar.slider("Training share", 0.5, 0.95, 0.8, 0.05,
                               help="Earliest part of the data used for training; the rest is the test period.")
split_row = int(len(df) * train_frac)
st.sidebar.markdown(
    f"**{len(df):,}** bars · {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  \n"
    f"Train: {split_row:,} · Test: {len(df) - split_row:,}"
)

st.title("Day Trading Forecaster")
st.caption(f"Forecasting the **{TARGET_COLUMN}** price of `{st.session_state['source']}`")

tab_data, tab_arima, tab_sarima, tab_lstm, tab_compare = st.tabs(
    ["Data & EDA", "ARIMA", "SARIMA", "LSTM", "Compare"]
)

# ---------------------------------------------------------------- data / EDA
with tab_data:
    ind = add_indicators(df)
    latest, first = df.iloc[-1], df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last open", f"${latest['Open']:.2f}", f"{latest['Open'] / first['Open'] - 1:+.1%} overall")
    c2.metric("Period high", f"${df['High'].max():.2f}")
    c3.metric("Period low", f"${df['Low'].min():.2f}")
    c4.metric("Bars per day", bars_per_day(df.index))

    st.subheader("Open price with moving averages")
    st.line_chart(ind[["Open", "SMA_20", "SMA_50"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Rolling 21-bar volatility")
        st.line_chart(ind["Volatility"])
    with right:
        st.subheader("Volume")
        st.bar_chart(df["Volume"])

    left, right = st.columns(2)
    with left:
        st.subheader("Distribution of returns")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(ind["Returns"].dropna(), bins=50, kde=True, ax=ax)
        ax.set_xlabel("Bar-to-bar return")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Correlation matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[["Open", "High", "Low", "Close", "Volume"]].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Summary statistics"):
        st.dataframe(df.describe())
    with st.expander("Raw data"):
        st.dataframe(df)

# ---------------------------------------------------------------- ARIMA
with tab_arima:
    st.write("Fits ARIMA(p, d, q) on training prices and forecasts the entire test period.")
    c1, c2, c3 = st.columns(3)
    p = c1.number_input("p (AR order)", 0, 10, 3, key="arima_p")
    q = c2.number_input("q (MA order)", 0, 10, 2, key="arima_q")
    auto_d = c3.checkbox("Choose d with ADF test", True, key="arima_auto_d")
    d = None if auto_d else c3.number_input("d", 0, 2, 1, key="arima_d")

    if st.button("Run ARIMA", type="primary"):
        with st.spinner("Fitting ARIMA..."):
            try:
                train, test, forecast, d_used = run_arima(df[TARGET_COLUMN], p=p, q=q, d=d, train_frac=train_frac)
                store_result("ARIMA", evaluate_forecast(train, test, forecast), test.index,
                             test.to_numpy(), forecast, f"ARIMA({p}, {d_used}, {q})")
            except Exception as exc:  # surface model errors in the UI instead of crashing
                st.error(f"ARIMA failed: {exc}")
    show_stored("ARIMA")

# ---------------------------------------------------------------- SARIMA
with tab_sarima:
    st.write("Seasonal ARIMA. The season length defaults to the number of bars per trading day.")
    c1, c2, c3 = st.columns(3)
    sp = c1.number_input("p", 0, 5, 1, key="sarima_p")
    sq = c2.number_input("q", 0, 5, 1, key="sarima_q")
    sd = c3.number_input("d", 0, 2, 1, key="sarima_d")
    c1, c2, c3, c4 = st.columns(4)
    SP = c1.number_input("P", 0, 5, 1, key="sarima_P")
    SQ = c2.number_input("Q", 0, 5, 1, key="sarima_Q")
    SD = c3.number_input("D", 0, 2, 1, key="sarima_D")
    s = c4.number_input("s (season)", 2, 48, bars_per_day(df.index), key="sarima_s")
    st.caption("Higher seasonal orders can take several minutes to fit.")

    if st.button("Run SARIMA", type="primary"):
        with st.spinner("Fitting SARIMA..."):
            try:
                train, test, forecast, order, seasonal = run_sarima(
                    df[TARGET_COLUMN], (sp, sd, sq), (SP, SD, SQ, s), train_frac=train_frac
                )
                store_result("SARIMA", evaluate_forecast(train, test, forecast), test.index,
                             test.to_numpy(), forecast, f"SARIMA{order}x{seasonal}")
            except Exception as exc:
                st.error(f"SARIMA failed: {exc}")
    show_stored("SARIMA")

# ---------------------------------------------------------------- LSTM
with tab_lstm:
    st.write("Two-layer LSTM predicting the next bar's open price from a sliding window of all features.")
    c1, c2, c3, c4 = st.columns(4)
    seq_length = c1.number_input("Window length", 2, 100, 10)
    epochs = c2.number_input("Epochs", 1, 300, 30, key="lstm_epochs")
    batch_size = c3.selectbox("Batch size", [16, 32, 64, 128], index=1)
    lr = c4.select_slider("Learning rate", [0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)

    if st.button("Train LSTM", type="primary"):
        bar = st.progress(0.0, text="Training...")

        def on_epoch(epoch, total, loss):
            bar.progress(epoch / total, text=f"Epoch {epoch}/{total} · loss {loss:.5f}")

        try:
            result = lstm.train_and_forecast(df, seq_length, epochs, batch_size, lr,
                                             train_frac=train_frac, progress=on_epoch)
            store_result("LSTM", result["metrics"], result["index"], result["targets"], result["predictions"],
                         f"window {seq_length}, {epochs} epochs, batch {batch_size}, lr {lr}")
            st.session_state["lstm_losses"] = result["losses"]
        except Exception as exc:
            st.error(f"LSTM training failed: {exc}")
        bar.empty()
    show_stored("LSTM")
    if "LSTM" in st.session_state.get("results", {}):
        with st.expander("Training loss"):
            st.line_chart(pd.Series(st.session_state["lstm_losses"], name="MSE loss"))

# ---------------------------------------------------------------- compare
with tab_compare:
    results = st.session_state.get("results", {})
    if not results:
        st.info("Run at least one model to compare results here.")
    else:
        table = pd.DataFrame({name: r["metrics"] for name, r in results.items()}).T
        table.columns = [METRIC_LABELS[c][0] for c in table.columns]
        table.insert(0, "Parameters", [r["params"] for r in results.values()])
        st.dataframe(table.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}",
                                         "Direction accuracy": "{:.1%}"}), use_container_width=True)
        st.caption("ARIMA/SARIMA forecast the whole test period from the end of training; "
                   "the LSTM predicts one bar ahead using actual recent bars, so it has more information.")

        first = next(iter(results.values()))
        chart = pd.DataFrame({"Actual": first["actual"]}, index=first["index"])
        for name, r in results.items():
            chart[name] = pd.Series(r["predicted"], index=r["index"])
        st.line_chart(chart)
