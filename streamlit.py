"""
Interactive Time Series Forecasting Dashboard
------------------------------------------------
Reproduces the exact model families used in Time_Series_Analysis.ipynb
(AR, MA, ARMA, ARIMA, SARIMA, ARIMAX) on live Yahoo Finance data, with
sliders/inputs to change every parameter and see the forecast update live.

Run with:
    streamlit run streamlit_dashboard.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")

st.title("📈 Time Series Forecasting Dashboard")
st.caption(
    "Same model families & parameter style as the source notebook "
    "(AR, MA, ARMA, ARIMA, SARIMA, ARIMAX) — applied to live Yahoo Finance data."
)

# ----------------------------------------------------------------------------
# Sidebar - data controls
# ----------------------------------------------------------------------------
st.sidebar.header("1. Data")

ticker = st.sidebar.text_input("Yahoo Finance Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("today"))
freq_choice = st.sidebar.selectbox(
    "Resample frequency", ["Monthly (MS)", "Weekly (W)", "Daily (raw)"], index=0
)

@st.cache_data(show_spinner=True)
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return data

if not ticker:
    st.stop()

raw = load_data(ticker, start_date, end_date)

# Newer yfinance versions return MultiIndex columns (e.g. ('Close','AAPL'))
# even for a single ticker -> flatten so raw['Close'] is a plain Series.
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

if raw.empty:
    st.error("No data returned for that ticker/date range. Try a different ticker.")
    st.stop()

if freq_choice.startswith("Monthly"):
    series = raw["Close"].resample("MS").mean().dropna()
    seasonal_period_default = 12
elif freq_choice.startswith("Weekly"):
    series = raw["Close"].resample("W").mean().dropna()
    seasonal_period_default = 52
else:
    series = raw["Close"].dropna()
    seasonal_period_default = 5

series.name = "Close"

st.subheader(f"{ticker} price series ({freq_choice})")
fig0, ax0 = plt.subplots(figsize=(10, 4))
ax0.plot(series.index, series.values, marker="o", markersize=3)
ax0.set_xlabel("Date")
ax0.set_ylabel("Price")
st.pyplot(fig0)

# ----------------------------------------------------------------------------
# Sidebar - model selection
# ----------------------------------------------------------------------------
st.sidebar.header("2. Model")
model_choice = st.sidebar.selectbox(
    "Choose forecasting model",
    ["AR", "MA", "ARMA", "ARIMA (grid search)", "SARIMA", "ARIMAX"],
)

horizon = st.sidebar.slider("Forecast horizon (steps ahead)", 1, 24, 3)

# ----------------------------------------------------------------------------
# Helper: forecast index
# ----------------------------------------------------------------------------
def make_future_index(series, steps, freq_choice):
    if freq_choice.startswith("Monthly"):
        return pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=steps, freq="MS")
    elif freq_choice.startswith("Weekly"):
        return pd.date_range(series.index[-1] + pd.DateOffset(weeks=1), periods=steps, freq="W")
    else:
        return pd.date_range(series.index[-1] + pd.DateOffset(days=1), periods=steps, freq="D")


def plot_forecast(series, forecast, future_index, title):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(series.index, series.values, marker="o", markersize=3, label="Actual")
    ax.plot(future_index, forecast, marker="o", linestyle="--", color="red", label="Forecast")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    st.pyplot(fig)


# ----------------------------------------------------------------------------
# AR MODEL
# ----------------------------------------------------------------------------
if model_choice == "AR":
    st.sidebar.subheader("AR params")
    show_pacf = st.sidebar.checkbox("Show PACF plot", value=True)
    max_lag_pacf = st.sidebar.slider("PACF max lags", 5, 40, 20)
    best_lag = st.sidebar.slider("AR lag (best_lag)", 1, 30, 1)

    if show_pacf:
        st.subheader("PACF Plot (used to pick best_lag)")
        fig_p, ax_p = plt.subplots(figsize=(9, 4))
        plot_pacf(series, lags=max_lag_pacf, ax=ax_p)
        st.pyplot(fig_p)

    model = AutoReg(series, lags=best_lag)
    model_fit = model.fit()

    prediction = model_fit.predict(start=len(series), end=len(series) + horizon - 1)
    future_index = make_future_index(series, horizon, freq_choice)

    st.subheader(f"AR({best_lag}) Forecast — next {horizon} steps")
    st.write(pd.DataFrame({"Forecast": prediction.values}, index=future_index))
    plot_forecast(series, prediction.values, future_index, f"AR({best_lag}) Forecast")

# ----------------------------------------------------------------------------
# MA MODEL
# ----------------------------------------------------------------------------
elif model_choice == "MA":
    st.sidebar.subheader("MA params")
    q = st.sidebar.slider("MA order (q)", 1, 5, 3)
    enforce_invertibility = st.sidebar.checkbox("enforce_invertibility", value=(q != 2))

    with st.spinner("Fitting MA(1), MA(2), MA(3)... for AIC comparison"):
        aic_rows = []
        for qi in range(1, max(q, 3) + 1):
            try:
                m = ARIMA(series, order=(0, 0, qi), enforce_invertibility=enforce_invertibility)
                mf = m.fit()
                aic_rows.append({"q": qi, "AIC": mf.aic})
            except Exception as e:
                aic_rows.append({"q": qi, "AIC": None})

    st.subheader("AIC comparison (lower is better)")
    st.write(pd.DataFrame(aic_rows))

    final_model = ARIMA(series, order=(0, 0, q), enforce_invertibility=enforce_invertibility)
    final_model_fit = final_model.fit()
    forecast = final_model_fit.forecast(steps=horizon)
    future_index = make_future_index(series, horizon, freq_choice)

    st.subheader(f"MA({q}) Forecast — next {horizon} steps")
    st.write(pd.DataFrame({"Forecast": forecast.values}, index=future_index))
    plot_forecast(series, forecast.values, future_index, f"MA({q}) Forecast")

# ----------------------------------------------------------------------------
# ARMA MODEL
# ----------------------------------------------------------------------------
elif model_choice == "ARMA":
    st.sidebar.subheader("ARMA params (p, q) — d fixed at 0")
    p = st.sidebar.slider("AR order (p)", 0, 5, 1)
    q = st.sidebar.slider("MA order (q)", 0, 5, 1)

    model = ARIMA(series, order=(p, 0, q))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    future_index = make_future_index(series, horizon, freq_choice)

    st.subheader(f"ARMA({p},0,{q}) Forecast — next {horizon} steps  |  AIC: {model_fit.aic:.2f}")
    st.write(pd.DataFrame({"Forecast": forecast.values}, index=future_index))
    plot_forecast(series, forecast.values, future_index, f"ARMA({p},0,{q}) Forecast")

# ----------------------------------------------------------------------------
# ARIMA (grid search)
# ----------------------------------------------------------------------------
elif model_choice == "ARIMA (grid search)":
    st.sidebar.subheader("ARIMA grid search bounds")
    p_max = st.sidebar.slider("max p", 1, 5, 3)
    d_max = st.sidebar.slider("max d", 0, 2, 1)
    q_max = st.sidebar.slider("max q", 1, 5, 3)
    manual_override = st.sidebar.checkbox("Manually set (p,d,q) instead of grid search")

    if manual_override:
        p = st.sidebar.number_input("p", 0, 5, 1)
        d = st.sidebar.number_input("d", 0, 2, 1)
        q = st.sidebar.number_input("q", 0, 5, 1)
        best_order = (p, d, q)
    else:
        with st.spinner("Grid searching ARIMA(p,d,q) by AIC..."):
            result = []
            for p in range(p_max + 1):
                for d in range(d_max + 1):
                    for q in range(q_max + 1):
                        try:
                            m = ARIMA(series, order=(p, d, q))
                            mf = m.fit()
                            result.append({"p": p, "d": d, "q": q, "Order": (p, d, q), "AIC": mf.aic})
                        except Exception:
                            continue
            result_df = pd.DataFrame(result).sort_values(by="AIC")
        st.subheader("Model comparison (top 10 by AIC)")
        st.write(result_df.head(10))
        best_order = result_df.iloc[0]["Order"]

    st.write(f"**Selected order:** {best_order}")

    final_model = ARIMA(series, order=best_order)
    final_result = final_model.fit()
    forecast = final_result.forecast(steps=horizon)
    future_index = make_future_index(series, horizon, freq_choice)

    st.subheader(f"ARIMA{best_order} Forecast — next {horizon} steps")
    st.write(pd.DataFrame({"Forecast": forecast.values}, index=future_index))
    plot_forecast(series, forecast.values, future_index, f"ARIMA{best_order} Forecast")

# ----------------------------------------------------------------------------
# SARIMA
# ----------------------------------------------------------------------------
elif model_choice == "SARIMA":
    st.sidebar.subheader("SARIMA non-seasonal order (p,d,q)")
    p = st.sidebar.slider("p", 0, 3, 1)
    d = st.sidebar.slider("d", 0, 2, 1)
    q = st.sidebar.slider("q", 0, 3, 1)

    st.sidebar.subheader("SARIMA seasonal order (P,D,Q,s)")
    P = st.sidebar.slider("P", 0, 3, 1)
    D = st.sidebar.slider("D", 0, 2, 1)
    Q = st.sidebar.slider("Q", 0, 3, 1)
    s = st.sidebar.number_input("s (seasonal period)", 2, 365, seasonal_period_default)

    with st.spinner("Fitting SARIMAX..."):
        model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=horizon)
    future_index = make_future_index(series, horizon, freq_choice)
    forecast.index = future_index

    st.subheader(f"SARIMA({p},{d},{q})x({P},{D},{Q},{s}) Forecast — next {horizon} steps")
    st.write(pd.DataFrame({"Forecast Sales": forecast}))
    plot_forecast(series, forecast.values, future_index,
                  f"SARIMA({p},{d},{q})x({P},{D},{Q},{s}) Forecast")

# ----------------------------------------------------------------------------
# ARIMAX
# ----------------------------------------------------------------------------
elif model_choice == "ARIMAX":
    st.sidebar.subheader("ARIMAX order (p,d,q)")
    p = st.sidebar.slider("p", 0, 3, 1)
    d = st.sidebar.slider("d", 0, 2, 1)
    q = st.sidebar.slider("q", 0, 3, 1)

    exog_cols = st.sidebar.multiselect(
        "Exogenous columns (from Yahoo Finance)",
        ["Volume", "Range (High-Low)"],
        default=["Volume", "Range (High-Low)"],
    )

    exog_raw = raw.copy()
    exog_raw["Range (High-Low)"] = exog_raw["High"] - exog_raw["Low"]

    if freq_choice.startswith("Monthly"):
        exog_full = exog_raw[["Volume", "Range (High-Low)"]].resample("MS").mean().dropna()
    elif freq_choice.startswith("Weekly"):
        exog_full = exog_raw[["Volume", "Range (High-Low)"]].resample("W").mean().dropna()
    else:
        exog_full = exog_raw[["Volume", "Range (High-Low)"]].dropna()

    if not exog_cols:
        st.warning("Select at least one exogenous column.")
        st.stop()

    common_idx = series.index.intersection(exog_full.index)
    y = series.loc[common_idx]
    X = exog_full.loc[common_idx, exog_cols]

    st.write("Future exogenous values (edit to test scenarios):")
    last_row = X.iloc[-1]
    future_exog_input = {}
    cols = st.columns(len(exog_cols))
    for i, c in enumerate(exog_cols):
        future_exog_input[c] = cols[i].number_input(
            f"Future {c}", value=float(last_row[c])
        )

    future_exog = pd.DataFrame({c: [future_exog_input[c]] * horizon for c in exog_cols})

    with st.spinner("Fitting SARIMAX (ARIMAX)..."):
        model = SARIMAX(y, exog=X, order=(p, d, q))
        model_fit = model.fit(disp=False)

    forecast = model_fit.forecast(steps=horizon, exog=future_exog)
    future_index = make_future_index(y, horizon, freq_choice)

    st.subheader(f"ARIMAX({p},{d},{q}) Forecast — next {horizon} steps")
    result_table = pd.DataFrame({"Month": future_index, **future_exog, "Forecast_Price": forecast.values})
    st.write(result_table)
    plot_forecast(y, forecast.values, future_index, f"ARIMAX({p},{d},{q}) Forecast")

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Yahoo Finance via `yfinance`. Educational use only — not financial advice.")
