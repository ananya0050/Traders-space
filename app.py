from pathlib import Path
import os
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model


MODEL_PATH = Path(__file__).with_name("Stock Predictions Model.keras")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calculate_r2(actual, predicted):
    residual_sum = np.sum((actual - predicted) ** 2)
    total_sum = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - residual_sum / total_sum if total_sum != 0 else 0


def scale_values(values, data_min, data_max):
    return (values - data_min) / (data_max - data_min)


def inverse_scale_values(values, data_min, data_max):
    return values * (data_max - data_min) + data_min


def stooq_symbol(symbol):
    symbol = symbol.strip().lower()
    if "." in symbol:
        return symbol
    return f"{symbol}.us"


@st.cache_data(ttl=3600)
def download_stock_data(symbol, start, end):
    params = {
        "s": stooq_symbol(symbol),
        "d1": pd.Timestamp(start).strftime("%Y%m%d"),
        "d2": pd.Timestamp(end).strftime("%Y%m%d"),
        "i": "d",
    }
    url = f"https://stooq.com/q/d/l/?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(request, timeout=20) as response:
        data = pd.read_csv(response)

    if data.empty or "Close" not in data.columns:
        return pd.DataFrame()

    data["Date"] = pd.to_datetime(data["Date"])
    return data.set_index("Date").sort_index()


st.set_page_config(page_title="Stock Market Predictor", layout="wide")
model = get_model()

st.title("Stock Market Predictor Using LSTM")

with st.sidebar:
    st.header("User Input")
    stock = st.text_input("Enter Stock Symbol (e.g., TSLA)", "TSLA")
    start = st.date_input("Start Date", pd.to_datetime("2012-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2022-12-31"))

if start >= end:
    st.error("Start date must be earlier than end date.")
    st.stop()

st.subheader("Stock Data")
with st.spinner("Loading stock data..."):
    try:
        data = download_stock_data(stock, start, end)
    except (URLError, TimeoutError, ValueError) as error:
        st.error(f"Unable to download stock data: {error}")
        st.stop()

if data.empty:
    st.error("No data found for the given stock symbol and date range.")
    st.stop()

if len(data) < 120:
    st.error("Please choose a wider date range. At least 120 trading days are needed for prediction.")
    st.stop()

st.write(data)

data_train = pd.DataFrame(data["Close"].iloc[0 : int(len(data) * 0.80)])
data_test = pd.DataFrame(data["Close"].iloc[int(len(data) * 0.80) :])

data_min = data_train["Close"].min()
data_max = data_train["Close"].max()

if data_max == data_min:
    st.error("The selected stock data does not have enough price variation for prediction.")
    st.stop()

data_train_scaled = scale_values(data_train[["Close"]].to_numpy(dtype=float), data_min, data_max)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scale_values(data_test[["Close"]].to_numpy(dtype=float), data_min, data_max)

x_train, y_train = [], []
for i in range(100, data_train_scaled.shape[0]):
    x_train.append(data_train_scaled[i - 100 : i])
    y_train.append(data_train_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i - 100 : i])
    y_test.append(data_test_scaled[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

with st.spinner("Predicting stock prices..."):
    predicted_prices = model.predict(x_test)
predicted_prices = inverse_scale_values(predicted_prices, data_min, data_max)
y_actual = inverse_scale_values(y_test.reshape(-1, 1), data_min, data_max)

st.markdown("### Model Accuracy Metrics")
col1, col2, col3 = st.columns(3)
mae = np.mean(np.abs(y_actual - predicted_prices))
mse = np.mean((y_actual - predicted_prices) ** 2)
r2 = calculate_r2(y_actual, predicted_prices)

col1.metric("MAE", f"{mae:.4f}")
col2.metric("MSE", f"{mse:.4f}")
col3.metric("R2", f"{r2:.4f}")

mape = calculate_mape(y_actual, predicted_prices)
st.metric("MAPE", f"{mape:.4f}%")

st.subheader("Actual vs Predicted Price Data")
comparison = pd.DataFrame(
    {
        "Actual Price": y_actual.flatten(),
        "Predicted Price": predicted_prices.flatten(),
    }
)
st.dataframe(comparison, use_container_width=True)

st.subheader("Model Accuracy & Comparison")
with st.expander("View Metric Details"):
    st.write("**MAE**: Measures average errors in prediction. Lower is better.")
    st.write("**MSE**: Gives larger errors more weight. Lower is better.")
    st.write("**R2**: Indicates how well predictions fit the actual data. Higher is better.")
    st.write("**MAPE**: Shows prediction error as a percentage. Lower is better.")
