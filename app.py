from pathlib import Path
import os

os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


MODEL_PATH = Path(__file__).with_name("Stock Predictions Model.keras")


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


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
    data = yf.download(stock, start=start, end=end, progress=False)

if data.empty:
    st.error("No data found for the given stock symbol and date range.")
    st.stop()

if len(data) < 120:
    st.error("Please choose a wider date range. At least 120 trading days are needed for prediction.")
    st.stop()

st.write(data)

data_train = pd.DataFrame(data["Close"][0 : int(len(data) * 0.80)])
data_test = pd.DataFrame(data["Close"][int(len(data) * 0.80) :])

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test)

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
predicted_prices = scaler.inverse_transform(predicted_prices)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

st.markdown("### Model Accuracy Metrics")
col1, col2, col3 = st.columns(3)
mae = mean_absolute_error(y_actual, predicted_prices)
mse = mean_squared_error(y_actual, predicted_prices)
r2 = r2_score(y_actual, predicted_prices)

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
