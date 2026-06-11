import numpy as np
import pandas as pd
import yfinance as yf # type: ignore
from keras.models import load_model # type: ignore
import streamlit as st # type: ignore
import plotly.graph_objects as go
import matplotlib.pyplot as plt # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = load_model(r"C:\Users\Ashok Kumar\Downloads\Stock_Market_Prediction_ML\Stock Predictions Model.keras")
st.title("📈 Stock Market Predictor Using LSTM")
with st.sidebar:
    st.header("User Input")
    stock = st.text_input('Enter Stock Symbol (e.g., TSLA)', 'TSLA')
    start = st.date_input("Start Date", pd.to_datetime("2012-01-01"))
    end = st.date_input("End Date", pd.to_datetime("2022-12-31"))

st.subheader("Stock Data 📊")
with st.spinner("Loading stock data..."):
    data = yf.download(stock, start=start, end=end)

if not data.empty:
    st.write(data)

    data_train = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
    data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train_scaled = scaler.fit_transform(data_train)

    past_100_days = data_train.tail(100)
    data_test = pd.concat([past_100_days, data_test], ignore_index=True)
    data_test_scaled = scaler.transform(data_test)

    
    x_train, y_train = [], []
    for i in range(100, data_train_scaled.shape[0]):
        x_train.append(data_train_scaled[i-100:i])
        y_train.append(data_train_scaled[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    
    x_test, y_test = [], []
    for i in range(100, data_test_scaled.shape[0]):
        x_test.append(data_test_scaled[i-100:i])
        y_test.append(data_test_scaled[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

   
    with st.spinner("Predicting stock prices..."):
        predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

   
    st.markdown("### Model Accuracy Metrics 📋")
    col1, col2, col3 = st.columns(3)
    mae = mean_absolute_error(y_actual, predicted_prices)
    mse = mean_squared_error(y_actual, predicted_prices)
    r2 = r2_score(y_actual, predicted_prices)

    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("R²", f"{r2:.4f}")

    
    def calculate_mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    mape = calculate_mape(y_actual, predicted_prices)
    st.metric("MAPE", f"{mape:.4f}%")

    st.subheader("Original Price vs Predicted Price 📈")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_actual.flatten(), mode='lines', name='Actual Price', line=dict(color='green')))
    fig.add_trace(go.Scatter(y=predicted_prices.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))
    fig.update_layout(title=f"{stock} Price Prediction vs Actual", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig)

   
    st.subheader("Moving Averages 📉")

    ma_50_days = data['Close'].rolling(50).mean()
    ma_100_days = data['Close'].rolling(100).mean()
    ma_200_days = data['Close'].rolling(200).mean()

    fig1 = plt.figure()
    plt.plot(data['Close'], label='Price', color='green')
    plt.plot(ma_50_days, label='MA 50', color='red')
    plt.title(f"{stock} Price vs MA50")
    plt.legend()
    st.pyplot(fig1)

    fig2 = plt.figure()
    plt.plot(data['Close'], label='Price', color='green')
    plt.plot(ma_50_days, label='MA 50', color='red')
    plt.plot(ma_100_days, label='MA 100', color='blue')
    plt.title(f"{stock} Price vs MA50 vs MA100")
    plt.legend()
    st.pyplot(fig2)

    
    fig3 = plt.figure()
    plt.plot(data['Close'], label='Price', color='green')
    plt.plot(ma_100_days, label='MA 100', color='red')
    plt.plot(ma_200_days, label='MA 200', color='blue')
    plt.title(f"{stock} Price vs MA100 vs MA200")
    plt.legend()
    st.pyplot(fig3)

   
    st.subheader("Model Accuracy & Comparison 💡")
    with st.expander("View Metric Details"):
        st.write("**MAE**: Measures average errors in prediction. Lower is better.")
        st.write("**MSE**: Gives larger errors more weight. Lower is better.")
        st.write("**R²**: Indicates how well predictions fit the actual data. Higher is better.")
        st.write("**MAPE**: Shows prediction error as a percentage. Lower is better.")
else:
    st.error("No data found for the given stock symbol and date range.")
