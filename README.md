# Traders Space - Stock Predictor

A Streamlit app that loads a trained LSTM model, fetches stock data with Yahoo Finance, and compares actual prices with predicted prices.

## Features

- Historical stock data download by ticker and date range.
- LSTM-based price prediction using the saved Keras model.
- Model metrics: MAE, MSE, R2, and MAPE.
- Interactive Plotly chart for actual vs predicted prices.
- Moving-average charts for 50, 100, and 200 day windows.

## Local setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## Deploy from GitHub to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud and create a new app.
3. Select your GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

The deployment uses these files:

- `requirements.txt` installs Python packages on the cloud server.
- `runtime.txt` asks the host to use Python 3.11.9.
- `.streamlit/config.toml` sets Streamlit server options for cloud hosting.
- `Stock Predictions Model.keras` is the saved model loaded by `app.py`.

## Troubleshooting

If deployment fails with `No matching distribution found for tensorflow`, the host is using a Python version that TensorFlow does not support. This project pins Python in `runtime.txt` and uses `tensorflow-cpu` in `requirements.txt`, which is the correct package for a CPU-only cloud app.

## Important notes

- GitHub Pages cannot run this Streamlit/Python app. Use Streamlit Community Cloud, Render, Railway, or another Python app host.
- Do not commit private API keys, database passwords, or `.streamlit/secrets.toml`.
- `server.js`, `index.html`, `news.html`, and `project1.html` are legacy/static files and are not required for the Streamlit deployment.
