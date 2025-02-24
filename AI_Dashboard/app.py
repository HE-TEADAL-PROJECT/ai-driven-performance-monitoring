import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging
import sys
import json

# Imposta il logging per stampare su stderr e scrivere su log.txt
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/app/log.txt"),  # Scrive su file
        logging.StreamHandler(sys.stderr)  # Stampa anche su stderr
    ]
)

# Forza Streamlit a stampare nel terminale
sys.stdout = sys.stderr

logging.info("🚀 Streamlit App Started! Logging initialized.")

# Carica la configurazione
with open('config.json') as f:
    config = json.load(f)
API_URL = config['API_URL']

st.title("Monitoring & AI/ML Dashboard")

# Sidebar - Configurazione globale
st.sidebar.header("Global Configuration")
default_query = '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
query = st.sidebar.text_input("PromQL Query", default_query)
hours = st.sidebar.number_input("Hours (for fetch/train/anomaly)", min_value=1, value=3, step=1)
input_steps = st.sidebar.number_input("Input Steps (RNN)", min_value=1, value=48, step=1)
output_steps = st.sidebar.number_input("Output Steps (prediction horizon)", min_value=1, value=20, step=1)

st.sidebar.write("---")
st.sidebar.write("**Models Available**: GRU, LSTM, ARIMA, Prophet, TimeGPT, LagLlama")

# Inizializza variabili di sessione
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame()
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = {}
if 'anomalies' not in st.session_state:
    st.session_state['anomalies'] = pd.DataFrame()
if 'rmse_results' not in st.session_state:
    st.session_state['rmse_results'] = pd.DataFrame()

# 1) Fetch Data
st.header("Fetch Historical Data from Thanos")
if st.button("Fetch Data"):
    response = requests.post(f"{API_URL}/fetch", json={"query": query, "hours": hours})
    if response.status_code == 200:
        data = pd.DataFrame(response.json()["data"])
        if not data.empty:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.sort_values('timestamp', inplace=True)
            st.session_state['data'] = data
            st.success(f"Fetched {data.shape[0]} records.")
            st.dataframe(data.head())
        else:
            st.warning("Fetched data is empty.")
    else:
        st.error(f"Fetch error: {response.status_code} - {response.text}")

# 2) Train Models
st.header("Train Local Models")
local_models = ["GRU", "LSTM", "ARIMA", "Prophet"]
selected_models_train = st.multiselect("Select local models to train", local_models)
if st.button("Train Models"):
    if st.session_state['data'].empty:
        st.error("No data available. Fetch data first.")
    else:
        for model in selected_models_train:
            response = requests.post(f"{API_URL}/train", json={
                "query": query,
                "model": model,
                "training_hours": hours,
                "input_steps": input_steps,
                "output_steps": output_steps
            })
            if response.status_code == 200:
                st.success(f"{model}: {response.json()['message']}")
            else:
                st.error(f"{model} training failed: {response.status_code} - {response.text}")

# 3) Compute RMSE
st.header("Compute error RMSE")
selected_models_rmse = st.multiselect("Select models for RMSE", local_models + ["TimeGPT", "LagLlama"], default=["GRU", "ARIMA"])
if st.button("Compute RMSE"):
    response = requests.post(f"{API_URL}/compute_rmse", json={
        "query": query,
        "training_hours": hours,
        "input_steps": input_steps,
        "output_steps": output_steps,
        "models": selected_models_rmse
    })
    if response.status_code == 200:
        rmse_results = pd.DataFrame(response.json())
        st.session_state['rmse_results'] = rmse_results
        st.dataframe(rmse_results)
    else:
        st.error(f"RMSE computation failed: {response.status_code} - {response.text}")

import json  # Aggiungi all'inizio del file, se non presente

# 4) Multi-Model Inference & Plot
st.header("Multi-Model Predictions")
models_infer = st.multiselect("Select models for inference", local_models + ["TimeGPT", "LagLlama"], default=["TimeGPT"])

if st.button("Run Inference"):
    if st.session_state['data'].empty:
        st.error("No data available. Fetch data first.")
    else:
        # Reset predictions dictionary to keep only current selected models
        st.session_state['predictions'] = {}
        df_hist = st.session_state['data'].copy()

        import json

        for model in models_infer:
            payload = {
                "thanos_url": None,  # Se necessario
                "query": str(query),
                "training_hours": int(hours),
                "input_steps": int(input_steps),
                "output_steps": int(output_steps),
                "model": str(model)
            }

            # Scrive il payload nei log
            logging.debug(f"DEBUG - JSON inviato:\n{json.dumps(payload, indent=4)}")

            # Manda la richiesta
            response = requests.post(f"{API_URL}/infer", json=payload)

            # Scrive la risposta nei log
            logging.debug(f"DEBUG - Response status code ({model}): {response.status_code}")
            logging.debug(f"DEBUG - Response text ({model}): {response.text}")

            if response.status_code != 200:
                st.error(f"Inference error: {response.status_code} - {response.text}")
                logging.error(f"ERROR - API {model} failed: {response.text}")




            if response.status_code == 200:
                pred_df = pd.DataFrame(response.json()["predictions"])
                if not pred_df.empty:
                    pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
                    st.session_state['predictions'][model] = pred_df
                    st.success(f"Inference OK for {model}")


        if st.session_state['predictions']:
            plt.figure(figsize=(12,6))
            plt.plot(df_hist['timestamp'], df_hist['value'], label="Historical Data", color='blue')
            for model, df_pred in st.session_state['predictions'].items():
                plt.plot(df_pred['timestamp'], df_pred['prediction'], label=model, linestyle='--')
            plt.legend()
            plt.xticks(rotation=45)
            plt.title("Predictions vs Historical Data")
            plt.grid(True)
            st.pyplot(plt)


# 5) Anomaly Detection
st.header("Anomaly Detection")
anom_methods = ["zscore", "iqr", "timegpt"]
selected_anom_method = st.selectbox("Select Anomaly Detection Method", anom_methods, index=0)

if st.button("Detect Anomalies"):
    # 🔹 Esegui il fetch dei dati per ottenere i più recenti
    response_fetch = requests.post(f"{API_URL}/fetch", json={"query": query, "hours": hours})
    
    if response_fetch.status_code == 200:
        fetched_data = pd.DataFrame(response_fetch.json()["data"])
        fetched_data['timestamp'] = pd.to_datetime(fetched_data['timestamp'])
        fetched_data.sort_values("timestamp", inplace=True)
        st.session_state['data'] = fetched_data  # 🔹 Aggiorna i dati in memoria
    else:
        st.error(f"Data fetch failed: {response_fetch.status_code} - {response_fetch.text}")
        st.stop()  # 🔹 Evita di eseguire l'anomaly detection con dati vecchi

    # 🔹 Ora eseguiamo l'anomaly detection sui dati aggiornati
    response_anomaly = requests.post(f"{API_URL}/anomaly", json={"query": query, "method": selected_anom_method, "hours": hours})
    
    if response_anomaly.status_code == 200:
        anom_df = pd.DataFrame(response_anomaly.json()["anomalies"])
        st.session_state['anomalies'] = anom_df

        # 🔹 Se ci sono anomalie, plottiamo
        if not anom_df.empty:
            anom_df['timestamp'] = pd.to_datetime(anom_df['timestamp'])
            plt.figure(figsize=(10,5))
            plt.plot(st.session_state['data']['timestamp'], st.session_state['data']['value'], label="Historical Data", color='blue')
            plt.scatter(anom_df['timestamp'], anom_df['value'], color='red', label="Anomalies", marker='o')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(plt)
            st.dataframe(anom_df)
        else:
            st.info("No anomalies detected.")
    else:
        st.error(f"Anomaly detection failed: {response_anomaly.status_code} - {response_anomaly.text}")
