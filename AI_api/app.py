import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
import polars as pl
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from datetime import datetime, timedelta

# Moduli personalizzati
from utils import setup_logging
from data_fetcher import fetch_data
from model import (
    train_gru, train_lstm,
    train_arima, train_prophet
)
from nixtla import NixtlaClient
# Tentativo di import di Lag-Llama e PyTorch, se disponibili
try:
    import torch
    from gluonts.dataset.common import ListDataset
    from lag_llama.gluon.estimator import LagLlamaEstimator
    LAG_LLAMA_AVAILABLE = True
except ImportError:
    LAG_LLAMA_AVAILABLE = False

# ------------------------------------------------------------------------------
# Carichiamo config.json
# ------------------------------------------------------------------------------
with open('config.json') as f:
    config = json.load(f)

DEFAULT_THANOS_URL = config['thanos_url']
TIMEGPT_API_KEY = config['timegpt_api_key']
LAG_LLAMA_CHECKPOINT = config.get("lag_llama_ckpt", "/app/checkpoints/lag-llama.ckpt")

# ------------------------------------------------------------------------------
# Setup logger
# ------------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Cartella dei modelli
# ------------------------------------------------------------------------------
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

import re
def sanitize_filename(filename: str, max_length: int = 50) -> str:
    """Pulisce il nome della query eliminando caratteri speciali e limitandone la lunghezza."""
    filename = filename.strip()
    filename = re.sub(r'[^\w\s-]', '', filename)  # Rimuove caratteri speciali tranne underscore e trattini
    filename = filename.replace(" ", "_")  # Sostituisce spazi con underscore
    filename = re.sub(r'_+', '_', filename)  # Sostituisce doppie underscore con una sola
    return filename[:max_length]  # Limita la lunghezza massima del nome

def save_model(model, query, training_hours, input_steps, output_steps, model_name, scaler=None):
    """Salva il modello con un nome che include solo i parametri necessari."""
    sanitized_query = sanitize_filename(query)  # Pulizia della query
    
    # Modelli GRU e LSTM hanno bisogno di input_steps
    if model_name in ["GRU", "LSTM"]:
        model_path = os.path.join(
            MODEL_DIR, f"{model_name}_{sanitized_query}_{training_hours}h_{input_steps}in_{output_steps}out.pkl"
        )
    else:
        model_path = os.path.join(
            MODEL_DIR, f"{model_name}_{sanitized_query}_{training_hours}h_{output_steps}out.pkl"
        )

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "query": query,
            "training_hours": training_hours,
            "input_steps": input_steps if model_name in ["GRU", "LSTM"] else None,
            "output_steps": output_steps,
        }, f)
    
    return model_path


def load_model(query, training_hours, input_steps, output_steps, model_name):
    """Carica un modello con i parametri giusti (con input_steps solo per GRU e LSTM)."""
    sanitized_query = sanitize_filename(query)

    if model_name in ["GRU", "LSTM"]:
        model_path = os.path.join(
            MODEL_DIR, f"{model_name}_{sanitized_query}_{training_hours}h_{input_steps}in_{output_steps}out.pkl"
        )
    else:
        model_path = os.path.join(
            MODEL_DIR, f"{model_name}_{sanitized_query}_{training_hours}h_{output_steps}out.pkl"
        )

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    return None



# ------------------------------------------------------------------------------
# In-memory dictionary (se vuoi tenere i modelli in RAM dopo /train)
# ------------------------------------------------------------------------------
trained_models = {}

# ------------------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------------------
class TrainRequest(BaseModel):
    thanos_url: Optional[str] = None
    query: str
    training_hours: int
    input_steps: int
    output_steps: int
    model: str  # "GRU","LSTM","ARIMA","Prophet","TimeGPT","LAGLLAMA"
class InferRequest(BaseModel):
    thanos_url: Optional[str] = None
    query: str
    training_hours: int
    input_steps: int
    output_steps: int
    model: str  # "GRU","LSTM","ARIMA","Prophet","TimeGPT","LAGLLAMA"


class PredictionItem(BaseModel):
    timestamp: str
    prediction: float

class TrainResponse(BaseModel):
    message: str

class InferResponse(BaseModel):
    model: str
    predictions: List[PredictionItem]

class AnomalyRequest(BaseModel):
    thanos_url: Optional[str] = None
    query: str
    hours: int
    method: str             # "zscore", "iqr", "timegpt"
    confidence: Optional[int] = 95

class AnomalyItem(BaseModel):
    timestamp: str
    value: float

class AnomalyResponse(BaseModel):
    method: str
    anomalies: List[AnomalyItem]

class FetchRequest(BaseModel):
    thanos_url: Optional[str] = None
    query: str
    hours: int

class FetchResponse(BaseModel):
    data: List[Dict[str, Any]]

class ComputeRMSERequest(BaseModel):
    thanos_url: Optional[str] = None
    query: str
    training_hours: int
    input_steps: int
    output_steps: int
    models: List[str]

class RMSEItem(BaseModel):
    Model: str
    Train_RMSE: Optional[float]
    Validation_RMSE: Optional[float]

# ------------------------------------------------------------------------------
# Creiamo l'app FastAPI
# ------------------------------------------------------------------------------
app = FastAPI(
    title="AI Prediction API: Train & Infer",
    description="Time Series prediction service with local or ephemeral models + anomaly detection",
    version="1.1.0"
)

# ------------------------------------------------------------------------------
# Endpoint /fetch
# ------------------------------------------------------------------------------
@app.post("/fetch", response_model=FetchResponse)
def fetch_endpoint(req: FetchRequest):
    thanos_url = req.thanos_url or DEFAULT_THANOS_URL
    logger.info(f"[FETCH] Using Thanos URL: {thanos_url}")

    data = fetch_data(thanos_url, req.query, req.hours)
    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="No data returned or data is empty.")

    records = data.to_dict(orient="records")
    return FetchResponse(data=records)

# ------------------------------------------------------------------------------
# Endpoint /train
# ------------------------------------------------------------------------------
@app.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    """
    Trains local models (GRU, LSTM, ARIMA, Prophet) or ephemeral (TimeGPT, LAGLLAMA).
    Saves local models to `models/`.
    """
    thanos_url = req.thanos_url or DEFAULT_THANOS_URL
    model_name = req.model.strip().upper()
    logger.info(f"[TRAIN] Using Thanos URL: {thanos_url}")

    # TimeGPT / LAGLLAMA => ephemeral
    if model_name in ["TIMEGPT", "LAGLLAMA"]:
        trained_models[model_name] = {"info": f"{model_name} ephemeral - no local training"}
        return TrainResponse(message=f"{model_name} does not require local training.")

    # Fetch dei dati
    data = fetch_data(thanos_url, req.query, req.training_hours)
    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="No data returned from Thanos or data is empty for training.")

    # Preprocess
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp').reset_index(drop=True)
    logger.info(f"[TRAIN] Training model: {model_name}")

    # GRU / LSTM
    if model_name in ["GRU", "LSTM"]:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['scaled_value'] = scaler.fit_transform(data[['value']])
        tmp_data = data[['timestamp', 'scaled_value']].rename(columns={'scaled_value': 'value'})

        if model_name == "GRU":
            model, _, _ = train_gru(tmp_data, req.input_steps, req.output_steps)
        else:  # LSTM
            model, _, _ = train_lstm(tmp_data, req.input_steps, req.output_steps)

        # Salva in RAM (opzionale)
        trained_models[model_name] = {
            "model": model,
            "scaler": scaler,
            "input_steps": req.input_steps,
            "output_steps": req.output_steps
        }
        # 🔹 Salva anche su disco con l'ordine corretto
        save_model(model, req.query, req.training_hours, req.input_steps, req.output_steps, model_name, scaler=scaler)

    elif model_name == "ARIMA":
        model_fit, preds, ts = train_arima(data, req.output_steps)
        trained_models[model_name] = {
            "model": model_fit,
            "output_steps": req.output_steps
        }
        # 🔹 Salva su disco con l'ordine corretto
        save_model(model_fit, req.query, req.training_hours, None, req.output_steps, model_name)

    elif model_name == "PROPHET":
        model_fb, preds, ts = train_prophet(data, req.output_steps)
        trained_models[model_name] = {
            "model": model_fb,
            "output_steps": req.output_steps
        }
        # 🔹 Salva su disco con l'ordine corretto
        save_model(model_fb, req.query, req.training_hours, None, req.output_steps, model_name)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model for training: {req.model}")

    return TrainResponse(message=f"Model {model_name} trained and saved successfully.")

# ------------------------------------------------------------------------------
# Endpoint /compute_rmse
# ------------------------------------------------------------------------------
@app.post("/compute_rmse", response_model=List[RMSEItem])
def compute_rmse_endpoint(req: ComputeRMSERequest):
    """
    Calcola Train e Validation RMSE per i modelli locali richiesti.
    1) Fetch dei dati storici (training_hours).
    2) Split 80/20.
    3) Per GRU/LSTM ricalcola le sequenze su train e val e confronta i pred.
    4) Per ARIMA/Prophet confronta i fitted values e forecast su val.
    """
    thanos_url = req.thanos_url or DEFAULT_THANOS_URL
    logger.info(f"[RMSE] Using Thanos URL: {thanos_url}")

    data = fetch_data(thanos_url, req.query, req.training_hours)
    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="No data returned for training RMSE.")

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp').reset_index(drop=True)

    # Split train/val
    train_size = int(len(data) * 0.8)
    train_df = data.iloc[:train_size].copy()
    val_df   = data.iloc[train_size:].copy()

    results_list = []

    for model_name in req.models:
        mdl = model_name.strip().upper()
        logger.info(f"Evaluating RMSE for {mdl} ...")

        # Carichiamo il modello da disco
        model_info = load_model(req.query, req.training_hours, req.input_steps, req.output_steps, mdl)
        if model_info is None:
            # Se non esiste, RMSE = None
            results_list.append({
                "Model": mdl,
                "Train_RMSE": None,
                "Validation_RMSE": None
            })
            continue

        model = model_info["model"]
        scaler = model_info.get("scaler", None)

        if mdl in ["GRU", "LSTM"]:
            if not scaler:
                raise ValueError(f"Scaler not found for {mdl} model.")
            def create_sequences(data_array, in_steps, out_steps):
                x, y = [], []
                for i in range(len(data_array) - in_steps - out_steps + 1):
                    x.append(data_array[i : i+in_steps])
                    y.append(data_array[i+in_steps : i+in_steps+out_steps])
                return np.array(x), np.array(y)

            # Train set
            train_scaled = scaler.transform(train_df[['value']])
            X_train, Y_train = create_sequences(train_scaled.flatten(), req.input_steps, req.output_steps)
            if len(X_train) > 0:
                X_train = np.expand_dims(X_train, axis=2)
                train_preds = model.predict(X_train, verbose=0)

                Y_train_inverted = scaler.inverse_transform(Y_train.reshape(-1,1))
                train_preds_inverted = scaler.inverse_transform(train_preds.reshape(-1,1))
                train_rmse = sqrt(mean_squared_error(Y_train_inverted, train_preds_inverted))
            else:
                train_rmse = None

            # Val set
            val_scaled = scaler.transform(val_df[['value']])
            X_val, Y_val = create_sequences(val_scaled.flatten(), req.input_steps, req.output_steps)
            if len(X_val) > 0:
                X_val = np.expand_dims(X_val, axis=2)
                val_preds = model.predict(X_val, verbose=0)

                Y_val_inverted = scaler.inverse_transform(Y_val.reshape(-1,1))
                val_preds_inverted = scaler.inverse_transform(val_preds.reshape(-1,1))
                val_rmse = sqrt(mean_squared_error(Y_val_inverted, val_preds_inverted))
            else:
                val_rmse = None

            results_list.append({
                "Model": mdl,
                "Train_RMSE": train_rmse,
                "Validation_RMSE": val_rmse
            })

        elif mdl == "ARIMA":
            train_len = len(train_df)
            val_len = len(val_df)
            train_rmse, val_rmse = None, None

            if train_len > 0:
                fitted_values = model.predict(start=0, end=train_len-1)
                train_rmse = sqrt(mean_squared_error(train_df['value'].values, fitted_values))

            if val_len > 0:
                forecast_val = model.forecast(steps=val_len)
                val_rmse = sqrt(mean_squared_error(val_df['value'].values, forecast_val))

            results_list.append({
                "Model": mdl,
                "Train_RMSE": train_rmse,
                "Validation_RMSE": val_rmse
            })

        elif mdl == "PROPHET":
            df_train = train_df.rename(columns={"timestamp":"ds","value":"y"})
            df_val   = val_df.rename(columns={"timestamp":"ds","value":"y"})
            train_rmse, val_rmse = None, None

            if not df_train.empty:
                train_forecast = model.predict(df_train[["ds"]])
                train_rmse = sqrt(mean_squared_error(df_train["y"], train_forecast["yhat"]))

            if not df_val.empty:
                val_forecast = model.predict(df_val[["ds"]])
                val_rmse = sqrt(mean_squared_error(df_val["y"], val_forecast["yhat"]))

            results_list.append({
                "Model": mdl,
                "Train_RMSE": train_rmse,
                "Validation_RMSE": val_rmse
            })

        else:
            # TimeGPT, LAGLLAMA => ephemeral => no local RMSE
            results_list.append({
                "Model": mdl,
                "Train_RMSE": None,
                "Validation_RMSE": None
            })

    return results_list

# ------------------------------------------------------------------------------
# Endpoint /infer
# ------------------------------------------------------------------------------
@app.post("/infer", response_model=InferResponse)
def infer_model(req: InferRequest):
    """
    Performs inference using a previously trained model (GRU, LSTM, ARIMA, Prophet),
    or ephemeral (TimeGPT, LAGLLAMA).
    """
    thanos_url = req.thanos_url or DEFAULT_THANOS_URL
    model_name = req.model.strip().upper()
    logger.info(f"[INFER] Inference on model: {model_name}")

    # Caso speciale: TIMEGPT (no local training)
    if model_name == "TIMEGPT":
        data = fetch_data(thanos_url, req.query, req.input_steps)
        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="No data returned for TimeGPT inference.")

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values(by='timestamp').reset_index(drop=True)

        nixtla_client = NixtlaClient(api_key=TIMEGPT_API_KEY)
        nixtla_client.validate_api_key()

        df_polars = pl.from_pandas(data)
        fcst_df = nixtla_client.forecast(
            df=df_polars.to_pandas(),
            h=req.output_steps,
            freq='T',
            time_col='timestamp',
            target_col='value'
        )
        if fcst_df is None or fcst_df.empty:
            raise HTTPException(status_code=400, detail="TimeGPT returned no predictions.")

        pred_col = [c for c in fcst_df.columns if c.lower() != 'timestamp'][0]
        predictions = [
            PredictionItem(timestamp=str(row['timestamp']), prediction=float(row[pred_col]))
            for _, row in fcst_df.iterrows()
        ]
        return InferResponse(model="TIMEGPT", predictions=predictions)

    # Caso speciale: LAGLLAMA
    if model_name == "LAGLLAMA":
        if not LAG_LLAMA_AVAILABLE:
            raise HTTPException(status_code=400, detail="Lag-Llama or Torch not installed/configured.")

        data = fetch_data(thanos_url, req.query, req.input_steps)
        if data is None or data.empty:
            raise HTTPException(status_code=400, detail="No data returned for Lag-Llama inference.")

        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values(by='timestamp').reset_index(drop=True)

        import torch
        from gluonts.dataset.common import ListDataset
        from lag_llama.gluon.estimator import LagLlamaEstimator

        ckpt_path = LAG_LLAMA_CHECKPOINT
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot load Lag-Llama checkpoint: {e}")

        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        zs_estimator = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=req.output_steps,
            context_length=req.input_steps,
            device=torch.device("cpu"),  # CPU
            input_size=estimator_args.get("input_size", 1),
            n_layer=estimator_args.get("n_layer", 8),
            n_embd_per_head=estimator_args.get("n_embd_per_head", 16),
            n_head=estimator_args.get("n_head", 8),
            scaling=estimator_args.get("scaling", 1.0),
            time_feat=estimator_args.get("time_feat", False),
            batch_size=32,
            num_parallel_samples=100
        )

        test_data = ListDataset(
            [
                {
                    "start": data["timestamp"].iloc[0],
                    "target": data["value"].values
                }
            ],
            freq="T"
        )
        transformation = zs_estimator.create_transformation()
        lightning_module = zs_estimator.create_lightning_module()
        zs_predictor = zs_estimator.create_predictor(transformation, lightning_module)

        forecast_iter = zs_predictor.predict(test_data)
        forecast_obj = list(forecast_iter)[0]

        last_ts = data['timestamp'].iloc[-1]
        future_ts = [last_ts + pd.Timedelta(minutes=i+1) for i in range(req.output_steps)]
        preds = forecast_obj.mean

        predictions = [
            PredictionItem(timestamp=str(t), prediction=float(p))
            for t, p in zip(future_ts, preds)
        ]
        return InferResponse(model="LAGLLAMA", predictions=predictions)

    # Modelli local
    data = fetch_data(thanos_url, req.query, req.input_steps)
    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="No data returned for inference.")

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp').reset_index(drop=True)

    # Carichiamo da file
    model_info = load_model(req.query, req.training_hours, req.input_steps, req.output_steps, model_name)
    if model_info is None:
        raise HTTPException(status_code=400, detail=f"Model {model_name} not found. Train it first.")

    model = model_info["model"]
    scaler = model_info.get("scaler", None)
    logger.info(f"[INFER] Loaded model {model_name} from checkpoint.")

    # GRU / LSTM
    if model_name in ["GRU", "LSTM"]:
        if not scaler:
            raise ValueError("Scaler not found for GRU/LSTM model.")

        scaled_series = scaler.transform(data[['value']])
        if len(scaled_series) < req.input_steps:
            raise HTTPException(status_code=400,
                                detail=f"Not enough data for {model_name} inference. Have {len(scaled_series)}, need {req.input_steps}.")

        last_window = scaled_series[-req.input_steps:].reshape(1, req.input_steps, 1)
        preds_scaled = model.predict(last_window, verbose=0)[0]
        preds = scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

        predictions = [
            PredictionItem(timestamp=str(data['timestamp'].iloc[-1] + pd.Timedelta(minutes=i+1)), prediction=float(p))
            for i, p in enumerate(preds)
        ]
        return InferResponse(model=model_name, predictions=predictions)

    # ARIMA
    elif model_name == "ARIMA":
        preds = model.forecast(steps=req.output_steps)
        timestamps = pd.date_range(
            start=data['timestamp'].iloc[-1],
            periods=req.output_steps+1,
            freq='T'
        )[1:]

        predictions = [
            PredictionItem(timestamp=str(t), prediction=float(p))
            for t, p in zip(timestamps, preds)
        ]
        return InferResponse(model="ARIMA", predictions=predictions)

    # PROPHET
    elif model_name == "PROPHET":
        future = model.make_future_dataframe(periods=req.output_steps, freq='T')
        forecast = model.predict(future)

        preds = forecast['yhat'].iloc[-req.output_steps:].values
        timestamps = future['ds'].iloc[-req.output_steps:].values

        predictions = [
            PredictionItem(timestamp=str(t), prediction=float(p))
            for t, p in zip(timestamps, preds)
        ]
        return InferResponse(model="PROPHET", predictions=predictions)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model for infer: {model_name}")


# ------------------------------------------------------------------------------
# Endpoint /anomaly
# ------------------------------------------------------------------------------
@app.post("/anomaly", response_model=AnomalyResponse)
def detect_anomaly(req: AnomalyRequest):
    thanos_url = req.thanos_url or DEFAULT_THANOS_URL
    data = fetch_data(thanos_url, req.query, req.hours)
    if data is None or data.empty:
        raise HTTPException(status_code=400, detail="No data returned for anomaly detection.")

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values('timestamp').reset_index(drop=True)

    method = req.method.lower()
    if method == "zscore":
        mean_val = data['value'].mean()
        std_val = data['value'].std()
        threshold = 3
        data['z_score'] = (data['value'] - mean_val)/std_val
        data['is_anomaly'] = data['z_score'].abs() > threshold
        anomalies = data[data['is_anomaly']]
        anomaly_list = [
            AnomalyItem(timestamp=str(row['timestamp']), value=float(row['value']))
            for _, row in anomalies.iterrows()
        ]
        return AnomalyResponse(method="zscore", anomalies=anomaly_list)

    elif method == "iqr":
        Q1 = data['value'].quantile(0.25)
        Q3 = data['value'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        data['is_anomaly'] = (data['value'] < lower_bound) | (data['value'] > upper_bound)
        anomalies = data[data['is_anomaly']]
        anomaly_list = [
            AnomalyItem(timestamp=str(row['timestamp']), value=float(row['value']))
            for _, row in anomalies.iterrows()
        ]
        return AnomalyResponse(method="iqr", anomalies=anomaly_list)

    elif method == "timegpt":
        nixtla_client = NixtlaClient(api_key=TIMEGPT_API_KEY)
        nixtla_client.validate_api_key()

        df_polars = pl.from_pandas(data)
        anomalies_df = nixtla_client.detect_anomalies(
            df=df_polars.to_pandas(),
            freq='T',
            level=req.confidence,
            time_col='timestamp',
            target_col='value'
        )
        anomalies_only = anomalies_df[anomalies_df['anomaly'] == 1]
        anomaly_list = []
        for _, row in anomalies_only.iterrows():
            tstamp = row.get('timestamp', None)
            val = row.get('value', None)
            if tstamp is not None and val is not None:
                anomaly_list.append(
                    AnomalyItem(timestamp=str(tstamp), value=float(val))
                )
        return AnomalyResponse(method="timegpt", anomalies=anomaly_list)

    else:
        raise HTTPException(status_code=400, detail="method must be one of: zscore, iqr, timegpt.")
