# AI-DPM API Service

## Overview
Complete AI-DPM workflow API with five main endpoints for time series forecasting and anomaly detection.
![alt text](image-1.png)

## API Endpoints

### `/fetch` - Historical Data Retrieval
- Retrieves time series data from Prometheus/Thanos
- **Parameters**: query, time range (hours), optional Thanos URL
- **Output**: Historical data for model training

### `/train` - Model Training
- Trains forecasting models (local and LLM)
- **Models**: GRU, LSTM, ARIMA, Prophet, Lag-Llama
- **Parameters**: query, duration, input/output steps, model type
- **Output**: Training confirmation, models saved to `models/` directory

### `/infer` - Prediction Generation
- Generates future predictions using trained models
- **Supports**: All model types (classical, local, cloud-based LLMs)
- **Parameters**: query, time range, input/output steps, model name
- **Output**: Time-stamped future predictions

### `/anomaly` - Anomaly Detection
- Identifies unusual patterns in time series data
- **Parameters**: query, detection method, confidence interval, duration
- **Output**: Time-stamped anomaly flags

### `/compute_rmse` - Model Evaluation
- Calculates Root Mean Square Error across models
- **Purpose**: Model comparison and selection
- **Output**: RMSE values (lower = better performance)

## Workflow
1. **Fetch** historical data
2. **Train** models with different algorithms
3. **Evaluate** performance using RMSE
4. **Generate** predictions via inference
5. **Monitor** for anomalies