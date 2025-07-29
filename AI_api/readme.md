# AI Time Series Prediction Service

A FastAPI service that provides time series forecasting capabilities using multiple models and integration with Thanos metrics.

## Overview

This service enables:
- Training different time series models on historical data
- Making predictions using trained models
- Integration with Thanos for fetching metrics
- Support for multiple model types: GRU, LSTM, ARIMA, Prophet, and TimeGPT

## Features

- **Multiple Model Support**:
  - RNN Models: GRU, LSTM  
  - Statistical Models: ARIMA, Prophet
  - Cloud API: TimeGPT

- **Flexible Data Source**: 
  - Fetches time series data from Thanos using PromQL queries
  - Configurable training window

- **Two Main Operations**:
  - Model Training: Train models on historical data
  - Inference: Generate predictions using trained models

## API Endpoints

### Train a Model (`/train`)

Trains a model on historical data from Thanos.

#### Request
```json
POST /train
{
    "thanos_url": "http://thanos:9090", // URL of the Thanos instance
    "query": "sum(rate(node_cpu_seconds_total{mode='idle'}[5m]))", // PromQL query to fetch data
    "training_hours": 48, // Number of hours of historical data to use for training
    "input_steps": 60, // Number of input steps for the model
    "output_steps": 20, // Number of output steps for the model
    "model": "GRU" // Model type to train (e.g., GRU, LSTM, ARIMA, Prophet, TimeGPT)
}
```

### Generate Predictions (`/infer`)

Generates predictions using a trained model.

#### Request
```json
POST /infer
{
    "thanos_url": "http://thanos:9090", // URL of the Thanos instance
    "query": "sum(rate(node_cpu_seconds_total{mode='idle'}[5m]))", // PromQL query to fetch data
    "input_steps": 60, // Number of input steps for the model
    "output_steps": 20, // Number of output steps for the model
    "model": "GRU" // Model type to use for inference (e.g., GRU, LSTM, ARIMA, Prophet, TimeGPT)
}
```

### Configuration

Configuration parameters for the service.

```json
{
    "thanos_url": "http://thanos:9090", // URL of the Thanos instance
    "training_hours": 48, // Number of hours of historical data to use for training
    "input_steps": 60, // Number of input steps for the model
    "output_steps": 20, // Number of output steps for the model
    "timegpt_api_key": "your-key-here" // API key for TimeGPT
}
```

### Model Details

- **RNN Models (GRU/LSTM)**
  - Use sliding windows of data for training
  - Normalize data using MinMaxScaler
  - Support multi-step predictions
  - Require training before inference

- **Statistical Models (ARIMA/Prophet)**
  - Traditional time series forecasting
  - No data normalization required
  - Support multi-step predictions
  - Require training before inference

- **TimeGPT**
  - Cloud-based API service
  - No local training required
  - Direct inference via API calls
  - Requires API key configuration

### Docker Deployment

Build and run using Docker:
```sh
docker build -t ai-prediction-service .
docker run -p 8504:8504 ai-prediction-service
```

### Integration with TEADAL Node
 
Registry for docker image:
 
registry.teadal.ubiwhere.com/teadal-public-images/ai-dpm:0.0.4
 
Installation note inside TEADAL node cluster with Kustomize:
 
https://gitlab.teadal.ubiwhere.com/teadal-tech/teadal.node/-/blob/main/docs/InstallTeadalTools.md?ref_type=heads#AI-DPM