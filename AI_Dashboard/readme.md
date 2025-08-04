# AI-DPM Service Dashboard

## Overview
The AI-DPM service dashboard goes hand-in-hand with AI-DPM API. It provides an intuitive interface for time series forecasting and anomaly detection, organized into two main sections: Global Configuration and Monitoring & ML Dashboard Panel.

## Global Configuration Panel

### Data Source Configuration
- **PromQL Query**: Define data source (defaults to CPU idle metrics)
- Compatible with Prometheus, Kepler, and Istio timeseries metadata
- Requires properly formatted and aggregated timeseries data

### Time Window Settings
- **Hours for fetch/train/anomaly**: Controls historical time window (default: 3 hours)
- Short windows (1-6 hours): Immediate patterns
- Medium windows (12-48 hours): Daily patterns  
- Long windows (72+ hours): Weekly patterns

### Model Parameters
- **Input Steps (RNN)**: Previous time points for predictions (default: 48 steps)
- **Output Steps**: Prediction horizon (default: 20 steps)

### Available Models
- Statistical: ARIMA, Prophet
- RNN: GRU, LSTM
- LLM: TimeGPT, LagLlama

## Operations Panel

### Core Functions
- **Fetch Historical Data**: Retrieve time-series data from Thanos backend
- **Train Local Models**: Select and train forecasting models
- **Compute RMSE**: Evaluate model performance with Root Mean Square Error
- **Anomaly Detection**: Identify unusual patterns using rolling z-score and prediction intervals

### Visualization Features
- Prediction results and plots
- Anomaly visualization with highlighted regions
- Tabular view of predicted values with timestamps
- Model performance comparison tools

## Key Benefits
- Intuitive interface for AI-DPM API functionality
- Complete ML cycle experimentation
- Model performance evaluation and comparison