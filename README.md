# Actia Intelligence Dashboard

A modern cloud-based monitoring system for **predictive failure detection** in electric vehicle (EV) and power grid infrastructure.  
Processes **real-time sensor data** with AI-driven forecasting and anomaly detection for actionable insights.

---

## 🚗 EV Battery Monitoring System

![SOC Anomaly Dashboard](https://via.placeholder.com/800x400?text=SOC+Anomaly+Dashboard)

Real-time battery monitoring dashboard with **predictive analytics** and **feature-level anomaly detection**.

---

## 📌 Overview
This system leverages **Azure IoT + AI models** to monitor commercial EV bus battery health in real time.  
It processes **4.2M+ data points** to:
- Predict **State of Charge (SOC)** using **LSTM forecasting**
- Detect anomalies using **LSTM Autoencoder**
- Visualize results in a **real-time web dashboard**

---

## ✨ Key Features
- ✅ **SOC Prediction** — 0.39 MAE (~0.39% SOC error) at 40-second forecast horizon  
- 🚨 **Anomaly Detection** — Feature-level diagnostics with 91% detection precision  
- 📊 **Real-time Dashboard** — Built with Angular + Chart.js for dynamic visualization  
- ☁️ **Azure Cloud Integration** — IoT Hub, Stream Analytics, Cosmos DB, App Services  
- ⚡ **Hybrid Architecture** — Edge pre-processing + cloud analytics

---

## 🏗 System Architecture

![System Architecture](https://github.com/user-attachments/assets/be65d605-aee6-4272-bdf3-4b4b38e2d42c)

---

## 🤖 Models Implemented

### 1. SOC Forecasting (LSTM)
- **Input:** 16-step sequence (160 seconds) of battery metrics  
- **Output:** SOC prediction at an hour horizon  
- **Performance (Test Set):**
  - MAE: **0.39 SOC** (~0.39%)
  - RMSE: **0.51 SOC**
  - R²: **0.9987**
  - MAPE: **0.56%**
- **Why LSTM?** Captures temporal dependencies in SOC trends for more accurate predictions.

---

### 2. Anomaly Detection (LSTM Autoencoder)
- **Approach:** Sequence reconstruction error  
- **Performance:**
  - Precision: **91%**
  - Detection Latency: **7.2s**
- **Why LSTM Autoencoder?** Learns normal sequence patterns and flags deviations under varying load/temperature conditions.

```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
