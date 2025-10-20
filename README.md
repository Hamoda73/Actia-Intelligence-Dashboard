# Real-Time EV Battery Monitoring System

A cloud-based platform for predictive battery health monitoring in electric vehicles, featuring AI-driven analytics and real-time anomaly detection.

## Overview

This system processes real-time EV battery telemetry using Azure cloud services and machine learning to enable predictive maintenance and safety monitoring. It transforms raw sensor data into actionable insights through advanced LSTM neural networks and an interactive dashboard.

## Key Features

- **SOC Prediction** - 1.82% MAE at 1-hour forecast horizon using LSTM forecasting
- **Anomaly Detection** - 87.3% true positive rate with LSTM Autoencoder
- **Real-time Dashboard** - Angular-based visualization with live metrics
- **Azure Cloud Integration** - IoT Hub, Stream Analytics, Cosmos DB, Machine Learning
- **Production Architecture** - End-to-end pipeline with 99.8% uptime

## System Architecture

![System Architecture](images/architecture-diagram.png)

## Models Implemented

### 1. SOC Forecasting (LSTM)
- **Input:** 16-step sequence (4 hours) of battery metrics
- **Output:** SOC prediction at 1-hour horizon
- **Performance:**
  - MAE: 1.82% SOC
  - R²: 0.94
- Captures temporal dependencies in battery behavior for accurate charge state forecasting

### 2. Anomaly Detection (LSTM Autoencoder)
- **Approach:** Sequence reconstruction error
- **Performance:**
  - True Positive Rate: 87.3%
  - False Positive Rate: 2.1%
- Identifies abnormal patterns in voltage, temperature, and current measurements

## Technology Stack

- **Cloud:** Microsoft Azure (IoT Hub, Stream Analytics, Cosmos DB, Machine Learning, Functions)
- **Backend:** Spring Boot, Java
- **Frontend:** Angular, TypeScript, Chart.js
- **Machine Learning:** Python, PyTorch, Scikit-learn
- **Development:** VS Code, IntelliJ, Git, Postman

## Dashboard

![Dashboard Dark Theme](images/dashboard-dark.png)
![Dashboard Light Theme](images/dashboard-light.png)
![Predictions Page](images/predictions-page.png)

## Performance

- **End-to-End Latency:** 1.2 seconds
- **System Uptime:** 99.8%
- **Cost Efficiency:** $362/month (40% savings vs on-premise)
- **Data Processing:** 720 messages/hour sustained throughput

## Business Impact

- 30% reduction in unexpected battery failures through predictive maintenance
- Extended battery lifespan via optimized charging strategies
- Enhanced safety through real-time anomaly detection
- Cost-effective cloud monitoring solution

## Future Enhancements

- WebSocket implementation for true real-time updates
- Multi-vehicle fleet management capabilities
- Mobile application development
- Edge computing deployment
- Advanced predictive maintenance features

## Acknowledgments

Developed during a summer internship at ACTIA Engineering Services under the supervision of Mr. Sofiane Sayahi and Mr. Youssef Allagui. Special thanks to ACTIA Engineering Services for providing the resources and collaborative environment.
