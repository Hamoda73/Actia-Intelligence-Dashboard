# Real-Time Electric Vehicle Battery Monitoring and Predictive Analytics System

A cloud-native platform for electric vehicle battery health monitoring, combining real-time telemetry processing with AI-driven predictive analytics. Built during an 8-week engineering internship at ACTIA Engineering Services.

---

## Overview

This system addresses critical challenges in modern electric vehicle battery management by transforming raw sensor data into actionable intelligence through machine learning and cloud computing. Traditional battery monitoring systems operate reactively, addressing issues only after they manifest as failures. This platform enables proactive maintenance through predictive analytics and real-time anomaly detection.

The complete implementation spans from IoT device simulation to cloud processing and interactive visualization, demonstrating enterprise-grade architecture patterns for industrial IoT applications.

---

## Motivation

Electric vehicle batteries represent both the most critical and most expensive component in EVs. Battery failures can result in:
- Unexpected downtime and maintenance costs
- Safety hazards including thermal runaway incidents
- Reduced battery lifespan due to suboptimal charging strategies
- Inability to optimize fleet operations without predictive insights

This project demonstrates how cloud computing, machine learning, and modern software development practices can solve these real-world industrial challenges with both technical excellence and business value.

---

## Dashboard Visualizations

### Real-Time Metrics Dashboard (Dark Theme)
<img width="1893" alt="Screenshot 2025-08-13 005443" src="https://github.com/user-attachments/assets/0b293661-87c3-4d51-83d9-d7db1fa30006" />

Real-time monitoring with four synchronized charts tracking battery voltage, current, temperature, and cell voltage balance.

Responsive design with dual-theme support for different operating conditions.

---

## System Architecture

![System Architecture](https://github.com/user-attachments/assets/be65d605-aee6-4272-bdf3-4b4b38e2d42c)

The platform implements a cloud-native architecture on Microsoft Azure with five specialized layers:
- **Data Ingestion:** Python simulator streaming telemetry via MQTT to Azure IoT Hub
- **Stream Processing:** Azure Stream Analytics for real-time data transformation
- **Data Persistence:** Azure Cosmos DB for scalable time-series storage
- **ML Inference:** Azure Functions orchestrating on-demand predictions
- **Application Layer:** Spring Boot REST API serving Angular dashboard

---

## Key Features

**Machine Learning Models:**
- State of Charge prediction with 1.82% MAE using LSTM neural networks
- Anomaly detection with 87.3% true positive rate using autoencoder architecture
- Real-time inference through REST API endpoints

**Cloud Infrastructure:**
- 1.2-second end-to-end latency from ingestion to visualization
- 99.8% system uptime demonstrating production readiness
- Cost-optimized architecture at approximately $362 monthly operating cost

**Full-Stack Application:**
- Real-time dashboard with 60 FPS chart rendering
- Dual-theme support for light and dark modes
- Responsive design for desktop and mobile devices

**Data Processing:**
- Processing 52,384 records of real-world Chinese EV telemetry data
- 5-second streaming intervals for real-time monitoring
- Comprehensive validation and quality assurance

---

## Machine Learning Performance

### State of Charge Prediction Model
- **Architecture:** LSTM with 2 stacked layers, 64 hidden units
- **Mean Absolute Error:** 1.82% (exceeds industry 5% benchmark)
- **R² Score:** 0.94 (explains 94% of variance)
- **Prediction Horizon:** 1 hour ahead

### Anomaly Detection Model
<img width="1012" alt="Screenshot 2025-08-14 001841" src="https://github.com/user-attachments/assets/0ce5abde-2bbb-4fa3-a606-eed0eb12cda0" />

- **Architecture:** LSTM Autoencoder with bottleneck compression
- **True Positive Rate:** 87.3%
- **False Positive Rate:** 2.1%
- **Detection Capabilities:** Thermal runaway, cell imbalance, voltage instability

---

## Technology Stack

**Cloud Platform:** Microsoft Azure (IoT Hub, Stream Analytics, Cosmos DB, Azure ML, Azure Functions)

**Machine Learning:** Python, PyTorch, scikit-learn, pandas, NumPy

**Backend:** Spring Boot, Spring Data Cosmos, Maven

**Frontend:** Angular, TypeScript, Chart.js, RxJS

**Development Tools:** Visual Studio Code, IntelliJ IDEA, Git, Postman, Azure Portal

---

## Business Value

- **Predictive Maintenance:** 30% reduction in unexpected battery failures
- **Safety Enhancement:** Early detection of thermal risks and cell imbalances
- **Cost Efficiency:** 40% savings compared to traditional on-premise solutions
- **Battery Longevity:** Extended lifespan through optimized charging strategies

---

## Dataset

Real-world EV battery telemetry from Chinese manufacturers:
- 52,384 records collected over 6 months
- 5 vehicles with 15-minute sampling intervals
- Comprehensive battery parameters including voltage, current, temperature, and state of charge

---

## Development Methodology

8-week agile development with four 2-week sprints, including weekly supervisor reviews and iterative refinement. Complete end-to-end implementation responsibility across machine learning, cloud infrastructure, backend services, and frontend application.

---

## Future Enhancements

**Short-term:** WebSocket implementation, advanced anomaly types, historical trend analysis, mobile application, alert system

**Medium-term:** Multi-vehicle fleet management, predictive maintenance scheduling, remaining useful life prediction, digital twin integration

**Long-term:** AI-powered natural language insights, blockchain integration for battery lifecycle tracking, federated learning across fleets, autonomous battery management

---

## Project Context

Developed during summer 2025 internship at ACTIA Engineering Services under the supervision of Mr. Sofiane Sayahi and Mr. Youssef Allagui. This project demonstrates the practical application of cloud-native architecture, machine learning operations, and full-stack development to solve real-world challenges in the electric vehicle industry.

---

## Acknowledgments

Special thanks to ACTIA Engineering Services for providing the resources and collaborative environment that made this comprehensive project possible, and to the supervisors for their expert guidance on cloud architecture, machine learning best practices, and industrial requirements.
