# AI-Powered IIoT Predictive Maintenance Stack

A complete, Docker-based IIoT data pipeline that uses advanced state estimation (EKF) and an AI model to detect anomalies from noisy sensor data and automatically trigger a control action. This project is a template for building robust, intelligent, closed-loop control systems for industrial automation.

## Final Dashboard

The final Grafana dashboard provides a real-time view of the motor's health, the EKF's performance in filtering sensor noise, and an immediate visual alert for AI-detected anomalies.

## Features

- **Realistic Data Simulation**: Simulates a motor's operational data, including a non-linear degradation process and realistic sensor noise.
- **Advanced State Estimation (EKF)**: Implements an Extended Kalman Filter to estimate the motor's true state from noisy sensor data, providing a robust, clean signal for the AI model.
- **AI-Powered Anomaly Detection**: Deploys a PyTorch-based Autoencoder model that acts as a Digital Twin, learning normal motor behavior to detect anomalies in the filtered data.
- **Closed-Loop Control**: When an anomaly is detected, the system automatically publishes a command to an MQTT topic, simulating a real-world control action (e.g., telling a PLC to enter a safe mode).
- **Fully Containerized**: The entire stack—including the EKF, AI model, control logic, broker, database, and dashboard—is managed by Docker Compose for one-command startup.

## Tech Stack

- **AI & Control Logic**: Python
  - **State Estimation**: NumPy (for EKF)
  - **AI Modeling**: PyTorch, Scikit-learn
  - **Messaging**: paho-mqtt
- **Infrastructure**: Docker & Docker Compose
- **Messaging Broker**: Mosquitto (MQTT)
- **Database**: InfluxDB
- **Visualization**: Grafana

## Use Case

### Robust, AI-Powered Closed-Loop Control

This project demonstrates a complete "filter -> detect -> act" loop, a core pattern for robust industrial automation.

Real-world industrial sensors are noisy. This system first addresses this challenge by using an Extended Kalman Filter (EKF) to process the raw, noisy vibration data. The EKF produces a clean, stable estimate of the motor's true state.

This clean, estimated data is then fed to an unsupervised neural network (an Autoencoder) that acts as a "digital twin" of the motor's healthy state. When the AI detects a significant deviation from normal behavior, it triggers a control action by publishing a command to an MQTT topic. A separate controller service, simulating an interface to a PLC, executes this command, closing the loop and transforming the system into an active, intelligent, and noise-tolerant control agent.

## Project Structure

```
iiot_project/
├── .github/
│   └── workflows/
│       └── ci-pipeline.yml
├── mosquitto/
│   └── config/
│       └── mosquitto.conf
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── publisher.py
│   ├── subscriber.py
│   ├── controller.py
│   ├── requirements.txt
│   ├── healthy_motor_data.csv
│   ├── model.pth
│   └── scaler.gz
├── docker-compose.yml
└── Dockerfile
```

## Getting Started

### Prerequisites

- Docker Desktop installed and running.

### Installation & Launch

1. Clone this repository to your local machine.
2. Navigate to the project's root directory (`iiot_project`) in your terminal.
3. Run the following command to build the Docker images and start all services:

```bash
docker-compose up -d --build
```

The entire pipeline is now running.
- **View the Dashboard**: http://localhost:3000 (login: admin/password)
- **Explore the Database**: http://localhost:8086 (login: admin/password123)

### Stopping the Application

To stop all running containers, run:

```bash
docker-compose down
```
