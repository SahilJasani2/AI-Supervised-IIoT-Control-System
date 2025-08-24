# 🏭 AI-Supervised IIoT Control System

A complete, Docker-based IIoT template for building a robust, intelligent industrial control system. This project demonstrates how an AI agent can act as a supervisor for a traditional PID controller.

## 🏗️ Architecture: AI-Supervised PID Control

This project demonstrates a hybrid control architecture, a core pattern for modern industrial automation.

### The PID Controller (The Worker ⚙️)
A standard PID controller runs a tight, closed-loop operation. Its only job is to maintain the motor's speed at a given setpoint (e.g., 1500 RPM) by constantly adjusting the output voltage.

### The AI Agent (The Supervisor 🧠)
An unsupervised Autoencoder model acts as a "digital twin," monitoring the motor's health (vibration) from the EKF-filtered data. It does not control the motor directly. Instead, when it detects a developing fault, it acts as a supervisor.

### The Action
Upon detecting an anomaly, the AI publishes a command. The PID controller receives this command and changes its setpoint to a safe, lower value (e.g., 500 RPM), automatically bringing the system to a stable, safe state.

## ✨ Features

- **Realistic Data Simulation**: Simulates a motor's speed and a non-linear degradation process (vibration) affected by sensor noise.
- **Advanced State Estimation (EKF)**: Implements an Extended Kalman Filter to provide a clean, robust estimate of the motor's health from noisy data.
- **Closed-Loop PID Control**: A PID controller actively manages the motor's speed to match a dynamic setpoint.
- **AI-Powered Supervision**: A PyTorch-based Autoencoder detects anomalies and intelligently modifies the PID controller's goal.
- **Fully Containerized**: The entire stack is managed by Docker Compose for one-command startup.

## 🛠️ Tech Stack

- **AI & Control Logic**: Python
  - **Control**: simple-pid
  - **State Estimation**: NumPy
  - **AI Modeling**: PyTorch, Scikit-learn
- **Infrastructure**: Docker & Docker Compose
- **Messaging**: Mosquitto (MQTT)
- **Database**: InfluxDB
- **Visualization**: Grafana

## 📁 Project Structure

```
iiot_project/
├── .github/
│   └── workflows/
│       └── ci-pipeline.yml
├── mosquitto/
│   └── config/
│       └── mosquitto.conf
├── src/
│   ├── motor_model.py
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

## 🚀 Getting Started

### 📋 Prerequisites

- Docker Desktop installed and running

### ⚙️ Installation & Launch

1. Clone this repository
2. Navigate to the project's root directory in your terminal
3. Run the following command to build the Docker images and start all services:

```bash
docker-compose up --build
```

The entire pipeline is now running.

- **📊 View the Dashboard**: [http://localhost:3000](http://localhost:3000) (login: `admin`/`password`)
- **💾 Explore the Database**: [http://localhost:8086](http://localhost:8086) (login: `admin`/`password123`)

### ⏹️ Stopping the Application

To stop all running containers, run:

```bash
docker-compose down
```
