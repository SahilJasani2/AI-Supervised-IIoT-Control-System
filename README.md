# AI-Powered IIoT Predictive Maintenance Stack

A complete, Docker-based IIoT data pipeline that simulates real-time sensor data, uses an AI model to detect anomalies, and automatically triggers a control action. This project serves as a template for building localized, intelligent, closed-loop control systems for industrial automation.

## Final Dashboard

The final Grafana dashboard provides a real-time view of the motor's key performance indicators and an immediate visual alert for AI-detected anomalies.

## Features

- **Real-time Data Simulation**: A Python script simulates a motor's RPM, temperature, and vibration, including a gradually developing fault.
- **AI-Powered Anomaly Detection**: Deploys a PyTorch-based Autoencoder model that learns normal motor behavior to detect anomalies in real-time, simulating an Edge AI / Digital Twin scenario.
- **Closed-Loop Control**: When an anomaly is detected, the system automatically publishes a command to an MQTT topic, simulating a real-world control action (e.g., telling a PLC to enter a safe mode).
- **Live Monitoring Dashboard**: A Grafana dashboard provides a visual representation of the motor's health and operational status.
- **Fully Containerized**: The entire stack—including the AI model, control logic, broker, database, and dashboard—is managed by Docker Compose for one-command startup.

## Tech Stack

- **AI & Control Logic**: Python
  - **AI Modeling**: PyTorch, Scikit-learn
  - **Messaging**: paho-mqtt
  - **Database Client**: influxdb-client
- **Infrastructure**: Docker & Docker Compose
- **Messaging Broker**: Mosquitto (MQTT)
- **Database**: InfluxDB (Time-Series DB)
- **Visualization**: Grafana

## Use Case

### AI-Powered Closed-Loop Control

This project demonstrates a complete "monitor -> detect -> act" loop, a core pattern in industrial automation.

The system uses an unsupervised neural network (an Autoencoder) to create a "digital twin" of a motor's healthy state. During live operation, it continuously compares real-time sensor data to this learned model.

When the AI detects a significant deviation (an anomaly), it doesn't just raise an alarm. It automatically publishes a command, such as `{"action": "enter_safe_mode"}`, to a control topic. A separate controller service, simulating an interface to a PLC, subscribes to this topic and executes the command. This closes the loop, transforming the system from a passive monitor into an active, intelligent control agent.

## Project Structure

```
iiot_project/
├── mosquitto/
│   └── config/
│       └── mosquitto.conf
├── src/
│   ├── generate_data.py       # Script to create the training dataset
│   ├── train_model.py         # Script to train the AI model
│   ├── publisher.py           # Live data simulation and AI inference
│   ├── subscriber.py          # Data ingestion to database
│   ├── controller.py          # Simulated PLC/control action listener
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
