# AI-Powered IIoT Predictive Maintenance Stack

A complete, Docker-based IIoT data pipeline that simulates real-time sensor data from an industrial motor, processes it, and uses an AI model to detect anomalies on a live dashboard. This project serves as a template for building localized, intelligent predictive maintenance systems.

## Final Dashboard

The final Grafana dashboard provides a real-time view of the motor's key performance indicators and an immediate visual alert for AI-detected anomalies.

## Features

- **Real-time Data Simulation**: A Python script simulates a motor's RPM, temperature, and vibration, including a gradually developing fault.
- **End-to-End Data Pipeline**: Data flows from a publisher script to an MQTT broker, is captured by a subscriber script, and stored in a time-series database.
- **AI-Powered Anomaly Detection**: Deploys a PyTorch-based Autoencoder model that learns normal motor behavior to detect anomalies in real-time, simulating an Edge AI / Digital Twin scenario.
- **Live Dashboard**: A Grafana dashboard provides a real-time, visual representation of the motor's health and operational status.
- **Fully Containerized**: The entire stack—including the AI-powered Python scripts, broker, database, and dashboard—is managed by Docker Compose for one-command startup.

## Tech Stack

- **AI & Data Processing**: Python
  - **AI Modeling**: PyTorch, Scikit-learn
  - **Messaging**: paho-mqtt
  - **Database Client**: influxdb-client
- **Infrastructure**: Docker & Docker Compose
- **Messaging Broker**: Mosquitto (MQTT)
- **Database**: InfluxDB (Time-Series DB)
- **Visualization**: Grafana

## Use Case

### AI-Powered Predictive Maintenance (Digital Twin)

Imagine a factory floor with critical motors running production lines. Instead of waiting for a motor to fail unexpectedly—causing costly downtime—this system provides an intelligent monitoring solution.

The system uses an unsupervised neural network (an Autoencoder) to create a "digital twin" of the motor's healthy operational state. The model learns the complex patterns of normal behavior from an initial dataset. During live operation, it continuously compares real-time sensor data to this learned model.

When live data deviates significantly from the healthy baseline, the system flags it as a potential anomaly. This approach is more robust and adaptive than simple thresholding and represents a core concept in Industry 4.0. The Grafana dashboard provides operators with a clear, immediate alert, allowing maintenance to be scheduled proactively before a critical failure occurs.

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
│   ├── requirements.txt
│   ├── healthy_motor_data.csv # Generated training data
│   ├── model.pth              # The trained PyTorch model
│   └── scaler.gz              # The data scaler for preprocessing
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
