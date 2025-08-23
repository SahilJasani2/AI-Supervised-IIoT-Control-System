# IIoT Predictive Maintenance Stack

A complete, Docker-based IIoT data pipeline that simulates real-time sensor data from an industrial motor, processes it, and visualizes it on a live dashboard. This project serves as a template for building localized predictive maintenance and industrial monitoring systems.

## Final Dashboard

The final Grafana dashboard provides a real-time view of the motor's key performance indicators and an immediate visual alert for detected anomalies.

## Features

- **Real-time Data Simulation**: A Python script simulates a motor's RPM, temperature, and vibration, including a gradually developing fault.
- **End-to-End Data Pipeline**: Data flows from a publisher script to an MQTT broker, is captured by a subscriber script, and stored in a time-series database.
- **Live Dashboard**: A Grafana dashboard provides a real-time, visual representation of the motor's health and operational status.
- **Rule-Based Anomaly Detection**: A simple predictive maintenance model flags data points as anomalies when vibration exceeds a predefined threshold.
- **Fully Containerized**: The entire stack—including the Python scripts, broker, database, and dashboard—is managed by Docker Compose for one-command startup and shutdown.

## Tech Stack

- **Data Simulation & Processing**: Python
  - `paho-mqtt` for messaging
  - `influxdb-client` for database communication
- **Infrastructure**: Docker & Docker Compose
- **Messaging Broker**: Mosquitto (MQTT)
- **Database**: InfluxDB (Time-Series DB)
- **Visualization**: Grafana

## Use Case

### Predictive Maintenance for Industrial Motors

Imagine a factory floor with dozens of critical motors running conveyor belts or pumps. Instead of waiting for a motor to fail unexpectedly—causing costly downtime—this system can monitor the health of each motor in real time.

By tracking key metrics like vibration, the system can detect early signs of wear and tear (e.g., a failing bearing). The Grafana dashboard provides operators with a clear view of a motor's status. When the "Motor Health Status" panel turns red, maintenance can be scheduled proactively, before a critical failure occurs, saving time and money.

## Project Structure

```
iiot_project/
├── mosquitto/
│   └── config/
│       └── mosquitto.conf
├── src/
│   ├── publisher.py
│   ├── subscriber.py
│   └── requirements.txt
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
