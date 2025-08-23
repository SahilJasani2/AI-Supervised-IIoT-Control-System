import paho.mqtt.client as mqtt
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np
import joblib

# --- AI Model Configuration ---
MODEL_FILE = 'model.pth'
SCALER_FILE = 'scaler.gz'
ANOMALY_THRESHOLD = 0.122610 

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_DATA_TOPIC = "iiot/motor1/data"
MQTT_COMMAND_TOPIC = "iiot/motor1/command"

# --- EKF Configuration ---
# State is a single value: [vibration]
# Process noise covariance (Q): How much we trust our process model.
# A small value means we trust our model a lot.
Q = np.array([[1e-5]]) 
# Measurement noise covariance (R): How much we trust the sensor measurement.
# A larger value means the sensor is noisy.
R = np.array([[0.75**2]]) 
# Initial state estimate
x_hat = np.array([[3.5]])
# Initial error covariance
P = np.array([[1.0]])

# --- AI Model Definition ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 2), nn.ReLU(), nn.Linear(2, 1))
        self.decoder = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, input_dim))
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Load AI Model and Scaler ---
print("--- Loading AI Model and Scaler ---")
try:
    model = Autoencoder(input_dim=3)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    print("Model loaded successfully.")
    scaler = joblib.load(SCALER_FILE)
    print("Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# --- EKF Functions ---
def f(x):
    """ Non-linear state transition function. """
    degradation_rate = 0.05 + (x[0, 0] / 100.0)
    return np.array([[x[0, 0] + degradation_rate]])

def h(x):
    """ Measurement function (linear in this case). """
    return x

def F_jacobian(x):
    """ Jacobian of the state transition function f(x). """
    # Derivative of f(x) with respect to x is 1 + 1/100
    return np.array([[1.01]])

def H_jacobian(x):
    """ Jacobian of the measurement function h(x). """
    # Derivative of h(x) with respect to x is 1
    return np.array([[1.0]])

def ekf_predict_update(x_hat, P, z):
    """ Performs one step of the Extended Kalman Filter. """
    # --- Predict Step ---
    x_hat_minus = f(x_hat)
    F = F_jacobian(x_hat)
    P_minus = F @ P @ F.T + Q

    # --- Update Step ---
    H = H_jacobian(x_hat_minus)
    y_tilde = z - h(x_hat_minus)
    S = H @ P_minus @ H.T + R
    K = P_minus @ H.T @ np.linalg.inv(S)
    x_hat_new = x_hat_minus + K @ y_tilde
    P_new = (np.eye(1) - K @ H) @ P_minus
    
    return x_hat_new, P_new

# --- Data Simulation & Prediction ---
true_vibration_state = 3.5

def get_sensor_data_and_predict():
    global true_vibration_state, x_hat, P
    
    # 1. Simulate the true process and noisy measurement
    degradation_rate = 0.05 + (true_vibration_state / 100.0)
    true_vibration_state += degradation_rate
    noise = np.random.normal(0, 0.75)
    noisy_vibration_measurement = true_vibration_state + noise
    
    # 2. Use EKF to get a clean estimate of the state
    z = np.array([[noisy_vibration_measurement]])
    x_hat, P = ekf_predict_update(x_hat, P, z)
    ekf_estimated_vibration = x_hat[0, 0]

    # Simulate other sensor data
    rpm = 700 + random.uniform(-20, 20)
    temperature = 85.5 + random.uniform(-1.5, 1.5)
    
    # 3. Perform AI Inference on the CLEAN, ESTIMATED data
    with torch.no_grad():
        raw_data = np.array([[rpm, temperature, ekf_estimated_vibration]], dtype='float32')
        scaled_data = scaler.transform(raw_data)
        data_tensor = torch.from_numpy(scaled_data)
        reconstruction = model(data_tensor)
        error = nn.MSELoss()(reconstruction, data_tensor).item()
        anomaly = 1 if error > ANOMALY_THRESHOLD else 0

    # 4. Package all data for publishing
    data = {
        "timestamp": time.time(),
        "rpm": round(rpm, 2),
        "temperature": round(temperature, 2),
        "true_vibration": round(true_vibration_state, 4),
        "noisy_vibration": round(noisy_vibration_measurement, 4),
        "ekf_vibration": round(ekf_estimated_vibration, 4),
        "reconstruction_error": round(error, 6),
        "anomaly": anomaly
    }
    return data

# --- MQTT Connection and Main Loop ---
def on_connect(client, userdata, flags, rc):
    if rc == 0: print("\nSuccessfully connected to MQTT Broker!")
    else: print(f"\nFailed to connect, return code {rc}")

client = mqtt.Client()
client.on_connect = on_connect
try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
except Exception as e:
    print(f"Error connecting to MQTT: {e}")
    exit()

client.loop_start()
time.sleep(1)

print("--- Starting Real-Time EKF Estimation and AI Anomaly Detection ---")
try:
    while True:
        payload = get_sensor_data_and_predict()
        json_payload = json.dumps(payload)
        client.publish(MQTT_DATA_TOPIC, json_payload)
        print(f"Published data: {json_payload}")

        if payload['anomaly'] == 1:
            command_payload = json.dumps({"action": "enter_safe_mode"})
            client.publish(MQTT_COMMAND_TOPIC, command_payload)
            print(f"--> ANOMALY DETECTED! Published command: {command_payload}")
        
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopping publisher.")
    client.loop_stop()
    client.disconnect()
