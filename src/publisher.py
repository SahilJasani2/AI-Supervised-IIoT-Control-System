import paho.mqtt.client as mqtt
import time
import json
import random
import torch
import torch.nn as nn
import joblib
import numpy as np

# Import the new Motor class
from motor_model import Motor

# --- AI Model Configuration ---
MODEL_FILE = 'model.pth'
SCALER_FILE = 'scaler.gz'
ANOMALY_THRESHOLD = 0.134833 

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_DATA_TOPIC = "iiot/motor1/data"
MQTT_AI_COMMAND_TOPIC = "iiot/motor1/command"
MQTT_SPEED_TOPIC = "iiot/motor1/speed"
MQTT_VOLTAGE_TOPIC = "iiot/motor1/control/voltage"

# --- Global variable to store the latest voltage from the controller ---
latest_voltage = 0.0

# --- AI Model Definition ---
# IMPORTANT: The input dimension will change. We'll handle this when we retrain.
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
    # The input dimension is now 2 (EKF Vibration, EKF Temperature)
    model = Autoencoder(input_dim=2) 
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    print("Model loaded successfully.")
    scaler = joblib.load(SCALER_FILE)
    print("Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("NOTE: You may need to run train_model.py again to create a model compatible with the new 2D data.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading the model: {e}")
    exit()


# --- Initialize the Motor Model ---
motor = Motor()

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("\nPublisher successfully connected to MQTT Broker!")
        client.subscribe(MQTT_VOLTAGE_TOPIC)
        print(f"Publisher subscribed to voltage topic: {MQTT_VOLTAGE_TOPIC}")
    else:
        print(f"\nPublisher failed to connect, return code {rc}")

def on_voltage_message(client, userdata, msg):
    """Callback to update the global voltage variable when a message is received."""
    global latest_voltage
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        latest_voltage = payload.get("voltage", latest_voltage)
    except Exception as e:
        print(f"Error in on_voltage_message: {e}")

# --- Data Simulation & Prediction ---
def get_sensor_data_and_predict():
    
    # 1. Get the latest motor state from our multi-sensor model
    motor_state = motor.update_state(voltage=latest_voltage)
    
    # 2. Perform AI Inference on the FUSED, EKF-estimated data
    with torch.no_grad():
        # The AI now uses both EKF-estimated vibration and temperature
        raw_data = np.array([[motor_state["ekf_vibration"], motor_state["ekf_temperature"]]], dtype='float32')
        scaled_data = scaler.transform(raw_data)
        data_tensor = torch.from_numpy(scaled_data)
        reconstruction = model(data_tensor)
        error = nn.MSELoss()(reconstruction, data_tensor).item()
        anomaly = 1 if error > ANOMALY_THRESHOLD else 0

    # 3. Package all data for publishing, including new temperature fields
    data = {
        "timestamp": time.time(),
        "true_vibration": round(motor_state["true_vibration"], 4),
        "noisy_vibration": round(motor_state["noisy_vibration"], 4),
        "ekf_vibration": round(motor_state["ekf_vibration"], 4),
        "true_temperature": round(motor_state["true_temperature"], 2),
        "noisy_temperature": round(motor_state["noisy_temperature"], 2),
        "ekf_temperature": round(motor_state["ekf_temperature"], 2),
        "speed": round(motor_state["speed"], 2),
        "reconstruction_error": round(error, 6),
        "anomaly": anomaly
    }
    return data

# --- Main Script ---
client = mqtt.Client()
client.on_connect = on_connect
client.message_callback_add(MQTT_VOLTAGE_TOPIC, on_voltage_message)

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
except Exception as e:
    print(f"Error connecting to MQTT: {e}")
    exit()

client.loop_start()
time.sleep(1)

print("--- Starting Real-Time Multi-Sensor Publisher ---")
try:
    while True:
        payload = get_sensor_data_and_predict()
        
        client.publish(MQTT_DATA_TOPIC, json.dumps(payload))
        
        speed_payload = json.dumps({"speed": payload["speed"]})
        client.publish(MQTT_SPEED_TOPIC, speed_payload)

        print(f"Published: Speed={payload['speed']:.2f}, Temp={payload['ekf_temperature']:.2f}, Vib={payload['ekf_vibration']:.2f}")

        if payload['anomaly'] == 1:
            command_payload = json.dumps({"action": "enter_safe_mode"})
            client.publish(MQTT_AI_COMMAND_TOPIC, command_payload)
            print(f"--> ANOMALY DETECTED! Published command: {command_payload}")
        
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopping publisher.")
    client.loop_stop()
    client.disconnect()
