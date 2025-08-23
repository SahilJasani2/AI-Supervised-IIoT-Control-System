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
# This threshold was determined by the train_model.py script
ANOMALY_THRESHOLD = 0.122610 

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "iiot/motor1/data"

# --- AI Model Definition (must match the training script) ---
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2), nn.ReLU(), nn.Linear(2, 1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, input_dim)
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- Load AI Model and Scaler ---
print("--- Loading AI Model and Scaler ---")
try:
    model = Autoencoder(input_dim=3)
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")
    
    scaler = joblib.load(SCALER_FILE)
    print("Scaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please ensure 'model.pth' and 'scaler.gz' are in the 'src' directory.")
    exit()

# --- Data Simulation ---
vibration_base = 3.5 

def get_sensor_data_and_predict():
    """Generates sensor data and uses the AI model to detect anomalies."""
    global vibration_base
    
    # 1. Simulate new sensor data
    rpm = 700 + random.uniform(-20, 20)
    temperature = 85.5 + random.uniform(-1.5, 1.5)
    vibration_base += 0.05 
    vibration = vibration_base + random.uniform(-0.2, 0.2)
    
    # 2. Perform AI Inference
    with torch.no_grad():
        # Prepare data for the model (must be in the same format as training)
        raw_data = np.array([[rpm, temperature, vibration]], dtype='float32')
        scaled_data = scaler.transform(raw_data)
        data_tensor = torch.from_numpy(scaled_data)
        
        # Get the model's reconstruction
        reconstruction = model(data_tensor)
        
        # Calculate the reconstruction error
        error = nn.MSELoss()(reconstruction, data_tensor).item()
        
        # Check if the error exceeds our threshold
        anomaly = 1 if error > ANOMALY_THRESHOLD else 0

    # 3. Package the data for publishing
    data = {
        "timestamp": time.time(),
        "rpm": round(rpm, 2),
        "temperature": round(temperature, 2),
        "vibration": round(vibration, 2),
        "reconstruction_error": round(error, 6), # Also send the error value
        "anomaly": anomaly
    }
    return data

# --- MQTT Connection Callback ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("\nSuccessfully connected to MQTT Broker!")
    else:
        print(f"\nFailed to connect, return code {rc}")

# --- Main Script ---
client = mqtt.Client()
client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
except Exception as e:
    print(f"Error connecting to MQTT: {e}")
    exit()

client.loop_start()
time.sleep(1)

print("--- Starting Real-Time Anomaly Detection ---")
try:
    while True:
        payload = get_sensor_data_and_predict()
        json_payload = json.dumps(payload)
        result = client.publish(MQTT_TOPIC, json_payload)
        
        if result[0] == 0:
            print(f"Published data: {json_payload}")
        else:
            print(f"Failed to publish message.")
        
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping publisher.")
    client.loop_stop()
    client.disconnect()

