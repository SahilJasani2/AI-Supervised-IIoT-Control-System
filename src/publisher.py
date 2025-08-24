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
ANOMALY_THRESHOLD = 0.122610 

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
# This service PUBLISHES sensor data...
MQTT_DATA_TOPIC = "iiot/motor1/data"
# ...and the AI's commands...
MQTT_AI_COMMAND_TOPIC = "iiot/motor1/command"
# ...and the motor's current speed for the PID controller.
MQTT_SPEED_TOPIC = "iiot/motor1/speed"
# This service LISTENS to the PID's voltage commands.
MQTT_VOLTAGE_TOPIC = "iiot/motor1/control/voltage"

# --- Global variable to store the latest voltage from the controller ---
latest_voltage = 0.0

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

# --- Initialize the Motor Model ---
motor = Motor()

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("\nPublisher successfully connected to MQTT Broker!")
        # Subscribe to the voltage topic to receive commands from the PID controller
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
    
    # 1. Get the latest motor state, passing the controller's latest voltage
    motor_state = motor.update_state(voltage=latest_voltage)
    
    # Simulate other sensor data
    rpm = 700 + random.uniform(-20, 20) # This is decorative, the true speed is in motor_state
    temperature = 85.5 + random.uniform(-1.5, 1.5)
    
    # 2. Perform AI Inference on the CLEAN, ESTIMATED vibration data
    with torch.no_grad():
        raw_data = np.array([[rpm, temperature, motor_state["ekf_vibration"]]], dtype='float32')
        scaled_data = scaler.transform(raw_data)
        data_tensor = torch.from_numpy(scaled_data)
        reconstruction = model(data_tensor)
        error = nn.MSELoss()(reconstruction, data_tensor).item()
        anomaly = 1 if error > ANOMALY_THRESHOLD else 0

    # 3. Package all data for publishing
    data = {
        "timestamp": time.time(),
        "rpm": round(rpm, 2),
        "temperature": round(temperature, 2),
        "true_vibration": round(motor_state["true_vibration"], 4),
        "noisy_vibration": round(motor_state["noisy_vibration"], 4),
        "ekf_vibration": round(motor_state["ekf_vibration"], 4),
        "speed": round(motor_state["speed"], 2), # Add the new speed data
        "reconstruction_error": round(error, 6),
        "anomaly": anomaly
    }
    return data

# --- Main Script ---
client = mqtt.Client()
client.on_connect = on_connect
# Route messages from the voltage topic to our specific callback
client.message_callback_add(MQTT_VOLTAGE_TOPIC, on_voltage_message)

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
except Exception as e:
    print(f"Error connecting to MQTT: {e}")
    exit()

client.loop_start() # Start a background thread to handle MQTT messages
time.sleep(1)

print("--- Starting Real-Time Publisher with PID Control Loop ---")
try:
    while True:
        # Get the latest data packet
        payload = get_sensor_data_and_predict()
        
        # Publish the full sensor data for the dashboard
        client.publish(MQTT_DATA_TOPIC, json.dumps(payload))
        
        # Publish the speed data on its own topic for the PID controller
        speed_payload = json.dumps({"speed": payload["speed"]})
        client.publish(MQTT_SPEED_TOPIC, speed_payload)

        print(f"Published Speed={payload['speed']:.2f} RPM | Vibration={payload['ekf_vibration']:.2f}")

        # If AI detects an anomaly, publish the supervisory command
        if payload['anomaly'] == 1:
            command_payload = json.dumps({"action": "enter_safe_mode"})
            client.publish(MQTT_AI_COMMAND_TOPIC, command_payload)
            print(f"--> ANOMALY DETECTED! Published command: {command_payload}")
        
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopping publisher.")
    client.loop_stop()
    client.disconnect()
