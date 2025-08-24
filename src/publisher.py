import asyncio
import json
import paho.mqtt.client as mqtt
from asyncua import Client, ua
import torch
import torch.nn as nn
import joblib
import numpy as np
import time

# Import the new Motor class
from motor_model import Motor

# --- AI Model Configuration ---
MODEL_FILE = 'model.pth'
SCALER_FILE = 'scaler.gz'
ANOMALY_THRESHOLD = 0.134833

# --- MQTT Configuration (for AI commands only) ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_DATA_TOPIC = "iiot/motor1/data" # For Grafana dashboard
MQTT_AI_COMMAND_TOPIC = "iiot/motor1/command" # To send anomaly alerts

# --- OPC UA Client Configuration ---
OPCUA_SERVER_URL = "opc.tcp://controller:4840/freeopcua/server/"
OPCUA_NAMESPACE = "http://examples.freeopcua.github.io"

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
    model = Autoencoder(input_dim=2) 
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
    print("Model loaded successfully.")
    scaler = joblib.load(SCALER_FILE)
    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model files: {e}")
    exit()

# --- Initialize the Motor Model ---
motor = Motor()

async def main():
    print("--- Starting Real-Time Publisher with OPC UA Client ---")

    # --- MQTT Client Setup ---
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    mqtt_client.loop_start()
    print(f"MQTT client connected for publishing data to {MQTT_DATA_TOPIC}")

    # --- Main Loop with OPC UA Client ---
    url = OPCUA_SERVER_URL
    
    # --- FIXED: Add a retry loop to handle server startup timing ---
    client = Client(url=url)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            await client.connect()
            print(f"OPC UA Client connected to {url}")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed to connect to OPC UA server: {e}")
            if attempt + 1 == max_retries:
                print("Could not connect to OPC UA server. Exiting.")
                return
            await asyncio.sleep(5) # Wait 5 seconds before retrying

    try:
        nsidx = await client.get_namespace_index(OPCUA_NAMESPACE)
        speed_node = await client.nodes.root.get_child(["0:Objects", f"{nsidx}:Motor1", f"{nsidx}:CurrentSpeed"])
        voltage_node = await client.nodes.root.get_child(["0:Objects", f"{nsidx}:Motor1", f"{nsidx}:OutputVoltage"])

        while True:
            latest_voltage = await voltage_node.get_value()
            motor_state = motor.update_state(voltage=latest_voltage)
            # --- FIXED: Ensure the value written is always a float ---
            await speed_node.write_value(float(motor_state["speed"]))

            with torch.no_grad():
                raw_data = np.array([[motor_state["ekf_vibration"], motor_state["ekf_temperature"]]], dtype='float32')
                scaled_data = scaler.transform(raw_data)
                data_tensor = torch.from_numpy(scaled_data)
                reconstruction = model(data_tensor)
                error = nn.MSELoss()(reconstruction, data_tensor).item()
                anomaly = 1 if error > ANOMALY_THRESHOLD else 0
            
            payload = {
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
            mqtt_client.publish(MQTT_DATA_TOPIC, json.dumps(payload))
            
            print(f"OPC Loop: Read Voltage={latest_voltage:.2f}, Wrote Speed={payload['speed']:.2f} | AI Error={payload['reconstruction_error']:.4f}")

            if payload['anomaly'] == 1:
                command_payload = json.dumps({"action": "enter_safe_mode"})
                mqtt_client.publish(MQTT_AI_COMMAND_TOPIC, command_payload)
                print(f"--> ANOMALY DETECTED! Published MQTT command: {command_payload}")
            
            await asyncio.sleep(2)
    finally:
        await client.disconnect()
        print("OPC UA Client disconnected.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Publisher stopped by user.")
    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
