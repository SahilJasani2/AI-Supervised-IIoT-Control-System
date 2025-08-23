import paho.mqtt.client as mqtt
import time
import json
import random

# --- Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "iiot/motor1/data"

# --- Data Simulation ---
vibration_base = 3.5 

def get_sensor_data():
    global vibration_base
    rpm = 700 + random.uniform(-20, 20)
    temperature = 85.5 + random.uniform(-1.5, 1.5)
    vibration_base += 0.05 
    vibration = vibration_base + random.uniform(-0.2, 0.2)
    timestamp = time.time()
    data = {
        "timestamp": timestamp,
        "rpm": round(rpm, 2),
        "temperature": round(temperature, 2),
        "vibration": round(vibration, 2)
    }
    return data

# --- MQTT Connection Callback (Corrected for paho-mqtt v1.6.1) ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

# --- Main Script ---
client = mqtt.Client()
client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
except ConnectionRefusedError:
    print("Connection refused. Is the MQTT broker running?")
    exit()

client.loop_start()
time.sleep(1)

print("Starting data publication...")
try:
    while True:
        payload = get_sensor_data()
        json_payload = json.dumps(payload)
        result = client.publish(MQTT_TOPIC, json_payload)
        
        if result[0] == 0:
            print(f"Published data: {json_payload}")
        else:
            print(f"Failed to send message to topic {MQTT_TOPIC}")
        
        time.sleep(2)

except KeyboardInterrupt:
    print("Publication stopped by user.")
    client.loop_stop()
    client.disconnect()