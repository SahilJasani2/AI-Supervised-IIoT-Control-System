import paho.mqtt.client as mqtt
import time
import json
import random

# --- Configuration ---
MQTT_BROKER_HOST = "localhost"  # Use 'mosquitto' when running in Docker network
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
        error_messages = {
            1: "Incorrect protocol version",
            2: "Invalid client identifier",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        print(f"Failed to connect: {error_messages.get(rc, f'Unknown error code: {rc}')}")

def on_disconnect(client, userdata, rc):
    if rc != 0:
        print(f"Unexpected disconnection: {rc}")
    else:
        print("Disconnected from MQTT Broker")

# --- Main Script ---
client = mqtt.Client()
client.on_connect = on_connect
client.on_disconnect = on_disconnect

# Enable debug messages
client.enable_logger()

try:
    print(f"Attempting to connect to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}...")
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.loop_start()
    # Wait for connection to complete
    for _ in range(5):
        if client.is_connected():
            break
        time.sleep(0.5)
    else:
        print("Connection timeout. Check if MQTT broker is running and accessible.")
        exit(1)
        
except Exception as e:
    print(f"Error connecting to MQTT broker: {str(e)}")
    print("Please check:")
    print("1. Is the MQTT broker running?")
    print("2. Is the broker accessible at the specified host and port?")
    print("3. Are there any firewall rules blocking the connection?")
    exit(1)

print("Starting data publication...")
try:
    while True:
        payload = get_sensor_data()
        json_payload = json.dumps(payload)
        try:
            result = client.publish(MQTT_TOPIC, json_payload, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published data: {json_payload}")
            else:
                print(f"Failed to publish message. Error code: {result.rc}")
                
        except Exception as e:
            print(f"Error publishing message: {str(e)}")
            if not client.is_connected():
                print("Not connected to MQTT broker. Attempting to reconnect...")
                try:
                    client.reconnect()
                except Exception as reconnect_error:
                    print(f"Reconnection failed: {str(reconnect_error)}")
        
        time.sleep(2)

except KeyboardInterrupt:
    print("Publication stopped by user.")
    client.loop_stop()
    client.disconnect()