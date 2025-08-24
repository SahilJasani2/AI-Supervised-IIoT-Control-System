import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

# --- InfluxDB Configuration ---
INFLUX_URL = "http://influxdb:8086"
INFLUX_TOKEN = "my-super-secret-auth-token"
INFLUX_ORG = "iiot"
INFLUX_BUCKET = "iiot-bucket"

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_DATA_TOPIC = "iiot/motor1/data"

# --- InfluxDB Client Setup ---
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Subscriber successfully connected to MQTT Broker!")
        client.subscribe(MQTT_DATA_TOPIC)
        print(f"Subscriber listening to topic: {MQTT_DATA_TOPIC}")
    else:
        print(f"Subscriber failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Processes messages and writes all new data fields to InfluxDB."""
    try:
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        
        print(f"Received message: {data}")

        # --- FIXED: Explicitly cast all numeric fields to float to prevent type conflicts ---
        point = Point("motor_data") \
            .tag("motor_id", "motor1") \
            .field("true_vibration", float(data.get("true_vibration", 0.0))) \
            .field("noisy_vibration", float(data.get("noisy_vibration", 0.0))) \
            .field("ekf_vibration", float(data.get("ekf_vibration", 0.0))) \
            .field("true_temperature", float(data.get("true_temperature", 0.0))) \
            .field("noisy_temperature", float(data.get("noisy_temperature", 0.0))) \
            .field("ekf_temperature", float(data.get("ekf_temperature", 0.0))) \
            .field("speed", float(data.get("speed", 0.0))) \
            .field("setpoint", float(data.get("setpoint", 0.0))) \
            .field("reconstruction_error", float(data.get("reconstruction_error", 0.0))) \
            .field("anomaly", int(data.get("anomaly", 0))) \
            .time(int(data.get("timestamp") * 1_000_000_000))

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print("Successfully wrote data to InfluxDB.")

    except json.JSONDecodeError:
        print(f"Error decoding JSON: {msg.payload.decode()}")
    except Exception as e:
        print(f"An error occurred in on_message: {e}")

# --- Main Script ---
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

try:
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    mqtt_client.loop_forever()
except KeyboardInterrupt:
    print("Subscriber stopped by user.")
    mqtt_client.disconnect()
except Exception as e:
    print(f"Connection error: {e}")
