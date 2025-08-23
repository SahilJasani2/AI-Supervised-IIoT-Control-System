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

        # Create an InfluxDB Point with all the new vibration fields
        point = Point("motor_data") \
            .tag("motor_id", "motor1") \
            .field("rpm", data.get("rpm")) \
            .field("temperature", data.get("temperature")) \
            .field("true_vibration", data.get("true_vibration")) \
            .field("noisy_vibration", data.get("noisy_vibration")) \
            .field("ekf_vibration", data.get("ekf_vibration")) \
            .field("reconstruction_error", data.get("reconstruction_error")) \
            .field("anomaly", data.get("anomaly")) \
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
