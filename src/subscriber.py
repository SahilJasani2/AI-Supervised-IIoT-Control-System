import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

# --- InfluxDB Configuration ---
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "my-super-secret-auth-token"  # The token from your docker-compose.yml
INFLUX_ORG = "iiot"
INFLUX_BUCKET = "iiot-bucket"

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "localhost" # We are running this script on our machine, not in a container
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "iiot/motor1/data"

# --- InfluxDB Client Setup ---
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
        print(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Processes messages received from the MQTT broker."""
    try:
        # Decode the message payload from bytes to a string
        payload_str = msg.payload.decode('utf-8')
        # Parse the JSON string into a Python dictionary
        data = json.loads(payload_str)

        print(f"Received message: {data}")

        # Create an InfluxDB Point
        point = Point("motor_data") \
            .tag("motor_id", "motor1") \
            .field("rpm", data.get("rpm")) \
            .field("temperature", data.get("temperature")) \
            .field("vibration", data.get("vibration")) \
            .time(int(data.get("timestamp") * 1_000_000_000)) # InfluxDB needs nanoseconds

        # Write the point to InfluxDB
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        print("Successfully wrote data to InfluxDB.")

    except json.JSONDecodeError:
        print(f"Error decoding JSON: {msg.payload.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main Script ---
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

try:
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    # loop_forever() is a blocking call that handles reconnection automatically
    mqtt_client.loop_forever()
except KeyboardInterrupt:
    print("Subscriber stopped by user.")
    mqtt_client.disconnect()
except Exception as e:
    print(f"Connection error: {e}")