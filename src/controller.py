import paho.mqtt.client as mqtt
import json
import time

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_COMMAND_TOPIC = "iiot/motor1/command"

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker."""
    if rc == 0:
        print("Controller successfully connected to MQTT Broker!")
        client.subscribe(MQTT_COMMAND_TOPIC)
        print(f"Controller subscribed to command topic: {MQTT_COMMAND_TOPIC}")
    else:
        print(f"Controller failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """
    Callback for when a message is received. This simulates taking a control action.
    """
    try:
        payload_str = msg.payload.decode('utf-8')
        command = json.loads(payload_str)
        action = command.get("action")

        if action == "enter_safe_mode":
            # In a real system, this is where you would interface with a PLC.
            # For example, using OPC UA to write to a specific tag on a Siemens S7-1500.
            print(f"\n>>> CONTROL ACTION: Command '{action}' received.")
            print(">>> SIMULATING: Sending signal to PLC to reduce motor speed by 20%.\n")
        else:
            print(f"Unknown command received: {action}")

    except json.JSONDecodeError:
        print(f"Error decoding JSON command: {msg.payload.decode()}")
    except Exception as e:
        print(f"An error occurred in on_message: {e}")

# --- Main Script ---
print("--- Starting Simulated Motor Controller ---")
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    # loop_forever() is a blocking call that handles reconnection
    client.loop_forever()
except KeyboardInterrupt:
    print("Controller stopped by user.")
    client.disconnect()
except Exception as e:
    print(f"Connection error: {e}")
