import paho.mqtt.client as mqtt
import json
import time
from simple_pid import PID

# --- MQTT Configuration ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
# This service LISTENS to the AI's commands...
MQTT_AI_COMMAND_TOPIC = "iiot/motor1/command" 
# ...and the motor's current speed...
MQTT_SPEED_TOPIC = "iiot/motor1/speed"
# ...and PUBLISHES the new control voltage.
MQTT_VOLTAGE_TOPIC = "iiot/motor1/control/voltage"

# --- PID Controller Setup ---
# These Kp, Ki, Kd values are a starting point and may need tuning.
pid = PID(Kp=0.8, Ki=0.1, Kd=0.05, setpoint=1500)
pid.output_limits = (0, 100) # Output voltage can be between 0 and 100
pid.sample_time = 2.0 # Update every 2 seconds, matching the publisher's rate

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the broker."""
    if rc == 0:
        print("Controller successfully connected to MQTT Broker!")
        # Subscribe to both the AI command topic and the motor speed topic
        client.subscribe([(MQTT_AI_COMMAND_TOPIC, 0), (MQTT_SPEED_TOPIC, 0)])
        print(f"Controller subscribed to: {MQTT_AI_COMMAND_TOPIC}")
        print(f"Controller subscribed to: {MQTT_SPEED_TOPIC}")
    else:
        print(f"Controller failed to connect, return code {rc}")

def on_ai_command(client, userdata, msg):
    """
    Callback for when a command is received from the AI/subscriber service.
    This is the SUPERVISORY action.
    """
    try:
        payload_str = msg.payload.decode('utf-8')
        command = json.loads(payload_str)
        action = command.get("action")

        if action == "enter_safe_mode":
            print(f"\n>>> AI SUPERVISION: Command '{action}' received.")
            print(f">>> ACTION: Changing PID setpoint from {pid.setpoint} to 500 RPM.\n")
            # The AI's action is to change the PID controller's goal
            pid.setpoint = 500
        else:
            print(f"Unknown command received: {action}")

    except Exception as e:
        print(f"An error occurred in on_ai_command: {e}")

def on_speed_message(client, userdata, msg):
    """
    Callback for when a new speed reading is received from the motor.
    This is the main PID control loop.
    """
    try:
        payload_str = msg.payload.decode('utf-8')
        data = json.loads(payload_str)
        current_speed = data.get("speed")

        if current_speed is not None:
            # 1. Calculate the new control variable (voltage) based on the current speed
            control_voltage = pid(current_speed)
            
            # 2. Publish the new voltage for the motor to use
            voltage_payload = json.dumps({"voltage": control_voltage})
            client.publish(MQTT_VOLTAGE_TOPIC, voltage_payload)
            
            print(f"PID Loop: Speed={current_speed:.2f} RPM, Target={pid.setpoint} RPM -> Publishing Voltage={control_voltage:.2f}V")

    except Exception as e:
        print(f"An error occurred in on_speed_message: {e}")

# --- Main Script ---
print("--- Starting PID Motor Controller ---")
client = mqtt.Client()
client.on_connect = on_connect

# Map different topics to different callback functions
client.message_callback_add(MQTT_AI_COMMAND_TOPIC, on_ai_command)
client.message_callback_add(MQTT_SPEED_TOPIC, on_speed_message)

try:
    client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("Controller stopped by user.")
    client.disconnect()
except Exception as e:
    print(f"Connection error: {e}")
