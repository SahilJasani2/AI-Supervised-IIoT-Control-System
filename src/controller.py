import asyncio
import json
import paho.mqtt.client as mqtt
from asyncua import Server, ua
from simple_pid import PID

# --- MQTT Configuration (for AI commands only) ---
MQTT_BROKER_HOST = "mosquitto"
MQTT_BROKER_PORT = 1883
MQTT_AI_COMMAND_TOPIC = "iiot/motor1/command"

# --- OPC UA Server Configuration ---
OPCUA_SERVER_URL = "opc.tcp://0.0.0.0:4840/freeopcua/server/"
OPCUA_NAMESPACE = "http://examples.freeopcua.github.io"

# --- PID Controller Setup ---
pid = PID(Kp=0.8, Ki=0.1, Kd=0.05, setpoint=1500.0) # Use float for setpoint
pid.output_limits = (0.0, 100.0) # Use floats for limits
pid.sample_time = 2.0

# --- Global Server object and Nodes ---
opcua_server = Server()
opcua_idx = 0
opcua_current_speed_node = None
opcua_setpoint_node = None
opcua_voltage_node = None

def on_ai_command(client, userdata, msg):
    """
    MQTT Callback for when a command is received from the AI.
    """
    try:
        payload_str = msg.payload.decode('utf-8')
        command = json.loads(payload_str)
        action = command.get("action")

        if action == "enter_safe_mode":
            print("\n>>> AI SUPERVISION (MQTT): Command 'enter_safe_mode' received.")
            print(">>> ACTION: Writing new setpoint 500.0 to OPC UA server.\n")
            # Ensure the value written is a float
            asyncio.run_coroutine_threadsafe(
                opcua_setpoint_node.write_value(500.0),
                loop=asyncio.get_event_loop()
            )
    except Exception as e:
        print(f"An error occurred in on_ai_command: {e}")

async def main():
    global opcua_server, opcua_idx, opcua_current_speed_node, opcua_setpoint_node, opcua_voltage_node

    print("--- Starting OPC UA Server and PID Controller ---")

    # --- MQTT Client Setup ---
    mqtt_client = mqtt.Client()
    mqtt_client.on_message = on_ai_command
    mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    mqtt_client.subscribe(MQTT_AI_COMMAND_TOPIC)
    mqtt_client.loop_start()
    print(f"MQTT client connected and listening on {MQTT_AI_COMMAND_TOPIC}")

    # --- OPC UA Server Initialization ---
    await opcua_server.init()
    opcua_server.set_endpoint(OPCUA_SERVER_URL)
    opcua_idx = await opcua_server.register_namespace(OPCUA_NAMESPACE)
    
    objects = opcua_server.get_objects_node()
    motor_obj = await objects.add_object(opcua_idx, "Motor1")
    
    # Initialize all nodes with float values (e.g., 0.0)
    opcua_current_speed_node = await motor_obj.add_variable(opcua_idx, "CurrentSpeed", 0.0)
    opcua_setpoint_node = await motor_obj.add_variable(opcua_idx, "SpeedSetpoint", 1500.0)
    opcua_voltage_node = await motor_obj.add_variable(opcua_idx, "OutputVoltage", 0.0)
    
    await opcua_current_speed_node.set_writable()
    await opcua_setpoint_node.set_writable()

    print(f"OPC UA Server initialized at {OPCUA_SERVER_URL}")
    print("OPC UA Nodes created: CurrentSpeed, SpeedSetpoint, OutputVoltage")

    # --- Main Control Loop ---
    async with opcua_server:
        while True:
            current_speed = await opcua_current_speed_node.get_value()
            current_setpoint = await opcua_setpoint_node.get_value()
            
            pid.setpoint = current_setpoint
            
            control_voltage = pid(current_speed)
            
            # --- FIXED: Ensure the value written to the node is always a float ---
            await opcua_voltage_node.write_value(float(control_voltage))
            
            print(f"PID Loop: Speed={current_speed:.2f}, Target={pid.setpoint:.2f} -> Voltage={control_voltage:.2f}")

            await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user.")
