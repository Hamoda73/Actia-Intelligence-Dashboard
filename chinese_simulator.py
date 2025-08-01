import pandas as pd
import time
import requests
from datetime import datetime
from azure.iot.device import IoTHubDeviceClient, Message
import json


# Config
CONFIG = {
    "connection_string":"",  
    "vehicle_id": "ev-vehicle-001",
    "transmission_interval": 5  
}

# Translation dictionaries  
vehicle_state_translation = {
    "车辆启动": "vehicle started",
    "熄火": "vehicle stopped",
    
}

charge_state_translation = {
    "未充电": "not charging",
    "停车充电": "charging while parked",
    
}

# Load the dataset
df = pd.read_excel("C:/Users/mdkhe/OneDrive/Desktop/actia/Dashboard/chinesedata.xlsx")

# Create Azure IoT Client
client = IoTHubDeviceClient.create_from_connection_string(CONFIG["connection_string"])

# Connect the client
client.connect()

# Loop through all rows
for idx, row in df.iterrows():
    try:
        # Parse timestamp safely, iso format or smth
        try:
            timestamp = pd.to_datetime(row["record_time"]).isoformat()
        except Exception:
            timestamp = datetime.now().isoformat()

        # Prepare payload with renamed keys
        payload = {
            "timestamp": timestamp,
            "deviceId": CONFIG["vehicle_id"],
            "sensorId": "battery-sensor-001",
            "type": "Battery",
            
            # Mapping columns directly:
            "vehicleState": str(row["vehicle_state"]),
            "chargeState": str(row["charge_state"]),
            "batteryVoltage": round(row["pack_voltage(V)"], 2),
            "motorCurrent": round(row["pack_current(A)"], 2),
            "stateOfCharge": round(row["SOC(%)"], 1),
            "maxCellVoltage": round(row["max_cell_voltage (V)"], 2),
            "minCellVoltage": round(row["min_cell_voltage (V)"], 2),
            "maxTemperature": round(row["max_probe_temperature (℃)"], 1),
            "minTemperature": round(row["min_probe_temperature (℃)"], 1),

            "metadata": {
                "source": "real-ev-xlsx-data",
                "row_index": idx
            }
        }

        # Translate Chinese states to English
        payload['vehicleState'] = vehicle_state_translation.get(payload['vehicleState'], payload['vehicleState'])
        payload['chargeState'] = charge_state_translation.get(payload['chargeState'], payload['chargeState'])

        print(f"\n[{timestamp}] Payload:")
        print(payload)

        # Create message from payload
       #  message = Message(str(payload))
        
        message = Message(json.dumps(payload))
        message.content_type = "application/json"
        message.content_encoding = "utf-8"

        
        # Send message to IoT Hub
        client.send_message(message)
        print("Message successfully sent to IoT Hub")
        
        

        time.sleep(CONFIG["transmission_interval"])

    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue
