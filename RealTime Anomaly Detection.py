import joblib
import serial
import time
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model(r'C:\Users\HP\Desktop\REVA\Capstone\Boeing 737 MAX\Capstone 2\Code\Implementation\LSTM\anomaly_detection_model.h5')
scaler = joblib.load(r'C:\Users\HP\Desktop\REVA\Capstone\Boeing 737 MAX\Capstone 2\Code\Implementation\LSTM\scaler.pkl')

# Prepare for real-time prediction
time_steps = 1
buffer = []

def extract_sensor_value(line):
    try:
        if line.startswith("Sensor Value: "):
            parts = line.split('\t')
            sensor_value_part = parts[0]
            sensor_value_str = sensor_value_part.split(': ')[1]
            sensor_value = float(sensor_value_str)
            return sensor_value
        else:
            raise ValueError("Line format is incorrect")
    except (IndexError, ValueError) as e:
        print(f"Error extracting sensor value: {e}")
        return None

# Set up the serial line
ser = serial.Serial('COM3', 9600)
time.sleep(2)

# Real-time anomaly detection
try:
    while True:
        #startTime = int(time.time() * 1_000_000)
        line = ser.readline().decode('utf-8').strip()
        #endTime = int(time.time() * 1_000_000)
        sensor_value = extract_sensor_value(line)
        if sensor_value is not None:
            buffer.append(sensor_value)
            if len(buffer) > time_steps:
                buffer.pop(0)
            
            if len(buffer) == time_steps:
                data_seq = np.array(buffer).reshape(1, time_steps, 1)
                data_seq_normalized = scaler.transform(data_seq.reshape(-1, 1)).reshape(1, time_steps, 1)
                prediction = model.predict(data_seq_normalized)
                #endTime = int(time.time() * 1_000_000)
                #elapsedTime = endTime - startTime
                #print(f"Elapsed time: {elapsedTime} microseconds")
            
                if prediction >= 0.5:
                    print(f'Anomaly detected: {buffer[-1]}')
                else:
                    print(f'Normal: {buffer[-1]}')
                    
        
except KeyboardInterrupt:
    pass
finally:
    ser.close()
