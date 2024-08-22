import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the data
file_path = r'C:\Users\HP\Desktop\REVA\Capstone\Boeing 737 MAX\Capstone 2\Code\Implementation\LSTM\synthetic_sensor_data.csv'
data = pd.read_csv(file_path)

# Prepare data
time_steps = 1
X = data.iloc[:, :-1].values.reshape(-1, time_steps, 1)
y = data['anomaly'].values

# Truncate to the highest multiple of time_steps
#n_samples = (len(data) // time_steps) * time_steps
#truncated_data = data.iloc[:n_samples, :-1].values
#X = truncated_data.reshape(-1, time_steps, 1)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(-1, time_steps, 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=40, batch_size=32, validation_split=0.2)

# Save the model and scaler
model.save(r'C:\Users\HP\Desktop\REVA\Capstone\Boeing 737 MAX\Capstone 2\Code\Implementation\LSTM\anomaly_detection_model.h5')
joblib.dump(scaler, r'C:\Users\HP\Desktop\REVA\Capstone\Boeing 737 MAX\Capstone 2\Code\Implementation\LSTM\scaler.pkl')

print("Model and scaler saved successfully.")

# Plot the data with anomalies
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data.index, data.iloc[:, 0], label='Sensor Value')
plt.scatter(data[data['anomaly'] == 1].index, data[data['anomaly'] == 1].iloc[:, 0], color='r', label='Anomalies')
plt.xlabel('Index')
plt.ylabel('Sensor Value')
plt.title('Sensor Data with Anomalies')
plt.legend()
plt.show()
