import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_sensor_data(total_length, normal_range, anomaly_range, anomaly_fraction):
    normal_data_length = int(total_length * (1 - anomaly_fraction))
    anomaly_data_length = total_length - normal_data_length

    normal_data = np.random.uniform(normal_range[0], normal_range[1], normal_data_length)
    anomaly_data_low = np.random.uniform(anomaly_range[0][0], anomaly_range[0][1], anomaly_data_length // 2)
    anomaly_data_high = np.random.uniform(anomaly_range[1][0], anomaly_range[1][1], anomaly_data_length // 2)
    
    anomaly_data = np.concatenate((anomaly_data_low, anomaly_data_high))

    data = np.concatenate((normal_data, anomaly_data))
    labels = np.concatenate((np.zeros(normal_data_length), np.ones(anomaly_data_length)))

    indices = np.arange(total_length)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    return data, labels

# Parameters
total_length = 50000
ldr_normal_range = (201, 700)
ldr_anomaly_range = [(0, 200), (701, 880)]
anomaly_fraction = 0.5

# Generate LDR data
ldr_data, ldr_labels = generate_sensor_data(total_length, ldr_normal_range, ldr_anomaly_range, anomaly_fraction)


# Save to CSV for use in training
df_ldr = pd.DataFrame({'sensor_value': ldr_data, 'anomaly': ldr_labels})
df_ldr.to_csv(r'C:\Users\HP\Desktop\REVA\Capstone\Boeing 737 MAX\Capstone 2\Code\Implementation\LSTM\synthetic_sensor_data.csv', index=False)

# Plot the generated data
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(ldr_data, label='LDR Data')
plt.scatter(np.where(ldr_labels == 1), ldr_data[ldr_labels == 1], color='r', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('LDR Value')
plt.title('Synthetic LDR Sensor Data with Anomalies')
plt.legend()

plt.tight_layout()
plt.show()
