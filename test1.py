import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Model definition (must match training model)
# ------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, (h_n, c_n) = self.encoder(x)
        code = torch.tanh(self.enc_fc(enc_out[:, -1, :]))  # latent vector from last timestep
        dec_in = torch.relu(self.dec_fc(code)).unsqueeze(1)  # expand dims for sequence
        dec_in_seq = dec_in.repeat(1, x.size(1), 1)  # repeat for each timestep
        dec_out, _ = self.decoder(dec_in_seq)
        return dec_out

# ------------------------------
# Parameters
# ------------------------------
sequence_length = 16  # must be same as training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load preprocessing and model objects
# ------------------------------
print("Loading saved model and preprocessing objects...")
vehicle_state_encoder = joblib.load("anomaly_vehicle_state_encoder.pkl")
charge_state_encoder = joblib.load("anomaly_charge_state_encoder.pkl")
scaler = joblib.load("anomaly_scaler.pkl")
thresholds = joblib.load("anomaly_thresholds.pkl")

# ------------------------------
# Load unseen data
# ------------------------------
file_path = "C:/Users/mdkhe/OneDrive/Desktop/lstm/car_data3/2022-04-17.xlsx"  # Change to your file path
df = pd.read_excel(file_path)

# Normalize columns names: lowercase, no spaces, remove (), %, ℃ replaced as words
df.columns = [c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'percent').replace('℃', 'degc') for c in df.columns]

print("Columns in the Excel file:", df.columns.tolist())

# ------------------------------
# Preprocess new data
# ------------------------------
print("Preprocessing data...")

# Rename columns to match training feature names
df.rename(columns={
    'pack_voltagev': 'pack_voltage(v)',
    'pack_currenta': 'pack_current(a)',
    'socpercent': 'soc(%)',
    'max_cell_voltage_v': 'max_cell_voltage_(v)',
    'min_cell_voltage_v': 'min_cell_voltage_(v)',
    'max_probe_temperature_degc': 'max_probe_temperature_(℃)',
    'min_probe_temperature_degc': 'min_probe_temperature_(℃)'
}, inplace=True)

# Translation dictionary for categoricals (Chinese → English)
translate = {
    "车辆启动": "started",
    "熄火": "stopped",
    "未充电": "not charging",
    "停车充电": "charging while parked",
    "充电完成熄火": "charge done + stopped"
}

if 'vehicle_state' in df.columns:
    df['vehicle_state'] = df['vehicle_state'].astype(str).map(translate).fillna(df['vehicle_state'])

if 'charge_state' in df.columns:
    df['charge_state'] = df['charge_state'].astype(str).map(translate).fillna(df['charge_state'])

# Encode categorical features
df['vehicle_state'] = vehicle_state_encoder.transform(df['vehicle_state'])
df['charge_state'] = charge_state_encoder.transform(df['charge_state'])

# Features in correct order for scaler and model input
FEATURES = [
    'vehicle_state', 'charge_state',
    'pack_voltage(v)', 'pack_current(a)', 'soc(%)',
    'max_cell_voltage_(v)', 'min_cell_voltage_(v)',
    'max_probe_temperature_(℃)', 'min_probe_temperature_(℃)'
]

# Verify all required columns exist
missing_cols = [col for col in FEATURES if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in input data: {missing_cols}")

df_features = df[FEATURES]

# Scale features
scaled_features = scaler.transform(df_features)

# Create sequences function
def create_sequences(data, seq_len):
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)

sequences = create_sequences(scaled_features, sequence_length)
sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)

# ------------------------------
# Load model and run inference
# ------------------------------
input_dim = sequences.shape[2]
model = LSTMAutoencoder(input_dim=input_dim).to(device)
model.load_state_dict(torch.load("best_anomaly_autoencoder.pt", map_location=device))
model.eval()

print("Running anomaly detection...")
with torch.no_grad():
    reconstructed = model(sequences_tensor)
    mse = torch.mean((sequences_tensor - reconstructed) ** 2, dim=(1, 2)).cpu().numpy()

# Flag anomalies by threshold
threshold = thresholds["mean_plus_3std"]
anomalies = mse > threshold

# ------------------------------
# Map anomalies to timestamps (last timestamp in sequence)
# ------------------------------
sequence_end_indices = [i + sequence_length - 1 for i in range(len(anomalies))]
timestamps = df['record_time'].iloc[sequence_end_indices].reset_index(drop=True)

# Results dataframe for anomalies only
import pandas as pd

results_df = pd.DataFrame({
    'sequence_end_index': sequence_end_indices,
    'timestamp': timestamps,
    'mse': mse,
    'anomaly': anomalies
})

print("\nSample results (first 10):")
print(results_df.head(10))

print(f"\nDetection complete. {np.sum(anomalies)} anomalies detected out of {len(anomalies)} sequences.")

# ------------------------------
# Plot reconstruction error & anomalies over time
# ------------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(15,5))
plt.plot(timestamps, mse, label='Reconstruction MSE')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(timestamps[anomalies], mse[anomalies], color='red', label='Anomalies')
plt.xlabel('Time')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Anomaly Detection Reconstruction Error Over Time')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------
# Save anomaly results CSV (optional)
# ------------------------------
results_df.to_csv("anomaly_detection_results.csv", index=False)
print("Anomaly detection results saved to anomaly_detection_results.csv")
