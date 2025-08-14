# anomaly_lstm_autoencoder.py
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Config
# =========================
folder_path = "./car_data1/"
sequence_length = 16
test_size = 0.2
batch_size = 64
epochs = 20
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Columns (exact names you gave)
FEATURES = [
    'vehicle_state', 'charge_state',
    'pack_voltage(v)', 'pack_current(a)', 'soc(%)',
    'max_cell_voltage_(v)', 'min_cell_voltage_(v)',
    'max_probe_temperature_(℃)', 'min_probe_temperature_(℃)'
]

# =========================
# 1. Load & preprocess
# =========================
print("🔍 Loading data...")
translate = {
    "车辆启动": "started", "熄火": "stopped",
    "未充电": "not charging", "停车充电": "charging while parked",
    "充电完成熄火": "charge done + stopped"
}

all_dfs = []
for f in tqdm(os.listdir(folder_path), desc="files"):
    if f.endswith(".xlsx"):
        try:
            df = pd.read_excel(os.path.join(folder_path, f), engine="openpyxl")
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            # unify names to match FEATURES (lowercase, replace)
            # map the translations if columns exist
            if 'vehicle_state' in df.columns:
                df['vehicle_state'] = df['vehicle_state'].astype(str).map(translate).fillna(df['vehicle_state'])
            if 'charge_state' in df.columns:
                df['charge_state'] = df['charge_state'].astype(str).map(translate).fillna(df['charge_state'])
            if 'record_time' in df.columns:
                df['record_time'] = pd.to_datetime(df['record_time'], errors='coerce')
                df = df.sort_values('record_time')
            df = df.dropna(subset=['soc(%)'])  # ensure SoC present
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

if not all_dfs:
    raise RuntimeError("No data found in folder_path")

df = pd.concat(all_dfs, ignore_index=True)

# Keep only rows that have the columns we need
# If column names in file slightly differ, ensure mapping — here we assume they match provided names
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns in data: {missing}")

# Encode categoricals (vehicle_state & charge_state)
v_encoder = LabelEncoder()
c_encoder = LabelEncoder()
df['vehicle_state'] = v_encoder.fit_transform(df['vehicle_state'].astype(str))
df['charge_state'] = c_encoder.fit_transform(df['charge_state'].astype(str))

joblib.dump(v_encoder, "anomaly_vehicle_state_encoder.pkl")
joblib.dump(c_encoder, "anomaly_charge_state_encoder.pkl")

# Feature scaling
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])
joblib.dump(scaler, "anomaly_scaler.pkl")

# =========================
# 2. Sequence creation
# =========================
print("⏳ Creating sequences...")
sequences = []
for i in tqdm(range(len(df) - sequence_length + 1), desc="seq"):
    seq = df[FEATURES].iloc[i:i+sequence_length].values  # shape (seq_len, n_features)
    sequences.append(seq)

X = np.array(sequences)  # (n_samples, seq_len, n_features)
print("X shape:", X.shape)

# Train/val split (no shuffling - timeseries)
train_X, val_X = train_test_split(X, test_size=test_size, shuffle=False)

# =========================
# 3. Dataset & DataLoader
# =========================
class SeqDataset(Dataset):
    def __init__(self, arrays):
        self.x = torch.tensor(arrays, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx]

train_loader = DataLoader(SeqDataset(train_X), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SeqDataset(val_X), batch_size=batch_size, shuffle=False)

# =========================
# 4. Model: LSTM Autoencoder
# =========================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)
        # decoder LSTM returns sequences; we expect output dim = input_dim
    def forward(self, x):
        # x: (B, T, input_dim)
        enc_out, (h_n, c_n) = self.encoder(x)  # enc_out: (B, T, hidden_dim)
        # take last timestep hidden (enc_out[:, -1, :]) or h_n[-1]
        code = torch.tanh(self.enc_fc(enc_out[:, -1, :]))  # (B, latent_dim)
        dec_in = torch.relu(self.dec_fc(code)).unsqueeze(1)  # (B, 1, hidden_dim)
        # we need to expand dec_in to seq length as decoder input
        dec_in_seq = dec_in.repeat(1, x.size(1), 1)  # (B, T, hidden_dim)
        dec_out, _ = self.decoder(dec_in_seq)  # (B, T, input_dim)
        return dec_out  # reconstructed sequence

input_dim = X.shape[2]
model = LSTMAutoencoder(input_dim=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss(reduction='none')  # we'll compute per-feature errors if needed

# =========================
# 5. Training loop
# =========================
print("⚙️  Training anomaly autoencoder...")
best_val_loss = float('inf')
for epoch in range(1, epochs+1):
    model.train()
    train_losses = []
    for xb in train_loader:
        xb = xb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, xb).mean()  # MSE per batch
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            loss = criterion(out, xb).mean()
            val_losses.append(loss.item())
    avg_train = np.mean(train_losses)
    avg_val = np.mean(val_losses)
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "best_anomaly_autoencoder.pt")
    print(f"Epoch {epoch}/{epochs} | train_loss: {avg_train:.6f} | val_loss: {avg_val:.6f}")

torch.save(model.state_dict(), "final_anomaly_autoencoder.pt")
print("✅ Training finished. Best val loss:", best_val_loss)

# =========================
# 6. Compute reconstruction errors on validation to set threshold
# =========================
print("🔎 Computing reconstruction errors on validation set...")
model.eval()
errors = []
with torch.no_grad():
    for xb in val_loader:
        xb = xb.to(device)
        out = model(xb)
        # per-sequence MSE
        seq_mse = ((out - xb)**2).mean(dim=(1,2)).cpu().numpy()  # shape (batch,)
        errors.extend(seq_mse.tolist())

errors = np.array(errors)
# Suggested thresholds:
th_mean3std = errors.mean() + 3 * errors.std()
th_99 = np.percentile(errors, 99)  # 99th percentile
th_95 = np.percentile(errors, 95)

print(f"Validation error mean: {errors.mean():.6e}, std: {errors.std():.6e}")
print(f"Suggested thresholds -> mean+3std: {th_mean3std:.6e}, 95%: {th_95:.6e}, 99%: {th_99:.6e}")

# Save threshold(s)
thresholds = {"mean_plus_3std": float(th_mean3std), "p95": float(th_95), "p99": float(th_99)}
joblib.dump(thresholds, "anomaly_thresholds.pkl")
print("Saved thresholds to anomaly_thresholds.pkl")

# =========================
# 7. Inference helper (example)
# =========================
def score_sequence(seq_array):
    """
    seq_array: numpy array shape (seq_len, n_features) already preprocessed (scaled & encoded)
    returns: reconstruction_error (float)
    """
    model.eval()
    x = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,F)
    with torch.no_grad():
        out = model(x)
    mse = ((out - x)**2).mean().item()
    return mse

# small example of using score_sequence on the last sequence from dataset
example_seq = X[-1]  # already scaled
err = score_sequence(example_seq)
print(f"Example sequence reconstruction MSE: {err:.6e}")
print("If MSE > chosen threshold -> flagged as anomaly.")
