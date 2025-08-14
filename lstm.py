import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib
from tqdm import tqdm

# ======================
# 1. Environment Checks
# ======================
print("🚀 Initializing...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
torch.cuda.empty_cache()

# ======================
# 2. Configuration
# ======================
folder_path = "./car_data1/"
sequence_length = 16  # Reduced from original if needed
prediction_shift = 4
test_size = 0.2
batch_size = 64
epochs = 10  # Reduced for testing

# ======================
# 3. Data Loading
# ======================
print("\n🔍 Loading data...")
translate = {
    "车辆启动": "started", "熄火": "stopped",
    "未充电": "not charging", "停车充电": "charging while parked",
    "充电完成熄火": "charge done + stopped"
}

all_data = []
for file in tqdm(os.listdir(folder_path), desc="Processing files"):
    if file.endswith(".xlsx"):
        try:
            df = pd.read_excel(os.path.join(folder_path, file), engine="openpyxl")
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            for col in ['vehicle_state', 'charge_state']:
                if col in df.columns:
                    df[col] = df[col].astype(str).map(translate)
            df['record_time'] = pd.to_datetime(df['record_time'], errors='coerce')
            df = df.sort_values('record_time').dropna()
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if not all_data:
    raise ValueError("No valid data files found!")

# ======================
# 4. Data Processing
# ======================
print("\n🧠 Processing data...")
df = pd.concat(all_data, ignore_index=True)
df = df.dropna(subset=['soc(%)'])
print(f"Total raw data points: {len(df)}")

# Encode categorical
v_encoder = LabelEncoder()
c_encoder = LabelEncoder()
df['vehicle_state'] = v_encoder.fit_transform(df['vehicle_state'])
df['charge_state'] = c_encoder.fit_transform(df['charge_state'])

# Save encoders
joblib.dump(v_encoder, "vehicle_state_encoder.pkl")
joblib.dump(c_encoder, "charge_state_encoder.pkl")

# Features and normalization
features = [
    'vehicle_state', 'charge_state',
    'pack_voltage(v)', 'pack_current(a)', 'soc(%)',
    'max_cell_voltage_(v)', 'min_cell_voltage_(v)',
    'max_probe_temperature_(℃)', 'min_probe_temperature_(℃)'
]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, "scaler.pkl")

# ======================
# 5. Sequence Creation
# ======================
print("\n⏳ Creating sequences...")
sequences = []
targets = []
for i in tqdm(range(len(df) - sequence_length - prediction_shift), desc="Generating sequences"):
    seq = df[features].iloc[i:i+sequence_length].values
    target = df['soc(%)'].iloc[i+sequence_length+prediction_shift-1]
    sequences.append(seq)
    targets.append(target)

X = np.array(sequences)
y = np.array(targets)
print(f"Final dataset shape - X: {X.shape}, y: {y.shape}")

# ======================
# 6. Dataset & Dataloaders
# ======================
class EVSoCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, shuffle=False)
train_dataset = EVSoCDataset(X_train, y_train)
val_dataset = EVSoCDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ======================
# 7. Model Definition
# ======================
class LSTMSoC(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

model = LSTMSoC(input_size=X.shape[2]).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ======================
# 8. Training Loop
# ======================
print("\n Starting training...")
best_val_loss = float('inf')

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += loss_fn(preds, yb).item()
    
    # Metrics
    avg_train_loss = train_loss/len(train_loader)
    avg_val_loss = val_loss/len(val_loader)
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_soc_lstm_model.pt")
    
    print(f"\nEpoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ======================
# 9. Final Save & Report
# ======================
torch.save(model.state_dict(), "final_soc_lstm_model.pt")
print("\n✅ Training complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Models saved to: best_soc_lstm_model.pt and final_soc_lstm_model.pt")