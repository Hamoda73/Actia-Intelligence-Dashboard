import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # This was missing
from tqdm import tqdm

# ======================
# 1. Configuration
# ======================
MODEL_PATH = "best_soc_lstm_model.pt"  # or "final_soc_lstm_model.pt"
SCALER_PATH = "scaler.pkl"
VEHICLE_ENCODER_PATH = "vehicle_state_encoder.pkl"
CHARGE_ENCODER_PATH = "charge_state_encoder.pkl"

# Should match training parameters
sequence_length = 16
prediction_shift = 4
batch_size = 64

# Test data path (could be same as training or separate)
test_folder_path = "./car_data3/"  # or "./test_data/" if you have separate test data

# ======================
# 2. Load Model and Preprocessing
# ======================
print("🚀 Loading model and preprocessing artifacts...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_soc(model, scaler, v_encoder, c_encoder, input_data):
    """
    Make SOC prediction on new data
    
    Args:
        input_data: List of dictionaries or DataFrame with the same structure as training data
                   Must contain at least 16 sequential records (sequence_length)
    """
    # Convert input to dataframe
    df_new = pd.DataFrame(input_data)
    
    # Preprocess exactly like training data
    df_new.columns = [c.strip().lower().replace(' ', '_') for c in df_new.columns]
    for col in ['vehicle_state', 'charge_state']:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(str).map(translate)
    df_new['record_time'] = pd.to_datetime(df_new['record_time'], errors='coerce')
    df_new = df_new.sort_values('record_time').dropna()
    
    # Encode categorical
    df_new['vehicle_state'] = v_encoder.transform(df_new['vehicle_state'])
    df_new['charge_state'] = c_encoder.transform(df_new['charge_state'])
    
    # Normalize features
    df_new[features] = scaler.transform(df_new[features])
    
    # Create sequence (last 16 records)
    seq = df_new[features].iloc[-sequence_length:].values
    seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(seq).cpu().numpy()
    
    # Inverse transform SOC
    prediction = soc_scaler.inverse_transform(prediction.reshape(-1, 1))[0][0]
    
    return prediction


# Load the model architecture and weights
class LSTMSoC(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# Determine input size (will be set after data loading)
model = LSTMSoC(input_size=9).to(device)  # Temporary, will be updated
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Load preprocessing artifacts
scaler = joblib.load(SCALER_PATH)
v_encoder = joblib.load(VEHICLE_ENCODER_PATH)
c_encoder = joblib.load(CHARGE_ENCODER_PATH)

# ======================
# 3. Data Loading and Processing
# ======================
print("\n🔍 Loading and processing test data...")

# Translation dictionary (should match training)
translate = {
    "车辆启动": "started", "熄火": "stopped",
    "未充电": "not charging", "停车充电": "charging while parked",
    "充电完成熄火": "charge done + stopped"
}

# Load test data (similar to training)
test_data = []
for file in tqdm(os.listdir(test_folder_path), desc="Processing test files"):
    if file.endswith(".xlsx"):
        try:
            df = pd.read_excel(os.path.join(test_folder_path, file), engine="openpyxl")
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            for col in ['vehicle_state', 'charge_state']:
                if col in df.columns:
                    df[col] = df[col].astype(str).map(translate)
            df['record_time'] = pd.to_datetime(df['record_time'], errors='coerce')
            df = df.sort_values('record_time').dropna()
            test_data.append(df)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if not test_data:
    raise ValueError("No valid test data files found!")

df_test = pd.concat(test_data, ignore_index=True)
df_test = df_test.dropna(subset=['soc(%)'])
print(f"Total test data points: {len(df_test)}")

# Encode categorical variables
df_test['vehicle_state'] = v_encoder.transform(df_test['vehicle_state'])
df_test['charge_state'] = c_encoder.transform(df_test['charge_state'])

# Features (should match training)
features = [
    'vehicle_state', 'charge_state',
    'pack_voltage(v)', 'pack_current(a)', 'soc(%)',
    'max_cell_voltage_(v)', 'min_cell_voltage_(v)',
    'max_probe_temperature_(℃)', 'min_probe_temperature_(℃)'
]

# Normalize features
df_test[features] = scaler.transform(df_test[features])

# ======================
# 4. Create Test Sequences
# ======================
print("\n⏳ Creating test sequences...")
test_sequences = []
test_targets = []
timestamps = []

for i in tqdm(range(len(df_test) - sequence_length - prediction_shift), desc="Generating test sequences"):
    seq = df_test[features].iloc[i:i+sequence_length].values
    target = df_test['soc(%)'].iloc[i+sequence_length+prediction_shift-1]
    timestamp = df_test['record_time'].iloc[i+sequence_length+prediction_shift-1]
    
    test_sequences.append(seq)
    test_targets.append(target)
    timestamps.append(timestamp)

X_test = np.array(test_sequences)
y_test = np.array(test_targets)
timestamps = pd.to_datetime(timestamps)

print(f"Test dataset shape - X: {X_test.shape}, y: {y_test.shape}")

# Update model input size if needed (should match training)
if X_test.shape[2] != model.lstm.input_size:
    print(f"⚠️ Warning: Input size mismatch. Updating model input size from {model.lstm.input_size} to {X_test.shape[2]}")
    model = LSTMSoC(input_size=X_test.shape[2]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

# ======================
# 5. Create DataLoader
# ======================
class TestSoCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

test_dataset = TestSoCDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ======================
# 6. Make Predictions
# ======================
print("\n🔮 Making predictions...")
predictions = []
actuals = []

with torch.no_grad():
    for xb, yb in tqdm(test_loader, desc="Predicting"):
        xb = xb.to(device)
        preds = model(xb).cpu().numpy()
        
        predictions.extend(preds)
        actuals.extend(yb.numpy())

# Inverse transform the SOC predictions (since SOC was normalized)
soc_scaler = StandardScaler()
soc_scaler.mean_ = scaler.mean_[features.index('soc(%)')]
soc_scaler.scale_ = scaler.scale_[features.index('soc(%)')]

predictions = soc_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals = soc_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

# ======================
# 7. Evaluate Performance
# ======================
print("\n📊 Evaluation Metrics:")
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actuals, predictions)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Calculate absolute percentage error
absolute_errors = np.abs(actuals - predictions)
percentage_errors = (absolute_errors / actuals) * 100
mean_ape = np.mean(percentage_errors)
print(f"Mean Absolute Percentage Error (MAPE): {mean_ape:.2f}%")

# ======================
# 8. Visualizations
# ======================
print("\n🎨 Generating visualizations...")
plt.figure(figsize=(15, 6))

# Time series plot
plt.subplot(1, 2, 1)
plt.plot(timestamps, actuals, label='Actual SOC', color='blue', alpha=0.7)
plt.plot(timestamps, predictions, label='Predicted SOC', color='red', alpha=0.7, linestyle='--')
plt.title('SOC Prediction vs Actual Over Time')
plt.xlabel('Time')
plt.ylabel('SOC (%)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

# Scatter plot
plt.subplot(1, 2, 2)
plt.scatter(actuals, predictions, alpha=0.5)
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
plt.title('Actual vs Predicted SOC')
plt.xlabel('Actual SOC (%)')
plt.ylabel('Predicted SOC (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('soc_prediction_results.png', dpi=300)
plt.show()

# Error distribution
plt.figure(figsize=(10, 5))
plt.hist(absolute_errors, bins=30, edgecolor='black')
plt.title('Distribution of Absolute Errors')
plt.xlabel('Absolute Error in SOC (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('error_distribution.png', dpi=300)
plt.show()

print("\n✅ Testing complete! Results saved to:")
print("- soc_prediction_results.png")
print("- error_distribution.png")

if __name__ == "__main__":
    # Example input data (replace with your actual new data)
    example_data = pd.read_csv("2022-02-07.xlsx").to_dict(orient='records')
    
    predicted_soc = predict_soc(model, scaler, v_encoder, c_encoder, example_data)
    print(f"\nPredicted SOC in {prediction_shift*15} minutes: {predicted_soc:.2f}%")
    