import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

MODEL_PATH = "transformer_t1.pt"
SX_PATH = "scaler_x.pkl"
SY_PATH = "scaler_y.pkl"

# -----------------------------
# 1. LOAD & HOURLY AGGREGATION
# -----------------------------
def load_hourly_data(path="remote_traffic_4year_full.csv"):
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df = df.sort_values("time")

    for col in ["traffic", "imdb_factor", "sub_multiplier"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hour"] = df["time"].dt.floor("h")
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_show"] = df["show_name"].notna().astype(int)

    hourly = df.groupby("hour").agg({
        "traffic": "max",
        "imdb_factor": "mean",
        "sub_multiplier": "mean",
        "hour_of_day": "first",
        "day_of_week": "first",
        "is_weekend": "first",
        "is_show": "max",
    }).reset_index()

    for lag in [1, 2, 24, 168]:
        hourly[f"lag_{lag}"] = hourly["traffic"].shift(lag)

    return hourly.dropna().reset_index(drop=True)

# -----------------------------
# 2. DATASET
# -----------------------------
class TSDataset(Dataset):
    def __init__(self, x, y, seq_len=168):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        return self.x[idx:idx+self.seq_len], self.y[idx+self.seq_len]

# -----------------------------
# 3. TRANSFORMER MODEL
# -----------------------------
class TransformerTS(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1])

# -----------------------------
# 4. TRAIN (ONLY IF NEEDED)
# -----------------------------
def train_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 168

    df = load_hourly_data()

    features = [
        "hour_of_day","day_of_week","is_weekend","is_show",
        "imdb_factor","sub_multiplier",
        "lag_1","lag_2","lag_24","lag_168"
    ]

    split_time = df["hour"].min() + pd.DateOffset(years=3)
    train_df = df[df["hour"] < split_time]
    test_df  = df[df["hour"] >= split_time]

    sx, sy = StandardScaler(), StandardScaler()
    X_train = sx.fit_transform(train_df[features])
    y_train = sy.fit_transform(train_df[["traffic"]]).flatten()

    train_ds = TSDataset(X_train, y_train, seq_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = TransformerTS(len(features)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Training Transformer...")
    for epoch in range(20):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(), yb)
            loss.backward()
            opt.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d} | Loss {loss.item():.4f}")

    # ---- SAVE EVERYTHING
    torch.save(model.state_dict(), MODEL_PATH)
    joblib.dump(sx, SX_PATH)
    joblib.dump(sy, SY_PATH)

    print("Model and scalers saved.")

# -----------------------------
# 5. LOAD & PLOT (NO TRAINING)
# -----------------------------
def load_and_plot():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = 168

    df = load_hourly_data()

    features = [
        "hour_of_day","day_of_week","is_weekend","is_show",
        "imdb_factor","sub_multiplier",
        "lag_1","lag_2","lag_24","lag_168"
    ]

    split_time = df["hour"].min() + pd.DateOffset(years=3)
    test_df = df[df["hour"] >= split_time]

    sx = joblib.load(SX_PATH)
    sy = joblib.load(SY_PATH)

    X_test = sx.transform(test_df[features])
    y_test = sy.transform(test_df[["traffic"]]).flatten()

    test_ds = TSDataset(X_test, y_test, seq_len)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = TransformerTS(len(features)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device)).cpu().numpy().flatten()
            preds.append(out)
            trues.append(yb.numpy())

    y_pred = sy.inverse_transform(np.concatenate(preds).reshape(-1,1)).flatten()
    y_true = sy.inverse_transform(np.concatenate(trues).reshape(-1,1)).flatten()

    print("\nLoaded Transformer")
    print("MAE :", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

    plt.figure(figsize=(14,6))
    plt.plot(y_true[-336:], label="Actual")
    plt.plot(y_pred[-336:], label="Transformer", linestyle="--")
    plt.title("Capacity Forecast – Last 14 Days (Hourly Max)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        train_and_save()
    load_and_plot()
