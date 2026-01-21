import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------
# 1. LOAD & HOURLY AGGREGATION (Max Minute Value)
# -----------------------------
def get_processed_data(path="remote_traffic_4year_full.csv"):
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)

    # Clean and Cast
    for col in ["traffic", "imdb_factor", "sub_multiplier"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1-Hour Bucket: Max traffic per hour
    df["hour_idx"] = df["time"].dt.floor("h")
    hourly = df.groupby("hour_idx").agg({
        "traffic": "max",  # Hourly Peak
        "imdb_factor": "mean",
        "sub_multiplier": "mean",
        "hour_idx": "first"
    }).reset_index(drop=True)

    # Features
    hourly["hour_of_day"] = hourly["hour_idx"].dt.hour
    hourly["day_of_week"] = hourly["hour_idx"].dt.dayofweek

    # Lag features
    for l in [1, 2, 24, 168]:
        hourly[f"lag_{l}"] = hourly["traffic"].shift(l)

    return hourly.dropna()


# -----------------------------
# 2. FAST DATASET (Tensor-based)
# -----------------------------
class FastTSDataset(Dataset):
    def __init__(self, features, targets, seq_len=168, pred_len=24):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        return self.x[idx: idx + self.seq_len], self.y[idx + self.seq_len: idx + self.seq_len + self.pred_len]


# -----------------------------
# 3. TRANSFORMER MODEL
# -----------------------------
class TransformerMultiStep(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, pred_len=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])  # Use the last context window to predict future


# -----------------------------
# 4. MAIN EXECUTION
# -----------------------------
def main():
    # Setup
    seq_len, pred_len = 168, 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = get_processed_data()
    feature_cols = ["hour_of_day", "day_of_week", "imdb_factor", "sub_multiplier", "lag_1", "lag_2", "lag_24"]

    # Split (3 years train, 1 year test)
    split_idx = int(len(data) * 0.75)
    train_df, test_df = data.iloc[:split_idx], data.iloc[split_idx:]

    # Scaling
    sc_x, sc_y = StandardScaler(), StandardScaler()
    train_x = sc_x.fit_transform(train_df[feature_cols])
    train_y = sc_y.fit_transform(train_df[["traffic"]]).flatten()

    test_x = sc_x.transform(test_df[feature_cols])
    test_y = sc_y.transform(test_df[["traffic"]]).flatten()

    # Loaders
    train_loader = DataLoader(FastTSDataset(train_x, train_y), batch_size=64, shuffle=True)
    test_ds = FastTSDataset(test_x, test_y)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Train
    model = TransformerMultiStep(len(feature_cols)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting fast training...")
    model.train()
    for epoch in range(20):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0: print(f"Epoch {epoch + 1} done, loss {loss}")

    # -----------------------------
    # 5. ERROR CALCULATION & PLOT
    # -----------------------------
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb.to(device)).cpu().numpy()
            preds.append(out)
            actuals.append(yb.numpy())

    # Reconstruct arrays (flattening for point-by-point comparison)
    y_pred_rescaled = sc_y.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()
    y_true_rescaled = sc_y.inverse_transform(np.concatenate(actuals).reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_true_rescaled, y_pred_rescaled))

    print(f"\n--- Results ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    # Plot Last 14 Days
    plt.figure(figsize=(12, 5))
    # We show only a slice to keep the plot readable
    plt.plot(y_true_rescaled[-336:], label="Actual (Peak)", alpha=0.8)
    plt.plot(y_pred_rescaled[-336:], label="Transformer Prediction", linestyle="--")
    plt.title("Traffic Forecast: Last 14 Days (Hourly Max)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()