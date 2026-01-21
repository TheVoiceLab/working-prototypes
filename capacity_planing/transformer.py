import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1. LOAD & PREP DATA
# -----------------------------
def load_data(path="remote_traffic_4year_full.csv"):
    df = pd.read_csv(path, parse_dates=["time"], low_memory=False)
    df = df.sort_values("time")

    df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
    df["imdb_factor"] = pd.to_numeric(df["imdb_factor"], errors="coerce")
    df["sub_multiplier"] = pd.to_numeric(df["sub_multiplier"], errors="coerce")

    df["hour"] = df["time"].dt.floor("h")
    df["hour_of_day"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_show"] = df["show_name"].notna().astype(int)

    return df.dropna(subset=["traffic"])

# -----------------------------
# 2. HOURLY AGGREGATION
# -----------------------------
def hourly_aggregate(df):
    hourly = df.groupby("hour").agg({
        "traffic": "max",
        "imdb_factor": "mean",
        "sub_multiplier": "mean",
        "hour_of_day": "first",
        "day_of_week": "first",
        "is_weekend": "first",
        "is_show": "max",
        "network": lambda x: x.mode().iloc[0] if not x.mode().empty else "NONE"
    }).reset_index()

    hourly["network"] = hourly["network"].astype("category")
    hourly["network_code"] = hourly["network"].cat.codes
    return hourly

# -----------------------------
# 3. LAG FEATURES
# -----------------------------
def add_lags(df, lags=(1,2,24,168)):
    for lag in lags:
        df[f"lag_{lag}"] = df["traffic"].shift(lag)
    return df.dropna()

# -----------------------------
# 4. PYTORCH DATASET (multi-step)
# -----------------------------
class MultiStepDataset(Dataset):
    def __init__(self, data, features, target_col="traffic", seq_len=168, pred_len=24):
        self.data = data
        self.features = features
        self.target_col = target_col
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[self.features].iloc[idx:idx+self.seq_len].values.astype(np.float32)
        y = self.data[self.target_col].iloc[idx+self.seq_len:idx+self.seq_len+self.pred_len].values.astype(np.float32)
        return torch.tensor(x), torch.tensor(y)

# -----------------------------
# 5. TRANSFORMER MODEL
# -----------------------------
class TransformerMultiStep(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_seq_len=168, pred_len=24):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc_out(x[:, -1, :])  # use last token to predict all future steps

# -----------------------------
# 6. MAIN PIPELINE
# -----------------------------
def main():
    df = load_data()
    hourly = hourly_aggregate(df)
    hourly = add_lags(hourly)

    features = ["hour_of_day","is_show","imdb_factor","sub_multiplier","network_code",
                "lag_1","lag_2","lag_24","lag_168"]

    # Train/Test split
    split_time = hourly["hour"].min() + pd.DateOffset(years=3)
    train = hourly[hourly["hour"] < split_time].copy()
    test  = hourly[hourly["hour"] >= split_time].copy()

    # -----------------------------
    # 6.1 NORMALIZATION
    # -----------------------------
    feature_scaler = StandardScaler()
    target_scaler  = StandardScaler()

    train[features] = feature_scaler.fit_transform(train[features])
    test[features]  = feature_scaler.transform(test[features])

    train["traffic_scaled"] = target_scaler.fit_transform(train[["traffic"]])
    test["traffic_scaled"]  = target_scaler.transform(test[["traffic"]])

    # -----------------------------
    # 6.2 DATASET & DATALOADER
    # -----------------------------
    seq_len = 168
    pred_len = 24

    train_ds = MultiStepDataset(train, features, target_col="traffic_scaled", seq_len=seq_len, pred_len=pred_len)
    test_ds  = MultiStepDataset(pd.concat([train.tail(seq_len), test]), features, target_col="traffic_scaled", seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)

    # -----------------------------
    # 6.3 MODEL
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerMultiStep(input_dim=len(features), pred_len=pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # -----------------------------
    # 6.4 TRAIN
    # -----------------------------
    model.train()
    epochs = 30
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

    # -----------------------------
    # 6.5 PREDICTION
    # -----------------------------
    model.eval()
    y_preds, y_trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_hat = model(x_batch).cpu().numpy()
            y_preds.append(y_hat)
            y_trues.append(y_batch.numpy())

    y_pred = target_scaler.inverse_transform(np.vstack(y_preds).reshape(-1,1)).flatten()
    y_true = target_scaler.inverse_transform(np.vstack(y_trues).reshape(-1,1)).flatten()

    print("\nTransformer Multi-Step")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))

    # -----------------------------
    # 6.6 PLOT LAST 14 DAYS
    # -----------------------------
    plt.figure(figsize=(14,6))
    last_hours = test.tail(24*14)
    plt.plot(last_hours["hour"], last_hours["traffic"], label="Actual")
    plt.plot(last_hours["hour"], y_pred[-len(last_hours):], label="Transformer", alpha=0.7)
    plt.title("Capacity Forecast – Last 14 Days (Transformer Multi-Step)")
    plt.ylabel("Peak Commands / Hour")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# -----------------------------
if __name__ == "__main__":
    main()
