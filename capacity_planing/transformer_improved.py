import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. LOAD & PREP DATA
# --------------------------------------------------
def load_data(path="remote_traffic_4year_full.csv"):
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.sort_values("time").set_index("time")

    df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
    df = df.dropna(subset=["traffic"])

    # time features
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)

    return df


# --------------------------------------------------
# 2. DATASET (LOG-DELTA TARGET)
# --------------------------------------------------
class TrafficDataset(Dataset):
    def __init__(self, df, features, seq_len=168):
        self.seq_len = seq_len
        self.X = df[features].values.astype(np.float32)

        self.y = df["traffic"].values.astype(np.float32)
        self.log_y = np.log(self.y + 1e-6)

    def __len__(self):
        return len(self.y) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]

        # log-delta target
        y_delta = (
            self.log_y[idx + self.seq_len + 1]
            - self.log_y[idx + self.seq_len]
        )

        last_value = self.y[idx + self.seq_len]

        return (
            torch.tensor(x),
            torch.tensor(y_delta),
            torch.tensor(last_value)
        )


# --------------------------------------------------
# 3. TRANSFORMER MODEL
# --------------------------------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.fc(x[:, -1]).squeeze(-1)


# --------------------------------------------------
# 4. TRAINING
# --------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, y_delta, _ in loader:
        x = x.to(device)
        y_delta = y_delta.to(device)

        optimizer.zero_grad()
        pred = model(x)

        loss = nn.MSELoss()(pred, y_delta)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# --------------------------------------------------
# 5. EVALUATION (RECONSTRUCT TRAFFIC)
# --------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for x, y_delta, last_val in loader:
            x = x.to(device)
            y_delta = y_delta.to(device)

            pred_delta = model(x)

            pred = last_val * torch.exp(pred_delta.cpu())
            actual = last_val * torch.exp(y_delta.cpu())

            preds.extend(pred.numpy())
            actuals.extend(actual.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))

    return mae, rmse, preds, actuals


# --------------------------------------------------
# 6. MAIN PIPELINE
# --------------------------------------------------
def main():
    df = load_data()

    FEATURES = [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos"
    ]

    # normalize features
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # train / test split (last year = test)
    split_time = df.index.max() - pd.DateOffset(years=1)
    train_df = df[df.index < split_time]
    test_df  = df[df.index >= split_time]

    train_ds = TrafficDataset(train_df, FEATURES)
    test_ds  = TrafficDataset(test_df, FEATURES)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerModel(input_dim=len(FEATURES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # training
    for epoch in range(30):
        loss = train_epoch(model, train_loader, optimizer, device)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/30, Loss: {loss:.6f}")

    # evaluation
    mae, rmse, preds, actuals = evaluate(model, test_loader, device)

    print("\nTransformer (Log-Delta)")
    print("MAE:", mae)
    print("RMSE:", rmse)

    # plot last 14 days
    plt.figure(figsize=(14,5))
    plt.plot(actuals[-336:], label="Actual")
    plt.plot(preds[-336:], label="Predicted")
    plt.title("Traffic Forecast – Transformer (Log-Delta)")
    plt.xlabel("Hour")
    plt.ylabel("Traffic")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# --------------------------------------------------
# 7. RUN
# --------------------------------------------------
if __name__ == "__main__":
    main()
