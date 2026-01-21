import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 1. LOAD & PREP DATA
# -----------------------------
def load_data(path="remote_traffic_4year_full.csv"):
    df = pd.read_csv(
        path,
        parse_dates=["time"],
        low_memory=False
    )

    df = df.sort_values("time")

    # Ensure numeric columns
    df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
    df["imdb_factor"] = pd.to_numeric(df["imdb_factor"], errors="coerce")
    df["sub_multiplier"] = pd.to_numeric(df["sub_multiplier"], errors="coerce")

    # Hourly features
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
        "traffic": "max",       # capacity cares about peaks
        "imdb_factor": "mean",
        "sub_multiplier": "mean",
        "hour_of_day": "first",
        "day_of_week": "first",
        "is_weekend": "first",
        "is_show": "max",
        "network": lambda x: x.mode().iloc[0] if not x.mode().empty else "NONE"
    }).reset_index()

    # Encode network as numeric
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
# 4. MAIN PIPELINE
# -----------------------------
def main():
    df = load_data()
    hourly = hourly_aggregate(df)

    # Train/Test split: first 3 years train, 4th year test
    split_time = hourly["hour"].min() + pd.DateOffset(years=3)
    train = hourly[hourly["hour"] < split_time].copy()
    test  = hourly[hourly["hour"] >= split_time].copy()

    y_train = train["traffic"].values
    y_test = test["traffic"].values

    # -----------------------------
    # 4.1 ARIMA
    # -----------------------------
    arima = ARIMA(y_train, order=(2,1,2))
    arima_fit = arima.fit()
    arima_pred = arima_fit.forecast(len(y_test))

    print("\nARIMA")
    print("MAE:", mean_absolute_error(y_test, arima_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, arima_pred)))

    # -----------------------------
    # 4.2 SARIMAX (improved)
    # -----------------------------
    # Use only most relevant features
    sarimax_cols = ["hour_of_day", "is_show", "imdb_factor", "sub_multiplier"]
    exog_train = train[sarimax_cols].values
    exog_test = test[sarimax_cols].values

    from numpy import log1p, expm1
    y_train_log = log1p(y_train)

    sarimax = SARIMAX(
        y_train_log,
        exog=exog_train,
        order=(2,1,2),
        seasonal_order=(2,0,2,24),  # daily seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarimax_fit = sarimax.fit(disp=False)
    y_pred_log = sarimax_fit.forecast(steps=len(y_test), exog=exog_test)
    y_pred_sarimax = expm1(y_pred_log)

    print("\nSARIMAX")
    print("MAE:", mean_absolute_error(y_test, y_pred_sarimax))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_sarimax)))

    # -----------------------------
    # 4.3 LIGHTGBM
    # -----------------------------
    train_lgb = add_lags(train)
    test_lgb = add_lags(pd.concat([train_lgb.tail(168), test]))

    features = sarimax_cols + ["lag_1","lag_2","lag_24","lag_168"]

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(train_lgb[features], train_lgb["traffic"])
    lgb_pred = model.predict(test_lgb[features])

    print("\nLightGBM")
    print("MAE:", mean_absolute_error(test_lgb["traffic"], lgb_pred))
    print("RMSE:", np.sqrt(mean_squared_error(test_lgb["traffic"], lgb_pred)))

    # -----------------------------
    # 5. PLOT LAST 14 DAYS
    # -----------------------------
    plt.figure(figsize=(14,6))
    last_hours = test_lgb.tail(24*14)
    plt.plot(last_hours["hour"], last_hours["traffic"], label="Actual")
    plt.plot(last_hours["hour"], arima_pred[-len(last_hours):], label="ARIMA", alpha=0.7)
    plt.plot(last_hours["hour"], y_pred_sarimax[-len(last_hours):], label="SARIMAX", alpha=0.7)
    plt.plot(last_hours["hour"], lgb_pred[-len(last_hours):], label="LightGBM", alpha=0.7)

    plt.title("Capacity Forecast – Last 14 Days")
    plt.ylabel("Peak Commands / Hour")
    plt.xlabel("Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# -----------------------------
if __name__ == "__main__":
    main()
