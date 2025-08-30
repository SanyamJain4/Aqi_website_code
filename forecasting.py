import pandas as pd

def forecast_next_days(df, model, target_col="PM2.5 (µg/m³)", horizon=3):
    latest_row = df.iloc[-1:].copy()
    forecasts = []

    for h in range(1, horizon + 1):
        X_latest = latest_row.drop(columns=["Timestamp", target_col])
        y_pred = model.predict(X_latest)[0]
        next_ts = latest_row["Timestamp"].values[0] + pd.Timedelta(days=1)

        forecasts.append({"Date": pd.to_datetime(next_ts), "Forecast_PM2.5": y_pred})

        # update latest row for recursive forecasting
        new_row = latest_row.copy()
        new_row["Timestamp"] = pd.to_datetime(next_ts)
        new_row[target_col] = y_pred
        new_row["lag_1d"] = latest_row[target_col].values[0]
        new_row["lag_2d"] = latest_row["lag_1d"].values[0]
        new_row["lag_3d"] = latest_row["lag_2d"].values[0]

        latest_row = new_row

    return pd.DataFrame(forecasts)
