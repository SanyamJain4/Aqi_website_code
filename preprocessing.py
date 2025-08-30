import pandas as pd
import numpy as np

def preprocess_data(file_path="preprocessed.csv"):
    df1 = pd.read_csv(file_path)
    df1['Timestamp'] = pd.to_datetime(df1['Timestamp'])
    df1.set_index('Timestamp', inplace=True)

    # Remove fully NA cols/rows
    df1 = df1.loc[:, ~(df1.isna()).all()]
    df1 = df1.loc[~(df1.isna()).all(axis=1)]

    # Lookup for filling future values
    lookup = {ts: df1.loc[ts].to_dict() for ts in df1.index}

    def fill_future(ts, col, max_years=7):
        for year_offset in range(1, max_years + 1):
            try:
                ts_new = ts.replace(year=ts.year + year_offset)
            except ValueError:
                continue
            if ts_new in lookup:
                val = lookup[ts_new].get(col)
                if pd.notna(val):
                    return val
        return np.nan

    # Fill missing values
    numeric_cols = df1.select_dtypes(include='number').columns
    for col in numeric_cols:
        df1[col] = [
            val if pd.notna(val) else fill_future(ts, col)
            for ts, val in zip(df1.index, df1[col])
        ]

    # Interpolation
    df1 = df1.infer_objects(copy=False)
    df1.interpolate(method='linear', limit_direction='both', inplace=True)

    # Outlier clipping
    for col in numeric_cols:
        Q1, Q3 = df1[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df1[col] = df1[col].clip(lower, upper)

    # Reset index
    df1 = df1.reset_index().sort_values("Timestamp")

    # Temporal features
    df1['hour'] = df1['Timestamp'].dt.hour
    df1['day_of_week'] = df1['Timestamp'].dt.dayofweek
    df1['month'] = df1['Timestamp'].dt.month

    # Seasonal weights
    month_weights = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2, 6: 0.1,
                     7: 0.1, 8: 0.1, 9: 0.2, 10: 0.6, 11: 0.9, 12: 1.0}
    hour_weights = {i: 0.9 - abs(12 - i) * 0.05 for i in range(24)}

    df1['PM2.5_month_weight'] = df1['month'].map(month_weights)
    df1['PM2.5_hour_weight'] = df1['hour'].map(hour_weights)

    # Lag features (1,2,3 days)
    if "PM2.5 (µg/m³)" in df1.columns:
        df1["lag_1d"] = df1["PM2.5 (µg/m³)"].shift(24)
        df1["lag_2d"] = df1["PM2.5 (µg/m³)"].shift(48)
        df1["lag_3d"] = df1["PM2.5 (µg/m³)"].shift(72)

    df1.dropna(inplace=True)
    return df1
