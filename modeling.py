import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_model(df, target_col="PM2.5 (µg/m³)"):
    last_year = df['Timestamp'].dt.year.max()
    train_df = df[df['Timestamp'].dt.year < last_year]
    test_df  = df[df['Timestamp'].dt.year == last_year]

    X_train = train_df.drop(columns=["Timestamp", target_col])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=["Timestamp", target_col])
    y_test = test_df[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    results = {
        "model": model,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "y_pred_train": y_pred_train, "y_pred_test": y_pred_test,
        "train_r2": r2_score(y_train, y_pred_train),
        "test_r2": r2_score(y_test, y_pred_test),
        "rmse": mean_squared_error(y_test, y_pred_test, squared=False)
    }
    return results
