from preprocessing import preprocess_data
from modeling import train_model
from plotting import plot_feature_importance, plot_actual_vs_pred
from forecasting import forecast_next_days

def main():
    print("🔹 Preprocessing Data...")
    df = preprocess_data("preprocessed.csv")

    print("🔹 Training Model...")
    results = train_model(df)

    print(f"Train R²: {results['train_r2']:.3f}")
    print(f"Test R²: {results['test_r2']:.3f}")
    print(f"Test RMSE: {results['rmse']:.3f}")

    print("🔹 Plotting Feature Importance...")
    plot_feature_importance(results["model"], results["X_train"])

    print("🔹 Plotting Actual vs Predicted...")
    plot_actual_vs_pred(results["y_train"], results["y_pred_train"],
                        results["y_test"], results["y_pred_test"])

    print("🔹 Forecasting Next 3 Days...")
    forecast_df = forecast_next_days(df, results["model"])
    print(forecast_df)

if __name__ == "__main__":
    main()
