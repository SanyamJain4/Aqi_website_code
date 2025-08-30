import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_importance(model, X, top_n=20):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp_df.head(top_n), x='Importance', y='Feature')
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def plot_actual_vs_pred(y_train, y_pred_train, y_test, y_pred_test):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(y_train.values, label="Actual Train", alpha=0.7)
    axes[0].plot(y_pred_train, label="Predicted Train", alpha=0.7)
    axes[0].set_title("Train Data: Actual vs Predicted")
    axes[0].legend()

    axes[1].plot(y_test.values, label="Actual Test", alpha=0.7)
    axes[1].plot(y_pred_test, label="Predicted Test", alpha=0.7)
    axes[1].set_title("Test Data: Actual vs Predicted")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
