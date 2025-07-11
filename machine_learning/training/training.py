import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from training.plotting import (
    plot_predictions,
    plot_residuals,
    plot_feature_importance,
)

OUTPUT_DIR = "machine_learning/training/mlruns/artifacts"


def train_evaluate_baseline_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        # Train
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        mse_per_output = mean_squared_error(y_test, y_pred, multioutput="raw_values")
        avg_mse = np.mean(mse_per_output)

        # Log parameters + metrics
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        for i, mse in enumerate(mse_per_output):
            mlflow.log_metric(f"mse_output_{i}", mse)
        mlflow.log_metric("avg_mse", avg_mse)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # === Generate and log plots ===
        os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

        # 1. Prediction vs. True
        pred_plot_path = f"{OUTPUT_DIR}/prediction_vs_truth.png"
        plot_predictions(y_test, y_pred, pred_plot_path)
        mlflow.log_artifact(pred_plot_path)

        # 2. Residuals
        res_plot_path = f"{OUTPUT_DIR}/residuals.png"
        plot_residuals(y_test, y_pred, res_plot_path)
        mlflow.log_artifact(res_plot_path)

        # 3. Feature importance
        feat_plot_path = f"{OUTPUT_DIR}/feature_importance.png"
        plot_feature_importance(model, X_test, y_test, feat_plot_path)
        mlflow.log_artifact(feat_plot_path)

        print("MSE per output coordinate:", mse_per_output)
        print("Average MSE:", avg_mse)

        return model, mse_per_output, avg_mse
