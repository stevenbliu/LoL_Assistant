import mlflow
import mlflow.sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score  # <-- added r2_score
import numpy as np
import os
from training.plotting import (
    plot_predictions,
    plot_residuals,
    plot_feature_importance,
    plot_learning_curve,
)

OUTPUT_DIR = "machine_learning/training/mlruns/artifacts"


def train_evaluate_baseline_model(X_train, y_train, X_test, y_test):
    with mlflow.start_run():
        base_model = HistGradientBoostingRegressor(
            max_iter=100,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_depth=10,
        )
        params = base_model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        model = MultiOutputRegressor(base_model)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse_per_output = mean_squared_error(y_test, y_pred, multioutput="raw_values")
        avg_mse = np.mean(mse_per_output)

        # Log parameters + metrics (updated model name)
        mlflow.log_param("model_type", "HistGradientBoostingRegressor")
        mlflow.log_param("max_iter", 100)
        mlflow.log_param("random_state", 42)

        for i, mse in enumerate(mse_per_output):
            mlflow.log_metric(f"mse_output_{i}", mse)
        mlflow.log_metric("avg_mse", avg_mse)

        mlflow.sklearn.log_model(model, "hist_gradient_boosting_model")

        os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)

        # 1. Prediction vs True
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

        # Train predictions and metrics for bias/variance check
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_pred)

        print(f"Train MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        print(f"Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        # Learning curve plot
        learning_curve_path = f"{OUTPUT_DIR}/learning_curve.png"
        plot_learning_curve(model, X_train, y_train, learning_curve_path)
        mlflow.log_artifact(learning_curve_path)

        return model, mse_per_output, avg_mse
