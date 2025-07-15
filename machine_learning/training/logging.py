# utils/logging.py

import os
import tempfile
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, r2_score
from training.plotting import plot_predictions, plot_residuals
import pandas as pd


def log_metrics_and_plots(y_test, y_pred, target_names=None, model_name="model"):
    if isinstance(y_test, pd.DataFrame):
        y_test_np = y_test.values
    else:
        y_test_np = y_test

    if isinstance(y_pred, pd.DataFrame):
        y_pred_np = y_pred.values
    else:
        y_pred_np = y_pred

    if hasattr(y_test, "columns"):
        target_names = target_names or y_test.columns.tolist()
    else:
        target_names = target_names or [f"target_{i}" for i in range(y_test.shape[1])]

    # Log metrics
    for i, name in enumerate(target_names):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mlflow.log_metric(f"{name}_mse", mse)
        mlflow.log_metric(f"{name}_r2", r2)
        print(f" - {name} | MSE: {mse:.4f} | R²: {r2:.4f}")

    avg_mse = np.mean(
        [mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    )
    mlflow.log_metric("avg_mse", avg_mse)

    # Save and log plots
    with tempfile.TemporaryDirectory() as tmpdir:
        scatter_path = os.path.join(tmpdir, "prediction_scatter.png")
        residuals_path = os.path.join(tmpdir, "residuals.png")

        plot_predictions(
            y_test, y_pred, output_path=scatter_path, target_names=target_names
        )
        plot_residuals(
            y_test, y_pred, output_path=residuals_path, target_names=target_names
        )

        mlflow.log_artifact(scatter_path)
        mlflow.log_artifact(residuals_path)
