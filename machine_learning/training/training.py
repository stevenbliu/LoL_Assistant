import mlflow
import mlflow.sklearn
import numpy as np
import os

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from training.plotting import (
    plot_predictions,
    plot_residuals,
    plot_feature_importance,
    plot_learning_curve,
)

OUTPUT_DIR = "machine_learning/training/mlruns/artifacts"


def train_evaluate_model(
    X_train, y_train, X_test, y_test, base_model, model_name="model"
):

    OUTPUT_DIR = f"machine_learning/training/mlruns/artifacts/{model_name}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with mlflow.start_run():
        # Wrap in MultiOutputRegressor if multi-output and model does not support it natively
        from sklearn.multioutput import MultiOutputRegressor

        if y_train.shape[1] > 1 and not hasattr(base_model, "predict_multioutput"):
            model = MultiOutputRegressor(base_model)
        else:
            model = base_model

        mlflow.log_param("model_type", model_name)
        for key, value in base_model.get_params().items():
            mlflow.log_param(key, value)

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mse_per_output = mean_squared_error(y_test, y_pred, multioutput="raw_values")
        avg_mse = np.mean(mse_per_output)

        for i, mse in enumerate(mse_per_output):
            mlflow.log_metric(f"mse_output_{i}", mse)
        mlflow.log_metric("avg_mse", avg_mse)

        # Log model
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        # Plots
        pred_plot_path = os.path.join(OUTPUT_DIR, "prediction_vs_truth.png")
        plot_predictions(y_test, y_pred, pred_plot_path)
        mlflow.log_artifact(pred_plot_path)

        res_plot_path = os.path.join(OUTPUT_DIR, "residuals.png")
        plot_residuals(y_test, y_pred, res_plot_path)
        mlflow.log_artifact(res_plot_path)

        # Feature importance plot — works only if model has feature_importances_ attribute
        feat_plot_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        try:
            # For MultiOutputRegressor, take first estimator's feature importance
            if hasattr(model, "estimators_"):
                est = model.estimators_[0]
            else:
                est = model
            plot_feature_importance(est, X_test, y_test.iloc[:, 0], feat_plot_path)
            mlflow.log_artifact(feat_plot_path)
        except Exception as e:
            print(f"Warning: Could not plot feature importance: {e}")

        # Learning curve
        learning_curve_path = os.path.join(OUTPUT_DIR, "learning_curve.png")
        try:
            plot_learning_curve(model, X_train, y_train, learning_curve_path)
            mlflow.log_artifact(learning_curve_path)
        except Exception as e:
            print(f"Warning: Could not plot learning curve: {e}")

        # Train/test evaluation
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_train_pred)

        test_mse = mean_squared_error(y_test, y_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(y_test, y_pred)

        print(
            f"\n📈 Train MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}"
        )
        print(f"📊 Test  MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

        mlflow.log_metric("train_mse", train_mse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        return model, mse_per_output, avg_mse
