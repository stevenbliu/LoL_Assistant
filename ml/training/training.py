import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def train_evaluate_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a RandomForestRegressor baseline model for multi-output regression.
    Tracks experiment with MLflow.

    Returns:
        model: Trained model
        mse_per_output: List of MSEs per output
        avg_mse: Average MSE
    """
    with mlflow.start_run():
        # Define and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_test)
        mse_per_output = mean_squared_error(y_test, y_pred, multioutput="raw_values")
        avg_mse = np.mean(mse_per_output)

        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)

        # Log metrics
        for i, mse in enumerate(mse_per_output):
            mlflow.log_metric(f"mse_output_{i}", mse)
        mlflow.log_metric("avg_mse", avg_mse)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Print
        print("MSE per output coordinate:", mse_per_output)
        print("Average MSE:", avg_mse)

    return model, mse_per_output, avg_mse
