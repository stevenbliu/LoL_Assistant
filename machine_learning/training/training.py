import mlflow
import mlflow.pyfunc
import joblib
import os
import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import training.plotting as plotting
from urllib.parse import urlparse
import tempfile
from mlflow.data import from_pandas


def get_local_path_from_uri(uri):
    parsed = urlparse(uri)
    path = parsed.path
    if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return path


class MultiOutputRegressorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)


from datetime import datetime


def train_evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    base_model,
    dataset,
    model_name="model",
    experiment_name="experiment",
):
    print("\nStarting model training...", flush=True)
    mlflow.set_experiment(experiment_name)
    run_name = f"run_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_param("dataset", dataset)
        # dataset = from_pandas(dataset, name="train_dataset", source="data/train.csv")
        # mlflow.log_input(dataset, context="training")

        # Wrap base model if multi-target regression needed
        if y_train.shape[1] > 1 and not isinstance(base_model, MultiOutputRegressor):
            print(
                f"Wrapping {model_name} with MultiOutputRegressor for multi-target regression.",
                flush=True,
            )
            model = MultiOutputRegressor(base_model)
        else:
            print(
                f"Using {model_name} directly for single-target regression.", flush=True
            )
            model = base_model

        print("Logging parameters...", flush=True)
        mlflow.log_param("model_type", model_name)
        for key, value in base_model.get_params().items():
            mlflow.log_param(key, value)

        # print("Fitting model...", flush=True)
        # model.fit(X_train, y_train)

        print("Predicting on test set...", flush=True)
        y_pred = model.predict(X_test)
        print("Predictions shape:", y_pred.shape, flush=True)

        print("Calculating metrics...", flush=True)
        mse_per_output = mean_squared_error(y_test, y_pred, multioutput="raw_values")
        avg_mse = np.mean(mse_per_output)
        for i, mse in enumerate(mse_per_output):
            print(f"MSE Output {i}: {mse}", flush=True)
            mlflow.log_metric(f"mse_output_{i}", mse)
        print(f"Average MSE: {avg_mse}", flush=True)
        mlflow.log_metric("avg_mse", avg_mse)

        print("Inferring model signature...", flush=True)
        signature = infer_signature(X_train, model.predict(X_train))

        # Save model temporarily and log to MLflow
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.joblib")
            joblib.dump(model, model_path)
            print(f"Model saved temporarily to: {model_path}", flush=True)

            mlflow.pyfunc.log_model(
                name=model_name,
                python_model=MultiOutputRegressorWrapper(),
                artifacts={"model_path": model_path},
                signature=signature,
                input_example=X_train[:5],
            )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{model_name}"
        print(f"Model URI: {model_uri}", flush=True)

        # Prepare evaluation DataFrame
        print("Preparing evaluation data...", flush=True)
        if not isinstance(y_test, pd.DataFrame):
            y_test_df = pd.DataFrame(
                y_test, columns=[f"target_{i}" for i in range(y_test.shape[1])]
            )
        else:
            y_test_df = y_test.copy()

        eval_data = pd.DataFrame(X_test).reset_index(drop=True)
        y_test_df = y_test_df.reset_index(drop=True)
        eval_data = pd.concat([eval_data, y_test_df], axis=1)
        print("Evaluation data shape:", eval_data.shape, flush=True)
        print("Target columns:", y_test_df.columns.tolist(), flush=True)

        print(
            "\nManual Evaluation (since mlflow.evaluate doesn't support multi-target):",
            flush=True,
        )
        for i, col in enumerate(y_test_df.columns):
            col_y_true = y_test_df.iloc[:, i]
            col_y_pred = y_pred[:, i]
            mse = mean_squared_error(col_y_true, col_y_pred)
            r2 = r2_score(col_y_true, col_y_pred)
            print(f"▶️ {col}: MSE = {mse:.4f}, R² = {r2:.4f}", flush=True)
            mlflow.log_metric(f"{col}_mse", mse)
            mlflow.log_metric(f"{col}_r2", r2)

        y_pred_df = pd.DataFrame(y_pred, columns=y_test_df.columns)

        # Use artifact URI to save plots
        artifact_root = get_local_path_from_uri(mlflow.get_artifact_uri())
        model_dir = os.path.join(artifact_root, model_name)
        os.makedirs(model_dir, exist_ok=True)

        print("\nGenerating and logging plots...", flush=True)

        pred_plot_path = os.path.join(model_dir, "prediction_scatter.png")
        plotting.plot_predictions(
            y_test=y_test_df.values,
            y_pred=y_pred_df.values,
            output_path=pred_plot_path,
            target_names=y_test_df.columns.tolist(),
        )
        mlflow.log_artifact(pred_plot_path)

        residual_plot_path = os.path.join(model_dir, "residuals.png")
        plotting.plot_residuals(
            y_test=y_test_df.values,
            y_pred=y_pred_df.values,
            output_path=residual_plot_path,
            target_names=y_test_df.columns.tolist(),
        )
        mlflow.log_artifact(residual_plot_path)

        if hasattr(model, "feature_importances_") or hasattr(
            base_model, "feature_importances_"
        ):
            print("Plotting feature importances...", flush=True)
            importance_plot_path = os.path.join(model_dir, "feature_importance.png")
            plotting.plot_feature_importance(
                model, pd.DataFrame(X_test), y_test_df, importance_plot_path
            )
            mlflow.log_artifact(importance_plot_path)

        print("Plotting learning curve...", flush=True)
        learning_curve_path = os.path.join(model_dir, "learning_curve.png")
        plotting.plot_learning_curve(
            model, pd.DataFrame(X_train), y_train, learning_curve_path
        )
        mlflow.log_artifact(learning_curve_path)
        print("✅ All plots saved and logged to MLflow.", flush=True)

    print("✅ Evaluation complete.\n", flush=True)
    return model, mse_per_output, avg_mse
