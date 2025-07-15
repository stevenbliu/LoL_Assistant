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
from sklearn.model_selection import RandomizedSearchCV
from training.logging import log_metrics_and_plots


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


from datetime import datetime
import os
import tempfile
import pandas as pd
import joblib
import mlflow
from sklearn.multioutput import MultiOutputRegressor
from mlflow.models.signature import infer_signature

from training.logging import log_metrics_and_plots


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
    print("\n🚀 Starting model training...", flush=True)
    mlflow.set_experiment(experiment_name)
    run_name = f"run_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("dataset", dataset)

        if y_train.shape[1] > 1 and not isinstance(base_model, MultiOutputRegressor):
            print(f"Wrapping {model_name} in MultiOutputRegressor...", flush=True)
            model = MultiOutputRegressor(base_model)
        else:
            model = base_model

        mlflow.log_param("model_type", model_name)
        for key, value in base_model.get_params().items():
            mlflow.log_param(key, value)

        print("📈 Fitting model...", flush=True)
        model.fit(X_train, y_train)

        print("📊 Predicting...", flush=True)
        y_pred = model.predict(X_test)

        # Log metrics and plots centrally
        log_metrics_and_plots(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            model=model,
        )

        # Save model
        print("💾 Saving model...", flush=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.joblib")
            joblib.dump(model, model_path)

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=X_train[:5],
            )

    print("✅ Training and evaluation complete.\n", flush=True)
    return model


def tune_model_randomizedsearchcv(
    base_model, param_dist, X_train, y_train, n_iter=10, cv=3
):
    model = MultiOutputRegressor(base_model)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions={f"estimator__{k}": v for k, v in param_dist.items()},
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_
