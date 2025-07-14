from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import joblib
import tempfile
import os

def train_evaluate_model(...):
    # your existing train_evaluate_model function
    pass

def tune_model_randomizedsearchcv(base_model, param_dist, X_train, y_train, n_iter=10, cv=3):
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
