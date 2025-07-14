from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor

from training.training import train_evaluate_model
from training.optuna_optimizer import optimize_with_optuna  # <- you define this
from training.data_prep import prepare_ml_data
from training.feature_selection import select_features_multioutput
from training.preprocess import preprocess_data
from training.eda import eda_raw, eda_processed
import pandas as pd
import numpy as np
import os

from training.config import DATA_DIR, VERSION, USE_MLFLOW

# 🔧 Config flags
USE_RANDOM_SEARCH = False
USE_OPTUNA = False
TESTING_MODE = True


def main():
    # --- Load and preprocess data ---
    raw_df_path = f"{DATA_DIR}/{VERSION}/all_jungler_data.csv"
    raw_df = pd.read_csv(raw_df_path)
    if TESTING_MODE:
        raw_df = raw_df.sample(n=1000, random_state=42).reset_index(drop=True)

    eda_raw(raw_df, eda_dir=f"{DATA_DIR}/{VERSION}/eda/raw")
    processed_df = preprocess_data(raw_df)
    eda_processed(
        processed_df,
        eda_dir=f"{DATA_DIR}/{VERSION}/eda/processed",
        log_to_mlflow=USE_MLFLOW,
    )

    target_cols = ["P2_X", "P2_Y"]
    all_features = processed_df.columns.drop(target_cols).tolist()
    selected_features = select_features_multioutput(
        processed_df, target_cols, all_features, top_n=20
    )

    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(
        processed_df, selected_features, target_cols
    )

    # --- Define models and parameter grids ---
    models_with_params = [
        (
            "RandomForest",
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10],
            },
        ),
        (
            "HistGradientBoosting",
            HistGradientBoostingRegressor(random_state=42),
            {
                "max_iter": [100, 200],
                "max_depth": [5, 10],
                "learning_rate": [0.05, 0.1],
            },
        ),
        (
            "XGBoost",
            XGBRegressor(
                objective="reg:squarederror", tree_method="hist", random_state=42
            ),
            {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        ),
        (
            "LightGBM",
            LGBMRegressor(random_state=42),
            {
                "n_estimators": [50, 100, 200],
                "num_leaves": [31, 50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.7, 1.0],
            },
        ),
    ]

    # --- Train or tune each model ---
    for model_name, base_model, param_dist in models_with_params:
        if USE_OPTUNA and model_name == "XGBoost":
            print(f"\n🔮 Tuning {model_name} with Optuna...")
            best_model = optimize_with_optuna(X_train, y_train, n_trials=20)
            train_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_test,
                best_model,
                raw_df_path,
                model_name=f"{model_name}_Optuna",
            )

        elif USE_RANDOM_SEARCH:
            print(f"\n🎯 Tuning {model_name} with RandomizedSearchCV...")
            model = MultiOutputRegressor(base_model)

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions={
                    f"estimator__{k}": v for k, v in param_dist.items()
                },
                n_iter=10,
                scoring="neg_mean_squared_error",
                cv=3,
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            print(f"✅ Best params for {model_name}: {search.best_params_}")
            train_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_test,
                best_model,
                raw_df_path,
                model_name=f"{model_name}_Tuned",
            )

        else:
            print(f"\n🚀 Training {model_name} without tuning...")
            model = MultiOutputRegressor(base_model)
            model.fit(X_train, y_train)
            train_evaluate_model(
                X_train,
                y_train,
                X_test,
                y_test,
                model,
                raw_df_path,
                model_name=model_name,
            )


if __name__ == "__main__":
    main()
