from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor

from training.training import train_evaluate_model
from training.optuna_optimizer import optimize_with_optuna  # <- you define this
from training.data_prep import prepare_ml_data, prepare_ml_data_sequences
from training.feature_selection import select_features_multioutput
from training.preprocess import preprocess_data
from training.eda import eda_raw, eda_processed
import pandas as pd
import numpy as np
import os
from training.neural_net import train_lstm_model

from training.config import DATA_DIR, VERSION, USE_MLFLOW
import hashlib

# 🔧 Config flags
USE_RANDOM_SEARCH = False
USE_OPTUNA = False
TESTING_MODE = False
USE_NEURAL_NETWORKS = True  # Set to True if you want to use neural networks


def create_historical_sequences(
    df, feature_cols, target_cols, history_steps=3, timestep=60, cache_dir="cache"
):
    os.makedirs(cache_dir, exist_ok=True)

    # 🔐 Create hash based on selected features and history_steps
    key_string = ",".join(sorted(feature_cols)) + f"_{history_steps}_{timestep}"
    cache_key = hashlib.md5(key_string.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"seq_{cache_key}.feather")

    if os.path.exists(cache_path):
        print(f"📂 Loading cached sequences from: {cache_path}")
        sequence_df = pd.read_feather(cache_path)
        return sequence_df

    print("\n🧮 Creating historical sequences from scratch...", flush=True)

    all_rows = []
    group_cols = ["MatchId", "P2_PlayerId"]
    sort_cols = group_cols + ["Timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    id_time_cols = ["MatchId", "P2_PlayerId", "Timestamp"]
    hist_feature_cols = [col for col in feature_cols if col not in id_time_cols]
    grouped = df.groupby(group_cols)

    for (match_id, player_id), group in grouped:
        group = group.reset_index(drop=True)
        for i in range(history_steps, len(group) - 1):
            input_features = []
            for h in range(history_steps):
                row = group.iloc[i - history_steps + h][hist_feature_cols]
                input_features.extend(row)
            row_now = group.iloc[i][hist_feature_cols]
            input_features.extend(row_now)
            id_time_values = group.iloc[i][id_time_cols].tolist()
            input_features.extend(id_time_values)
            target_row = group.iloc[i + 1][target_cols]
            all_rows.append(input_features + target_row.tolist())

    input_feature_names = []
    for h in range(-history_steps, 0):
        for col in hist_feature_cols:
            input_feature_names.append(f"{col}_t{h * timestep}s")
    for col in hist_feature_cols:
        input_feature_names.append(f"{col}_t0s")
    input_feature_names.extend(id_time_cols)
    col_names = input_feature_names + target_cols

    sequence_df = pd.DataFrame(all_rows, columns=col_names)
    print(f"✅ Created historical sequences with shape: {sequence_df.shape}")

    # 🧊 Cache result
    sequence_df.to_feather(cache_path)
    print(f"💾 Cached to: {cache_path}")

    return sequence_df


def main():
    # --- Load and preprocess data ---
    raw_df_path = f"{DATA_DIR}/{VERSION}/all_jungler_data.csv"
    raw_df = pd.read_csv(raw_df_path)
    if TESTING_MODE:
        raw_df = raw_df.head(1000).reset_index(drop=True)

    # eda_raw(raw_df, eda_dir=f"{DATA_DIR}/{VERSION}/eda/raw")
    processed_df = preprocess_data(raw_df)
    # eda_processed(
    #     processed_df,
    #     eda_dir=f"{DATA_DIR}/{VERSION}/eda/processed",
    #     log_to_mlflow=USE_MLFLOW,
    # )

    target_cols = ["P2_X", "P2_Y"]
    all_features = processed_df.columns.drop(target_cols).tolist()
    selected_features = select_features_multioutput(
        processed_df, target_cols, all_features, top_n=100
    )
    keep_features = ["MatchId", "P2_PlayerId", "Timestamp"]
    selected_features = list(set(selected_features + keep_features))

    if USE_NEURAL_NETWORKS:
        # Create historical sequences first
        processed_df = create_historical_sequences(
            df=processed_df,
            feature_cols=selected_features,
            target_cols=target_cols,
            history_steps=3,
            timestep=60,
        )
        id_time_cols = ["MatchId", "P2_PlayerId", "Timestamp"]

        # Pass all features including id/time columns (so prepare_ml_data_sequences can drop them internally)
        X_train, X_test, y_train, y_test, scaler, target_scaler = (
            prepare_ml_data_sequences(
                processed_df,
                feature_cols=processed_df.columns.drop(target_cols).tolist(),
                target_cols=target_cols,
                id_time_cols=id_time_cols,
            )
        )
    else:
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
        # (
        #     "HistGradientBoosting",
        #     HistGradientBoostingRegressor(random_state=42),
        #     {
        #         "max_iter": [100, 200],
        #         "max_depth": [5, 10],
        #         "learning_rate": [0.05, 0.1],
        #     },
        # ),
        # (
        #     "XGBoost",
        #     XGBRegressor(
        #         objective="reg:squarederror", tree_method="hist", random_state=42
        #     ),
        #     {
        #         "n_estimators": [50, 100, 200],
        #         "max_depth": [3, 5, 7],
        #         "learning_rate": [0.01, 0.1, 0.2],
        #     },
        # ),
        # (
        #     "LightGBM",
        #     LGBMRegressor(random_state=42),
        #     {
        #         "n_estimators": [50, 100, 200],
        #         "num_leaves": [31, 50, 100],
        #         "learning_rate": [0.01, 0.1, 0.2],
        #         "subsample": [0.7, 1.0],
        #     },
        # ),
    ]

    # --- Train or tune each model ---
    if USE_NEURAL_NETWORKS:
        print("\n🔁 Training Neural Network Model...")
        from training.neural_net import train_lstm_model

        print("Shapes before training LSTM:")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        print("X_test:", X_test.shape)
        print("y_test:", y_test.shape)

        train_lstm_model(
            X_train, y_train, X_test, y_test, target_scaler, model_name="LSTM_P2_XY"
        )
        return  # Skip traditional models

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
