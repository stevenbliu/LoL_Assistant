import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow  # optional, enable if needed

from sklearn.preprocessing import LabelEncoder

# ---- Configuration ----
DATA_DIR = "database/riot_data"
VERSION = "v3"
USE_MLFLOW = False  # Set True to enable MLflow logging


# ---- EDA: RAW DATA ----
def eda_raw(df, eda_dir):
    os.makedirs(eda_dir, exist_ok=True)
    print("\n📊 Running Raw Data EDA...")

    print("\n📐 Dataset shape:", df.shape)
    print("\n🧾 Column types:\n", df.dtypes)
    print("\n🔍 Missing values:\n", df.isnull().sum())

    # Missing values heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Value Heatmap (Raw Data)")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "missing_values_raw.png"))
    plt.close()

    # Histograms of numeric features
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols].hist(bins=20, figsize=(20, 15))
    plt.suptitle("Raw Feature Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "feature_distributions_raw.png"))
    plt.close()

    # Correlation heatmap
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Raw Data)")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "correlation_heatmap_raw.png"))
    plt.close()

    print(f"✅ Raw EDA plots saved to: {eda_dir}")


# ---- PREPROCESSING ----
def preprocess_data(raw_df):
    print("🔧 Starting preprocessing...")
    df = raw_df.copy()

    # Handle missing values
    thresh = len(df) * 0.1
    df = df.loc[:, df.isnull().sum() < thresh]

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))

    # Encode categorical
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Interaction features
    df["delta_gold"] = df["P1_currentGold"] - df["P2_currentGold"]
    df["delta_level"] = df["P1_level"] - df["P2_level"]
    df["delta_minionsKilled"] = df["P1_MinionsKilled"] - df["P2_MinionsKilled"]
    # df["distance_x"] = df["P1_X"] - df["P2_X"]
    # df["distance_y"] = df["P1_Y"] - df["P2_Y"]

    # Time-based features
    bins = [0, 600000, 1200000, np.inf]
    labels = ["early", "mid", "late"]
    df["game_phase"] = pd.cut(df["Timestamp"], bins=bins, labels=labels).cat.codes
    df["gold_per_minute_P1"] = df["P1_currentGold"] / (df["Timestamp"] / 60000 + 1e-5)
    df["gold_per_minute_P2"] = df["P2_currentGold"] / (df["Timestamp"] / 60000 + 1e-5)

    # Domain features
    df["P1_damage_efficiency"] = df["P1_totalDamageDone"] / (
        df["P1_totalDamageTaken"] + 1e-5
    )
    df["P2_damage_efficiency"] = df["P2_totalDamageDone"] / (
        df["P2_totalDamageTaken"] + 1e-5
    )
    df["P1_aggression"] = (df["P1_totalDamageDone"] + df["P1_currentGold"]) / (
        df["Timestamp"] + 1e-5
    )
    df["P2_aggression"] = (df["P2_totalDamageDone"] + df["P2_currentGold"]) / (
        df["Timestamp"] + 1e-5
    )

    # Downcast numeric to save memory
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, downcast="float")
    df["delta_level"] = df["delta_level"].astype("int8")
    df["game_phase"] = df["game_phase"].astype("int8")

    df = df.copy()  # defragment

    print(f"✅ Preprocessing complete. Final shape: {df.shape}")
    return df


# ---- EDA: PROCESSED DATA ----
def eda_processed(df, eda_dir, log_to_mlflow=False):
    print("\n📊 Processed Data Preview:")
    print(df.head(10))
    print("\n🧾 Summary Statistics:\n", df.describe().T)

    os.makedirs(eda_dir, exist_ok=True)

    # Save CSV summaries
    df.head(10).to_csv(os.path.join(eda_dir, "head_processed.csv"), index=False)
    df.describe().T.to_csv(os.path.join(eda_dir, "summary_stats_processed.csv"))
    df.dtypes.to_frame("dtype").to_csv(os.path.join(eda_dir, "dtypes_processed.csv"))
    missing = df.isnull().sum()
    if missing.any():
        missing[missing > 0].to_csv(
            os.path.join(eda_dir, "missing_values_processed.csv")
        )

    # Plot important features
    important_cols = [
        # "delta_gold",
        # "delta_level",
        # "distance_x",
        # "gold_per_minute_P1",
        # "P1_damage_efficiency",
        # "P1_aggression",
    ]
    for col in important_cols:
        print(f"\n📈 Plotting feature: {col}")
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            data = df[col].copy()

            # Clip data to 1st and 99th percentile to avoid extreme outliers
            lower = data.quantile(0.01)
            upper = data.quantile(0.99)
            data = data.clip(lower, upper)

            # Plot without KDE for lower memory usage
            sns.histplot(data, bins=50, kde=False)

            plt.title(f"Distribution of {col} (clipped)")
            plt.tight_layout()

            plot_path = os.path.join(eda_dir, f"feature_dist_{col}.png")
            plt.savefig(plot_path)
            plt.close()

            if log_to_mlflow:
                mlflow.log_artifact(plot_path)

    # Correlation heatmap
    print("\n📊 Plotting correlation heatmap...")
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Processed Data)")
    plt.tight_layout()
    corr_path = os.path.join(eda_dir, "correlation_heatmap_processed.png")
    plt.savefig(corr_path)
    plt.close()
    if log_to_mlflow:
        mlflow.log_artifact(corr_path)

    print(f"\n✅ Processed EDA plots and summaries saved to: {eda_dir}")


from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas as pd
import numpy as np


def visualize_feature_importances(feature_importances, top_n=20):
    save_dir = f"{DATA_DIR}/{VERSION}/eda/feature_selection"
    os.makedirs(save_dir, exist_ok=True)
    top_features = feature_importances.head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/feature_importances.png")
    # plt.show()

    print("\nInterpretation:")
    print(
        "- Features with higher importance values contribute more to predicting P1_X and P1_Y."
    )
    print(
        "- Look out for intuitive features related to position, gold, level, and damage metrics."
    )
    print(
        "- If surprising features rank high, consider their domain meaning or possible correlation."
    )


def select_features_multioutput(
    df, target_cols, feature_cols, top_n=20, random_state=42
):
    X = df[feature_cols]
    y = df[target_cols]

    # Fit multi-output RandomForestRegressor
    model = MultiOutputRegressor(RandomForestRegressor(random_state=random_state))
    model.fit(X, y)

    # Aggregate feature importances across outputs
    importances = np.mean(
        [est.feature_importances_ for est in model.estimators_], axis=0
    )

    feature_importances = pd.Series(importances, index=feature_cols)
    feature_importances = feature_importances.sort_values(ascending=False)

    print(f"\nTop {top_n} features by importance:")
    print(feature_importances.head(top_n))

    visualize_feature_importances(feature_importances, top_n)

    selected_features = feature_importances.head(top_n).index.tolist()

    return selected_features


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_ml_data(df, selected_features, target_cols, test_size=0.2, random_state=42):
    X = df[selected_features]
    y = df[target_cols]

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale numeric features (fit scaler on train, transform both)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set shape: {X_train_scaled.shape}, {y_train.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}, {y_test.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


TESTING_MODE = False  # Set to True for quick tests with smaller data


# ---- MAIN ----
def main():
    raw_df_path = f"{DATA_DIR}/{VERSION}/all_jungler_data.csv"
    raw_df = pd.read_csv(raw_df_path)
    if (
        TESTING_MODE
    ):  # You can set TESTING_MODE = True somewhere in your config or environment
        raw_df = raw_df.sample(n=1000, random_state=42).reset_index(drop=True)
    # Raw EDA
    eda_raw(raw_df, eda_dir=f"{DATA_DIR}/{VERSION}/eda/raw")

    # Preprocess
    processed_df = preprocess_data(raw_df)

    # Processed EDA
    eda_processed(
        processed_df,
        eda_dir=f"{DATA_DIR}/{VERSION}/eda/processed",
        log_to_mlflow=USE_MLFLOW,
    )

    target_cols = ["P2_X", "P2_Y"]
    all_features = processed_df.columns.drop(target_cols).tolist()

    selected_features = select_features_multioutput(
        processed_df, target_cols=target_cols, feature_cols=all_features, top_n=20
    )

    print(f"Selected features for modeling: {selected_features}")

    X_train, X_test, y_train, y_test, scaler = prepare_ml_data(
        processed_df, selected_features, target_cols
    )

    # processed_df = preprocess_for_training(raw_df)
    # print(raw_df.groupby("MatchId").size())

    # print(f"Processed data shape: {raw_df.head(20).to_dict(orient='records')}")
    # processed_df.to_csv(f"{DATA_DIR}/jungler_training_data.csv", index=False)
    # print(f"Processed data shape: {processed_df}")

    # exit()
    # X_train, X_test, y_train, y_test, encoders = prepare_ml_data(processed_df)

    # exit()
    # if VERBOSE:
    #     print_dataset_info(
    #         raw_df, processed_df, X_train, y_train, X_test, y_test, encoders
    #     )

    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from training.training import train_evaluate_model

    models = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=42)),
        (
            "HistGradientBoosting",
            HistGradientBoostingRegressor(
                max_iter=100,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_depth=10,
            ),
        ),
    ]
    from xgboost import XGBRegressor
    from sklearn.model_selection import RandomizedSearchCV
    from lightgbm import LGBMRegressor

    models_with_params = [
        # (
        #     "RandomForest",
        #     RandomForestRegressor(random_state=42),
        #     {
        #         "n_estimators": [100, 200, 300],
        #         "max_depth": [None, 10, 20],
        #         "min_samples_split": [2, 5, 10],
        #     },
        # ),
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
        #         tree_method="hist", objective="reg:squarederror", random_state=42
        #     ),
        #     {
        #         "n_estimators": [50, 100, 200],
        #         "max_depth": [3, 5, 7],
        #         "learning_rate": [0.01, 0.1, 0.2],
        #     },
        # ),
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

    # for model_name, model_instance in models:
    #     print(f"\n🚀 Training {model_name}...")
    #     train_evaluate_model(
    #         X_train,
    #         y_train,
    #         X_test,
    #         y_test,
    #         model_instance,
    #         raw_df_path,
    #         model_name=model_name,
    #     )

    for model_name, base_model, param_dist in models_with_params:
        print(f"\n🎯 Tuning {model_name} with RandomizedSearchCV...")

        # Wrap for multi-output
        model = MultiOutputRegressor(base_model)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions={f"estimator__{k}": v for k, v in param_dist.items()},
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


if __name__ == "__main__":
    main()
