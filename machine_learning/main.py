import pandas as pd
from preprocessing.preprocessing import (
    preprocess_for_training,
    prepare_ml_data,
    print_dataset_info,
)
from training.training import train_evaluate_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

DATA_DIR = "database/riot_data"

VERBOSE = False  # Set to False to disable detailed output


def main():
    raw_df = pd.read_csv(f"{DATA_DIR}/all_jungler_data.csv")
    processed_df = preprocess_for_training(raw_df)
    print(raw_df.groupby("MatchId").size())

    # print(f"Processed data shape: {raw_df.head(20).to_dict(orient='records')}")
    # processed_df.to_csv(f"{DATA_DIR}/jungler_training_data.csv", index=False)
    print(f"Processed data shape: {processed_df}")

    # exit()
    X_train, X_test, y_train, y_test, encoders = prepare_ml_data(processed_df)
    print(X_train, y_train)
    # exit()
    if VERBOSE:
        print_dataset_info(
            raw_df, processed_df, X_train, y_train, X_test, y_test, encoders
        )

    # Example with Linear Regression
    lr_model = LinearRegression()
    train_evaluate_model(
        X_train, y_train, X_test, y_test, lr_model, model_name="LinearRegression"
    )

    # Example with Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    train_evaluate_model(
        X_train, y_train, X_test, y_test, rf_model, model_name="RandomForest"
    )

    # Example with HistGradientBoosting
    hgb_model = HistGradientBoostingRegressor(
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_depth=10,
    )
    train_evaluate_model(
        X_train, y_train, X_test, y_test, hgb_model, model_name="HistGradientBoosting"
    )


if __name__ == "__main__":
    main()
