import pandas as pd
from preprocessing.preprocessing import (
    preprocess_for_training,
    prepare_ml_data,
    print_dataset_info,
)
from training.training import train_evaluate_baseline_model

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

    model, mse_per_output, avg_mse = train_evaluate_baseline_model(
        X_train, y_train, X_test, y_test
    )


if __name__ == "__main__":
    main()
