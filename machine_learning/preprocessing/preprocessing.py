import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_for_training(df):
    # Sort for consistent order
    df = df.sort_values(by=["MatchId", "Second"]).reset_index(drop=True)

    # Drop rows where either jungler's position is missing
    df = df.dropna(subset=["P1_X", "P1_Y", "P2_X", "P2_Y"]).copy()

    # Create next positions as targets by shifting positions up by one row within each match
    df["P1_x_next"] = df.groupby("MatchId")["P1_X"].shift(-1)
    df["P1_y_next"] = df.groupby("MatchId")["P1_Y"].shift(-1)
    df["P2_x_next"] = df.groupby("MatchId")["P2_X"].shift(-1)
    df["P2_y_next"] = df.groupby("MatchId")["P2_Y"].shift(-1)

    # Remove last row of each match (no next position)
    df = df.dropna(subset=["P1_x_next", "P1_y_next", "P2_x_next", "P2_y_next"])

    # Normalize position columns (map to [0,1], assuming max map coordinate ~15000)
    pos_cols = [
        "P1_X",
        "P1_Y",
        "P1_x_next",
        "P1_y_next",
        "P2_X",
        "P2_Y",
        "P2_x_next",
        "P2_y_next",
    ]

    for col in pos_cols:
        df[col] = df[col] / 15000.0

    # Optionally normalize distance as well (max possible distance ~21213)
    # df["Distance_Between_Junglers"] /= 21213

    return df


def prepare_ml_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["P1_Player", "P2_Player", "P2_Position"])

    # Encode categorical columns
    cat_cols = ["P1_Champion", "P2_Champion", "P1_Position"]
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Convert P2_Team to boolean (0/1)
    df["P2_Team"] = df["P2_Team"].map({100: 0, 200: 1})  # or .astype(bool)
    df["P1_Team"] = df["P1_Team"].map({100: 0, 200: 1})  # or .astype(bool)

    # Define features and targets
    target_cols = ["P1_x_next", "P1_y_next", "P2_x_next", "P2_y_next"]
    feature_cols = [c for c in df.columns if c not in target_cols + ["MatchId"]]

    # Avoid data leakage by splitting on MatchId
    match_ids = df["MatchId"].unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)

    train_df = df[df["MatchId"].isin(train_ids)]
    test_df = df[df["MatchId"].isin(test_ids)]

    X_train = train_df[feature_cols]
    y_train = train_df[target_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[target_cols]

    return X_train, X_test, y_train, y_test, label_encoders


def print_dataset_info(
    raw_df, processed_df, X_train, y_train, X_test, y_test, encoders
):
    print("Raw data loaded:")
    print(raw_df.head())
    print(raw_df.info())
    print(f"Number of rows: {len(raw_df)}")
    print(f"Columns: {raw_df.columns.tolist()}\n")

    print("Processed data preview:")
    print(processed_df.head())
    print(processed_df.info())
    print(f"Number of rows after processing: {len(processed_df)}")

    missing_rows = processed_df[processed_df.isnull().any(axis=1)]
    if not missing_rows.empty:
        print("\nRows with missing values after processing:")
        print(missing_rows)
    else:
        print("\nNo missing values detected after processing.")

    print("\nSample features (X_train):")
    print(X_train.head())

    print("\nSample targets (y_train):")
    print(y_train.head())

    print(f"\nNumber of training samples: {len(X_train)}")
    print(f"Number of testing samples: {len(X_test)}")

    print("\nLabel encoders for categorical columns:")
    for col, le in encoders.items():
        print(f"  {col}: classes = {list(le.classes_)}")
