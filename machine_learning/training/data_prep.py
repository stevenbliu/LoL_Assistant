import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_ml_data(
    df: pd.DataFrame,
    selected_features: List[str],
    target_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Prepares the dataset for machine learning training and testing.

    Parameters:
    - df: Raw input dataframe.
    - selected_features: List of column names to use as features.
    - target_cols: List of column names to use as targets.
    - test_size: Fraction of data to use for testing.
    - random_state: Random seed for reproducibility.

    Returns:
    - X_train_scaled: Numpy array of scaled training features.
    - X_test_scaled: Numpy array of scaled testing features.
    - y_train: Training targets (DataFrame).
    - y_test: Testing targets (DataFrame).
    - scaler: StandardScaler fitted on training features.
    """
    minutes = 5
    max_timestep = 60 * minutes  # Define the maximum timestep for early-game data
    print(
        f"\nPreparing data for machine learning. timestep: {max_timestep} test_size:{test_size}",
        flush=True,
    )

    early_df = df[df["Timestamp"] <= max_timestep].copy()

    X = early_df[selected_features]
    y = early_df[target_cols]

    # Match-level split
    match_ids = X["MatchId"].unique()
    match_train_ids, match_test_ids = train_test_split(
        match_ids, test_size=test_size, random_state=random_state
    )

    train_mask = X["MatchId"].isin(match_train_ids)
    test_mask = X["MatchId"].isin(match_test_ids)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Filtered for early timesteps (<= {max_timestep})")
    print(f"Training set shape: {X_train_scaled.shape}, {y_train.shape}")
    print(f"Testing set shape: {X_test_scaled.shape}, {y_test.shape}")
    print("✅ Data preparation complete.", flush=True)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def prepare_ml_data_sequences(
    df,
    feature_cols: List[str],
    target_cols,
    history_steps=3,
    id_time_cols=["MatchId", "P2_PlayerId", "Timestamp"],
    test_size=0.2,
    random_state=42,
):
    print(
        "\nPreparing data for machine learning with historical sequences...", flush=True
    )
    # Drop ID/time cols for input features
    feature_cols = [col for col in feature_cols if col not in id_time_cols]

    X = df[feature_cols]
    y = df[target_cols]

    # Split by MatchId (or grouping key)
    match_ids = df["MatchId"].unique()
    train_ids, test_ids = train_test_split(
        match_ids, test_size=test_size, random_state=random_state
    )

    train_mask = df["MatchId"].isin(train_ids)
    test_mask = df["MatchId"].isin(test_ids)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    total_timesteps = history_steps + 1
    features_per_sample = X_train_scaled.shape[1]
    if features_per_sample % total_timesteps != 0:
        raise ValueError(
            f"Features {features_per_sample} not divisible by timesteps {total_timesteps}"
        )

    input_dim = features_per_sample // total_timesteps

    num_train_samples = X_train_scaled.shape[0]
    num_test_samples = X_test_scaled.shape[0]

    # RESHAPE PER SAMPLE (flattened features -> sequence)
    X_train_reshaped = X_train_scaled.reshape(
        num_train_samples, total_timesteps, input_dim
    )
    X_test_reshaped = X_test_scaled.reshape(
        num_test_samples, total_timesteps, input_dim
    )
    print(f"X_train_reshaped shape: {X_train_reshaped.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Finished preparing sequences with {total_timesteps} timesteps.", flush=True)
    return X_train_reshaped, X_test_reshaped, y_train.values, y_test.values, scaler
