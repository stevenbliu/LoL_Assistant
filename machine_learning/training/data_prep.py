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
