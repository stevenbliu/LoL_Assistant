import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
