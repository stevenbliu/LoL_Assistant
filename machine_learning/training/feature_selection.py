import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
from training.config import DATA_DIR, VERSION, USE_MLFLOW
from lightgbm import LGBMRegressor


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
    print("\n🔍 Selecting top features for multi-output regression...", flush=True)
    X = df[feature_cols]
    y = df[target_cols]

    # Fit multi-output RandomForestRegressor
    # model = MultiOutputRegressor(RandomForestRegressor(random_state=random_state))
    model = MultiOutputRegressor(
        LGBMRegressor(random_state=random_state, n_estimators=100)
    )
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
