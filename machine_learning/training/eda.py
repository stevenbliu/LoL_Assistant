import os
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow


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


# ---- EDA: PROCESSED DATA ----
def eda_processed(df, eda_dir, log_to_mlflow=False):
    print("\n📊 Processed Data Preview:")
    print(df.head(3))
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
