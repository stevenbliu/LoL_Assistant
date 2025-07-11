import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance


def plot_predictions(y_test, y_pred, output_path):
    coords = ["P1_x_next", "P1_y_next", "P2_x_next", "P2_y_next"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, coord in enumerate(coords):
        axes[i].scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5)
        axes[i].plot([0, 1], [0, 1], "r--")
        axes[i].set_xlabel(f"True {coord}")
        axes[i].set_ylabel(f"Predicted {coord}")
        axes[i].set_title(f"{coord} Prediction vs True")

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_residuals(y_test, y_pred, output_path):
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals.flatten(), bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Residuals (All Coordinates)")
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_importance(model, X_test, y_test, output_path):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx = result.importances_mean.argsort()[::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(X_test.columns[sorted_idx], result.importances_mean[sorted_idx])
    ax.set_title("Permutation Feature Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
