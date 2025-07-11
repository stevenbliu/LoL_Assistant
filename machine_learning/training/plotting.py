import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd


def plot_predictions(y_test, y_pred, output_path):
    target_names = y_test.columns.tolist()
    num_targets = len(target_names)
    num_cols = 2
    num_rows = int(np.ceil(num_targets / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 5 * num_rows))
    axes = np.atleast_1d(axes).flatten()

    print("\n📊 Prediction Metrics:")

    for i, name in enumerate(target_names):
        true_vals = y_test.iloc[:, i]
        pred_vals = y_pred[:, i]

        axes[i].scatter(true_vals, pred_vals, alpha=0.5)
        axes[i].plot([0, 1], [0, 1], "r--")
        axes[i].set_xlabel(f"True {name}")
        axes[i].set_ylabel(f"Predicted {name}")
        axes[i].set_title(f"{name} Prediction vs True")

        mae = mean_absolute_error(true_vals, pred_vals)
        print(f" - {name} MAE: {mae:.4f}")

    # Overall metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"\nOverall MSE: {mse:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall R² Score: {r2:.4f}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # clean up unused axes

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_residuals(y_test, y_pred, output_path):
    residuals = y_test.values - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals.flatten(), bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Residuals (All Outputs Combined)")
    fig.savefig(output_path)
    plt.close(fig)

    print("\n📉 Residual Analysis:")
    print("Residuals Mean:", np.mean(residuals))
    print("Residuals Std Dev:", np.std(residuals))


def plot_feature_importance(model, X_test, y_test, output_path):
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx = result.importances_mean.argsort()[::-1]
    sorted_features = X_test.columns[sorted_idx]
    sorted_importance = result.importances_mean[sorted_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(sorted_features, sorted_importance)
    ax.set_title("Permutation Feature Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print("\n📌 Feature Importances:")
    for name, importance in zip(sorted_features, sorted_importance):
        print(f" - {name}: {importance:.5f}")


def plot_learning_curve(model, X, y, output_path, scoring="neg_mean_squared_error"):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1,
    )
    train_errors = -train_scores.mean(axis=1)
    val_errors = -val_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors, label="Training Error")
    plt.plot(train_sizes, val_errors, label="Validation Error")
    plt.xlabel("Training Set Size")
    plt.ylabel("MSE")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
