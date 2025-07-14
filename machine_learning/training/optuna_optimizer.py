import optuna
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    model = MultiOutputRegressor(XGBRegressor(objective="reg:squarederror", **params))
    score = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=3).mean()
    return score


def optimize_with_optuna(X, y, n_trials=20):
    study = optuna.create_study(direction="maximize")  # Because score is negative MSE
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    print("✅ Optuna best params:", study.best_params)

    best_params = study.best_params
    base_model = XGBRegressor(objective="reg:squarederror", **best_params)
    return MultiOutputRegressor(base_model)
