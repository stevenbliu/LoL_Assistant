Lightweight Python Flask/FastAPI service

Loads pre-trained model

Accepts real-time state from Live Client Monitor

Returns predicted jungle location map or X/Y positions with probabilities

Example response:

{
  "prediction_time": "3:20",
  "likely_positions": [
    {"x": 6000, "y": 10000, "confidence": 0.82},
    {"x": 4800, "y": 9600, "confidence": 0.12}
  ]
}

Models to try:
Model	Why try it?	What to expect	Parameter intuition
Linear Regression (Baseline)	Easy, fast, interpretable	Will underfit if relationships are nonlinear	No parameters besides regularization if Ridge/Lasso used
Random Forest Regressor	Handles nonlinearities, robust to overfitting on smaller data	May generalize better than complex boosting on small data	n_estimators (trees), max_depth (complexity control), min_samples_leaf (leaf size)
HistGradientBoostingRegressor	Powerful, fast, good with tabular data, early stopping	Good if enough data, risk of overfitting small data	max_iter, max_depth, learning_rate, early stopping
CatBoost / LightGBM	Can handle categorical features natively, generally strong	Might need tuning, better for bigger data	Similar params to HGBR, plus categorical features
Neural Network (MLP)	Potentially powerful but requires lots of data and tuning	Likely to overfit small data, needs regularization	layers, neurons, learning_rate, batch_size, dropout