 ML Training Service
Cron job or nightly training loop

Loads historical data from DB

Trains a model to predict position over time (given partial observations)

Model type options:

Classical: Hidden Markov Model (HMM), Decision Trees, Bayesian Inference

ML: LSTM (for sequential pathing), XGBoost (faster training), or even transformer-based if you scale

Output format:

Input: {game state at t} → Output: {probable X,Y + confidence}

Trained model saved locally or into object storage for inference (e.g. models/latest_pathing.pt).