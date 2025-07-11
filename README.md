**🧠 Project Summary: League of Legends Jungler Behavior Prediction**
🎯 Goal
To predict jungler movement over time by analyzing match timelines, using machine learning to forecast their next position on the map (x_next, y_next).

🏗️ Pipeline Overview
✅ 1. Data Collection
Input: List of summoners in found_summoners.csv

Uses Riot API to fetch:

Ranked Solo matches (queue=420)

Match data and timelines

Extracts per-match jungler data using extract_jungler_data()

Each match saved as a CSV file under match_data/

Final dataset is consolidated into all_jungler_data.csv

Supports resumable execution via the processed flag in summoner CSV

✅ 2. Data Structure
Each row (time step) contains:

Match metadata (e.g. Second, MatchId)

Player-specific features:

P1_Champion, P1_X, P1_Y, P1_JungleMinionsKilled, etc.

P2_* (opposing jungler)

Labels (target values): P1_x_next, P1_y_next, P2_x_next, P2_y_next

Derived stats like Distance_Between_Junglers

Total data so far:
~4,500 rows (~90–100 matches), with updates appending new match data to avoid duplicates

🤖 Modeling
Regression model predicts each jungler's next map position every 60 seconds

Evaluated using:

MAE, MSE, RMSE, R²

Logged using MLflow

📉 Latest Metrics:
Test R² = 0.056 → indicates limited predictive power

RMSE ≈ 0.21 → decent spatial error but room for improvement

Top features:

P1_JungleMinionsKilled, P2_JungleMinionsKilled, P2_Y, P1_Y, Second

🚧 Limitations & Challenges
Low R² suggests the model is only marginally better than a baseline

Some features have zero or negative importance

~100 matches = not enough to learn meaningful behavior patterns

✅ Next Steps 
🔁 Data
Increase to 1,000+ ranked jungle matches (Games are limited by Riot's rate limiters. Feel free to reach out to me if you want what I have so far.)

Improve feature engineering (e.g. dx/dy, map zones, path memory)

📈 Model
Try LSTM or sequence-based models (movement is not memoryless)

Add map-objective features (camp proximity, dragon timers)

📊 Analysis
Visualize actual vs predicted positions

Analyze differences by champ, patch, or role matchup


**Sample high-level pipeline flow:**

Summoner List CSV
       ↓
Riot API Calls (Ranked Match IDs, Match Info, Timeline)
       ↓
Data Extraction (extract_jungler_data)
       ↓
Match CSV files (one per match) + processed tracking
       ↓
Concatenate to full dataset CSV (all_jungler_data.csv)
       ↓
ML Model Training & Evaluation (Predictions, Metrics, Logging)
       ↓
Feature Importance & Residual Analysis


![CI](https://github.com/stevenbliu/LoL_assistant/actions/workflows/ci.yml/badge.svg)
