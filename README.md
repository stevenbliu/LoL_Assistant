*League of Legends Jungler Behavior Prediction*
🧠 Project Summary
Predict jungler movement on the League of Legends map over time by analyzing match timelines. The goal is to forecast each jungler’s next position coordinates (x_next, y_next) using machine learning.

🏗️ Pipeline Overview
1. Data Collection
Input: List of summoners in found_summoners.csv

Uses Riot API to fetch:

Ranked Solo matches (queue=420)

Match info and timelines

Extracts jungler-specific data per match via extract_jungler_data()

Saves each match as a CSV in match_data/ folder

Consolidates all matches into all_jungler_data.csv

Supports resumable runs by tracking processed matches

2. Data Structure
Each row represents a time step in a match

Includes:

Match metadata (e.g., Second, MatchId)

Player-specific features (P1_Champion, P1_X, P1_Y, P1_JungleMinionsKilled, etc.)

Opponent jungler features (P2_*)

Target labels: P1_x_next, P1_y_next, P2_x_next, P2_y_next

Derived stats like distance between junglers

Current data: ~4,500 rows (~90–100 matches)

New matches append to dataset without duplication

🤖 Modeling Approach
Regression models predict next map coordinates every 60 seconds

Evaluation metrics: MAE, MSE, RMSE, R²

Results tracked and visualized with MLflow

📉 Latest Performance Highlights
Test R² ≈ 0.056 (indicates limited predictive power)

RMSE ≈ 0.21 (decent spatial error, room to improve)

Important features: P1_JungleMinionsKilled, P2_JungleMinionsKilled, P2_Y, P1_Y, Second

🚧 Limitations & Challenges
Low R² suggests model only marginally outperforms baseline

Some features have zero or negative importance

Current dataset (~300 matches) likely too small for complex patterns

✅ Next Steps
Data
Expand dataset to 1,000+ ranked jungle matches

Handle Riot API rate limits during data collection

Contact for existing dataset if interested

Feature Engineering
Add movement deltas (dx/dy), map zone features, path memory

Incorporate map objectives (camp proximity, dragon timers)

Modeling
Explore sequence models (e.g., LSTM, Transformer) for temporal dependencies

Improve model interpretability and validation

Analysis & Visualization
Compare actual vs predicted positions visually

Analyze performance by champion, patch version, role matchups

📊 Pipeline Flow Summary
sql
Copy
Edit
Summoner List CSV 
      ↓
Riot API Calls 
      ↓
Data Extraction (extract_jungler_data) 
      ↓
Per-match CSV files + processed tracking 
      ↓
Concatenate to full dataset CSV (all_jungler_data.csv) 
      ↓
ML Model Training & Evaluation 
      ↓
Feature Importance & Residual Analysis