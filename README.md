#League of Legends Jungler Pathing Prediction – ML & Computer Vision Project
# Stack: Python, Scikit-learn, PyTorch, PyTesseract, PyQT, Riot Games API, Docker, Matplotlib, FAISS
Developed an ML system to predict enemy jungler pathing in League of Legends using historical match data and real-time visual inputs.
## 🧠 Project Summary
Predict jungler movement on the League of Legends map over time by analyzing match timelines. The goal is to forecast each jungler’s next position coordinates (x_next, y_next) using machine learning.

Computer Vision Overlay: Built a PyQT interface to capture in-game screen regions; used Tesseract-OCR to extract creep score (CS) in real time, enabling data collection from matches and replays.

ML Pipeline: Trained a multi-target regression model to predict enemy jungler X/Y positions from engineered features (delta gold, XP, aggression metrics); incorporated timestamped spatial data and match phase classification.

Data Engineering: Processed 1000+ ranked jungle matches from the Riot API with resumable ETL pipeline; integrated game metadata and time-series position tracking; built automated deduplication and progress tracking tools.

Feature Design: Engineered domain-specific features like jungle aggression, gold efficiency, and temporal bins (early/mid/late); designed a two-stage modeling approach for incorporating player-specific embeddings.

Performance & Evaluation: Visualized prediction accuracy via scatter/residual plots; initial R² ~0.05–0.10 indicated baseline model learning patterns but highlighted need for temporal models (LSTM/Transformer) and larger datasets.

Roadmap (Ongoing):

Reverse-engineering .rofl replay files for finer granularity

Exploring real-time prediction integration via computer vision

Scaling dataset and refining model with sequence-based architectures


🏗️ Pipeline Overview
# 1. Data Collection
Input: List of summoners in found_summoners.csv

Uses Riot API to fetch:

Ranked Solo matches (queue=420)

Match info and timelines

Extracts jungler-specific data per match via extract_jungler_data()

Saves each match as a CSV in match_data/ folder

Consolidates all matches into all_jungler_data.csv

Supports resumable runs by tracking processed matches

# 2. Data Structure
Each row represents a time step in a match

Includes:

      Match metadata (e.g., Second, MatchId)
      
      Player-specific features (P1_Champion, P1_X, P1_Y, P1_JungleMinionsKilled, etc.)
      
      Opponent jungler features (P2_*)
      
      Target labels: P1_x_next, P1_y_next, P2_x_next, P2_y_next
      
      Derived stats like distance between junglers
      
      Current data: ~4,500 rows (~90–100 matches)

New matches append to dataset without duplication

## 🤖 Modeling Approach
Regression models predict next map coordinates every 60 seconds

Evaluation metrics: MAE, MSE, RMSE, R²

Results tracked and visualized with MLflow

## 📉 Latest Performance Highlights
Test R² ≈ 0.056 (indicates limited predictive power)

RMSE ≈ 0.21 (decent spatial error, room to improve)

Important features: P1_JungleMinionsKilled, P2_JungleMinionsKilled, P2_Y, P1_Y, Second

## 🚧 Limitations & Challenges
Low R² suggests model only marginally outperforms baseline

Some features have zero or negative importance

Current dataset (~300 matches) likely too small for complex patterns

## ✅ Next Steps
Data
- Expand dataset to 1,000+ ranked jungle matches

- Handle Riot API rate limits during data collection

- Contact for existing dataset if interested

Feature Engineering
- Add movement deltas (dx/dy), map zone features, path memory

Incorporate map objectives (camp proximity, dragon timers)

Modeling
- Explore sequence models (e.g., LSTM, Transformer) for temporal dependencies

- Improve model interpretability and validation

Analysis & Visualization
- Compare actual vs predicted positions visually

- Analyze performance by champion, patch version, role matchups

## 📊 Pipeline Flow Summary


Summoner List CSV 
      ->
Riot API Calls 
      ->
Data Extraction (extract_jungler_data) 
      ->
Per-match CSV files + processed tracking 
      ->
Concatenate to full dataset CSV (all_jungler_data.csv) 
      ->
ML Model Training & Evaluation 
      ->
Feature Importance & Residual Analysis
