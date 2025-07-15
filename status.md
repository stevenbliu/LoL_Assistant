# TODO:
- [~] Computer Vision
    - [x] Use PyQT to create an overlay that allows users to select parts of the screen to capture
    - [x] Use Tesseract-OCR to process selected parts as numbers
    - [x] Successfully able to get CS from top-right.
    - [~] Should also be able to get enemy/teammate CS via Scoreboard.
    - [ ] Potentially track appearance and locations of players via minimap
        - Very difficult
    - So we currently have real-time data for CS! This is great!

- [ ] **ML Project: Predict Enemy Jungler Pathing**
    - [ ] **🎯 Main Goal**
        - Predict the **enemy jungler's pathing** using their historical match data.
            - Use **similarity checks** on team compositions and other contextual factors to guide prediction.
            - Evaluate **similarity over time** — early, mid, late — since compositions and objectives shift.
    - [x] **📦 Preprocessing**
        - Raw data from Riot API cleaned, engineered, and encoded.
    - [x] **📈 Baseline Model (Scikit-learn)**
        - [x] Multi-target regression (predicting `P2_X`, `P2_Y`)
        - [~] Hyperparameter tuning in progress
        - [x] Basic metrics & visualizations:
            - [x] ✅ True vs Predicted Scatter Plot
            - [x] ✅ Residuals Plot
            - [x] ✅ Feature Importance Bar Chart
    - [~] **📉 Model Performance**
        - Current R² ≈ **0.05–0.10**, shows model is learning but **not generalizing well yet**
        - Indicates a need for:
            - Better feature engineering
            - More advanced models (e.g., LightGBM, XGBoost, or deep learning)
            - Possibly temporal models (sequence models like RNNs or Transformers)
    - [x] **📊 Data Expansion**
        - Goal: Use **1000+ ranked jungle matches**
        - Larger dataset expected to improve generalization and pattern capture
        - Incorporate player-specific features (e.g. win rate, champ mastery) into predictions
            1. Use a two-stage model:
                Stage 1: Predict outcome with general match features
                Stage 2: Recalibrate using player-specific features when available
            2. Train a unified model that handles missing player data, and substitutes real values once available

            Cache precomputed player embeddings to enable fast feature injection in real-time
    - [ ] **🧠 Feature Set Overview**
        - [x] **Raw Features (from Riot API)**
            - `P1_X`, `P1_Y`, `P2_X`, `P2_Y`: Absolute map positions of both junglers
            - `P1_currentGold`, `P2_currentGold`
            - `P1_level`, `P2_level`
            - `P1_MinionsKilled`, `P2_MinionsKilled`
            - `P1_totalDamageDone`, `P1_totalDamageTaken`, etc.
            - `Timestamp`: Current match time in milliseconds

        - [x] **Engineered Features**
            - 📊 **Delta Metrics**
                - `delta_gold` = `P1_currentGold` - `P2_currentGold`
                - `delta_level` = `P1_level` - `P2_level`
                - `delta_minionsKilled` = `P1_MinionsKilled` - `P2_MinionsKilled`
            - 📏 **Relative Positioning**
                - ⚠️ `distance_x` = `P1_X` - `P2_X` *(REMOVED — can't use enemy positions during inference)*
                - ⚠️ `distance_y` = `P1_Y` - `P2_Y` *(REMOVED — same reason as above)*
            - ⏱ **Temporal Features**
                - `game_phase`: early, mid, late (based on time bins)
                - `gold_per_minute_P1`, `gold_per_minute_P2`
            - 💥 **Domain-Aware Features**
                - `P1_damage_efficiency` = damage dealt / (damage taken + ε)
                - `P2_damage_efficiency` = same for enemy
                - `P1_aggression` = (damage + gold) / time
                - `P2_aggression` = same

        - [x] **Label / Target Columns**
            - Currently predicting: `P2_X`, `P2_Y`
                - Intended to model **enemy jungler position over time**

        - [~] **Planned / Future Features**
            - Team composition similarity (e.g., % AD/AP, CC, scaling)
            - Lane states (pushed lanes, roamed lanes)
            - Ally/enemy ward coverage (from vision events or replays)
            - Objective timers (next dragon/baron, tower HP)
            - Augment with Map Context
                - Add jungle camp state: which camps are up? (from last clear)
                - Add objective timers: dragon/baron spawn, tower HP
            - Team Composition
                - Add laner data, lane status, vision info etc.


    - [x] **Removed / Invalid Features**
        - ❌ `distance_x`, `distance_y` — requires knowing enemy location at prediction time
        - ⚠️ Features directly derived from future or unseen info must be filtered

- [ ] Riot API Data
    - [~] Determine where to get historical match data.
        - OP.gg considered for lane score data
        - [x] Integrated with Riot's Dev API
            - Successfully collected data with x,y coordinates of players and jungle minion stats
            - Match timeline granularity limited to 60 seconds per timestep — may limit model accuracy
        - [~] Investigating replay data via Live Client API for finer granularity
        - [x] Current system fetches ranked solo queue matches (queue=420) and extracts jungler timeline data
        - [~] Implemented resumable data pipeline with progress tracking to gather large datasets over time

- [ ] Live Client
    - [x] Successfully integrated for real games, but data is too limited. No useful detailed data can be collected during live matches via Live Client API (failed)
        - So, to collect real-game data, we implemented a Computer Vision program to capture data from the screen.
    - [~] Working on integrating with replays.
        - [~] Developing a system to download and parse replay files (.rofl) for finer data granularity
        - Riot only provides replay files for games the user has played — limits scale of data collection
        - Reverse engineering .rofl files is very difficult and time-consuming, so not currently feasible
        - May fallback to Computer Vision approaches for more granular data extraction from replays

# NOTES:
- Current ML pipeline is functional but model performance is limited by dataset size and feature set.
- Need to prioritize collecting more ranked jungle matches (~1000+) to improve training data volume.
- Feature engineering improvements and potentially sequence models (LSTM) should be explored to better capture jungler movement patterns.
- Data appending implemented to avoid overwriting; duplicates filtered on MatchId.
- Progress tracking enabled for summoner processing, allowing incremental dataset growth.

# Project Overview

This project aims to analyze and predict the pathing of enemy junglers in League of Legends using historical match data. By leveraging Riot’s API, computer vision, and machine learning techniques, we seek to build a predictive model that forecasts jungler positions and movements during matches. The ultimate goal is to provide strategic insights to players and enhance real-time game decision making.

---

# Roadmap

### Phase 1: Data Collection & Preprocessing
- Integrate Riot API to collect ranked match data focused on jungle role
- Develop and refine computer vision tools to extract real-time in-game data (CS, player positions)
- Implement data pipeline with resumable processing and progress tracking

### Phase 2: Exploratory Data Analysis & Feature Engineering
- Analyze movement and jungle path patterns from match timelines
- Engineer new features such as movement deltas, camp proximity, and time-relative stats
- Assess data quality and address granularity limitations

### Phase 3: Baseline Modeling
- Train initial ML models (e.g. sklearn regressors) to predict jungler next position
- Evaluate performance using MAE, RMSE, R², and residual analysis
- Visualize feature importance and prediction errors

### Phase 4: Model Improvement & Scaling
- Expand dataset to 1000+ ranked jungle matches for improved generalization
- Experiment with sequence models (LSTM, RNN) to capture temporal dependencies
- Incorporate team composition similarity and mid-game dynamics

### Phase 5: Integration & Real-Time Application
- Explore integration with live client and replay data for finer granularity
- Enhance computer vision pipeline for real-time prediction support
- Deploy model for strategic assistance in live matches

---

# Current Status

- Data collection pipeline operational with Riot API and computer vision components
- Initial ML model developed and evaluated, showing room for improvement
- Dataset size growing incrementally with resumable processing and duplication checks
- Replay data parsing remains a challenge due to file access and complexity

---

