# TODO:
- [~] Computer Vision
    - [x] Use PyQT to create an overlay that allows users to select parts of the screen to capture
    - [x] Use Tesseract-OCR to process selected parts as numbers
    - [x] Successfully able to get CS from top-right.
    - [~] Should also be able to get enemy/teammate CS via Scoreboard.
    - [ ] Potentially track appearance and locations of players via minimap
        - Very difficult
    - So we currently have real-time data for CS! This is great!

- [ ] ML
    - [ ] Main goal!
        - Predict pathing of enemy jungler! From their historical matches, we should know how they typically path. 
            - We can use similarity checks for team composition + other ML methods to determine it.
            - It would be key to use similarity checks at different time ranges of the game, as compositions can change mid-game.
        - [x] Pre-processing of raw data from Riot API Data
        - [x] SKLEARN model as a baseline
            - [~] Tuning ongoing
            - [x] Generate some metrics/plots
                - [x] Prediction vs True Scatter
                - [x] Residuals
                - [x] Feature Importance
        - [~] Current model achieves limited R² (~0.05), indicating room for improved feature engineering and model complexity
        - [~] Need to increase dataset size (~1000+ ranked jungle matches) for better model generalization

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

