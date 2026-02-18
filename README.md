<<<<<<< HEAD
# Flight Delay Prediction using Airline Operational Data

## Overview
A production-quality ML system to predict flight delays based on airline, airport, schedule, and weather features.

## Structure
- `src/`: Core logic modules
- `data/`: Dataset storage
- `models/`: Saved ML models
- `plots/`: Generated visualizations

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Place your `flights.csv` in the `data/` directory. (Recommended: 2015 Flight Delays and Cancellations from Kaggle)
   - Note: The `data_loader.py` will attempt to download a sample if no file is found, but the full dataset is recommended for better results.
3. Run the pipeline: `python main.py`

## Features
- **Target**: `is_delayed` (Arrival Delay > 15 minutes)
- **Models**: Logistic Regression, Random Forest, XGBoost
=======
# Flight-Delay-Prediction
ml project
>>>>>>> ba6318cd46120d88b746b23f617f63f036e83d76
