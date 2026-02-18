import pandas as pd
import os
import requests

# Valid sample URL (2018 Delays)
SAMPLE_DATA_URL = "https://raw.githubusercontent.com/nickdcox/learn-airline-delays/main/delays_2018.csv"

DATA_PATH = os.path.join("data", "flights.csv")

def load_data():
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(DATA_PATH):
        print(f"Downloading sample flight data from {SAMPLE_DATA_URL}...")
        try:
            # Stream download to avoid memory issues if large
            response = requests.get(SAMPLE_DATA_URL, stream=True)
            response.raise_for_status()
            with open(DATA_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise
            
    print("Loading flight data...")
    # Load first 200k rows for efficiency as requested
    df = pd.read_csv(DATA_PATH, nrows=200000)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df
