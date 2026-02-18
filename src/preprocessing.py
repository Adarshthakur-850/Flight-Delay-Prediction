import pandas as pd
import numpy as np

def preprocess_data(df):
    print("Preprocessing data...")
    
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Rename columns based on actual dataset header (nickdcox/learn-airline-delays)
    rename_map = {
        'carrier': 'airline',
        'origin': 'origin_airport',
        'destination': 'destination_airport',
        'date': 'flight_date',
        'arr_delay': 'arrival_delay'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Handle missing target (arrival_delay)
    if 'arrival_delay' in df.columns:
        df = df.dropna(subset=['arrival_delay'])
    else:
         raise KeyError("Dataset missing 'arrival_delay' column")
            
    # Create target: is_delayed (delayed > 15 mins)
    df['is_delayed'] = (df['arrival_delay'] > 15).astype(int)
    
    # Date/Time parsing
    if 'flight_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['flight_date']):
        df['flight_date'] = pd.to_datetime(df['flight_date'])

    # Backfill/Proxy scheduled_departure if missing
    if 'scheduled_departure' not in df.columns and 'departure_time' in df.columns:
        # Assuming departure_time is HHMM float/int
        df['scheduled_departure'] = df['departure_time']
        
    # Drop rows with minimal missing essential features
    essential_cols = ['airline', 'origin_airport', 'destination_airport']
    # Check intersection to be safe
    existing_essential = [c for c in essential_cols if c in df.columns]
    df = df.dropna(subset=existing_essential)
    
    return df
