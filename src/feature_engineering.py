import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def engineer_features(df):
    print("Engineering features...")
    
    # Use copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    # 1. Temporal Features
    # Create valid hour from scheduled_departure (format HHMM or HMM, need to handle)
    # lhbelfanti/flights seems to have integer HHMM maybe?
    # Let's assume standard int format like 1730 for 5:30 PM.
    if 'scheduled_departure' in df.columns:
        # Handle cases where it might be float or NA
        df['scheduled_departure'] = df['scheduled_departure'].fillna(0).astype(int)
        df['dep_hour'] = (df['scheduled_departure'] // 100).clip(0, 23)
        df['dep_min'] = df['scheduled_departure'] % 100
    
    if 'scheduled_arrival' in df.columns:
        df['scheduled_arrival'] = df['scheduled_arrival'].fillna(0).astype(int)
        df['arr_hour'] = (df['scheduled_arrival'] // 100).clip(0, 23)

    if 'month' not in df.columns and 'flight_date' in df.columns:
        df['month'] = df['flight_date'].dt.month
        df['day_of_week'] = df['flight_date'].dt.dayofweek
    elif 'day_of_week' not in df.columns and 'month' in df.columns:
        # Just use what we have. 
        pass
        
    # 2. Historical Delay Features (Target Encoding-ish, but using past averages requires temporal split logic)
    # For simplicity in this project (and avoid leakage), we'll skip complex historical avg per row
    # and use simple Label Encoding for categories.
    
    # 3. Encode Categoricals
    le_airline = LabelEncoder()
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()
    
    # Normalize string cols
    for col in ['airline', 'origin_airport', 'destination_airport']:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    if 'airline' in df.columns:
        df['airline_encoded'] = le_airline.fit_transform(df['airline'])
    
    if 'origin_airport' in df.columns:
        # Handle high cardinality: maybe hash or top-k? 
        # For simplicity, label encode all
        df['origin_airport_encoded'] = le_origin.fit_transform(df['origin_airport'])
        
    if 'destination_airport' in df.columns:
        df['dest_airport_encoded'] = le_dest.fit_transform(df['destination_airport'])
        
    # Select feature columns
    feature_cols = ['month', 'day', 'day_of_week', 'airline_encoded', 'origin_airport_encoded', 
                    'dest_airport_encoded', 'distance', 'dep_hour', 'arr_hour']
    
    # Filter only existing columns
    final_features = [c for c in feature_cols if c in df.columns]
    
    print(f"selected features: {final_features}")
    
    return df, final_features
