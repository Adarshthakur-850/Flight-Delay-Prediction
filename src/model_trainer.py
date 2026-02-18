from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
import numpy as np
import pandas as pd

def train_models(df, feature_cols, target_col='is_delayed'):
    print("Training models...")
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Simple split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale data for Logistic Regression? Simple StandardScaler is good.
    # But RF and XGB don't strictly need it. Let's do it for consistency/safety.
    # Actually, let's keep it raw for trees and rely on robustness, or handle in preprocessing.
    # Given requirements, "Production-quality" usually implies pipelines, but for this constraint (single files), 
    # we'll stick to raw features for trees and maybe specific for LogReg if needed, but LogReg usually handles reasonable scales.
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
    }
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    trained_models = {}
    best_score = -1
    best_name = ""
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test) # Accuracy
            print(f"{name} Test Accuracy: {score:.4f}")
            
            trained_models[name] = model
            
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
        except Exception as e:
            print(f"Failed to train {name}: {e}")
            
    print(f"Best Model: {best_name} (Acc: {best_score:.4f})")
    if best_model:
        joblib.dump(best_model, f"models/best_model.pkl")
        
    return trained_models, X_test, y_test
