import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.eda import perform_eda
from src.feature_engineering import engineer_features
from src.model_trainer import train_models
from src.evaluation import evaluate_models
from src.visualization import plot_roc_curves, plot_feature_importance

def main():
    print("Starting Flight Delay Prediction Pipeline...")
    
    # 1. Load Data
    try:
        df = load_data()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 2. Preprocessing
    df = preprocess_data(df)
    
    # 3. EDA
    try:
        perform_eda(df)
    except Exception as e:
        print(f"EDA failed: {e}")
    
    # 4. Feature Engineering
    df, features = engineer_features(df)
    
    # 5. Model Training
    models, X_test, y_test = train_models(df, features)
    
    # 6. Evaluation
    evaluate_models(models, X_test, y_test)
    
    # 7. Visualization
    plot_roc_curves(models, X_test, y_test)
    
    if 'RandomForest' in models:
        plot_feature_importance(models['RandomForest'], features)
    elif 'XGBoost' in models:
        plot_feature_importance(models['XGBoost'], features)
        
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Pipeline Failed: {e}")
