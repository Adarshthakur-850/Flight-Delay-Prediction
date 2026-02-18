from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def evaluate_models(models, X_test, y_test):
    print("Evaluating models...")
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    metrics = []
    
    # Compare Metrics
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        metrics.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC': auc
        })
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(f"plots/confusion_matrix_{name}.png")
        plt.close()

    metrics_df = pd.DataFrame(metrics)
    print("\nModel Evaluation:")
    print(metrics_df)
    metrics_df.to_csv("models/metrics.csv", index=False)
    
    return metrics_df
