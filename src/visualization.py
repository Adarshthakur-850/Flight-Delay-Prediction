import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc

def plot_roc_curves(models, X_test, y_test):
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("plots/roc_curves.png")
    plt.close()

def plot_feature_importance(model, feature_names):
    if not os.path.exists("plots"):
        os.makedirs("plots")
        
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]
        
        top_n = min(20, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title("Top Feature Importances")
        plt.bar(range(top_n), importances[top_indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90)
        plt.tight_layout()
        plt.savefig("plots/feature_importance.png")
        plt.close()
