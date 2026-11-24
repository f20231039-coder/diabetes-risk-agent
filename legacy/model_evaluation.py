"""
Model Evaluation Module
Handles model evaluation, metrics calculation, and cross-validation.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, brier_score_loss
)
from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # ROC and PR curves
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Brier score
    brier = brier_score_loss(y_test, y_proba)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} EVALUATION")
    print(f"{'='*80}")
    
    print(f"\n[Performance Metrics]")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    print(f"\n[Confusion Matrix]")
    print(f"  True Positives (TP):  {tp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    
    print(f"\n[ROC & Calibration]")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    
    return metrics, cm, roc_auc, pr_auc


def cross_validate_models(models, X, y):
    """Perform 5-fold cross-validation."""
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION (5-FOLD)")
    print(f"{'='*80}")
    
    cv_results = {}
    
    for name, model in models.items():
        print(f"\n[{name}]")
        scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        print(f"  F1-Score: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  All folds: {[f'{s:.4f}' for s in scores]}")
    
    return cv_results
