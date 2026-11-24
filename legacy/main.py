"""
Diabetes Prediction System - Main Pipeline
==========================================

A comprehensive machine learning system for predicting diabetes using
Naive Bayes and MLP classifiers with enhanced preprocessing.

Usage:
    python main.py

Features:
    - Enhanced preprocessing (zero handling, outlier capping, feature engineering)
    - Model training (Naive Bayes and MLP)
    - Comprehensive evaluation (metrics, confusion matrix, ROC, cross-validation)
    - Visualization (EDA plots)
    - Optional hyperparameter tuning

Author: ML Assignment 4
Date: 2025-11-19
"""

import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import warnings

# Import custom modules
from data_loader import load_data, explore_data
from visualization import create_visualizations
from preprocessing import preprocess_data
from model_training import train_models
from model_evaluation import evaluate_model, cross_validate_models
from model_persistence import save_pipeline, save_manifest

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution pipeline."""
    print("\n" + "="*80)
    print("DIABETES PREDICTION SYSTEM - COMPLETE PIPELINE")
    print("="*80)
    
    # Step 1: Load data
    df = load_data('diabetes.csv')
    
    # Step 2: Explore data
    explore_data(df)
    
    # Step 3: Create visualizations
    create_visualizations(df)
    
    # Step 4: Preprocess data
    df_processed = preprocess_data(df)
    
    # Step 5: Prepare features and target
    print(f"\n{'='*80}")
    print("PREPARING DATA FOR TRAINING")
    print(f"{'='*80}")
    
    feature_cols = [col for col in df_processed.columns if col != 'Outcome']
    X = df_processed[feature_cols]
    y = df_processed['Outcome']
    
    print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
    print(f"Target: Outcome")
    
    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Step 7: Train models
    # Automatically use default parameters (no tuning) for faster execution
    use_tuning = False
    
    models = train_models(X_train, y_train, use_tuning=use_tuning)
    
    # Step 8: Evaluate models
    all_metrics = {}
    for name, model in models.items():
        metrics, cm, roc_auc, pr_auc = evaluate_model(model, X_test, y_test, name)
        all_metrics[name] = metrics
    
    # Step 9: Cross-validation
    cv_results = cross_validate_models(models, X, y)
    
    # Step 10: Save models
    print(f"\n{'='*80}")
    print("SAVING MODELS")
    print(f"{'='*80}")
    
    # Prepare metadata
    manifest_data = {}
    
    for name, model in models.items():
        # Determine filename
        if name == 'Naive Bayes':
            filename = 'models/nb_pipeline_v1.pkl'
            manifest_key = 'naive_bayes'
        else:  # MLP
            filename = 'models/mlp_pipeline_v1.pkl'
            manifest_key = 'mlp'
        
        # Save pipeline
        print(f"\n[{name}]")
        save_pipeline(model, filename)
        
        # Prepare metadata
        metrics = all_metrics[name]
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        test_auc = auc(fpr, tpr)
        
        manifest_data[manifest_key] = {
            'version': 'v1',
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': feature_cols,
            'test_accuracy': float(metrics['accuracy']),
            'test_precision': float(metrics['precision']),
            'test_recall': float(metrics['recall']),
            'test_f1_score': float(metrics['f1_score']),
            'test_auc': float(test_auc),
            'cv_mean_f1': float(cv_results[name]['mean']),
            'cv_std_f1': float(cv_results[name]['std'])
        }
    
    # Save manifest
    print(f"\n[Metadata]")
    save_manifest(manifest_data)
    
    print(f"\n[OK] All models saved successfully")
    print(f"\n[SECURITY WARNING]")
    print(f"   Pickle files can execute arbitrary code when loaded.")
    print(f"   Only load .pkl files from trusted sources!")
    print(f"   Never load pickle files from unknown or untrusted origins.")
    
    # Step 11: Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n[Test Set Performance]")
    for name, metrics in all_metrics.items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    
    # Determine best model
    best_model = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n[Best Model] {best_model[0]}")
    print(f"   F1-Score: {best_model[1]['f1_score']:.4f}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    
    print(f"\n[OK] Pipeline execution complete!")
    print(f"[OK] Visualizations saved to 'plots/' directory")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
