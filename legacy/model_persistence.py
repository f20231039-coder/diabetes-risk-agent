"""
Model Persistence Module
Handles saving and loading of trained models and metadata.
"""

import pickle
import json
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def save_pipeline(pipeline, filepath, metadata=None):
    """
    Save trained pipeline (scaler + model) using pickle.
    
    Args:
        pipeline: Trained sklearn Pipeline object
        filepath: Path to save the .pkl file
        metadata: Optional dictionary with model metadata
    
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save pipeline using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline, f)
        
        logger.info(f"Pipeline saved successfully: {filepath}")
        print(f"  [OK] Saved pipeline: {filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save pipeline to {filepath}: {str(e)}")
        print(f"  [ERROR] Failed to save pipeline: {filepath}")
        return False


def load_pipeline(filepath):
    """
    Load a serialized pipeline from pickle file.
    
    Args:
        filepath: Path to the .pkl file
    
    Returns:
        Pipeline object if successful, None otherwise
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"Pipeline file not found: {filepath}")
            print(f"  [ERROR] File not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        logger.info(f"Pipeline loaded successfully: {filepath}")
        print(f"  [OK] Loaded pipeline: {filepath}")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to load pipeline from {filepath}: {str(e)}")
        print(f"  [ERROR] Failed to load pipeline: {filepath}")
        print(f"    Error: {str(e)}")
        return None


def save_manifest(metadata, filepath='models/manifest.json'):
    """
    Save model metadata to JSON manifest file.
    
    Args:
        metadata: Dictionary containing model metadata
        filepath: Path to save the manifest.json file
    
    Returns:
        bool: True if save successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Load existing manifest if it exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                existing_manifest = json.load(f)
        else:
            existing_manifest = {}
        
        # Update with new metadata
        existing_manifest.update(metadata)
        
        # Save manifest
        with open(filepath, 'w') as f:
            json.dump(existing_manifest, f, indent=2)
        
        logger.info(f"Manifest saved successfully: {filepath}")
        print(f"  [OK] Saved manifest: {filepath}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save manifest to {filepath}: {str(e)}")
        print(f"  [ERROR] Failed to save manifest: {filepath}")
        return False


def load_manifest(filepath='models/manifest.json'):
    """
    Load model metadata from JSON manifest file.
    
    Args:
        filepath: Path to the manifest.json file
    
    Returns:
        Dictionary with metadata if successful, None otherwise
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Manifest file not found: {filepath}")
            print(f"  [WARNING] Manifest not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            manifest = json.load(f)
        
        logger.info(f"Manifest loaded successfully: {filepath}")
        print(f"  [OK] Loaded manifest: {filepath}")
        
        return manifest
        
    except Exception as e:
        logger.error(f"Failed to load manifest from {filepath}: {str(e)}")
        print(f"  [ERROR] Failed to load manifest: {filepath}")
        return None


def demonstrate_model_loading():
    """
    Demonstrate how to load saved models and make predictions.
    This function shows the complete workflow for using saved models.
    """
    print(f"\n{'='*80}")
    print("MODEL LOADING DEMONSTRATION")
    print(f"{'='*80}")
    
    # Load manifest
    print("\n[1/3] Loading model metadata...")
    manifest = load_manifest('models/manifest.json')
    
    if manifest:
        print("\n📋 Available Models:")
        for model_name, metadata in manifest.items():
            print(f"\n  {model_name.upper()}:")
            print(f"    Version: {metadata.get('version', 'N/A')}")
            print(f"    Created: {metadata.get('created_date', 'N/A')}")
            print(f"    Test Accuracy: {metadata.get('test_accuracy', 0):.4f}")
            print(f"    Test AUC: {metadata.get('test_auc', 0):.4f}")
            print(f"    Features: {len(metadata.get('features', []))} features")
    
    # Load Naive Bayes pipeline
    print("\n[2/3] Loading Naive Bayes pipeline...")
    nb_pipeline = load_pipeline('models/nb_pipeline_v1.pkl')
    
    if nb_pipeline:
        print("  [OK] Naive Bayes pipeline ready for predictions")
    
    # Load MLP pipeline
    print("\n[3/3] Loading MLP pipeline...")
    mlp_pipeline = load_pipeline('models/mlp_pipeline_v1.pkl')
    
    if mlp_pipeline:
        print("  [OK] MLP pipeline ready for predictions")
    
    # Example prediction (if models loaded successfully)
    if nb_pipeline and mlp_pipeline:
        print(f"\n{'='*80}")
        print("EXAMPLE PREDICTION")
        print(f"{'='*80}")
        
        # Example patient data (must match training features)
        example_data = np.array([[148.0, 0.0, 33.6, 50, 4968.0, 0.0]])  # BMI_x_Age, Glucose_x_Insulin
        
        print("\nExample patient data:")
        print("  Glucose: 148.0, Insulin: 0.0, BMI: 33.6, Age: 50")
        
        # Predict with Naive Bayes
        nb_pred = nb_pipeline.predict(example_data)[0]
        nb_proba = nb_pipeline.predict_proba(example_data)[0, 1]
        print(f"\n  Naive Bayes: Prediction = {nb_pred}, Probability = {nb_proba:.4f}")
        
        # Predict with MLP
        mlp_pred = mlp_pipeline.predict(example_data)[0]
        mlp_proba = mlp_pipeline.predict_proba(example_data)[0, 1]
        print(f"  MLP:         Prediction = {mlp_pred}, Probability = {mlp_proba:.4f}")
    
    print(f"\n[OK] Model loading demonstration complete")
    print(f"{'='*80}\n")
