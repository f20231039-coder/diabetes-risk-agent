"""
Flask API for Diabetes Prediction
==================================

A REST API for serving diabetes predictions using trained ML models.

Endpoints:
    POST /predict - Make diabetes prediction
    GET /health - Check API health status

Author: ML Assignment 4
Date: 2025-11-19
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import prediction utilities
from model_persistence import load_pipeline, load_manifest

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

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for loaded models
models = {
    'naive_bayes': None,
    'mlp': None
}
manifest = None

# Model file paths
MODEL_PATHS = {
    'naive_bayes': 'models/nb_pipeline_v1.pkl',
    'mlp': 'models/mlp_pipeline_v1.pkl'
}


def load_models_at_startup():
    """
    Load both pickled pipelines at application startup.
    This ensures models are ready before handling any requests.
    """
    global models, manifest
    
    logger.info("="*80)
    logger.info("FLASK API STARTUP - LOADING MODELS")
    logger.info("="*80)
    
    # Load manifest
    logger.info("Loading model manifest...")
    manifest = load_manifest('models/manifest.json')
    
    if manifest:
        logger.info("✓ Manifest loaded successfully")
    else:
        logger.warning("⚠ Manifest not found or failed to load")
    
    # Load Naive Bayes pipeline
    logger.info("Loading Naive Bayes pipeline...")
    models['naive_bayes'] = load_pipeline(MODEL_PATHS['naive_bayes'])
    
    if models['naive_bayes']:
        logger.info("✓ Naive Bayes pipeline loaded successfully")
    else:
        logger.error("✗ Failed to load Naive Bayes pipeline")
    
    # Load MLP pipeline
    logger.info("Loading MLP pipeline...")
    models['mlp'] = load_pipeline(MODEL_PATHS['mlp'])
    
    if models['mlp']:
        logger.info("✓ MLP pipeline loaded successfully")
    else:
        logger.error("✗ Failed to load MLP pipeline")
    
    # Check if at least one model loaded
    if not any(models.values()):
        logger.error("CRITICAL: No models loaded! API will not function properly.")
        logger.error("Please train models first by running: python main.py")
    else:
        logger.info("✓ Model loading complete - API ready")
    
    logger.info("="*80)


def validate_input(data):
    """
    Validate JSON input for prediction request.
    
    Args:
        data: Dictionary containing request data
    
    Returns:
        tuple: (is_valid, error_message, validated_data)
    """
    required_fields = ['glucose', 'insulin', 'bmi', 'age', 'model']
    
    # Check for missing fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}", None
    
    # Extract and validate values
    try:
        glucose = float(data['glucose'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        age = int(data['age'])
        model_name = str(data['model']).lower()
    except (ValueError, TypeError) as e:
        return False, f"Invalid data type: {str(e)}", None
    
    # Validate ranges
    errors = []
    
    if glucose < 0:
        errors.append("Glucose must be non-negative")
    elif glucose > 300:
        errors.append("Glucose value seems unrealistic (>300 mg/dL)")
    
    if insulin < 0:
        errors.append("Insulin must be non-negative")
    elif insulin > 1000:
        errors.append("Insulin value seems unrealistic (>1000 μU/mL)")
    
    if bmi < 10:
        errors.append("BMI must be at least 10")
    elif bmi > 70:
        errors.append("BMI value seems unrealistic (>70)")
    
    if age < 0:
        errors.append("Age must be non-negative")
    elif age > 120:
        errors.append("Age value seems unrealistic (>120 years)")
    
    if model_name not in ['naive_bayes', 'mlp']:
        errors.append("Model must be 'naive_bayes' or 'mlp'")
    
    if errors:
        return False, "; ".join(errors), None
    
    validated_data = {
        'glucose': glucose,
        'insulin': insulin,
        'bmi': bmi,
        'age': age,
        'model': model_name
    }
    
    return True, None, validated_data


def preprocess_input(glucose, insulin, bmi, age):
    """
    Apply the same preprocessing used during training.
    
    The saved pipeline already includes StandardScaler, so we only need to:
        1. Handle zero values (median imputation) 
        2. Cap outliers using IQR method
        3. Create interaction features
    
    Note: These values are computed from the actual training data (diabetes.csv)
    
    Args:
        glucose: Blood glucose level
        insulin: Insulin level
        bmi: Body Mass Index
        age: Patient age
    
    Returns:
        np.ndarray: Preprocessed feature array (6 features)
    """
    # Create DataFrame for easier preprocessing
    df = pd.DataFrame({
        'Glucose': [glucose],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Age': [age]
    })
    
    # Step 1: Handle zero values with median imputation
    # These medians are computed from the actual training data (diabetes.csv)
    training_medians = {
        'Glucose': 117.0,
        'Insulin': 125.0,
        'BMI': 32.3
    }
    
    for col in ['Glucose', 'Insulin', 'BMI']:
        if df[col].iloc[0] == 0:
            df.loc[0, col] = training_medians[col]
    
    # Step 2: Cap outliers using IQR method (multiplier=1.5)
    # These bounds are computed from the training data AFTER zero handling
    iqr_bounds = {
        'Glucose': (39.0, 201.0),
        'Insulin': (112.875, 135.875),
        'BMI': (13.85, 50.25),
        'Age': (-1.5, 66.5)
    }
    
    for col, (lower, upper) in iqr_bounds.items():
        df[col] = df[col].clip(lower=lower, upper=upper)
    
    # Step 3: Create interaction features (same as training)
    df['BMI_x_Age'] = df['BMI'] * df['Age']
    df['Glucose_x_Insulin'] = df['Glucose'] * df['Insulin']
    
    # Return as numpy array in the exact order used during training
    # The pipeline expects: [Glucose, Insulin, BMI, Age, BMI_x_Age, Glucose_x_Insulin]
    feature_order = ['Glucose', 'Insulin', 'BMI', 'Age', 'BMI_x_Age', 'Glucose_x_Insulin']
    return df[feature_order].values


@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request errors."""
    logger.warning(f"Bad Request: {error}")
    return jsonify({
        'error': 'Bad Request',
        'message': str(error.description) if hasattr(error, 'description') else 'Invalid request'
    }), 400


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 Internal Server Error."""
    logger.error(f"Internal Server Error: {error}", exc_info=True)
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred. Please try again later.'
    }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint.
    
    Returns API status and information about loaded models.
    """
    logger.info("Health check requested")
    
    # Check which models are loaded
    loaded_models = []
    failed_models = []
    
    for model_name, pipeline in models.items():
        if pipeline is not None:
            loaded_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # Determine overall status
    if len(loaded_models) == 2:
        status = 'healthy'
        message = 'All models loaded successfully'
    elif len(loaded_models) == 1:
        status = 'degraded'
        message = f'Only {loaded_models[0]} model loaded'
    else:
        status = 'unhealthy'
        message = 'No models loaded - please train models first'
    
    response = {
        'status': status,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'loaded_models': loaded_models,
        'failed_models': failed_models,
        'api_version': '1.0.0'
    }
    
    # Add manifest info if available
    if manifest:
        response['model_metadata'] = manifest
    
    logger.info(f"Health check: {status}")
    
    return jsonify(response), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Accepts JSON input with patient data and returns diabetes prediction.
    
    Request JSON:
        {
            "glucose": float,
            "insulin": float,
            "bmi": float,
            "age": int,
            "model": "naive_bayes" | "mlp"
        }
    
    Response JSON:
        {
            "prediction": 0 | 1,
            "probability": float,
            "model_used": str,
            "message": str
        }
    """
    # Log request
    logger.info(f"Prediction request received from {request.remote_addr}")
    
    try:
        # Get JSON data
        if not request.is_json:
            logger.warning("Request is not JSON")
            return jsonify({
                'error': 'Invalid Content-Type',
                'message': 'Request must be JSON (Content-Type: application/json)'
            }), 400
        
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        # Validate input
        is_valid, error_msg, validated_data = validate_input(data)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            return jsonify({
                'error': 'Invalid Input',
                'message': error_msg
            }), 400
        
        # Extract validated values
        glucose = validated_data['glucose']
        insulin = validated_data['insulin']
        bmi = validated_data['bmi']
        age = validated_data['age']
        model_name = validated_data['model']
        
        # Check if requested model is loaded
        if models[model_name] is None:
            logger.error(f"Requested model not loaded: {model_name}")
            return jsonify({
                'error': 'Model Not Available',
                'message': f'The {model_name} model is not loaded. Please train models first.'
            }), 500
        
        # Preprocess input
        logger.info("Preprocessing input data...")
        X_input = preprocess_input(glucose, insulin, bmi, age)
        
        # Make prediction
        logger.info(f"Making prediction with {model_name}...")
        pipeline = models[model_name]
        prediction = int(pipeline.predict(X_input)[0])
        probability = float(pipeline.predict_proba(X_input)[0, 1])
        
        # Prepare response
        if prediction == 0:
            message = "No diabetes detected"
        else:
            message = "Diabetes detected"
        
        response = {
            'prediction': prediction,
            'probability': probability,
            'model_used': model_name.replace('_', ' ').title(),
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction successful: {prediction} (probability: {probability:.4f})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Prediction Failed',
            'message': 'An error occurred during prediction. Please check your input and try again.'
        }), 500


if __name__ == '__main__':
    # Load models before starting the server
    load_models_at_startup()
    
    # Start Flask server
    logger.info("Starting Flask server on http://localhost:5000")
    logger.info("Press CTRL+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
