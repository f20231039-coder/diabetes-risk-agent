"""
Diabetes Prediction Script
===========================

Command-line interface for making diabetes predictions using trained models.

Usage:
    python predict.py --glucose 148 --insulin 0 --bmi 33.6 --age 50 --model mlp
    python predict.py --glucose 85 --insulin 66 --bmi 26.6 --age 31 --model naive_bayes

Author: ML Assignment 4
Date: 2025-11-20
"""

import argparse
import sys
import numpy as np
import pandas as pd
from model_persistence import load_pipeline, load_manifest


def preprocess_input(glucose, insulin, bmi, age):
    """
    Apply the same preprocessing used during training.
    
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
    training_medians = {
        'Glucose': 117.0,
        'Insulin': 125.0,
        'BMI': 32.3
    }
    
    for col in ['Glucose', 'Insulin', 'BMI']:
        if df[col].iloc[0] == 0:
            df.loc[0, col] = training_medians[col]
    
    # Step 2: Cap outliers using IQR method
    iqr_bounds = {
        'Glucose': (39.0, 201.0),
        'Insulin': (112.875, 135.875),
        'BMI': (13.85, 50.25),
        'Age': (-1.5, 66.5)
    }
    
    for col, (lower, upper) in iqr_bounds.items():
        df[col] = df[col].clip(lower=lower, upper=upper)
    
    # Step 3: Create interaction features
    df['BMI_x_Age'] = df['BMI'] * df['Age']
    df['Glucose_x_Insulin'] = df['Glucose'] * df['Insulin']
    
    # Return as numpy array in the exact order used during training
    feature_order = ['Glucose', 'Insulin', 'BMI', 'Age', 'BMI_x_Age', 'Glucose_x_Insulin']
    return df[feature_order].values


def validate_input(glucose, insulin, bmi, age):
    """
    Validate input values.
    
    Args:
        glucose: Blood glucose level
        insulin: Insulin level
        bmi: Body Mass Index
        age: Patient age
    
    Returns:
        tuple: (is_valid, error_message)
    """
    errors = []
    
    if glucose < 0 or glucose > 300:
        errors.append("Glucose must be between 0 and 300 mg/dL")
    
    if insulin < 0 or insulin > 1000:
        errors.append("Insulin must be between 0 and 1000 μU/mL")
    
    if bmi < 10 or bmi > 70:
        errors.append("BMI must be between 10 and 70")
    
    if age < 1 or age > 120:
        errors.append("Age must be between 1 and 120 years")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, None


def predict_diabetes(glucose, insulin, bmi, age, model_name='mlp'):
    """
    Make diabetes prediction using trained model.
    
    Args:
        glucose: Blood glucose level
        insulin: Insulin level
        bmi: Body Mass Index
        age: Patient age
        model_name: 'mlp' or 'naive_bayes'
    
    Returns:
        dict: Prediction results
    """
    # Validate input
    is_valid, error_msg = validate_input(glucose, insulin, bmi, age)
    if not is_valid:
        return {'error': error_msg}
    
    # Validate model name
    if model_name not in ['mlp', 'naive_bayes']:
        return {'error': "Model must be 'mlp' or 'naive_bayes'"}
    
    # Load model
    model_paths = {
        'mlp': 'models/mlp_pipeline_v1.pkl',
        'naive_bayes': 'models/nb_pipeline_v1.pkl'
    }
    
    pipeline = load_pipeline(model_paths[model_name])
    
    if pipeline is None:
        return {'error': f"Failed to load {model_name} model. Please train models first by running: python main.py"}
    
    # Preprocess input
    X_input = preprocess_input(glucose, insulin, bmi, age)
    
    # Make prediction
    prediction = int(pipeline.predict(X_input)[0])
    probability = float(pipeline.predict_proba(X_input)[0, 1])
    
    # Prepare result
    result = {
        'prediction': prediction,
        'probability': probability,
        'model_used': model_name,
        'message': 'Diabetes detected' if prediction == 1 else 'No diabetes detected'
    }
    
    return result


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description='Diabetes Prediction using Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --glucose 148 --insulin 0 --bmi 33.6 --age 50 --model mlp
  python predict.py --glucose 85 --insulin 66 --bmi 26.6 --age 31 --model naive_bayes
        """
    )
    
    parser.add_argument('--glucose', type=float, required=True,
                        help='Blood glucose level (mg/dL)')
    parser.add_argument('--insulin', type=float, required=True,
                        help='Insulin level (μU/mL)')
    parser.add_argument('--bmi', type=float, required=True,
                        help='Body Mass Index')
    parser.add_argument('--age', type=int, required=True,
                        help='Patient age (years)')
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'naive_bayes'],
                        help='Model to use for prediction (default: mlp)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("DIABETES PREDICTION")
    print("="*80)
    
    print(f"\nInput Values:")
    print(f"  Glucose: {args.glucose} mg/dL")
    print(f"  Insulin: {args.insulin} μU/mL")
    print(f"  BMI:     {args.bmi}")
    print(f"  Age:     {args.age} years")
    print(f"  Model:   {args.model.replace('_', ' ').title()}")
    
    # Make prediction
    result = predict_diabetes(args.glucose, args.insulin, args.bmi, args.age, args.model)
    
    # Display result
    if 'error' in result:
        print(f"\n❌ Error: {result['error']}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print("PREDICTION RESULT")
    print(f"{'='*80}")
    
    if result['prediction'] == 1:
        print(f"\n⚠️  {result['message'].upper()}")
        print(f"   Prediction: Positive (1)")
        print(f"   Confidence: {result['probability']*100:.1f}%")
    else:
        print(f"\n✅ {result['message'].upper()}")
        print(f"   Prediction: Negative (0)")
        print(f"   Confidence: {(1-result['probability'])*100:.1f}%")
    
    print(f"\n⚕️  Medical Disclaimer:")
    print(f"   This prediction is for educational purposes only.")
    print(f"   Always consult with a qualified healthcare provider.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
