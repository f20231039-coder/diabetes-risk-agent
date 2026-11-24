"""
Data Loading and Exploration Module
Handles loading and initial exploration of the diabetes dataset.
"""

import pandas as pd


def load_data(filepath='diabetes.csv'):
    """Load diabetes dataset from CSV file."""
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")
    
    df = pd.read_csv(filepath)
    print(f"\n[OK] Data loaded successfully")
    print(f"  Shape: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    
    return df


def explore_data(df):
    """Display data overview and statistics."""
    print(f"\n{'='*80}")
    print("DATA EXPLORATION")
    print(f"{'='*80}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  [OK] No missing values")
    else:
        print(missing[missing > 0])
    
    print("\nClass distribution:")
    if 'Outcome' in df.columns:
        class_counts = df['Outcome'].value_counts()
        print(f"  Class 0 (No Diabetes): {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
        print(f"  Class 1 (Diabetes): {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")
