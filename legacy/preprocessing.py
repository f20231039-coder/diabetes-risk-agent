"""
Data Preprocessing Module
Handles zero values, outliers, and feature engineering.
"""

import numpy as np


def handle_zero_values(df, columns=None):
    """Replace zero values in medical features with median (best performing method)."""
    if columns is None:
        columns = ['Glucose', 'Insulin', 'BMI']
    
    df_processed = df.copy()
    
    print(f"\n{'='*80}")
    print("HANDLING ZERO VALUES (Median Imputation)")
    print(f"{'='*80}")
    
    for col in columns:
        if col in df_processed.columns:
            zero_count = (df_processed[col] == 0).sum()
            if zero_count > 0:
                df_processed.loc[df_processed[col] == 0, col] = np.nan
                fill_value = df_processed[col].median()
                df_processed[col] = df_processed[col].fillna(fill_value)
                print(f"  {col}: Replaced {zero_count} zeros with median = {fill_value:.2f}")
    
    return df_processed


def handle_outliers(df, columns=None, multiplier=1.5):
    """Cap outliers using IQR method."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Outcome' in columns:
            columns.remove('Outcome')
    
    df_processed = df.copy()
    
    print(f"\n{'='*80}")
    print("HANDLING OUTLIERS (IQR METHOD)")
    print(f"{'='*80}")
    
    for col in columns:
        if col in df_processed.columns:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_count = ((df_processed[col] < lower_bound) | 
                           (df_processed[col] > upper_bound)).sum()
            
            df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"  {col}: Capped {outlier_count} outliers to [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return df_processed


def create_interaction_features(df):
    """Create interaction features (BMI_x_Age and Glucose_x_Insulin - best performing)."""
    df_processed = df.copy()
    
    print(f"\n{'='*80}")
    print("CREATING INTERACTION FEATURES")
    print(f"{'='*80}")
    
    if 'BMI' in df_processed.columns and 'Age' in df_processed.columns:
        df_processed['BMI_x_Age'] = df_processed['BMI'] * df_processed['Age']
        print(f"  [OK] Created: BMI_x_Age")
    
    if 'Glucose' in df_processed.columns and 'Insulin' in df_processed.columns:
        df_processed['Glucose_x_Insulin'] = df_processed['Glucose'] * df_processed['Insulin']
        print(f"  [OK] Created: Glucose_x_Insulin")
    
    return df_processed


def preprocess_data(df):
    """Apply complete preprocessing pipeline."""
    print(f"\n{'='*80}")
    print("ENHANCED PREPROCESSING PIPELINE")
    print(f"{'='*80}")
    print(f"\nOriginal shape: {df.shape}")
    
    # Apply all preprocessing steps
    df_processed = handle_zero_values(df)
    df_processed = handle_outliers(df_processed)
    df_processed = create_interaction_features(df_processed)
    
    print(f"\nFinal shape: {df_processed.shape}")
    print(f"{'='*80}")
    
    return df_processed
