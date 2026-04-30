"""
Feature preprocessing pipeline - FIXED
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Define feature groups
NUMERIC_FEATURES = [
    'prior_gpa', 'attendance_pct', 'quiz_avg', 'assign_avg', 
    'midterm', 'study_hours_wk', 'on_time_submit_pct',
    'lms_logins_wk', 'forum_posts', 'commute_min'
]

CATEGORICAL_FEATURES = [
    'gender', 'school_type', 'parent_edu'
]

def create_preprocessing_pipeline():
    """Create scikit-learn preprocessing pipeline"""
    
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combined preprocessor
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    
    return preprocessor

def prepare_data(df):
    """Prepare features and target"""
    
    # Features (exclude target and identifiers)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    
    # Target
    y = df['passed'].copy()
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"✅ Target distribution: \n{y.value_counts(normalize=True)}")
    
    return X, y

if __name__ == "__main__":
    # Test pipeline
    print("Loading data...")
    df = pd.read_csv('data/students.csv')  # Use CSV instead of parquet
    print(f"Loaded {len(df)} rows")
    
    X, y = prepare_data(df)
    
    preprocessor = create_preprocessing_pipeline()
    X_transformed = preprocessor.fit_transform(X)
    
    print(f"\n✅ Pipeline created successfully")
    print(f"   Input features: {X.shape[1]}")
    print(f"   Output features: {X_transformed.shape[1]}")
    
    # Save preprocessor
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("✅ Preprocessor saved to 'models/preprocessor.joblib'")