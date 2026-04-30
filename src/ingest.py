"""
Data ingestion with schema validation
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Define schema
SCHEMA = {
    'student_id': 'string',
    'gender': 'category',
    'school_type': 'category',
    'parent_edu': 'category',
    'commute_min': 'int64',
    'prior_gpa': 'float64',
    'attendance_pct': 'float64',
    'quiz_avg': 'float64',
    'assign_avg': 'float64',
    'midterm': 'float64',
    'study_hours_wk': 'float64',
    'on_time_submit_pct': 'float64',
    'lms_logins_wk': 'int64',
    'forum_posts': 'int64',
    'final_score': 'float64',
    'grade_band': 'category',
    'passed': 'int64'
}

def load_and_validate_data(filepath):
    """Load data and validate schema"""
    
    # Load data
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)
    
    print(f"📁 Loaded {len(df)} rows from {filepath}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"⚠️ Missing values found:\n{missing[missing > 0]}")
    
    # Validate columns
    expected_cols = set(SCHEMA.keys())
    actual_cols = set(df.columns)
    
    missing_cols = expected_cols - actual_cols
    extra_cols = actual_cols - expected_cols
    
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if extra_cols:
        print(f"ℹ️ Extra columns found: {extra_cols}")
    
    # Convert dtypes
    for col, dtype in SCHEMA.items():
        if col in df.columns:
            if dtype == 'category':
                df[col] = df[col].astype('category')
            elif dtype in ['int64', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("✅ Data validation passed")
    return df

if __name__ == "__main__":
    df = load_and_validate_data('data/students.parquet')
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"🎯 Pass rate: {df['passed'].mean():.1%}")