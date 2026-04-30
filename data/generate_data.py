"""
Synthetic Student Data Generator - FIXED VERSION
Creates realistic student performance dataset with balanced pass/fail distribution
"""

import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

def generate_student_data(n_students=10000):
    """Generate synthetic student data with realistic correlations and balanced classes"""
    
    # Demographics
    genders = np.random.choice(['M', 'F'], size=n_students, p=[0.48, 0.52])
    school_types = np.random.choice(['Public', 'Private', 'Charter'], 
                                     size=n_students, p=[0.6, 0.25, 0.15])
    parent_edu = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                   size=n_students, p=[0.4, 0.35, 0.15, 0.1])
    commute_min = np.random.exponential(30, n_students).astype(int)
    commute_min = np.clip(commute_min, 5, 120)
    
    # Prior academic performance (GPA 0-4) - more realistic spread
    prior_gpa = np.random.beta(a=1.5, b=1.5, size=n_students) * 3.5 + 0.5
    prior_gpa = np.clip(prior_gpa, 1.0, 4.0)
    
    # Engagement metrics (correlated with prior_gpa)
    # Add more variation to create realistic fail cases
    attendance_pct = np.random.normal(75, 18, n_students)
    attendance_pct = np.clip(attendance_pct + (prior_gpa - 2.5)*4, 30, 100)
    
    quiz_avg = np.random.normal(65, 20, n_students)
    quiz_avg = np.clip(quiz_avg + (prior_gpa - 2.5)*10, 20, 100)
    
    assign_avg = np.random.normal(65, 18, n_students)
    assign_avg = np.clip(assign_avg + (prior_gpa - 2.5)*9, 25, 100)
    
    midterm = np.random.normal(62, 20, n_students)
    midterm = np.clip(midterm + (prior_gpa - 2.5)*11, 20, 100)
    
    study_hours_wk = np.random.exponential(12, n_students)
    study_hours_wk = np.clip(study_hours_wk + (prior_gpa - 2.5)*3, 1, 40)
    
    on_time_submit_pct = np.random.normal(70, 22, n_students)
    on_time_submit_pct = np.clip(on_time_submit_pct + (prior_gpa - 2.5)*8, 20, 100)
    
    lms_logins_wk = np.random.negative_binomial(3, 0.25, n_students)
    lms_logins_wk = np.clip(lms_logins_wk + (prior_gpa - 2.5)*2, 0, 35)
    
    forum_posts = np.random.poisson(1.5, n_students)
    forum_posts = np.clip(forum_posts + (prior_gpa - 2.5)*0.5, 0, 12)
    
    # Final score calculation (weighted sum with noise)
    # Adjusted to create realistic pass/fail distribution
    final_score = (
        prior_gpa * 12 +
        attendance_pct * 0.12 +
        quiz_avg * 0.18 +
        assign_avg * 0.18 +
        midterm * 0.18 +
        study_hours_wk * 0.6 +
        on_time_submit_pct * 0.08
    ) + np.random.normal(0, 8, n_students)
    
    final_score = np.clip(final_score, 0, 100)
    
    # Grade bands with realistic thresholds
    grade_band = pd.cut(final_score, 
                        bins=[0, 50, 60, 70, 80, 100],
                        labels=['F', 'D', 'C', 'B', 'A'])
    
    # Pass/Fail (threshold 60) - will create balanced ~70-80% pass rate
    passed = (final_score >= 60).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'student_id': [f'STU_{i:05d}' for i in range(n_students)],
        'gender': genders,
        'school_type': school_types,
        'parent_edu': parent_edu,
        'commute_min': commute_min,
        'prior_gpa': np.round(prior_gpa, 2),
        'attendance_pct': np.round(attendance_pct, 1),
        'quiz_avg': np.round(quiz_avg, 1),
        'assign_avg': np.round(assign_avg, 1),
        'midterm': np.round(midterm, 1),
        'study_hours_wk': np.round(study_hours_wk, 1),
        'on_time_submit_pct': np.round(on_time_submit_pct, 1),
        'lms_logins_wk': lms_logins_wk.astype(int),
        'forum_posts': forum_posts.astype(int),
        'final_score': np.round(final_score, 1),
        'grade_band': grade_band,
        'passed': passed
    })
    
    return df

if __name__ == "__main__":
    # Generate 10,000 student records
    df = generate_student_data(10000)
    
    # Save to CSV (always works)
    df.to_csv('data/students.csv', index=False)
    print(f"✅ Generated {len(df)} student records")
    print(f"✅ Saved to data/students.csv")
    
    print("\n📊 Data Preview:")
    print(df.head())
    
    print("\n📈 Data Types:")
    print(df.dtypes)
    
    print("\n🎯 Target Distribution (Pass/Fail):")
    print(df['passed'].value_counts())
    print(f"Pass rate: {df['passed'].mean():.1%}")
    print(f"Fail rate: {(1-df['passed'].mean()):.1%}")
    
    # Try to save as Parquet only if pyarrow is available
    try:
        df.to_parquet('data/students.parquet', index=False)
        print("\n✅ Saved as students.parquet")
    except ImportError:
        print("\n⚠️ Pyarrow not installed. Using CSV only.")
        print("   Run: pip install pyarrow")
    
    # Quick stats
    print("\n📊 Grade Distribution:")
    print(df['grade_band'].value_counts().sort_index())
    
    print("\n📈 Feature Statistics:")
    print(df[['prior_gpa', 'attendance_pct', 'quiz_avg', 'assign_avg', 'midterm', 'study_hours_wk']].describe())