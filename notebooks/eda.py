"""
Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_parquet('data/students.parquet')
print("=" * 60)
print("STUDENT PERFORMANCE DATA - EDA REPORT")
print("=" * 60)

# 1. Basic info
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"\n📈 Basic Statistics:")
print(df.describe())

# 2. Target distribution
print(f"\n🎯 Target Distribution:")
print(f"Pass: {df['passed'].sum():,} ({df['passed'].mean():.1%})")
print(f"Fail: {(len(df)-df['passed'].sum()):,} ({1-df['passed'].mean():.1%})")

# 3. Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Pass/Fail distribution
df['passed'].value_counts().plot(kind='bar', ax=axes[0,0], color=['green', 'red'])
axes[0,0].set_title('Pass/Fail Distribution')
axes[0,0].set_xticklabels(['Pass', 'Fail'], rotation=0)

# Plot 2: Grade distribution
df['grade_band'].value_counts().sort_index().plot(kind='bar', ax=axes[0,1], color='skyblue')
axes[0,1].set_title('Grade Distribution')

# Plot 3: Prior GPA vs Final Score
axes[0,2].scatter(df['prior_gpa'], df['final_score'], alpha=0.3)
axes[0,2].set_xlabel('Prior GPA')
axes[0,2].set_ylabel('Final Score')
axes[0,2].set_title('Prior GPA vs Final Score')

# Plot 4: Study hours vs Pass rate
study_bins = pd.cut(df['study_hours_wk'], bins=[0,5,10,15,20,25,30,50])
pass_rate = df.groupby(study_bins)['passed'].mean()
pass_rate.plot(kind='bar', ax=axes[1,0], color='coral')
axes[1,0].set_title('Study Hours vs Pass Rate')
axes[1,0].set_xlabel('Study Hours/Week')

# Plot 5: Attendance vs Pass rate
attendance_bins = pd.cut(df['attendance_pct'], bins=[0,50,60,70,80,90,100])
pass_by_attendance = df.groupby(attendance_bins)['passed'].mean()
pass_by_attendance.plot(kind='bar', ax=axes[1,1], color='lightgreen')
axes[1,1].set_title('Attendance vs Pass Rate')

# Plot 6: Correlation heatmap
numeric_cols = ['prior_gpa', 'attendance_pct', 'quiz_avg', 'assign_avg', 
                'midterm', 'study_hours_wk', 'on_time_submit_pct', 
                'lms_logins_wk', 'forum_posts', 'final_score']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1,2])
axes[1,2].set_title('Feature Correlations')

plt.tight_layout()
plt.savefig('images/eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Visualizations saved to 'images/eda_plots.png'")

# 4. Feature importance insights
print("\n🔍 Top Correlations with Final Score:")
corr_with_target = corr['final_score'].sort_values(ascending=False)
for feature, corr_val in corr_with_target.items():
    if feature != 'final_score':
        print(f"  {feature}: {corr_val:.3f}")