"""
Simple script to generate all visualizations
This will definitely create images in the images folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create directories if they don't exist
os.makedirs('images', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("="*60)
print("📊 GENERATING VISUALIZATIONS")
print("="*60)

# Load data
try:
    df = pd.read_parquet('data/students.parquet')
    print(f"✅ Loaded {len(df)} records")
except:
    print("❌ Could not load parquet, trying CSV...")
    df = pd.read_csv('data/students.csv')
    print(f"✅ Loaded {len(df)} records from CSV")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# ============================================
# PLOT 1: Pass/Fail Distribution
# ============================================
plt.figure(figsize=(8, 6))
colors = ['green', 'red']
df['passed'].value_counts().plot(kind='bar', color=colors)
plt.title('Student Performance Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.xticks([0, 1], ['Pass', 'Fail'], rotation=0)
# Add value labels
for i, v in enumerate(df['passed'].value_counts()):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('images/pass_fail_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/pass_fail_distribution.png")

# ============================================
# PLOT 2: Grade Distribution
# ============================================
plt.figure(figsize=(10, 6))
grade_order = ['A', 'B', 'C', 'D', 'F']
grade_counts = df['grade_band'].value_counts()
# Reorder
grade_counts = grade_counts.reindex(grade_order)
colors_grades = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
grade_counts.plot(kind='bar', color=colors_grades)
plt.title('Grade Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Grade', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)
plt.xticks(rotation=0)
# Add value labels
for i, v in enumerate(grade_counts):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('images/grade_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/grade_distribution.png")

# ============================================
# PLOT 3: Study Hours vs Final Score
# ============================================
plt.figure(figsize=(10, 6))
plt.scatter(df['study_hours_wk'], df['final_score'], alpha=0.5, c='blue', s=20)
plt.xlabel('Study Hours per Week', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.title('Study Hours vs Final Score', fontsize=14, fontweight='bold')
# Add trend line
z = np.polyfit(df['study_hours_wk'], df['final_score'], 1)
p = np.poly1d(z)
plt.plot(df['study_hours_wk'].sort_values(), p(df['study_hours_wk'].sort_values()), 
         "r--", linewidth=2, label=f'Trend (R²={np.corrcoef(df["study_hours_wk"], df["final_score"])[0,1]**2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/study_hours_vs_score.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/study_hours_vs_score.png")

# ============================================
# PLOT 4: Attendance vs Pass Rate
# ============================================
plt.figure(figsize=(10, 6))
# Create attendance bins
attendance_bins = pd.cut(df['attendance_pct'], bins=[0, 50, 60, 70, 80, 90, 100])
pass_rate = df.groupby(attendance_bins)['passed'].mean() * 100
pass_rate.plot(kind='bar', color='lightgreen', edgecolor='darkgreen')
plt.xlabel('Attendance Percentage', fontsize=12)
plt.ylabel('Pass Rate (%)', fontsize=12)
plt.title('Attendance vs Pass Rate', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.axhline(y=60, color='red', linestyle='--', linewidth=2, label='60% Threshold')
plt.legend()
# Add value labels
for i, v in enumerate(pass_rate):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('images/attendance_vs_passrate.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/attendance_vs_passrate.png")

# ============================================
# PLOT 5: Feature Correlation Heatmap
# ============================================
plt.figure(figsize=(12, 10))
numeric_cols = ['prior_gpa', 'attendance_pct', 'quiz_avg', 'assign_avg', 
                'midterm', 'study_hours_wk', 'on_time_submit_pct', 
                'lms_logins_wk', 'forum_posts', 'final_score']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/correlation_heatmap.png")

# ============================================
# PLOT 6: Performance by Demographics
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# By Gender
gender_pass = df.groupby('gender')['passed'].mean() * 100
gender_pass.plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#4ecdc4'])
axes[0].set_title('Pass Rate by Gender', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Pass Rate (%)')
axes[0].set_ylim(0, 100)
axes[0].set_xticklabels(['Male', 'Female'], rotation=0)
for i, v in enumerate(gender_pass):
    axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# By School Type
school_pass = df.groupby('school_type')['passed'].mean() * 100
school_pass.plot(kind='bar', ax=axes[1], color=['#95e77b', '#ffd93d', '#6c5ce7'])
axes[1].set_title('Pass Rate by School Type', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Pass Rate (%)')
axes[1].set_ylim(0, 100)
axes[1].set_xtickrotation=0
for i, v in enumerate(school_pass):
    axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# By Parent Education
parent_pass = df.groupby('parent_edu')['passed'].mean() * 100
parent_order = ['High School', 'Bachelor', 'Master', 'PhD']
parent_pass = parent_pass.reindex(parent_order)
parent_pass.plot(kind='bar', ax=axes[2], color=['#fd79a8', '#e84393', '#d63031', '#fdcb6e'])
axes[2].set_title('Pass Rate by Parent Education', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Pass Rate (%)')
axes[2].set_ylim(0, 100)
axes[2].tick_params(axis='x', rotation=45)
for i, v in enumerate(parent_pass):
    axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('images/demographics_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/demographics_performance.png")

# ============================================
# PLOT 7: Box Plot - Scores by Grade
# ============================================
plt.figure(figsize=(10, 6))
grade_groups = [df[df['grade_band'] == g]['final_score'] for g in grade_order]
bp = plt.boxplot(grade_groups, labels=grade_order, patch_artist=True)
colors_box = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
plt.xlabel('Grade', fontsize=12)
plt.ylabel('Final Score', fontsize=12)
plt.title('Score Distribution by Grade', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/scores_by_grade.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/scores_by_grade.png")

# ============================================
# PLOT 8: Feature Importance (Correlation with final score)
# ============================================
plt.figure(figsize=(10, 8))
corr_with_target = corr['final_score'].drop('final_score').sort_values(ascending=True)
colors_imp = plt.cm.viridis(np.linspace(0, 1, len(corr_with_target)))
corr_with_target.plot(kind='barh', color=colors_imp)
plt.xlabel('Correlation with Final Score', fontsize=12)
plt.title('Feature Importance Analysis', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
# Add value labels
for i, v in enumerate(corr_with_target):
    plt.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/feature_importance.png")

# ============================================
# PLOT 9: Risk Analysis Pie Chart
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart - Pass/Fail
pass_fail = df['passed'].value_counts()
colors_pie = ['#2ecc71', '#e74c3c']
explode = (0.05, 0.05)
ax1.pie(pass_fail, explode=explode, labels=['Pass', 'Fail'], colors=colors_pie,
        autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Overall Pass/Fail Ratio', fontsize=12, fontweight='bold')

# Risk level distribution
def get_risk_level(score):
    if score >= 70:
        return 'Low Risk'
    elif score >= 50:
        return 'Medium Risk'
    else:
        return 'High Risk'

df['risk_level'] = df['final_score'].apply(get_risk_level)
risk_counts = df['risk_level'].value_counts()
colors_risk = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}
risk_colors = [colors_risk[r] for r in risk_counts.index]
ax2.pie(risk_counts, labels=risk_counts.index, colors=risk_colors,
        autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Student Risk Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('images/risk_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/risk_analysis.png")

# ============================================
# PLOT 10: Study Habits Analysis
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Study hours distribution by outcome
pass_study = df[df['passed'] == 1]['study_hours_wk']
fail_study = df[df['passed'] == 0]['study_hours_wk']
axes[0].hist(pass_study, bins=20, alpha=0.7, label='Pass', color='green', edgecolor='darkgreen')
axes[0].hist(fail_study, bins=20, alpha=0.7, label='Fail', color='red', edgecolor='darkred')
axes[0].set_xlabel('Study Hours per Week', fontsize=12)
axes[0].set_ylabel('Number of Students', fontsize=12)
axes[0].set_title('Study Hours Distribution: Pass vs Fail', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].axvline(x=df[df['passed']==1]['study_hours_wk'].mean(), color='darkgreen', 
                linestyle='--', linewidth=2, label='Avg Pass Study Hours')
axes[0].axvline(x=df[df['passed']==0]['study_hours_wk'].mean(), color='darkred', 
                linestyle='--', linewidth=2, label='Avg Fail Study Hours')

# LMS activity comparison
lms_activity = df.groupby('passed')[['lms_logins_wk', 'forum_posts']].mean()
lms_activity.plot(kind='bar', ax=axes[1])
axes[1].set_xlabel('Outcome', fontsize=12)
axes[1].set_ylabel('Average Count', fontsize=12)
axes[1].set_title('LMS Activity: Pass vs Fail', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(['Fail', 'Pass'], rotation=0)
axes[1].legend(['LMS Logins/Week', 'Forum Posts'])

plt.tight_layout()
plt.savefig('images/study_habits_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Created: images/study_habits_analysis.png")

# ============================================
# Create a summary report in outputs folder
# ============================================
with open('outputs/analysis_summary.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("STUDENT PERFORMANCE ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Total Students: {len(df)}\n")
    f.write(f"Pass Rate: {df['passed'].mean():.1%}\n")
    f.write(f"Fail Rate: {(1-df['passed'].mean()):.1%}\n\n")
    
    f.write("GRADE DISTRIBUTION:\n")
    for grade in grade_order:
        count = len(df[df['grade_band'] == grade])
        pct = count/len(df)*100
        f.write(f"  {grade}: {count} students ({pct:.1f}%)\n")
    
    f.write("\nKEY INSIGHTS:\n")
    f.write(f"1. Students who study >15 hours/week have {df[df['study_hours_wk']>15]['passed'].mean():.1%} pass rate\n")
    f.write(f"2. Students with >80% attendance have {df[df['attendance_pct']>80]['passed'].mean():.1%} pass rate\n")
    f.write(f"3. Students with quiz avg >70 have {df[df['quiz_avg']>70]['passed'].mean():.1%} pass rate\n")
    f.write(f"4. Students with assignment avg >70 have {df[df['assign_avg']>70]['passed'].mean():.1%} pass rate\n")
    
    f.write("\nTOP CORRELATIONS with Final Score:\n")
    for feature, corr_val in corr_with_target.sort_values(ascending=False).head(5).items():
        f.write(f"  {feature}: {corr_val:.3f}\n")

print("✅ Created: outputs/analysis_summary.txt")

print("\n" + "="*60)
print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print(f"📁 Images saved in: {os.path.abspath('images')}")
print(f"📁 Outputs saved in: {os.path.abspath('outputs')}")
print("="*60)

# List all generated files
print("\n📋 Generated Files:")
for file in os.listdir('images'):
    print(f"   - images/{file}")
for file in os.listdir('outputs'):
    print(f"   - outputs/{file}")