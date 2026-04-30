"""
Model training with multiple algorithms - FIXED
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from preprocessing import create_preprocessing_pipeline, prepare_data

# Load data (use CSV since parquet might have issues)
print("📂 Loading data...")
df = pd.read_csv('data/students.csv')  # Changed from parquet to CSV
print(f"Loaded {len(df)} rows")

X, y = prepare_data(df)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Training set: {len(X_train)} samples")
print(f"📊 Test set: {len(X_test)} samples")
print(f"📊 Train pass rate: {y_train.mean():.1%}")
print(f"📊 Test pass rate: {y_test.mean():.1%}")

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Train and evaluate
results = {}
best_model = None
best_score = 0

print("\n" + "="*60)
print("MODEL TRAINING & EVALUATION")
print("="*60)

for name, model in models.items():
    print(f"\n🔧 Training {name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', create_preprocessing_pipeline()),
        ('classifier', model)
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
        print(f"   CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    except Exception as e:
        print(f"   CV Error: {e}")
        cv_scores = np.array([0])
    
    # Train on full training set
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    results[name] = {
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'test_accuracy': accuracy,
        'test_f1': f1,
        'test_auc': auc,
        'pipeline': pipeline
    }
    
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Test F1: {f1:.4f}")
    print(f"   Test AUC: {auc:.4f}")
    
    if f1 > best_score:
        best_score = f1
        best_model = pipeline
        best_name = name

# Results summary
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
comparison = pd.DataFrame(results).T
print(comparison[['test_accuracy', 'test_f1', 'test_auc']].round(4))

print(f"\n🏆 BEST MODEL: {best_name}")
print(f"   F1 Score: {best_score:.4f}")

# Save best model
import os
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/best_model.joblib')
print("\n✅ Best model saved to 'models/best_model.joblib'")

# Detailed evaluation of best model
print("\n" + "="*60)
print("BEST MODEL - DETAILED EVALUATION")
print("="*60)

y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Fail', 'Pass']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(f"   True Negatives (correctly identified Fail): {cm[0,0]}")
print(f"   False Positives (wrongly identified Pass): {cm[0,1]}")
print(f"   False Negatives (wrongly identified Fail): {cm[1,0]}")
print(f"   True Positives (correctly identified Pass): {cm[1,1]}")