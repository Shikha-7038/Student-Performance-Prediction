"""
Hyperparameter tuning with Optuna for XGBoost
"""

import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

from preprocessing import create_preprocessing_pipeline, prepare_data

# Load data
df = pd.read_parquet('data/students.parquet')
X, y = prepare_data(df)

# Preprocessor (fit once)
preprocessor = create_preprocessing_pipeline()
X_transformed = preprocessor.fit_transform(X)

def objective(trial):
    """Optuna objective function"""
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'random_state': 42
    }
    
    model = XGBClassifier(**params, eval_metric='logloss', use_label_encoder=False)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_transformed, y, cv=cv, scoring='f1')
    
    return scores.mean()

print("🔧 Starting hyperparameter tuning with Optuna...")
print("   (This may take 5-10 minutes)")

# Create study
study = optuna.create_study(direction='maximize', study_name='student_performance')
study.optimize(objective, n_trials=30, show_progress_bar=True)

# Best parameters
print("\n" + "="*60)
print("BEST PARAMETERS FOUND:")
print("="*60)
best_params = study.best_params
for param, value in best_params.items():
    print(f"   {param}: {value}")

print(f"\n📈 Best F1 Score: {study.best_value:.4f}")

# Train final model with best parameters
print("\n🚀 Training final model with best parameters...")
best_model = XGBClassifier(**best_params, eval_metric='logloss', use_label_encoder=False, random_state=42)

# Create pipeline
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

# Train on full data
final_pipeline.fit(X, y)

# Save
joblib.dump(final_pipeline, 'models/tuned_xgboost.joblib')
print("✅ Tuned model saved to 'models/tuned_xgboost.joblib'")

# Save study results
import json
study_results = {
    'best_params': best_params,
    'best_value': study.best_value,
    'n_trials': len(study.trials)
}
with open('models/optuna_results.json', 'w') as f:
    json.dump(study_results, f, indent=2)