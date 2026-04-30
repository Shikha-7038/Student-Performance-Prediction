"""
FastAPI inference service for student performance prediction
FIXED: NumPy type serialization issues
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json
import os

# Initialize app
app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict student performance based on academic and engagement metrics",
    version="1.0.0"
)

# Load model (lazy loading)
model = None
preprocessor = None

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def load_model_and_preprocessor():
    """Load model and preprocessor"""
    global model, preprocessor
    if model is None:
        try:
            # Try to load tuned model first
            if os.path.exists('models/tuned_xgboost.joblib'):
                model = joblib.load('models/tuned_xgboost.joblib')
                print("✅ Tuned XGBoost model loaded")
            elif os.path.exists('models/best_model.joblib'):
                model = joblib.load('models/best_model.joblib')
                print("✅ Best model loaded")
            else:
                print("⚠️ No model found, training new model...")
                import subprocess
                subprocess.run(['python', 'src/train.py'], check=True)
                model = joblib.load('models/best_model.joblib')
            
            # Load preprocessor if exists
            if os.path.exists('models/preprocessor.joblib'):
                preprocessor = joblib.load('models/preprocessor.joblib')
                print("✅ Preprocessor loaded")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    return model

# Request schema
class StudentFeatures(BaseModel):
    prior_gpa: float = Field(..., ge=0, le=4, description="Prior GPA (0-4)")
    attendance_pct: float = Field(..., ge=0, le=100, description="Attendance percentage")
    quiz_avg: float = Field(..., ge=0, le=100, description="Quiz average score")
    assign_avg: float = Field(..., ge=0, le=100, description="Assignment average score")
    midterm: float = Field(..., ge=0, le=100, description="Midterm exam score")
    study_hours_wk: float = Field(..., ge=0, le=60, description="Study hours per week")
    on_time_submit_pct: float = Field(..., ge=0, le=100, description="On-time submission percentage")
    lms_logins_wk: int = Field(..., ge=0, le=50, description="LMS logins per week")
    forum_posts: int = Field(..., ge=0, le=50, description="Forum posts count")
    commute_min: int = Field(..., ge=0, le=180, description="Commute time in minutes")
    gender: str = Field(..., pattern="^(M|F)$", description="Gender (M/F)")
    school_type: str = Field(..., pattern="^(Public|Private|Charter)$", description="School type")
    parent_edu: str = Field(..., pattern="^(High School|Bachelor|Master|PhD)$", description="Parent education level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prior_gpa": 3.2,
                "attendance_pct": 85.0,
                "quiz_avg": 78.5,
                "assign_avg": 82.0,
                "midterm": 75.0,
                "study_hours_wk": 15.0,
                "on_time_submit_pct": 90.0,
                "lms_logins_wk": 12,
                "forum_posts": 5,
                "commute_min": 30,
                "gender": "M",
                "school_type": "Public",
                "parent_edu": "Bachelor"
            }
        }

# Prediction response
class PredictionResponse(BaseModel):
    risk_probability: float = Field(..., description="Probability of passing (0-1)")
    predicted_outcome: str = Field(..., description="Predicted outcome: Pass/Fail")
    risk_level: str = Field(..., description="Risk level: Low/Medium/High")
    timestamp: str = Field(..., description="Prediction timestamp")

# Intervention suggestions
def get_interventions(features: StudentFeatures, risk_prob: float) -> List[str]:
    """Generate personalized intervention suggestions"""
    interventions = []
    
    if risk_prob < 0.5:
        interventions.append("⚠️ **HIGH RISK** - Immediate intervention recommended")
        
        if features.attendance_pct < 70:
            interventions.append("📅 Schedule mandatory attendance counseling session")
            interventions.append("👥 Assign peer buddy for accountability")
        
        if features.quiz_avg < 55:
            interventions.append("📚 Enroll in remedial quiz workshops")
            interventions.append("🎯 Daily micro-quiz practice recommended")
        
        if features.assign_avg < 55:
            interventions.append("✏️ Weekly assignment review with TA")
            interventions.append("📝 Assignment planning workshop")
        
        if features.study_hours_wk < 8:
            interventions.append("⏰ Study skills workshop and schedule planning")
        
        if features.on_time_submit_pct < 70:
            interventions.append("📅 Deadline management training")
            interventions.append("🔔 Enable automated deadline reminders")
    
    elif risk_prob < 0.7:
        interventions.append("🟡 **MEDIUM RISK** - Monitoring recommended")
        interventions.append("📊 Weekly progress check-in")
        interventions.append("💪 Targeted support in weak areas")
    else:
        interventions.append("🟢 **LOW RISK** - On track for success")
        interventions.append("⭐ Continue current study habits")
        interventions.append("🎓 Consider advanced enrichment activities")
    
    return interventions

@app.get("/")
async def root():
    return {
        "service": "Student Performance Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict student performance",
            "/predict/with-interventions": "POST - Predict with interventions",
            "/predict/batch": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
            "/web": "GET - Web interface"
        }
    }

@app.get("/health")
async def health_check():
    try:
        load_model_and_preprocessor()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/web", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface HTML"""
    html_path = "src/web_interface.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    else:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>Web Interface Not Found</h1>
                <p>Please ensure src/web_interface.html exists.</p>
                <p>You can still use the API at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """)

@app.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentFeatures):
    """
    Predict student performance based on features
    """
    try:
        # Load model
        model = load_model_and_preprocessor()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([student.model_dump()])
        
        # Make prediction
        proba = model.predict_proba(input_df)[0, 1]  # Probability of passing
        proba = float(proba)  # Convert to Python float
        prediction = 1 if proba >= 0.5 else 0
        
        # Determine risk level
        if proba < 0.5:
            risk_level = "High"
        elif proba < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return PredictionResponse(
            risk_probability=round(proba, 4),
            predicted_outcome="Pass" if prediction == 1 else "Fail",
            risk_level=risk_level,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/with-interventions")
async def predict_with_interventions(student: StudentFeatures):
    """
    Predict student performance and get intervention suggestions
    """
    try:
        model = load_model_and_preprocessor()
        input_df = pd.DataFrame([student.model_dump()])
        proba = model.predict_proba(input_df)[0, 1]
        proba = float(proba)  # Convert to Python float
        prediction = 1 if proba >= 0.5 else 0
        
        interventions = get_interventions(student, proba)
        
        response = {
            "risk_probability": round(proba, 4),
            "predicted_outcome": "Pass" if prediction == 1 else "Fail",
            "risk_level": "High" if proba < 0.5 else "Medium" if proba < 0.7 else "Low",
            "interventions": interventions,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert any numpy types
        response = convert_numpy_types(response)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(students: List[StudentFeatures]):
    """
    Batch prediction for multiple students
    """
    try:
        model = load_model_and_preprocessor()
        
        results = []
        for i, student in enumerate(students):
            input_df = pd.DataFrame([student.model_dump()])
            proba = model.predict_proba(input_df)[0, 1]
            proba = float(proba)
            
            results.append({
                "student_index": i,
                "risk_probability": round(proba, 4),
                "predicted_outcome": "Pass" if proba >= 0.5 else "Fail",
                "risk_level": "High" if proba < 0.5 else "Medium" if proba < 0.7 else "Low"
            })
        
        response = {
            "total_students": len(students),
            "results": results,
            "summary": {
                "pass_count": sum(1 for r in results if r["predicted_outcome"] == "Pass"),
                "fail_count": sum(1 for r in results if r["predicted_outcome"] == "Fail"),
                "high_risk_count": sum(1 for r in results if r["risk_level"] == "High")
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Convert any numpy types
        response = convert_numpy_types(response)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)