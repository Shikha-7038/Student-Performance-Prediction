"""
Main script to run the complete Student Performance Prediction System
"""

import subprocess
import sys
import os
import webbrowser
import time
import threading

def run_command(command, cwd=None):
    """Run a command and return the process"""
    return subprocess.Popen(command, shell=True, cwd=cwd)

def open_browser():
    """Open browser after delay"""
    time.sleep(3)
    webbrowser.open('http://127.0.0.1:8000/docs')
    webbrowser.open('src/web_interface.html')

def main():
    print("="*60)
    print("🎓 STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("="*60)
    
    # Check if data exists
    if not os.path.exists('data/students.parquet'):
        print("\n📊 Generating synthetic data...")
        subprocess.run([sys.executable, 'data/generate_data.py'], check=True)
    
    # Check if model exists
    if not os.path.exists('models/tuned_xgboost.joblib'):
        print("\n🔧 Training model...")
        subprocess.run([sys.executable, 'src/train.py'], check=True)
        
        print("\n🎯 Hyperparameter tuning...")
        subprocess.run([sys.executable, 'src/tune.py'], check=True)
    
    print("\n" + "="*60)
    print("🚀 STARTING SERVICES")
    print("="*60)
    
    # Start API server
    print("\n▶️ Starting FastAPI server on http://127.0.0.1:8000")
    api_process = run_command(f'{sys.executable} -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload')
    
    # Open browser for API docs and web interface
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("\n✅ System is ready!")
    print("\n📌 Available endpoints:")
    print("   - API Documentation: http://127.0.0.1:8000/docs")
    print("   - Health Check: http://127.0.0.1:8000/health")
    print("   - Web Interface: src/web_interface.html (open in browser)")
    print("\n📋 Press Ctrl+C to stop the server")
    
    try:
        api_process.wait()
    except KeyboardInterrupt:
        print("\n\n⏹️ Shutting down...")
        api_process.terminate()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main()