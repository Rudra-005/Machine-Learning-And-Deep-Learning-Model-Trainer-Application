#!/usr/bin/env python
"""
ML/DL Trainer Application Launcher
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("=" * 70)
    print("🤖 ML/DL TRAINER - MACHINE LEARNING & DEEP LEARNING PLATFORM")
    print("=" * 70)
    print()
    print("📊 Starting Application...")
    print()
    print("🌐 Access the application at: http://localhost:8501")
    print()
    print("📋 Features:")
    print("   ✅ Data Upload & Exploration")
    print("   ✅ Exploratory Data Analysis (EDA)")
    print("   ✅ Model Training (ML & DL)")
    print("   ✅ Performance Evaluation")
    print("   ✅ Model Download (PKL Format)")
    print()
    print("🎯 Supported Models:")
    print("   • Classification: Logistic Regression, Random Forest, SVM, Gradient Boosting")
    print("   • Regression: Linear Regression, Random Forest, SVM, Gradient Boosting")
    print()
    print("⏹️  Press Ctrl+C to stop the application")
    print()
    print("=" * 70)
    print()
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/main.py",
            "--logger.level=error"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Application stopped.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
