#!/usr/bin/env python3
"""
Simple verification and runner for the Streamlit app.
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("ML/DL Trainer - Streamlit Application")
    print("=" * 70)
    
    print("\nStarting Streamlit app...")
    print("The app will open at: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app/main.py"],
            cwd=Path(__file__).parent
        )
    except KeyboardInterrupt:
        print("\n\nStreamlit server stopped.")
        sys.exit(0)

if __name__ == "__main__":
    main()
