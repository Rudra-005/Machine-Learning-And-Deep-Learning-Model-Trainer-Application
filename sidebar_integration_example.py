"""
Sidebar Refactor - Integration Example

Minimal example showing how to integrate refactored sidebar into main app.
"""

import streamlit as st
from sidebar_refactored import render_sidebar_navigation


def initialize_session_state():
    """Initialize session state."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False


def main():
    """Main app with refactored sidebar."""
    st.set_page_config(page_title="ML/DL Trainer", page_icon="🤖", layout="wide")
    
    initialize_session_state()
    
    # Render sidebar and get current page
    pages = ["🏠 Home", "📊 Data Loading", "🧠 AutoML Training", "📈 Results"]
    page = render_sidebar_navigation(pages)
    
    # Route to pages
    if page == "🏠 Home":
        st.title("🤖 ML/DL Trainer")
        st.write("Welcome to ML/DL Trainer")
    
    elif page == "📊 Data Loading":
        st.title("📊 Data Loading")
        if st.button("Load Sample"):
            import pandas as pd
            import numpy as np
            np.random.seed(42)
            X = np.random.randn(100, 5)
            st.session_state.dataset = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
            st.success("✓ Loaded")
        
        if st.session_state.dataset is not None:
            st.write(st.session_state.dataset.head())
    
    elif page == "🧠 AutoML Training":
        st.title("🧠 AutoML Training")
        if st.session_state.dataset is None:
            st.warning("Load data first")
        else:
            st.write("Training...")
            if st.button("Train"):
                st.session_state.model_trained = True
                st.success("✓ Trained")
    
    elif page == "📈 Results":
        st.title("📈 Results")
        if not st.session_state.model_trained:
            st.warning("Train model first")
        else:
            st.write("Results here")


if __name__ == "__main__":
    main()
