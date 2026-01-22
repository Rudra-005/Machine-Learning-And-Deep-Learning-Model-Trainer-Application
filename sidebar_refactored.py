"""
Refactored Sidebar Status Section

Checks ONLY for st.session_state["dataset"] existence.
No deprecated flags like "data_loaded" or "data_preprocessed".
"""

import streamlit as st


def render_sidebar_status():
    """Render sidebar status section with canonical checks."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status**")
    
    # Data status: check ONLY if dataset exists
    if st.session_state.get("dataset") is not None:
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.warning("⚠️ No Data Loaded")
    
    # Model status: check if model is trained
    if st.session_state.get("model_trained"):
        st.sidebar.success("✅ Model Trained")
    else:
        st.sidebar.info("ℹ️ No Model Trained")


def render_sidebar_navigation(pages):
    """Render sidebar navigation."""
    st.sidebar.title("🤖 ML/DL Trainer")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", options=pages)
    
    render_sidebar_status()
    
    return page


# Example usage in main app
if __name__ == "__main__":
    # Initialize session state
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    
    # Render sidebar
    pages = ["🏠 Home", "📊 Data Loading", "🧠 AutoML Training", "📈 Results"]
    page = render_sidebar_navigation(pages)
    
    st.write(f"Current page: {page}")
