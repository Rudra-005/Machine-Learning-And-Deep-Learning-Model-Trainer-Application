"""
Canonical Session State Contract for Dataset Loading

RULE: st.session_state["dataset"] is the ONLY indicator that data is loaded.
- If dataset is None: no data loaded
- If dataset is not None: data is loaded and ready
- All pages must use is_data_loaded() helper

This ensures single source of truth across all pages.
"""

import streamlit as st


def is_data_loaded():
    """
    Canonical check: data is loaded if dataset exists.
    
    This is the ONLY function that should be used to check if data is loaded.
    Use this in all pages instead of checking multiple flags.
    
    Returns:
        bool: True if dataset is loaded, False otherwise
    """
    return st.session_state.get("dataset") is not None


def initialize_session_state():
    """
    Initialize all session state variables with canonical dataset as source of truth.
    
    Call this once at app startup.
    """
    # Dataset is the canonical indicator
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    
    # Preprocessed data
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_val" not in st.session_state:
        st.session_state.X_val = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_val" not in st.session_state:
        st.session_state.y_val = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    
    # Model training
    if "model" not in st.session_state:
        st.session_state.model = None
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "training_history" not in st.session_state:
        st.session_state.training_history = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False


def clear_dataset():
    """Clear dataset and all dependent data."""
    st.session_state.dataset = None
    st.session_state.X_train = None
    st.session_state.X_val = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_val = None
    st.session_state.y_test = None
    st.session_state.preprocessor = None
