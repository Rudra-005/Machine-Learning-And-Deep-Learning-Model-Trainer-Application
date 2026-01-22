"""
Session State Manager

Centralized state access to prevent bugs and ensure consistency.
All pages use these functions instead of directly accessing st.session_state.
"""

import streamlit as st
import pandas as pd


def is_data_loaded():
    """Check if dataset is loaded."""
    return 'data' in st.session_state and st.session_state.data is not None


def get_dataset():
    """Get loaded dataset or None."""
    return st.session_state.get('data', None)


def set_dataset(data):
    """Set dataset in session state."""
    st.session_state.data = data


def clear_dataset():
    """Clear dataset and related preprocessing state."""
    st.session_state.data = None
    st.session_state.data_preprocessed = False
    st.session_state.X_train = None
    st.session_state.X_val = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_val = None
    st.session_state.y_test = None
    st.session_state.preprocessor = None


def is_model_trained():
    """Check if model is trained."""
    return st.session_state.get('model_trained', False) and st.session_state.get('trained_model') is not None


def get_trained_model():
    """Get trained model or None."""
    return st.session_state.get('trained_model', None)


def set_trained_model(model):
    """Set trained model in session state."""
    st.session_state.trained_model = model
    st.session_state.model_trained = True


def get_metrics():
    """Get evaluation metrics or None."""
    return st.session_state.get('metrics', None)


def set_metrics(metrics):
    """Set metrics in session state."""
    st.session_state.metrics = metrics


def clear_training_state():
    """Clear all training-related state."""
    st.session_state.trained_model = None
    st.session_state.model_trained = False
    st.session_state.training_history = None
    st.session_state.metrics = None
    st.session_state.last_task_type = None
    st.session_state.last_model_name = None


def initialize_defaults():
    """Initialize all session state defaults."""
    defaults = {
        'data': None,
        'data_preprocessed': False,
        'X_train': None,
        'X_val': None,
        'X_test': None,
        'y_train': None,
        'y_val': None,
        'y_test': None,
        'preprocessor': None,
        'model': None,
        'trained_model': None,
        'training_history': None,
        'metrics': None,
        'model_trained': False,
        'last_task_type': None,
        'last_model_name': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
