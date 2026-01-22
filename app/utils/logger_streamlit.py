"""
Streamlit Integration for Training Logger

Displays training logs in user-friendly format.
"""

import streamlit as st
from app.utils.training_logger import TrainingLogger, log_training_decision


def display_training_explanation(model_name, task_type, cv_folds=None, epochs=None):
    """Display training strategy explanation before training."""
    
    with st.expander("📖 How This Training Works", expanded=True):
        st.markdown(TrainingLogger.log_strategy_decision(model_name))
        
        st.divider()
        
        st.markdown("**Parameters Shown/Hidden:**")
        st.markdown(TrainingLogger.log_parameter_decisions(model_name, {}))
        
        st.divider()
        
        st.markdown(TrainingLogger.log_cv_explanation(model_name, cv_folds or 5))


def display_training_log_during(model_name, task_type, params_dict):
    """Display training log during training."""
    
    with st.expander("📋 Training Configuration", expanded=True):
        st.markdown(TrainingLogger.log_training_start(model_name, task_type, params_dict))


def display_training_log_after(model_name, metrics):
    """Display training log after training."""
    
    with st.expander("📋 Training Summary", expanded=True):
        st.markdown(TrainingLogger.log_training_complete(model_name, metrics))


def display_quick_summary(model_name, task_type, cv_folds=None, epochs=None):
    """Display quick summary of training decision."""
    
    summary = log_training_decision(model_name, task_type, cv_folds, epochs)
    
    st.markdown("### Training Configuration")
    st.markdown(summary)
