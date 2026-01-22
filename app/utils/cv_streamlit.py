"""
Streamlit Integration for K-Fold Cross-Validation

Provides UI components and training pipeline for k-fold CV.
"""

import streamlit as st
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from evaluation.kfold_validator import (
    KFoldCrossValidator, train_ml_with_cv, display_cv_summary, get_cv_config
)
from models.model_config import is_deep_learning, is_tree_based, is_iterative
from evaluation.metrics import MetricsCalculator


def render_cv_config(model_name):
    """
    Render CV configuration UI.
    
    Returns:
        k: Number of folds
        enable_cv: Whether to use CV
    """
    if is_deep_learning(model_name):
        return None, False
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_cv = st.checkbox(
            "Enable Cross-Validation",
            value=True,
            help="Use k-fold cross-validation for robust evaluation"
        )
    
    with col2:
        if enable_cv:
            k = st.slider(
                "K-Fold Splits",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
        else:
            k = None
    
    return k, enable_cv


def train_with_cv_pipeline(model, X_train, y_train, X_test, y_test, 
                           k, task_type, model_name):
    """
    Complete training pipeline with k-fold CV.
    
    Returns:
        trained_model: Fitted model
        cv_results: CV metrics
        predictions: Test predictions
        metrics: Test set metrics
    """
    
    with st.spinner("⏳ Training with cross-validation..."):
        # Train with CV
        trained_model, cv_results, predictions = train_ml_with_cv(
            model, X_train, y_train, X_test, y_test, k, task_type, model_name
        )
        
        # Compute test metrics
        if task_type.lower() == "classification":
            metrics = MetricsCalculator.classification_metrics(y_test, predictions)
        else:
            metrics = MetricsCalculator.regression_metrics(y_test, predictions)
        
        return trained_model, cv_results, predictions, metrics


def display_training_results(cv_results, metrics, task_type):
    """Display complete training results."""
    
    st.divider()
    st.markdown("### Training Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cross-Validation**")
        if cv_results:
            display_cv_summary(cv_results)
    
    with col2:
        st.markdown("**Test Set Performance**")
        metric_count = 0
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and metric_count < 4:
                st.metric(
                    key.replace("_", " ").title(),
                    f"{value:.4f}"
                )
                metric_count += 1


def get_cv_info_text(k, task_type):
    """Get informational text about CV configuration."""
    cv_config = get_cv_config(k)
    
    info = f"""
    **Cross-Validation Configuration:**
    - Folds: {cv_config['n_splits']}
    - Shuffle: {cv_config['shuffle']}
    - Stratified: {cv_config['stratified']} (for {task_type})
    - Random State: {cv_config['random_state']}
    """
    
    return info
