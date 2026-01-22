"""
Streamlit Integration for Hyperparameter Optimization

Provides UI for optional HP tuning in advanced settings.
"""

import streamlit as st
from evaluation.hp_optimizer import (
    HyperparameterOptimizer, train_with_hp_optimization
)
from models.model_config import is_deep_learning


def render_hp_optimization_config(model_name):
    """
    Render HP optimization configuration in advanced settings.
    
    Returns:
        enable_tuning: Whether to enable HP tuning
        n_iter: Number of search iterations
    """
    
    if is_deep_learning(model_name):
        return False, None
    
    enable_tuning = st.checkbox(
        "🔍 Enable Hyperparameter Optimization",
        value=False,
        help="Use RandomizedSearchCV to find best hyperparameters (advanced)"
    )
    
    if enable_tuning:
        n_iter = st.slider(
            "Search Iterations",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Number of parameter combinations to test"
        )
        
        st.info(
            f"Will test {n_iter} random parameter combinations using k-fold CV. "
            "This may take longer but finds better hyperparameters."
        )
        
        return enable_tuning, n_iter
    
    return False, None


def train_with_optional_tuning(model, X_train, y_train, X_test, y_test,
                               model_name, task_type, cv_folds,
                               enable_tuning=False, n_iter=20):
    """
    Train model with optional HP optimization.
    
    Returns:
        trained_model: Fitted model
        search_results: HP optimization results (or None)
        predictions: Test predictions
    """
    
    if enable_tuning and not is_deep_learning(model_name):
        with st.spinner(f"🔍 Optimizing hyperparameters ({n_iter} iterations)..."):
            trained_model, search_results, predictions = train_with_hp_optimization(
                model, X_train, y_train, X_test, y_test,
                model_name, task_type, n_iter, cv_folds
            )
        return trained_model, search_results, predictions
    else:
        # Standard training without HP optimization
        with st.spinner("⏳ Training model..."):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
        return model, None, predictions


def display_hp_optimization_results(search_results):
    """Display HP optimization results."""
    if search_results is None:
        return
    
    HyperparameterOptimizer.display_optimization_results(search_results)


def get_hp_optimization_info():
    """Get information about HP optimization."""
    return """
    **Hyperparameter Optimization:**
    - Uses RandomizedSearchCV for efficient search
    - Reuses k-fold cross-validation
    - Tests random parameter combinations
    - Selects best estimator automatically
    - Optional advanced feature
    """
