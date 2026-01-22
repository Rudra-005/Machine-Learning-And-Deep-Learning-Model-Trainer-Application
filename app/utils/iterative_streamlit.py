"""
Streamlit Integration for Iterative ML Models

Exposes max_iter parameter and integrates with cross-validation.
"""

import streamlit as st
from models.iterative_models import IterativeModelHandler
from evaluation.kfold_validator import KFoldCrossValidator


def render_iterative_model_config(model_name):
    """
    Render configuration UI for iterative models.
    
    Returns:
        max_iter: Maximum iterations for convergence
        additional_params: Dict of additional parameters
    """
    
    if not IterativeModelHandler.is_iterative_model(model_name):
        return None, {}
    
    model_info = IterativeModelHandler.get_model_info(model_name)
    max_iter_range = model_info['max_iter_range']
    default_max_iter = model_info['default_max_iter']
    
    st.markdown("**Iterative Model Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_iter = st.slider(
            "Max Iterations",
            min_value=max_iter_range[0],
            max_value=max_iter_range[1],
            value=default_max_iter,
            step=100,
            help="Maximum iterations for convergence (NOT epochs). "
                 "Iterative models converge when loss stops improving or max_iter reached."
        )
    
    with col2:
        st.info(
            f"**{model_info['name']}**\n\n"
            f"{model_info['description']}\n\n"
            f"Max iterations: {max_iter_range[0]}-{max_iter_range[1]}"
        )
    
    additional_params = {}
    
    # Model-specific parameters
    if model_name == 'sgd_classifier':
        loss = st.selectbox(
            "Loss Function",
            ['hinge', 'log', 'modified_huber', 'squared_hinge'],
            help="Loss function for SGD"
        )
        additional_params['loss'] = loss
    
    elif model_name == 'logistic_regression':
        solver = st.selectbox(
            "Solver",
            ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
            help="Algorithm for optimization"
        )
        additional_params['solver'] = solver
    
    return max_iter, additional_params


def train_iterative_model(model_name, X_train, y_train, X_test, y_test,
                          max_iter, cv_folds=5, **kwargs):
    """
    Train iterative model with CV integration.
    
    Returns:
        trained_model: Fitted model
        cv_scores: Cross-validation scores
        predictions: Test predictions
    """
    
    with st.spinner(f"⏳ Training {model_name} (max_iter={max_iter})..."):
        # Create model with max_iter
        model = IterativeModelHandler.create_iterative_model(
            model_name, max_iter=max_iter, **kwargs
        )
        
        # Train with CV
        trained_model, cv_scores, predictions = IterativeModelHandler.train_iterative_with_cv(
            model, X_train, y_train, X_test, y_test, model_name, cv_folds
        )
    
    return trained_model, cv_scores, predictions


def display_iterative_model_info(model_name):
    """Display information about iterative model."""
    
    if not IterativeModelHandler.is_iterative_model(model_name):
        return
    
    model_info = IterativeModelHandler.get_model_info(model_name)
    
    st.markdown("### Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Model**: {model_info['name']}")
        st.write(f"**Type**: Iterative ML")
        st.write(f"**Parameter**: max_iter (convergence iterations)")
    
    with col2:
        st.write(f"**Description**: {model_info['description']}")
        st.write(f"**Range**: {model_info['max_iter_range'][0]}-{model_info['max_iter_range'][1]}")
        st.write(f"**Default**: {model_info['default_max_iter']}")


def display_iterative_training_results(cv_scores, metrics):
    """Display training results for iterative model."""
    
    st.divider()
    st.markdown("### Training Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cross-Validation**")
        st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
        st.metric("Std Dev", f"{cv_scores.std():.4f}")
        st.metric("Folds", len(cv_scores))
    
    with col2:
        st.markdown("**Test Set Performance**")
        metric_count = 0
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and metric_count < 3:
                st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                metric_count += 1
