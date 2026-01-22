"""
Streamlit AutoML UI: Render dynamic controls based on auto-detected model type.

Shows only relevant parameters for the selected model without user confusion.
"""

import streamlit as st
from typing import Any, Dict, Tuple
from models.automl import AutoMLConfig, get_strategy_explanation


def render_automl_mode(model: Any) -> Dict[str, Any]:
    """
    Render AutoML mode UI with auto-detected strategy and relevant controls.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with user-selected parameters
    """
    automl = AutoMLConfig(model)
    ui_config = automl.get_ui_config()
    
    # Display strategy info
    st.info(
        f"🤖 **AutoML Mode Active**\n\n"
        f"**Model:** {ui_config['model_name']}\n\n"
        f"**Strategy:** {ui_config['strategy']}\n\n"
        f"Optimal parameters are automatically configured for this model type."
    )
    
    # Render visible parameters
    params = {}
    visible = ui_config['visible_parameters']
    defaults = ui_config['defaults']
    
    col1, col2 = st.columns(2)
    
    # K-Fold CV
    if visible['cv_folds']:
        with col1:
            params['cv_folds'] = st.slider(
                "K-Fold Cross-Validation",
                min_value=3,
                max_value=10,
                value=defaults['cv_folds'],
                help="Number of folds for cross-validation. More folds = more robust but slower."
            )
    
    # Max Iterations (for iterative models)
    if visible['max_iter']:
        with col2:
            params['max_iter'] = st.slider(
                "Max Iterations (Convergence)",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                help="Maximum iterations for convergence. Increase if model doesn't converge."
            )
    
    # Epochs (for deep learning)
    if visible['epochs']:
        with col1:
            params['epochs'] = st.slider(
                "Epochs",
                min_value=10,
                max_value=200,
                value=defaults['epochs'],
                help="Number of passes through the training data."
            )
        
        with col2:
            params['batch_size'] = st.slider(
                "Batch Size",
                min_value=16,
                max_value=128,
                value=defaults['batch_size'],
                step=16,
                help="Number of samples per gradient update."
            )
    
    # Learning Rate (for iterative and DL models)
    if visible['learning_rate']:
        with col1:
            params['learning_rate'] = st.slider(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=defaults['learning_rate'],
                step=0.0001,
                format="%.4f",
                help="Step size for optimization. Smaller = slower but more stable."
            )
    
    # Hyperparameter Tuning
    if visible['hp_tuning']:
        with col2:
            params['enable_hp_tuning'] = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=False,
                help="Search for optimal hyperparameters (slower but better results)."
            )
            
            if params['enable_hp_tuning']:
                params['hp_iterations'] = st.slider(
                    "Tuning Iterations",
                    min_value=5,
                    max_value=100,
                    value=defaults['hp_iterations'],
                    step=5,
                    help="Number of random hyperparameter combinations to try."
                )
    
    # Early Stopping (for DL)
    if visible['early_stopping']:
        with col2:
            params['early_stopping'] = st.checkbox(
                "Enable Early Stopping",
                value=True,
                help="Stop training if validation loss doesn't improve."
            )
    
    return params


def render_automl_summary(model: Any, params: Dict[str, Any]) -> None:
    """
    Display summary of AutoML configuration.
    
    Args:
        model: Model instance
        params: Selected parameters
    """
    automl = AutoMLConfig(model)
    config = automl.config
    
    st.subheader("📋 Training Configuration Summary")
    
    summary_cols = st.columns(3)
    
    with summary_cols[0]:
        st.metric("Model Type", config['model_name'])
    
    with summary_cols[1]:
        st.metric("Strategy", config['description'].split('(')[0].strip())
    
    with summary_cols[2]:
        if config['use_epochs']:
            st.metric("Epochs", params.get('epochs', config.get('epochs', 50)))
        elif config['use_epochs'] is False:
            st.metric("CV Folds", params.get('cv_folds', config.get('cv_folds', 5)))
    
    # Detailed explanation
    with st.expander("📖 Why this strategy?"):
        st.markdown(get_strategy_explanation(model))


def render_automl_comparison(model: Any) -> None:
    """
    Show why this strategy was chosen vs alternatives.
    
    Args:
        model: Model instance
    """
    automl = AutoMLConfig(model)
    category = automl.config['category']
    
    st.subheader("🔍 Strategy Comparison")
    
    if category == 'tree_based':
        st.markdown("""
        **Why K-Fold CV (not epochs)?**
        - Tree-based models converge in a single pass
        - Epochs would redundantly retrain the same model
        - K-Fold CV provides robust overfitting detection
        - All data used for training (no wasted validation set)
        """)
    
    elif category == 'iterative':
        st.markdown("""
        **Why K-Fold CV with max_iter (not epochs)?**
        - Iterative models need convergence control (max_iter)
        - K-Fold CV ensures robust generalization
        - max_iter prevents infinite training loops
        - More efficient than epochs for this model class
        """)
    
    elif category == 'deep_learning':
        st.markdown("""
        **Why Epochs with Early Stopping (not CV)?**
        - Neural networks require multiple passes through data
        - Epochs track training progress across batches
        - Early stopping prevents overfitting automatically
        - K-Fold CV too expensive for deep learning
        """)
    
    elif category == 'svm':
        st.markdown("""
        **Why K-Fold CV (not epochs)?**
        - SVMs solve optimization in a single pass
        - K-Fold CV validates kernel and regularization choices
        - Hyperparameter tuning more important than iterations
        """)


def get_automl_training_info(model: Any) -> str:
    """
    Get training information message for AutoML mode.
    
    Args:
        model: Model instance
        
    Returns:
        Information message
    """
    automl = AutoMLConfig(model)
    config = automl.config
    
    if config['use_epochs']:
        return (
            f"Training {config['model_name']} with **{config['description']}**. "
            f"Model will train for multiple epochs with early stopping to prevent overfitting."
        )
    else:
        return (
            f"Training {config['model_name']} with **{config['description']}**. "
            f"Model will be evaluated across {config.get('cv_folds', 5)} folds for robust performance estimation."
        )


def display_automl_training_progress(model: Any, progress_info: Dict[str, Any]) -> None:
    """
    Display training progress in AutoML mode.
    
    Args:
        model: Model instance
        progress_info: Training progress information
    """
    automl = AutoMLConfig(model)
    config = automl.config
    
    if config['use_epochs']:
        # DL training progress
        st.subheader("📊 Training Progress")
        
        if 'epoch' in progress_info:
            col1, col2, col3 = st.columns(3)
            col1.metric("Epoch", progress_info['epoch'])
            col2.metric("Train Loss", f"{progress_info.get('train_loss', 0):.4f}")
            col3.metric("Val Loss", f"{progress_info.get('val_loss', 0):.4f}")
    
    else:
        # ML training progress (CV)
        st.subheader("📊 Cross-Validation Progress")
        
        if 'fold' in progress_info:
            col1, col2, col3 = st.columns(3)
            col1.metric("Fold", progress_info['fold'])
            col2.metric("Fold Score", f"{progress_info.get('fold_score', 0):.4f}")
            col3.metric("Mean CV Score", f"{progress_info.get('mean_score', 0):.4f}")


def display_automl_results(model: Any, results: Dict[str, Any]) -> None:
    """
    Display training results in AutoML mode.
    
    Args:
        model: Model instance
        results: Training results
    """
    automl = AutoMLConfig(model)
    config = automl.config
    
    st.subheader("✅ Training Complete")
    
    if config['use_epochs']:
        # DL results
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Train Loss", f"{results.get('train_loss', 0):.4f}")
        col2.metric("Final Val Loss", f"{results.get('val_loss', 0):.4f}")
        col3.metric("Test Accuracy", f"{results.get('test_accuracy', 0):.4f}")
    
    else:
        # ML results (CV)
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean CV Score", f"{results.get('cv_mean', 0):.4f}")
        col2.metric("Std Dev", f"{results.get('cv_std', 0):.4f}")
        col3.metric("Test Score", f"{results.get('test_score', 0):.4f}")
        
        # Confidence interval
        cv_mean = results.get('cv_mean', 0)
        cv_std = results.get('cv_std', 0)
        ci_lower = cv_mean - 1.96 * cv_std
        ci_upper = cv_mean + 1.96 * cv_std
        
        st.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Hyperparameter tuning results
    if results.get('hp_tuning_enabled'):
        st.subheader("🎯 Best Hyperparameters")
        best_params = results.get('best_params', {})
        
        param_cols = st.columns(len(best_params))
        for i, (param, value) in enumerate(best_params.items()):
            with param_cols[i % len(param_cols)]:
                st.metric(param, value)
