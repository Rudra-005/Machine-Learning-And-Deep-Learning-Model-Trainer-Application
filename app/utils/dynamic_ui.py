"""
Dynamic UI Parameter Logic

Conditionally renders training parameters based on:
- Model category (tree-based, iterative, deep learning)
- Tuning enabled/disabled
- Parameter validity rules
"""

import streamlit as st
from models.model_config import (
    is_tree_based, is_iterative, is_deep_learning,
    get_model_params, get_cv_folds_config
)


class ParameterValidator:
    """Validates parameter values against rules."""
    
    @staticmethod
    def validate_cv_folds(value, model_name):
        """Validate k-fold value."""
        config = get_cv_folds_config(model_name)
        if not config:
            return False, "Model does not support cross-validation"
        
        if value < config["min"] or value > config["max"]:
            return False, f"K-fold must be between {config['min']} and {config['max']}"
        
        return True, None
    
    @staticmethod
    def validate_max_iter(value):
        """Validate max iterations for iterative models."""
        if value < 100 or value > 10000:
            return False, "Max iterations must be between 100 and 10000"
        return True, None
    
    @staticmethod
    def validate_epochs(value):
        """Validate epochs for deep learning."""
        if value < 1 or value > 500:
            return False, "Epochs must be between 1 and 500"
        return True, None
    
    @staticmethod
    def validate_hp_iterations(value):
        """Validate hyperparameter search iterations."""
        if value < 5 or value > 100:
            return False, "HP search iterations must be between 5 and 100"
        return True, None


def render_training_parameters(model_name, task_type):
    """
    Render training parameters conditionally.
    
    Rules:
    - Show CV folds for ALL ML models
    - Show HP search iterations only when tuning enabled
    - Show max_iter ONLY for iterative ML
    - Show epochs ONLY for deep learning
    - Hide invalid parameters automatically
    """
    
    params = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Parameters**")
        
        # Rule 1: Show CV folds for ALL ML models
        if is_tree_based(model_name) or is_iterative(model_name):
            cv_config = get_cv_folds_config(model_name)
            if cv_config:
                cv_value = st.slider(
                    "Cross-Validation Folds (k)",
                    cv_config["min"],
                    cv_config["max"],
                    cv_config["default"],
                    help="Number of folds for k-fold cross-validation"
                )
                is_valid, error = ParameterValidator.validate_cv_folds(cv_value, model_name)
                if is_valid:
                    params['cv_folds'] = cv_value
                else:
                    st.error(f"❌ {error}")
        
        # Rule 3: Show max_iter ONLY for iterative ML models
        if is_iterative(model_name):
            max_iter_value = st.slider(
                "Max Iterations",
                100,
                10000,
                100,
                step=100,
                help="Maximum iterations for convergence"
            )
            is_valid, error = ParameterValidator.validate_max_iter(max_iter_value)
            if is_valid:
                params['max_iter'] = max_iter_value
            else:
                st.error(f"❌ {error}")
        
        # Rule 4: Show epochs ONLY for deep learning models
        if is_deep_learning(model_name):
            epochs_value = st.slider(
                "Epochs",
                1,
                500,
                50,
                help="Number of training epochs"
            )
            is_valid, error = ParameterValidator.validate_epochs(epochs_value)
            if is_valid:
                params['epochs'] = epochs_value
            else:
                st.error(f"❌ {error}")
        
        # Model-specific parameters
        model_params = get_model_params(model_name)
        for param_name, param_cfg in model_params.items():
            if param_name not in params:
                param_value = _render_model_param(param_name, param_cfg)
                if param_value is not None:
                    params[param_name] = param_value
    
    with col2:
        st.markdown("**Advanced Options**")
        
        # Rule 2: Show HP search iterations only when tuning enabled
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
        
        if enable_tuning:
            hp_iter_value = st.slider(
                "Hyperparameter Search Iterations",
                5,
                100,
                20,
                step=5,
                help="Number of iterations for hyperparameter search"
            )
            is_valid, error = ParameterValidator.validate_hp_iterations(hp_iter_value)
            if is_valid:
                params['hp_search_iterations'] = hp_iter_value
            else:
                st.error(f"❌ {error}")
        
        # Show strategy info
        st.markdown("**Strategy**")
        if is_tree_based(model_name) or is_iterative(model_name):
            st.write("• K-fold cross-validation")
            st.write("• Stratified split (classification)")
            st.write("• Random state: 42")
        elif is_deep_learning(model_name):
            st.write("• Iterative training")
            st.write("• Batch-based updates")
            st.write("• Optimizer: Adam")
    
    return params, enable_tuning


def _render_model_param(param_name, param_cfg):
    """Render a single model parameter."""
    param_type = param_cfg.get("type")
    label = param_cfg.get("label", param_name)
    
    if param_type == "slider":
        return st.slider(
            label,
            param_cfg["min"],
            param_cfg["max"],
            param_cfg["default"],
            step=param_cfg.get("step", 1)
        )
    
    elif param_type == "selectbox":
        return st.selectbox(
            label,
            param_cfg["options"],
            index=param_cfg["options"].index(param_cfg["default"])
        )
    
    elif param_type == "number_input":
        return st.number_input(
            label,
            param_cfg["min"],
            param_cfg["max"],
            param_cfg["default"]
        )
    
    return None


def validate_training_params(params, model_name, task_type):
    """
    Validate all training parameters.
    Returns: (is_valid: bool, errors: list)
    """
    errors = []
    
    # Validate CV folds for ML models
    if 'cv_folds' in params:
        is_valid, error = ParameterValidator.validate_cv_folds(params['cv_folds'], model_name)
        if not is_valid:
            errors.append(error)
    
    # Validate max_iter for iterative models
    if 'max_iter' in params:
        is_valid, error = ParameterValidator.validate_max_iter(params['max_iter'])
        if not is_valid:
            errors.append(error)
    
    # Validate epochs for deep learning
    if 'epochs' in params:
        is_valid, error = ParameterValidator.validate_epochs(params['epochs'])
        if not is_valid:
            errors.append(error)
    
    # Validate HP search iterations
    if 'hp_search_iterations' in params:
        is_valid, error = ParameterValidator.validate_hp_iterations(params['hp_search_iterations'])
        if not is_valid:
            errors.append(error)
    
    return len(errors) == 0, errors


def display_parameter_summary(params, model_name):
    """Display summary of selected parameters."""
    st.markdown("### Parameter Summary")
    
    summary_cols = st.columns(2)
    
    with summary_cols[0]:
        st.markdown("**Selected Parameters**")
        for key, value in params.items():
            st.write(f"• {key}: {value}")
    
    with summary_cols[1]:
        st.markdown("**Model Info**")
        st.write(f"• Model: {model_name}")
        st.write(f"• Category: {_get_category_name(model_name)}")
        st.write(f"• Strategy: {_get_strategy_name(model_name)}")


def _get_category_name(model_name):
    """Get human-readable category name."""
    if is_tree_based(model_name):
        return "Tree-Based ML"
    elif is_iterative(model_name):
        return "Iterative ML"
    elif is_deep_learning(model_name):
        return "Deep Learning"
    return "Unknown"


def _get_strategy_name(model_name):
    """Get human-readable strategy name."""
    if is_tree_based(model_name) or is_iterative(model_name):
        return "K-Fold Cross-Validation"
    elif is_deep_learning(model_name):
        return "Epochs"
    return "Unknown"
