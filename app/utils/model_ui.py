"""
Streamlit UI Utilities for Model Configuration

Dynamically renders hyperparameters based on model config.
"""

import streamlit as st
from models.model_config import (
    get_model_config, get_model_params, get_category_strategy,
    get_cv_folds_config, is_tree_based, is_iterative, is_deep_learning
)


def render_hyperparameters(model_name):
    """
    Render hyperparameter UI for a model.
    Returns dictionary of selected hyperparameters.
    """
    hyperparams = {}
    config = get_model_config(model_name)
    
    if not config:
        return hyperparams
    
    strategy = get_category_strategy(model_name)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if strategy == "k-fold_cv":
            st.info("🎯 K-fold cross-validation for robust evaluation")
            cv_config = get_cv_folds_config(model_name)
            if cv_config:
                hyperparams['cv_folds'] = st.slider(
                    "K-Fold Splits",
                    cv_config["min"],
                    cv_config["max"],
                    cv_config["default"],
                    help="Number of cross-validation folds"
                )
        else:
            st.info("🧠 Epochs for iterative training")
        
        params = get_model_params(model_name)
        for param_name, param_cfg in params.items():
            hyperparams[param_name] = _render_param(param_name, param_cfg)
    
    with col2:
        if strategy == "k-fold_cv":
            st.markdown("**Validation Strategy**")
            cv_config = get_cv_folds_config(model_name)
            if cv_config:
                st.write(f"• K-Fold: {hyperparams.get('cv_folds', cv_config['default'])} splits")
            st.write("• Stratified: Yes")
            st.write("• Shuffle: Yes")
            st.write("• Random State: 42")
        else:
            st.markdown("**Training Config**")
            st.write(f"• Epochs: {hyperparams.get('epochs', 50)}")
            st.write(f"• Batch Size: {hyperparams.get('batch_size', 32)}")
            st.write(f"• Learning Rate: {hyperparams.get('learning_rate', 0.001)}")
            st.write("• Optimizer: Adam")
    
    return hyperparams


def _render_param(param_name, param_cfg):
    """Render a single parameter based on its configuration."""
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


def get_strategy_info(model_name):
    """Get training strategy information for a model."""
    strategy = get_category_strategy(model_name)
    
    if strategy == "k-fold_cv":
        return {
            "type": "cross_validation",
            "description": "K-fold cross-validation",
            "uses_epochs": False
        }
    else:
        return {
            "type": "epochs",
            "description": "Iterative training with epochs",
            "uses_epochs": True
        }
