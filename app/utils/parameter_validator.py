"""
Parameter Validation Layer

Prevents:
- Epochs being passed to ML models
- max_iter being passed to tree-based models
- CV being applied incorrectly to DL models

Raises clean, user-friendly warnings instead of errors.
"""

import streamlit as st
from models.model_config import is_tree_based, is_iterative, is_deep_learning


class ParameterValidator:
    """Validates parameter combinations for models."""
    
    @staticmethod
    def validate_epochs_usage(model_name, epochs):
        """
        Validate epochs parameter.
        
        Epochs should ONLY be used for DL models.
        ML models should use max_iter or CV folds.
        """
        if epochs is None:
            return True, None
        
        if is_deep_learning(model_name):
            return True, None
        
        # ML model with epochs - warning
        warning = (
            f"⚠️ **Epochs not applicable for {model_name}**\n\n"
            f"Epochs are for deep learning models (Sequential, CNN, RNN).\n\n"
            f"For ML models:\n"
            f"- Use **K-Fold Cross-Validation** for robust evaluation\n"
            f"- Use **Max Iterations** for iterative models (LogisticRegression, SGD)\n"
            f"- Tree-based models don't need iterations\n\n"
            f"Epochs parameter will be ignored."
        )
        return False, warning
    
    @staticmethod
    def validate_max_iter_usage(model_name, max_iter):
        """
        Validate max_iter parameter.
        
        max_iter should ONLY be used for iterative ML models.
        Tree-based models don't need max_iter.
        DL models use epochs instead.
        """
        if max_iter is None:
            return True, None
        
        if is_iterative(model_name):
            return True, None
        
        # Tree-based or DL model with max_iter - warning
        if is_tree_based(model_name):
            warning = (
                f"⚠️ **Max Iterations not applicable for {model_name}**\n\n"
                f"Max iterations are for iterative models (LogisticRegression, SGD, Perceptron).\n\n"
                f"Tree-based models ({model_name}) don't use iterations.\n"
                f"They use:\n"
                f"- **n_estimators**: Number of trees\n"
                f"- **max_depth**: Tree depth\n"
                f"- **K-Fold CV**: For evaluation\n\n"
                f"Max iterations parameter will be ignored."
            )
        else:  # DL model
            warning = (
                f"⚠️ **Max Iterations not applicable for {model_name}**\n\n"
                f"Max iterations are for iterative ML models.\n\n"
                f"Deep learning models use:\n"
                f"- **Epochs**: Multiple passes through data\n"
                f"- **Batch Size**: Samples per update\n"
                f"- **Learning Rate**: Optimizer step size\n\n"
                f"Max iterations parameter will be ignored."
            )
        
        return False, warning
    
    @staticmethod
    def validate_cv_usage(model_name, cv_folds):
        """
        Validate cross-validation usage.
        
        CV should NOT be applied to DL models.
        ML models should use CV for robust evaluation.
        """
        if cv_folds is None:
            return True, None
        
        if is_deep_learning(model_name):
            warning = (
                f"⚠️ **K-Fold CV not applicable for {model_name}**\n\n"
                f"K-fold cross-validation is for ML models.\n\n"
                f"Deep learning models use:\n"
                f"- **Train/Validation/Test split**: Separate datasets\n"
                f"- **Epochs**: Multiple passes through training data\n"
                f"- **Early Stopping**: Monitor validation loss\n\n"
                f"CV parameter will be ignored. Using train/val/test split instead."
            )
            return False, warning
        
        # ML model with CV - this is correct
        return True, None
    
    @staticmethod
    def validate_batch_size_usage(model_name, batch_size):
        """
        Validate batch_size parameter.
        
        Batch size should ONLY be used for DL models.
        ML models don't use batches.
        """
        if batch_size is None:
            return True, None
        
        if is_deep_learning(model_name):
            return True, None
        
        # ML model with batch_size - warning
        warning = (
            f"⚠️ **Batch Size not applicable for {model_name}**\n\n"
            f"Batch size is for deep learning models.\n\n"
            f"ML models train on full dataset at once.\n"
            f"They don't use batches or epochs.\n\n"
            f"Batch size parameter will be ignored."
        )
        return False, warning
    
    @staticmethod
    def validate_learning_rate_usage(model_name, learning_rate):
        """
        Validate learning_rate parameter.
        
        Learning rate is primarily for DL models.
        Some iterative ML models use it, but it's not primary.
        """
        if learning_rate is None:
            return True, None
        
        if is_deep_learning(model_name) or is_iterative(model_name):
            return True, None
        
        # Tree-based model with learning_rate - warning
        warning = (
            f"⚠️ **Learning Rate not applicable for {model_name}**\n\n"
            f"Learning rate is for:\n"
            f"- **Deep Learning**: Neural networks, CNN, RNN\n"
            f"- **Iterative ML**: LogisticRegression, SGD, Perceptron\n\n"
            f"Tree-based models ({model_name}) don't use learning rates.\n\n"
            f"Learning rate parameter will be ignored."
        )
        return False, warning
    
    @staticmethod
    def validate_all_parameters(model_name, params_dict):
        """
        Validate all parameters for a model.
        
        Returns:
            is_valid: bool
            warnings: list of warning messages
        """
        warnings = []
        
        # Check each parameter
        if 'epochs' in params_dict:
            is_valid, warning = ParameterValidator.validate_epochs_usage(
                model_name, params_dict['epochs']
            )
            if not is_valid and warning:
                warnings.append(warning)
        
        if 'max_iter' in params_dict:
            is_valid, warning = ParameterValidator.validate_max_iter_usage(
                model_name, params_dict['max_iter']
            )
            if not is_valid and warning:
                warnings.append(warning)
        
        if 'cv_folds' in params_dict:
            is_valid, warning = ParameterValidator.validate_cv_usage(
                model_name, params_dict['cv_folds']
            )
            if not is_valid and warning:
                warnings.append(warning)
        
        if 'batch_size' in params_dict:
            is_valid, warning = ParameterValidator.validate_batch_size_usage(
                model_name, params_dict['batch_size']
            )
            if not is_valid and warning:
                warnings.append(warning)
        
        if 'learning_rate' in params_dict:
            is_valid, warning = ParameterValidator.validate_learning_rate_usage(
                model_name, params_dict['learning_rate']
            )
            if not is_valid and warning:
                warnings.append(warning)
        
        return len(warnings) == 0, warnings
    
    @staticmethod
    def display_warnings(warnings):
        """Display warnings in Streamlit."""
        if not warnings:
            return
        
        for warning in warnings:
            st.warning(warning)
    
    @staticmethod
    def get_valid_parameters(model_name, params_dict):
        """
        Filter parameters to only valid ones for the model.
        
        Returns:
            filtered_params: Dict with only valid parameters
        """
        filtered = {}
        
        for key, value in params_dict.items():
            if key == 'epochs' and is_deep_learning(model_name):
                filtered[key] = value
            elif key == 'max_iter' and is_iterative(model_name):
                filtered[key] = value
            elif key == 'cv_folds' and not is_deep_learning(model_name):
                filtered[key] = value
            elif key == 'batch_size' and is_deep_learning(model_name):
                filtered[key] = value
            elif key == 'learning_rate' and (is_deep_learning(model_name) or is_iterative(model_name)):
                filtered[key] = value
            elif key not in ['epochs', 'max_iter', 'cv_folds', 'batch_size', 'learning_rate']:
                # Keep model-specific parameters
                filtered[key] = value
        
        return filtered
