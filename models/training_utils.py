"""
Model Training Utilities

Handles model training with appropriate strategies based on model type.
"""

from sklearn.model_selection import cross_val_score
from models.model_config import (
    get_category_strategy, get_cv_folds_config, is_tree_based,
    is_iterative, is_deep_learning
)


def train_model_with_strategy(model, X_train, y_train, model_name, task_type):
    """
    Train model using appropriate strategy.
    Returns: (trained_model, cv_scores or None)
    """
    strategy = get_category_strategy(model_name)
    
    if strategy == "k-fold_cv":
        cv_config = get_cv_folds_config(model_name)
        cv_folds = cv_config["default"] if cv_config else 5
        
        scoring = 'accuracy' if task_type == "classification" else 'r2'
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
        
        model.fit(X_train, y_train)
        return model, cv_scores
    
    else:  # epochs strategy
        model.fit(X_train, y_train)
        return model, None


def get_training_info(model_name):
    """Get training information for a model."""
    strategy = get_category_strategy(model_name)
    
    info = {
        "model_name": model_name,
        "strategy": strategy,
        "is_tree_based": is_tree_based(model_name),
        "is_iterative": is_iterative(model_name),
        "is_deep_learning": is_deep_learning(model_name),
    }
    
    if strategy == "k-fold_cv":
        cv_config = get_cv_folds_config(model_name)
        info["cv_folds"] = cv_config["default"] if cv_config else 5
        info["uses_cv"] = True
    else:
        info["uses_cv"] = False
    
    return info


def apply_hyperparams(model, hyperparams):
    """Apply hyperparameters to model (excluding cv_folds)."""
    params_to_apply = {k: v for k, v in hyperparams.items() if k != 'cv_folds'}
    
    if params_to_apply:
        model.set_params(**params_to_apply)
    
    return model
