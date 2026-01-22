"""
AutoML Mode: Automatic model type detection and strategy selection.

Detects model category, selects optimal training strategy, and applies
CV, tuning, or epochs intelligently without user configuration.
"""

from enum import Enum
from typing import Dict, Tuple, Any
import numpy as np
from sklearn.base import BaseEstimator


class ModelCategory(Enum):
    """Model categories with distinct training strategies."""
    TREE_BASED = "tree_based"
    ITERATIVE = "iterative"
    DEEP_LEARNING = "deep_learning"
    SVM = "svm"


class TrainingStrategy(Enum):
    """Training strategies for different model types."""
    K_FOLD_CV = "k_fold_cv"
    K_FOLD_CV_WITH_CONVERGENCE = "k_fold_cv_with_convergence"
    EPOCHS_WITH_EARLY_STOPPING = "epochs_with_early_stopping"


# Model categorization
MODEL_REGISTRY = {
    ModelCategory.TREE_BASED: [
        'RandomForestClassifier', 'RandomForestRegressor',
        'GradientBoostingClassifier', 'GradientBoostingRegressor',
        'DecisionTreeClassifier', 'DecisionTreeRegressor',
        'ExtraTreesClassifier', 'ExtraTreesRegressor'
    ],
    ModelCategory.ITERATIVE: [
        'LogisticRegression', 'SGDClassifier', 'SGDRegressor',
        'Perceptron', 'Ridge', 'Lasso', 'ElasticNet'
    ],
    ModelCategory.SVM: [
        'SVC', 'SVR', 'LinearSVC', 'LinearSVR'
    ],
    ModelCategory.DEEP_LEARNING: [
        'Sequential', 'CNN', 'LSTM', 'RNN', 'Functional'
    ]
}

# Strategy configuration per category
STRATEGY_CONFIG = {
    ModelCategory.TREE_BASED: {
        'strategy': TrainingStrategy.K_FOLD_CV,
        'cv_folds': 5,
        'use_epochs': False,
        'use_max_iter': False,
        'use_hp_tuning': True,
        'hp_iterations': 30,
        'description': 'K-Fold Cross-Validation (single pass, no epochs)'
    },
    ModelCategory.ITERATIVE: {
        'strategy': TrainingStrategy.K_FOLD_CV_WITH_CONVERGENCE,
        'cv_folds': 5,
        'use_epochs': False,
        'use_max_iter': True,
        'use_hp_tuning': True,
        'hp_iterations': 30,
        'description': 'K-Fold CV with convergence iterations'
    },
    ModelCategory.SVM: {
        'strategy': TrainingStrategy.K_FOLD_CV,
        'cv_folds': 5,
        'use_epochs': False,
        'use_max_iter': False,
        'use_hp_tuning': True,
        'hp_iterations': 30,
        'description': 'K-Fold Cross-Validation (kernel optimization)'
    },
    ModelCategory.DEEP_LEARNING: {
        'strategy': TrainingStrategy.EPOCHS_WITH_EARLY_STOPPING,
        'cv_folds': None,
        'use_epochs': True,
        'use_max_iter': False,
        'use_hp_tuning': False,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping_patience': 5,
        'description': 'Epochs with Early Stopping'
    }
}


def detect_model_category(model: Any) -> ModelCategory:
    """
    Auto-detect model category from model instance or class name.
    
    Args:
        model: Model instance or class
        
    Returns:
        ModelCategory enum value
    """
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else str(model)
    
    for category, models in MODEL_REGISTRY.items():
        if model_name in models:
            return category
    
    # Default fallback based on module
    if hasattr(model, '__module__'):
        if 'keras' in model.__module__ or 'tensorflow' in model.__module__:
            return ModelCategory.DEEP_LEARNING
        elif 'sklearn' in model.__module__:
            return ModelCategory.TREE_BASED
    
    raise ValueError(f"Unknown model type: {model_name}")


def get_training_config(model: Any) -> Dict[str, Any]:
    """
    Get optimal training configuration for a model.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary with training configuration
    """
    category = detect_model_category(model)
    config = STRATEGY_CONFIG[category].copy()
    config['category'] = category.value
    config['model_name'] = model.__class__.__name__
    return config


def get_visible_parameters(model: Any) -> Dict[str, bool]:
    """
    Determine which hyperparameters should be visible in UI.
    
    Args:
        model: Model instance
        
    Returns:
        Dictionary mapping parameter names to visibility
    """
    config = get_training_config(model)
    
    return {
        'cv_folds': config['use_epochs'] is False,
        'max_iter': config['use_max_iter'],
        'epochs': config['use_epochs'],
        'batch_size': config['use_epochs'],
        'learning_rate': config['use_epochs'] or config['category'] == 'iterative',
        'hp_tuning': config['use_hp_tuning'],
        'hp_iterations': config['use_hp_tuning'],
        'early_stopping': config['use_epochs']
    }


def should_use_cv(model: Any) -> bool:
    """Check if model should use cross-validation."""
    config = get_training_config(model)
    return config['strategy'] in [
        TrainingStrategy.K_FOLD_CV,
        TrainingStrategy.K_FOLD_CV_WITH_CONVERGENCE
    ]


def should_use_epochs(model: Any) -> bool:
    """Check if model should use epochs."""
    config = get_training_config(model)
    return config['use_epochs']


def get_default_cv_folds(model: Any) -> int:
    """Get default number of CV folds."""
    config = get_training_config(model)
    return config.get('cv_folds', 5)


def get_default_epochs(model: Any) -> int:
    """Get default number of epochs."""
    config = get_training_config(model)
    return config.get('epochs', 50)


def get_default_batch_size(model: Any) -> int:
    """Get default batch size."""
    config = get_training_config(model)
    return config.get('batch_size', 32)


def get_default_learning_rate(model: Any) -> float:
    """Get default learning rate."""
    config = get_training_config(model)
    return config.get('learning_rate', 0.001)


def get_hp_tuning_iterations(model: Any) -> int:
    """Get default HP tuning iterations."""
    config = get_training_config(model)
    return config.get('hp_iterations', 30)


def get_strategy_explanation(model: Any) -> str:
    """
    Get human-readable explanation of selected strategy.
    
    Args:
        model: Model instance
        
    Returns:
        Explanation string
    """
    config = get_training_config(model)
    return config['description']


def validate_parameters(model: Any, params: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that provided parameters match model strategy.
    
    Args:
        model: Model instance
        params: Parameter dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    config = get_training_config(model)
    
    # Check epochs
    if 'epochs' in params and params['epochs'] is not None:
        if not config['use_epochs']:
            return False, f"Epochs not applicable for {config['model_name']}. Using {config['description']} instead."
    
    # Check max_iter
    if 'max_iter' in params and params['max_iter'] is not None:
        if not config['use_max_iter']:
            return False, f"max_iter not applicable for {config['model_name']}. Using {config['description']} instead."
    
    # Check CV
    if 'cv_folds' in params and params['cv_folds'] is not None:
        if not should_use_cv(model):
            return False, f"Cross-validation not applicable for {config['model_name']}. Using {config['description']} instead."
    
    return True, ""


class AutoMLConfig:
    """AutoML configuration manager."""
    
    def __init__(self, model: Any):
        """Initialize with model."""
        self.model = model
        self.config = get_training_config(model)
        self.visible_params = get_visible_parameters(model)
    
    def get_training_params(self, user_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get final training parameters, merging defaults with user input.
        
        Args:
            user_params: User-provided parameters
            
        Returns:
            Final training parameters
        """
        params = self.config.copy()
        
        if user_params:
            # Only apply user params that are valid for this model
            for key, value in user_params.items():
                if self.visible_params.get(key, False) and value is not None:
                    params[key] = value
        
        return params
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get configuration for UI rendering."""
        return {
            'model_name': self.config['model_name'],
            'category': self.config['category'],
            'strategy': self.config['description'],
            'visible_parameters': self.visible_params,
            'defaults': {
                'cv_folds': self.config.get('cv_folds', 5),
                'epochs': self.config.get('epochs', 50),
                'batch_size': self.config.get('batch_size', 32),
                'learning_rate': self.config.get('learning_rate', 0.001),
                'hp_iterations': self.config.get('hp_iterations', 30)
            }
        }
