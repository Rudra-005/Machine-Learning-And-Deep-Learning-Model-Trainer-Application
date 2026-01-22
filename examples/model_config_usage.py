"""
Model Configuration Usage Examples

Demonstrates how to use model_config and related utilities.
"""

from models.model_config import (
    MODEL_CONFIG, get_model_category, get_model_config,
    get_models_by_task, get_category_strategy, get_cv_folds_config,
    is_tree_based, is_iterative, is_deep_learning, get_model_params
)
from models.training_utils import train_model_with_strategy, get_training_info, apply_hyperparams
from app.utils.model_ui import render_hyperparameters, get_strategy_info


# ============ EXAMPLE 1: Get all models for a task ============
def example_get_models_by_task():
    """Get all models available for classification."""
    classification_models = get_models_by_task("classification")
    print("Classification Models:")
    for model_name, config in classification_models.items():
        print(f"  - {config['name']} ({model_name})")


# ============ EXAMPLE 2: Check model category ============
def example_check_model_category():
    """Check what category a model belongs to."""
    print("\nModel Categories:")
    print(f"  RandomForest: {get_model_category('random_forest')}")
    print(f"  LogisticRegression: {get_model_category('logistic_regression')}")
    print(f"  CNN: {get_model_category('cnn')}")


# ============ EXAMPLE 3: Get model parameters ============
def example_get_model_params():
    """Get parameters for a specific model."""
    print("\nRandom Forest Parameters:")
    params = get_model_params('random_forest')
    for param_name, param_cfg in params.items():
        print(f"  - {param_cfg['label']}: {param_cfg['type']}")


# ============ EXAMPLE 4: Check training strategy ============
def example_check_training_strategy():
    """Check training strategy for models."""
    print("\nTraining Strategies:")
    print(f"  RandomForest: {get_category_strategy('random_forest')}")
    print(f"  LogisticRegression: {get_category_strategy('logistic_regression')}")
    print(f"  CNN: {get_category_strategy('cnn')}")


# ============ EXAMPLE 5: Type checking ============
def example_type_checking():
    """Example of type checking utilities."""
    print("\nType Checking:")
    print(f"  is_tree_based('random_forest'): {is_tree_based('random_forest')}")
    print(f"  is_iterative('logistic_regression'): {is_iterative('logistic_regression')}")
    print(f"  is_deep_learning('cnn'): {is_deep_learning('cnn')}")


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL CONFIGURATION USAGE EXAMPLES")
    print("=" * 60)
    
    example_get_models_by_task()
    example_check_model_category()
    example_get_model_params()
    example_check_training_strategy()
    example_type_checking()
