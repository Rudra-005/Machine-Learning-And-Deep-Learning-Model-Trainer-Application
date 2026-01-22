"""
Model Configuration & Categorization

Defines model categories, valid parameters, and training strategies.
Reusable across UI and backend for consistent model handling.
"""

MODEL_CONFIG = {
    "tree_based": {
        "category": "Tree-Based ML",
        "strategy": "k-fold_cv",
        "models": {
            "random_forest": {
                "name": "Random Forest",
                "task_types": ["classification", "regression"],
                "params": {
                    "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100, "label": "Trees"},
                    "max_depth": {"type": "slider", "min": 2, "max": 20, "default": 10, "label": "Max Depth"},
                    "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2, "label": "Min Samples Split"},
                },
                "cv_folds": {"min": 3, "max": 10, "default": 5}
            },
            "gradient_boosting": {
                "name": "Gradient Boosting",
                "task_types": ["classification", "regression"],
                "params": {
                    "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100, "label": "Estimators"},
                    "learning_rate": {"type": "slider", "min": 0.001, "max": 0.5, "default": 0.1, "label": "Learning Rate", "step": 0.01},
                    "max_depth": {"type": "slider", "min": 2, "max": 20, "default": 5, "label": "Max Depth"},
                },
                "cv_folds": {"min": 3, "max": 10, "default": 5}
            }
        }
    },
    "iterative": {
        "category": "Iterative ML",
        "strategy": "k-fold_cv",
        "models": {
            "logistic_regression": {
                "name": "Logistic Regression",
                "task_types": ["classification"],
                "params": {
                    "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "label": "Regularization (C)", "step": 0.1},
                    "max_iter": {"type": "slider", "min": 100, "max": 1000, "default": 100, "label": "Max Iterations"},
                },
                "cv_folds": {"min": 3, "max": 10, "default": 5}
            },
            "linear_regression": {
                "name": "Linear Regression",
                "task_types": ["regression"],
                "params": {},
                "cv_folds": {"min": 3, "max": 10, "default": 5}
            },
            "svm": {
                "name": "Support Vector Machine",
                "task_types": ["classification", "regression"],
                "params": {
                    "kernel": {"type": "selectbox", "options": ["linear", "rbf", "poly"], "default": "rbf", "label": "Kernel"},
                    "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "label": "Regularization (C)", "step": 0.1},
                },
                "cv_folds": {"min": 3, "max": 10, "default": 5}
            }
        }
    },
    "deep_learning": {
        "category": "Deep Learning",
        "strategy": "epochs",
        "models": {
            "sequential": {
                "name": "Sequential Neural Network",
                "task_types": ["classification", "regression"],
                "params": {
                    "epochs": {"type": "slider", "min": 10, "max": 200, "default": 50, "label": "Epochs"},
                    "batch_size": {"type": "selectbox", "options": [16, 32, 64, 128], "default": 32, "label": "Batch Size"},
                    "learning_rate": {"type": "number_input", "min": 0.0001, "max": 0.1, "default": 0.001, "label": "Learning Rate"},
                }
            },
            "cnn": {
                "name": "Convolutional Neural Network",
                "task_types": ["classification"],
                "params": {
                    "epochs": {"type": "slider", "min": 10, "max": 200, "default": 50, "label": "Epochs"},
                    "batch_size": {"type": "selectbox", "options": [16, 32, 64, 128], "default": 32, "label": "Batch Size"},
                    "learning_rate": {"type": "number_input", "min": 0.0001, "max": 0.1, "default": 0.001, "label": "Learning Rate"},
                }
            },
            "rnn": {
                "name": "Recurrent Neural Network",
                "task_types": ["classification", "regression"],
                "params": {
                    "epochs": {"type": "slider", "min": 10, "max": 200, "default": 50, "label": "Epochs"},
                    "batch_size": {"type": "selectbox", "options": [16, 32, 64, 128], "default": 32, "label": "Batch Size"},
                    "learning_rate": {"type": "number_input", "min": 0.0001, "max": 0.1, "default": 0.001, "label": "Learning Rate"},
                }
            }
        }
    }
}


def get_model_category(model_name):
    """Get category of a model."""
    for category, config in MODEL_CONFIG.items():
        if model_name in config["models"]:
            return category
    return None


def get_model_config(model_name):
    """Get full config for a model."""
    category = get_model_category(model_name)
    if category:
        return MODEL_CONFIG[category]["models"][model_name]
    return None


def get_models_by_task(task_type):
    """Get all models available for a task type."""
    models = {}
    for category, config in MODEL_CONFIG.items():
        for model_name, model_cfg in config["models"].items():
            if task_type.lower() in model_cfg["task_types"]:
                models[model_name] = model_cfg
    return models


def get_category_strategy(model_name):
    """Get training strategy for a model (k-fold_cv or epochs)."""
    category = get_model_category(model_name)
    if category:
        return MODEL_CONFIG[category]["strategy"]
    return None


def get_cv_folds_config(model_name):
    """Get k-fold configuration for a model."""
    category = get_model_category(model_name)
    if category and category != "deep_learning":
        return MODEL_CONFIG[category]["models"][model_name].get("cv_folds", {"min": 3, "max": 10, "default": 5})
    return None


def is_tree_based(model_name):
    """Check if model is tree-based."""
    return get_model_category(model_name) == "tree_based"


def is_iterative(model_name):
    """Check if model is iterative."""
    return get_model_category(model_name) == "iterative"


def is_deep_learning(model_name):
    """Check if model is deep learning."""
    return get_model_category(model_name) == "deep_learning"


def get_model_params(model_name):
    """Get parameters for a model."""
    config = get_model_config(model_name)
    return config.get("params", {}) if config else {}


def get_param_config(model_name, param_name):
    """Get configuration for a specific parameter."""
    params = get_model_params(model_name)
    return params.get(param_name, None)
