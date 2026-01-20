"""
Model Factory Module

A flexible, extensible factory for creating machine learning and deep learning models.
Supports dynamic model instantiation with custom hyperparameters.

Author: ML Engineer
Date: 2026-01-19
"""

from typing import Dict, Any, Optional, Union, Callable, Type
from abc import ABC, abstractmethod
import logging

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Sequential
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available. Deep learning models disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model Registries and Default Hyperparameters
# ============================================================================

DEFAULT_HYPERPARAMETERS = {
    'classification': {
        'logistic_regression': {
            'max_iter': 1000,
            'solver': 'lbfgs',
            'random_state': 42
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'svm': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    },
    'regression': {
        'linear_regression': {},
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'svm': {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale'
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
    }
}


# ============================================================================
# Model Builders
# ============================================================================

def build_logistic_regression(**hyperparams) -> LogisticRegression:
    """Build Logistic Regression classifier."""
    logger.info(f"Building Logistic Regression with params: {hyperparams}")
    return LogisticRegression(**hyperparams)


def build_random_forest_classifier(**hyperparams) -> RandomForestClassifier:
    """Build Random Forest classifier."""
    logger.info(f"Building Random Forest Classifier with params: {hyperparams}")
    return RandomForestClassifier(**hyperparams)


def build_random_forest_regressor(**hyperparams) -> RandomForestRegressor:
    """Build Random Forest regressor."""
    logger.info(f"Building Random Forest Regressor with params: {hyperparams}")
    return RandomForestRegressor(**hyperparams)


def build_svm_classifier(**hyperparams) -> SVC:
    """Build Support Vector Machine classifier."""
    logger.info(f"Building SVM Classifier with params: {hyperparams}")
    return SVC(**hyperparams)


def build_svm_regressor(**hyperparams) -> SVR:
    """Build Support Vector Machine regressor."""
    logger.info(f"Building SVM Regressor with params: {hyperparams}")
    return SVR(**hyperparams)


def build_gradient_boosting_classifier(**hyperparams) -> GradientBoostingClassifier:
    """Build Gradient Boosting classifier."""
    logger.info(f"Building Gradient Boosting Classifier with params: {hyperparams}")
    return GradientBoostingClassifier(**hyperparams)


def build_gradient_boosting_regressor(**hyperparams) -> GradientBoostingRegressor:
    """Build Gradient Boosting regressor."""
    logger.info(f"Building Gradient Boosting Regressor with params: {hyperparams}")
    return GradientBoostingRegressor(**hyperparams)


def build_neural_network_classifier(
    input_dim: int,
    num_classes: int,
    hidden_layers: Optional[list] = None,
    activation: str = 'relu',
    output_activation: str = 'softmax',
    dropout_rate: float = 0.2,
    **hyperparams
) -> Sequential:
    """
    Build Neural Network classifier using Keras Sequential API.
    
    Args:
        input_dim (int): Input dimension (number of features)
        num_classes (int): Number of output classes
        hidden_layers (list): List of hidden layer sizes. Default: [128, 64]
        activation (str): Activation function for hidden layers. Default: 'relu'
        output_activation (str): Activation function for output. Default: 'softmax'
        dropout_rate (float): Dropout rate. Default: 0.2
        **hyperparams: Additional parameters (ignored for Keras model)
        
    Returns:
        tf.keras.Sequential: Compiled neural network model
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required for neural network models")
    
    if hidden_layers is None:
        hidden_layers = [128, 64]
    
    logger.info(
        f"Building Neural Network Classifier: "
        f"input={input_dim}, classes={num_classes}, "
        f"hidden_layers={hidden_layers}, dropout={dropout_rate}"
    )
    
    model = Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation=output_activation))
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Neural Network model compiled successfully")
    return model


def build_neural_network_regressor(
    input_dim: int,
    hidden_layers: Optional[list] = None,
    activation: str = 'relu',
    dropout_rate: float = 0.2,
    **hyperparams
) -> Sequential:
    """
    Build Neural Network regressor using Keras Sequential API.
    
    Args:
        input_dim (int): Input dimension (number of features)
        hidden_layers (list): List of hidden layer sizes. Default: [128, 64]
        activation (str): Activation function for hidden layers. Default: 'relu'
        dropout_rate (float): Dropout rate. Default: 0.2
        **hyperparams: Additional parameters (ignored for Keras model)
        
    Returns:
        tf.keras.Sequential: Compiled neural network model
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required for neural network models")
    
    if hidden_layers is None:
        hidden_layers = [128, 64]
    
    logger.info(
        f"Building Neural Network Regressor: "
        f"input={input_dim}, hidden_layers={hidden_layers}, dropout={dropout_rate}"
    )
    
    model = Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation=activation))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer (single neuron for regression)
    model.add(layers.Dense(1, activation='linear'))
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    logger.info("Neural Network regressor compiled successfully")
    return model


# ============================================================================
# Model Factory Class
# ============================================================================

class ModelFactory:
    """
    Factory for creating ML/DL models dynamically.
    
    Supports flexible model creation with custom hyperparameters and
    easy extensibility for new models.
    """
    
    # Registry of model builders
    _BUILDERS = {
        'classification': {
            'logistic_regression': build_logistic_regression,
            'random_forest': build_random_forest_classifier,
            'svm': build_svm_classifier,
            'gradient_boosting': build_gradient_boosting_classifier,
            'neural_network': build_neural_network_classifier,
        },
        'regression': {
            'linear_regression': None,  # sklearn native
            'random_forest': build_random_forest_regressor,
            'svm': build_svm_regressor,
            'gradient_boosting': build_gradient_boosting_regressor,
            'neural_network': build_neural_network_regressor,
        }
    }
    
    @staticmethod
    def get_available_models(task_type: str) -> list:
        """
        Get list of available models for a task type.
        
        Args:
            task_type (str): 'classification' or 'regression'
            
        Returns:
            list: Available model names
            
        Raises:
            ValueError: If task_type invalid
        """
        if task_type not in ModelFactory._BUILDERS:
            raise ValueError(
                f"Invalid task_type: {task_type}. "
                f"Must be 'classification' or 'regression'"
            )
        
        models = list(ModelFactory._BUILDERS[task_type].keys())
        logger.info(f"Available {task_type} models: {models}")
        return models
    
    @staticmethod
    def create_model(
        task_type: str,
        model_name: str,
        **hyperparams
    ) -> Union[Any, Sequential]:
        """
        Create a model instance dynamically.
        
        Args:
            task_type (str): 'classification' or 'regression'
            model_name (str): Name of the model to create
            **hyperparams: Custom hyperparameters (override defaults)
            
        Returns:
            Instantiated model (sklearn estimator or Keras Sequential)
            
        Raises:
            ValueError: If task_type or model_name invalid
            
        Examples:
            >>> # Scikit-learn model
            >>> clf = ModelFactory.create_model(
            ...     'classification', 
            ...     'random_forest',
            ...     n_estimators=200
            ... )
            
            >>> # Neural network (requires input_dim, num_classes)
            >>> nn = ModelFactory.create_model(
            ...     'classification',
            ...     'neural_network',
            ...     input_dim=20,
            ...     num_classes=3,
            ...     hidden_layers=[256, 128]
            ... )
        """
        # Validate task_type
        if task_type not in ModelFactory._BUILDERS:
            raise ValueError(
                f"Invalid task_type: {task_type}. "
                f"Must be 'classification' or 'regression'"
            )
        
        # Validate model_name
        if model_name not in ModelFactory._BUILDERS[task_type]:
            available = ModelFactory.get_available_models(task_type)
            raise ValueError(
                f"Invalid model_name: {model_name}. "
                f"Available models for {task_type}: {available}"
            )
        
        # Special case: Linear Regression (no builder needed)
        if task_type == 'regression' and model_name == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression()
        
        # Get builder and defaults
        builder = ModelFactory._BUILDERS[task_type][model_name]
        defaults = DEFAULT_HYPERPARAMETERS[task_type].get(model_name, {})
        
        # Merge defaults with user hyperparameters
        final_hyperparams = {**defaults, **hyperparams}
        
        logger.info(
            f"Creating {task_type} model: {model_name} "
            f"with hyperparameters: {final_hyperparams}"
        )
        
        # Build and return model
        return builder(**final_hyperparams)
    
    @staticmethod
    def register_model(
        task_type: str,
        model_name: str,
        builder: Callable,
        defaults: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new model builder (for extensibility).
        
        Args:
            task_type (str): 'classification' or 'regression'
            model_name (str): Name for the new model
            builder (Callable): Function that builds and returns the model
            defaults (Dict): Default hyperparameters
            
        Example:
            >>> def build_my_model(**params):
            ...     return MyCustomModel(**params)
            
            >>> ModelFactory.register_model(
            ...     'classification',
            ...     'my_model',
            ...     build_my_model,
            ...     defaults={'param1': 'value1'}
            ... )
        """
        if task_type not in ModelFactory._BUILDERS:
            ModelFactory._BUILDERS[task_type] = {}
        
        ModelFactory._BUILDERS[task_type][model_name] = builder
        
        if defaults is None:
            defaults = {}
        
        if task_type not in DEFAULT_HYPERPARAMETERS:
            DEFAULT_HYPERPARAMETERS[task_type] = {}
        
        DEFAULT_HYPERPARAMETERS[task_type][model_name] = defaults
        
        logger.info(
            f"Registered new model: {task_type}/{model_name} "
            f"with defaults: {defaults}"
        )
    
    @staticmethod
    def get_default_hyperparameters(task_type: str, model_name: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for a model.
        
        Args:
            task_type (str): 'classification' or 'regression'
            model_name (str): Model name
            
        Returns:
            Dict: Default hyperparameters
        """
        return DEFAULT_HYPERPARAMETERS.get(task_type, {}).get(model_name, {})


# ============================================================================
# Convenience Functions
# ============================================================================

def create_classification_model(model_name: str, **hyperparams):
    """
    Quick wrapper to create a classification model.
    
    Args:
        model_name (str): Name of classifier model
        **hyperparams: Model hyperparameters
        
    Returns:
        Classifier model instance
        
    Example:
        >>> rf_clf = create_classification_model('random_forest', n_estimators=200)
    """
    return ModelFactory.create_model('classification', model_name, **hyperparams)


def create_regression_model(model_name: str, **hyperparams):
    """
    Quick wrapper to create a regression model.
    
    Args:
        model_name (str): Name of regressor model
        **hyperparams: Model hyperparameters
        
    Returns:
        Regressor model instance
        
    Example:
        >>> rf_reg = create_regression_model('random_forest', n_estimators=150)
    """
    return ModelFactory.create_model('regression', model_name, **hyperparams)


def create_neural_classifier(
    input_dim: int,
    num_classes: int,
    hidden_layers: Optional[list] = None,
    **hyperparams
):
    """
    Quick wrapper to create a neural network classifier.
    
    Args:
        input_dim (int): Number of input features
        num_classes (int): Number of output classes
        hidden_layers (list): Architecture of hidden layers
        **hyperparams: Additional parameters
        
    Returns:
        Compiled Keras Sequential model
        
    Example:
        >>> nn = create_neural_classifier(
        ...     input_dim=20,
        ...     num_classes=3,
        ...     hidden_layers=[256, 128]
        ... )
    """
    return ModelFactory.create_model(
        'classification',
        'neural_network',
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_layers=hidden_layers,
        **hyperparams
    )


def create_neural_regressor(
    input_dim: int,
    hidden_layers: Optional[list] = None,
    **hyperparams
):
    """
    Quick wrapper to create a neural network regressor.
    
    Args:
        input_dim (int): Number of input features
        hidden_layers (list): Architecture of hidden layers
        **hyperparams: Additional parameters
        
    Returns:
        Compiled Keras Sequential model
        
    Example:
        >>> nn = create_neural_regressor(
        ...     input_dim=20,
        ...     hidden_layers=[256, 128]
        ... )
    """
    return ModelFactory.create_model(
        'regression',
        'neural_network',
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        **hyperparams
    )


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Model Factory - ML/DL Model Creation System")
    print("=" * 70)
    
    print("\n1. Available Classification Models:")
    print(f"   {ModelFactory.get_available_models('classification')}")
    
    print("\n2. Available Regression Models:")
    print(f"   {ModelFactory.get_available_models('regression')}")
    
    print("\n3. Usage Examples:")
    print("""
    # Scikit-learn models
    from model_factory import ModelFactory
    
    # Classification
    clf = ModelFactory.create_model(
        'classification',
        'random_forest',
        n_estimators=200,
        max_depth=15
    )
    
    # Regression
    reg = ModelFactory.create_model(
        'regression',
        'svm',
        C=10.0,
        kernel='poly'
    )
    
    # Neural Networks (if TensorFlow installed)
    from model_factory import create_neural_classifier, create_neural_regressor
    
    nn_clf = create_neural_classifier(
        input_dim=20,
        num_classes=3,
        hidden_layers=[256, 128, 64]
    )
    
    nn_reg = create_neural_regressor(
        input_dim=15,
        hidden_layers=[128, 64]
    )
    
    # Register custom models
    def build_my_model(**params):
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(**params)
    
    ModelFactory.register_model(
        'classification',
        'adaboost',
        build_my_model,
        defaults={'n_estimators': 50}
    )
    
    # Use custom model
    ada = ModelFactory.create_model('classification', 'adaboost')
    """)
    
    print("\n" + "=" * 70)
