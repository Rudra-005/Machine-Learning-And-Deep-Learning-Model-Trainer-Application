"""
Iterative ML Models Handler

Handles LogisticRegression, SGDClassifier, Perceptron.
Exposes "Max Iterations" parameter mapped to max_iter.

Why not epochs?
- Epochs are for deep learning (multiple passes through entire dataset)
- max_iter is for iterative ML (convergence iterations in optimization)
- Iterative ML trains once on full dataset with max_iter limit
- Deep learning trains multiple times (epochs) on batches
- Different concepts, different parameters
"""

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import streamlit as st


# Iterative models that use max_iter
ITERATIVE_MODELS = {
    'logistic_regression': LogisticRegression,
    'sgd_classifier': SGDClassifier,
    'perceptron': Perceptron
}


class IterativeModelHandler:
    """Handles iterative ML models with max_iter parameter."""
    
    @staticmethod
    def is_iterative_model(model_name):
        """Check if model is iterative."""
        return model_name in ITERATIVE_MODELS
    
    @staticmethod
    def get_iterative_models():
        """Get list of iterative models."""
        return list(ITERATIVE_MODELS.keys())
    
    @staticmethod
    def create_iterative_model(model_name, max_iter=100, **kwargs):
        """
        Create iterative model with max_iter parameter.
        
        Args:
            model_name: Name of iterative model
            max_iter: Maximum iterations for convergence
            **kwargs: Additional model parameters
        
        Returns:
            model: Configured sklearn model
        """
        if model_name not in ITERATIVE_MODELS:
            raise ValueError(f"Unknown iterative model: {model_name}")
        
        model_class = ITERATIVE_MODELS[model_name]
        
        # Set max_iter (convergence iterations, NOT epochs)
        kwargs['max_iter'] = max_iter
        
        # Set random_state for reproducibility
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42
        
        return model_class(**kwargs)
    
    @staticmethod
    def train_iterative_with_cv(model, X_train, y_train, X_test, y_test,
                                model_name, cv_folds=5):
        """
        Train iterative model with k-fold cross-validation.
        
        Note: max_iter is set during model creation, not during training.
        CV evaluates the model's convergence quality across folds.
        
        Args:
            model: Sklearn iterative model
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            model_name: Name of model
            cv_folds: Number of CV folds
        
        Returns:
            trained_model: Fitted model
            cv_scores: Cross-validation scores
            predictions: Test predictions
        """
        
        # Compute CV scores
        cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_splitter,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Get predictions
        predictions = model.predict(X_test)
        
        return model, cv_scores, predictions
    
    @staticmethod
    def get_model_info(model_name):
        """Get information about iterative model."""
        info = {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'description': 'Linear model for binary/multiclass classification',
                'max_iter_range': (100, 10000),
                'default_max_iter': 100
            },
            'sgd_classifier': {
                'name': 'SGD Classifier',
                'description': 'Stochastic Gradient Descent for classification',
                'max_iter_range': (100, 10000),
                'default_max_iter': 1000
            },
            'perceptron': {
                'name': 'Perceptron',
                'description': 'Linear classifier using perceptron algorithm',
                'max_iter_range': (100, 10000),
                'default_max_iter': 1000
            }
        }
        return info.get(model_name, {})
