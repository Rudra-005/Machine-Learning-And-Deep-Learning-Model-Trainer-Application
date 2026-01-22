"""
Hyperparameter Optimization for ML Models

Uses RandomizedSearchCV for efficient hyperparameter tuning.
Reuses k-fold cross-validation.
"""

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
import numpy as np
import streamlit as st
from models.model_config import is_deep_learning


# Parameter search spaces for each model
PARAM_DISTRIBUTIONS = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'gradient_boosting': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    },
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [100, 500, 1000],
        'solver': ['lbfgs', 'liblinear']
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    'linear_regression': {
        'fit_intercept': [True, False]
    }
}


class HyperparameterOptimizer:
    """Handles hyperparameter optimization using RandomizedSearchCV."""
    
    @staticmethod
    def get_param_distribution(model_name):
        """Get parameter distribution for a model."""
        return PARAM_DISTRIBUTIONS.get(model_name, {})
    
    @staticmethod
    def optimize(model, X_train, y_train, model_name, task_type, n_iter=20, cv=5):
        """
        Optimize hyperparameters using RandomizedSearchCV.
        
        Args:
            model: Sklearn model
            X_train: Training features
            y_train: Training target
            model_name: Name of model
            task_type: 'classification' or 'regression'
            n_iter: Number of search iterations
            cv: Number of CV folds
        
        Returns:
            best_model: Best estimator found
            search_results: Dict with optimization results
        """
        
        param_dist = HyperparameterOptimizer.get_param_distribution(model_name)
        
        if not param_dist:
            return model, None
        
        # Get CV splitter
        if task_type.lower() == 'classification':
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
            scoring = 'r2'
        
        # Run RandomizedSearchCV
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        # Package results
        search_results = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'n_iter': n_iter,
            'cv_folds': cv,
            'scoring': scoring,
            'cv_results': random_search.cv_results_
        }
        
        return random_search.best_estimator_, search_results
    
    @staticmethod
    def display_optimization_results(search_results):
        """Display optimization results in Streamlit."""
        if search_results is None:
            return
        
        st.markdown("### Hyperparameter Optimization Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Best Score",
                f"{search_results['best_score']:.4f}",
                help="Best cross-validation score"
            )
        
        with col2:
            st.metric(
                "Iterations",
                search_results['n_iter'],
                help="Number of parameter combinations tested"
            )
        
        with col3:
            st.metric(
                "CV Folds",
                search_results['cv_folds'],
                help="Cross-validation folds used"
            )
        
        # Display best parameters
        st.markdown("**Best Parameters**")
        params_df = pd.DataFrame(
            list(search_results['best_params'].items()),
            columns=['Parameter', 'Value']
        )
        st.dataframe(params_df, use_container_width=True)
        
        # Display top 5 parameter combinations
        st.markdown("**Top 5 Parameter Combinations**")
        cv_results = search_results['cv_results']
        
        top_indices = np.argsort(cv_results['mean_test_score'])[-5:][::-1]
        
        top_results = []
        for idx in top_indices:
            params = {k.replace('param_', ''): v[idx] for k, v in cv_results.items() if k.startswith('param_')}
            score = cv_results['mean_test_score'][idx]
            top_results.append({'Score': score, 'Parameters': str(params)})
        
        top_df = pd.DataFrame(top_results)
        st.dataframe(top_df, use_container_width=True)


def train_with_hp_optimization(model, X_train, y_train, X_test, y_test,
                               model_name, task_type, n_iter=20, cv=5):
    """
    Train model with hyperparameter optimization.
    
    Returns:
        best_model: Optimized model
        search_results: Optimization results
        predictions: Test predictions
    """
    
    best_model, search_results = HyperparameterOptimizer.optimize(
        model, X_train, y_train, model_name, task_type, n_iter, cv
    )
    
    # Get predictions
    predictions = best_model.predict(X_test)
    
    return best_model, search_results, predictions


import pandas as pd
