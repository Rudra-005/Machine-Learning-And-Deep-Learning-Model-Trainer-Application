"""
K-Fold Cross-Validation for Traditional ML Models

Implements k-fold CV using sklearn cross_val_score.
Skipped for deep learning models.
"""

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import numpy as np
import streamlit as st
from models.model_config import is_deep_learning


class KFoldCrossValidator:
    """Handles k-fold cross-validation for ML models."""
    
    @staticmethod
    def get_cv_splitter(task_type, k):
        """Get appropriate CV splitter based on task type."""
        if task_type.lower() == "classification":
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        else:
            return KFold(n_splits=k, shuffle=True, random_state=42)
    
    @staticmethod
    def compute_cv_scores(model, X, y, k, task_type):
        """
        Compute k-fold cross-validation scores.
        
        Args:
            model: Sklearn model
            X: Features
            y: Target
            k: Number of folds
            task_type: 'classification' or 'regression'
        
        Returns:
            cv_scores: Array of fold scores
            mean_score: Mean CV score
            std_score: Standard deviation of CV scores
        """
        cv_splitter = KFoldCrossValidator.get_cv_splitter(task_type, k)
        
        scoring = 'accuracy' if task_type.lower() == 'classification' else 'r2'
        
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1
        )
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        
        return cv_scores, mean_score, std_score
    
    @staticmethod
    def display_cv_results(cv_scores, mean_score, std_score, task_type):
        """Display CV results in Streamlit."""
        st.markdown("### Cross-Validation Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mean CV Score",
                f"{mean_score:.4f}",
                help="Average score across all folds"
            )
        
        with col2:
            st.metric(
                "Std Dev",
                f"{std_score:.4f}",
                help="Standard deviation of fold scores"
            )
        
        with col3:
            st.metric(
                "Folds",
                len(cv_scores),
                help="Number of cross-validation folds"
            )
        
        # Display individual fold scores
        st.markdown("**Fold Scores**")
        fold_data = {
            'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
            'Score': cv_scores
        }
        fold_df = st.dataframe(fold_data, use_container_width=True)
        
        # Confidence interval
        ci_lower = mean_score - 1.96 * std_score
        ci_upper = mean_score + 1.96 * std_score
        st.write(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")


def train_ml_with_cv(model, X_train, y_train, X_test, y_test, k, task_type, model_name):
    """
    Train ML model with k-fold cross-validation.
    
    Args:
        model: Sklearn model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        k: Number of folds
        task_type: 'classification' or 'regression'
        model_name: Name of model
    
    Returns:
        trained_model: Fitted model
        cv_results: Dict with CV scores and metrics
        test_metrics: Dict with test set metrics
    """
    
    # Skip CV for deep learning
    if is_deep_learning(model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return model, None, predictions
    
    # Compute CV scores
    cv_scores, mean_score, std_score = KFoldCrossValidator.compute_cv_scores(
        model, X_train, y_train, k, task_type
    )
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Get test predictions
    predictions = model.predict(X_test)
    
    # Package results
    cv_results = {
        'cv_scores': cv_scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'k_folds': k,
        'task_type': task_type
    }
    
    return model, cv_results, predictions


def display_cv_summary(cv_results):
    """Display CV summary in Streamlit."""
    if cv_results is None:
        return
    
    cv_scores = cv_results['cv_scores']
    mean_score = cv_results['mean_score']
    std_score = cv_results['std_score']
    task_type = cv_results['task_type']
    
    KFoldCrossValidator.display_cv_results(cv_scores, mean_score, std_score, task_type)


def get_cv_config(k):
    """Get CV configuration for display."""
    return {
        'n_splits': k,
        'shuffle': True,
        'random_state': 42,
        'stratified': True
    }
