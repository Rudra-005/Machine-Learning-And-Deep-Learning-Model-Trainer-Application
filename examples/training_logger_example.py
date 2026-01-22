"""
Training Logger Examples

Demonstrates training logs for different model types.
Shows explanations in user-friendly language.
"""

import streamlit as st
from app.utils.training_logger import TrainingLogger, log_training_decision
from app.utils.logger_streamlit import (
    display_training_explanation, display_training_log_during,
    display_training_log_after, display_quick_summary
)


def example_tree_based_logging():
    """Show logging for tree-based model."""
    st.markdown("## Example 1: Tree-Based Model (Random Forest)")
    
    model_name = 'random_forest'
    task_type = 'classification'
    cv_folds = 5
    
    st.markdown("### Before Training")
    display_training_explanation(model_name, task_type, cv_folds)
    
    st.divider()
    
    st.markdown("### During Training")
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'cv_folds': 5
    }
    display_training_log_during(model_name, task_type, params)
    
    st.divider()
    
    st.markdown("### After Training")
    metrics = {
        'accuracy': 0.92,
        'precision': 0.91,
        'recall': 0.93,
        'f1_score': 0.92
    }
    display_training_log_after(model_name, metrics)


def example_iterative_logging():
    """Show logging for iterative model."""
    st.markdown("## Example 2: Iterative Model (Logistic Regression)")
    
    model_name = 'logistic_regression'
    task_type = 'classification'
    cv_folds = 5
    
    st.markdown("### Before Training")
    display_training_explanation(model_name, task_type, cv_folds)
    
    st.divider()
    
    st.markdown("### During Training")
    params = {
        'max_iter': 1000,
        'C': 1.0,
        'cv_folds': 5
    }
    display_training_log_during(model_name, task_type, params)
    
    st.divider()
    
    st.markdown("### After Training")
    metrics = {
        'accuracy': 0.88,
        'precision': 0.87,
        'recall': 0.89,
        'f1_score': 0.88
    }
    display_training_log_after(model_name, metrics)


def example_dl_logging():
    """Show logging for deep learning model."""
    st.markdown("## Example 3: Deep Learning Model (Sequential NN)")
    
    model_name = 'sequential'
    task_type = 'classification'
    epochs = 50
    
    st.markdown("### Before Training")
    display_training_explanation(model_name, task_type, epochs=epochs)
    
    st.divider()
    
    st.markdown("### During Training")
    params = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'early_stopping': True
    }
    display_training_log_during(model_name, task_type, params)
    
    st.divider()
    
    st.markdown("### After Training")
    metrics = {
        'accuracy': 0.95,
        'loss': 0.15,
        'val_accuracy': 0.93,
        'val_loss': 0.18
    }
    display_training_log_after(model_name, metrics)


def example_strategy_comparison():
    """Compare strategies for different models."""
    st.markdown("## Example 4: Strategy Comparison")
    
    models = [
        ('random_forest', 'classification', 5, None),
        ('logistic_regression', 'classification', 5, None),
        ('sequential', 'classification', None, 50)
    ]
    
    for model_name, task_type, cv_folds, epochs in models:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {model_name.replace('_', ' ').title()}")
            display_quick_summary(model_name, task_type, cv_folds, epochs)
        
        with col2:
            st.markdown("### Why This Strategy?")
            st.markdown(TrainingLogger.log_strategy_decision(model_name))
        
        st.divider()


def example_parameter_explanation():
    """Show parameter explanations."""
    st.markdown("## Example 5: Parameter Explanations")
    
    model_choice = st.selectbox(
        "Select Model",
        ['random_forest', 'logistic_regression', 'sequential']
    )
    
    st.divider()
    
    st.markdown("### Parameters Shown/Hidden")
    st.markdown(TrainingLogger.log_parameter_decisions(model_choice, {}))


def example_cv_explanation():
    """Show CV explanation."""
    st.markdown("## Example 6: Cross-Validation Explanation")
    
    model_choice = st.selectbox(
        "Select Model",
        ['random_forest', 'logistic_regression', 'sequential'],
        key='cv_model'
    )
    
    cv_folds = st.slider("CV Folds", 3, 10, 5)
    
    st.divider()
    
    st.markdown(TrainingLogger.log_cv_explanation(model_choice, cv_folds))


def main():
    """Main example app."""
    st.set_page_config(page_title="Training Logger Examples", layout="wide")
    
    st.title("Training Logger Examples")
    st.divider()
    
    example_choice = st.selectbox(
        "Select Example",
        [
            "Tree-Based Model Logging",
            "Iterative Model Logging",
            "Deep Learning Model Logging",
            "Strategy Comparison",
            "Parameter Explanations",
            "Cross-Validation Explanation"
        ]
    )
    
    st.divider()
    
    if example_choice == "Tree-Based Model Logging":
        example_tree_based_logging()
    elif example_choice == "Iterative Model Logging":
        example_iterative_logging()
    elif example_choice == "Deep Learning Model Logging":
        example_dl_logging()
    elif example_choice == "Strategy Comparison":
        example_strategy_comparison()
    elif example_choice == "Parameter Explanations":
        example_parameter_explanation()
    elif example_choice == "Cross-Validation Explanation":
        example_cv_explanation()


if __name__ == "__main__":
    main()
