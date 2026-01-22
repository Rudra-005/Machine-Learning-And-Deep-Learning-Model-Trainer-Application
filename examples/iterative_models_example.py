"""
Iterative ML Models Examples

Demonstrates LogisticRegression, SGDClassifier, Perceptron usage.
Shows max_iter parameter and CV integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.iterative_models import IterativeModelHandler
from app.utils.iterative_streamlit import (
    render_iterative_model_config, train_iterative_model,
    display_iterative_model_info, display_iterative_training_results
)
from evaluation.metrics import MetricsCalculator


def example_iterative_models_list():
    """Show available iterative models."""
    st.markdown("## Example 1: Available Iterative Models")
    
    models = IterativeModelHandler.get_iterative_models()
    
    st.write("**Iterative ML Models (use max_iter, NOT epochs):**")
    for model_name in models:
        info = IterativeModelHandler.get_model_info(model_name)
        st.write(f"- **{info['name']}**: {info['description']}")


def example_max_iter_explanation():
    """Explain max_iter vs epochs."""
    st.markdown("## Example 2: Max Iterations vs Epochs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Max Iterations (Iterative ML)**")
        st.write("""
        - Single pass through training data
        - Optimization iterations until convergence
        - Stops when loss plateaus or max_iter reached
        - Example: max_iter=1000
        - Used by: LogisticRegression, SGD, Perceptron
        """)
    
    with col2:
        st.markdown("**Epochs (Deep Learning)**")
        st.write("""
        - Multiple passes through training data
        - Each epoch processes all data in batches
        - Typically 10-100+ epochs
        - Example: epochs=50
        - Used by: Neural Networks, CNN, RNN
        """)


def example_basic_training():
    """Basic iterative model training."""
    st.markdown("## Example 3: Basic Training")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features (important for iterative models)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create model with max_iter
    model = IterativeModelHandler.create_iterative_model(
        'logistic_regression', max_iter=1000
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    st.write(f"Train Score: {train_score:.4f}")
    st.write(f"Test Score: {test_score:.4f}")


def example_cv_integration():
    """Iterative model with CV integration."""
    st.markdown("## Example 4: CV Integration")
    
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Select Iterative Model",
            IterativeModelHandler.get_iterative_models()
        )
        max_iter = st.slider("Max Iterations", 100, 10000, 1000, step=100)
    
    with col2:
        cv_folds = st.slider("K-Fold Splits", 3, 10, 5)
    
    if st.button("🚀 Train with CV", type="primary"):
        trained_model, cv_scores, predictions = train_iterative_model(
            model_choice, X_train, y_train, X_test, y_test,
            max_iter, cv_folds
        )
        
        metrics = MetricsCalculator.classification_metrics(y_test, predictions)
        display_iterative_training_results(cv_scores, metrics)
        
        st.success("✅ Training complete!")


def example_max_iter_impact():
    """Show impact of max_iter on convergence."""
    st.markdown("## Example 5: Impact of Max Iterations")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = []
    for max_iter in [100, 500, 1000, 5000]:
        model = IterativeModelHandler.create_iterative_model(
            'logistic_regression', max_iter=max_iter
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results.append({'Max Iterations': max_iter, 'Test Score': score})
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    st.write("Higher max_iter allows more convergence iterations, "
             "but may not always improve performance.")


def example_model_comparison():
    """Compare different iterative models."""
    st.markdown("## Example 6: Model Comparison")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    results = []
    for model_name in IterativeModelHandler.get_iterative_models():
        model = IterativeModelHandler.create_iterative_model(
            model_name, max_iter=1000
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        info = IterativeModelHandler.get_model_info(model_name)
        results.append({
            'Model': info['name'],
            'Test Score': score
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)


def main():
    """Main example app."""
    st.set_page_config(page_title="Iterative Models Examples", layout="wide")
    
    st.title("Iterative ML Models Examples")
    st.divider()
    
    example_choice = st.selectbox(
        "Select Example",
        [
            "Available Models",
            "Max Iterations vs Epochs",
            "Basic Training",
            "CV Integration",
            "Impact of Max Iterations",
            "Model Comparison"
        ]
    )
    
    st.divider()
    
    if example_choice == "Available Models":
        example_iterative_models_list()
    elif example_choice == "Max Iterations vs Epochs":
        example_max_iter_explanation()
    elif example_choice == "Basic Training":
        example_basic_training()
    elif example_choice == "CV Integration":
        example_cv_integration()
    elif example_choice == "Impact of Max Iterations":
        example_max_iter_impact()
    elif example_choice == "Model Comparison":
        example_model_comparison()


if __name__ == "__main__":
    main()
