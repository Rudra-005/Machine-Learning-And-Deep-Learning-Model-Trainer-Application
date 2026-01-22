"""
Hyperparameter Optimization Examples

Demonstrates RandomizedSearchCV integration for ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.hp_optimizer import HyperparameterOptimizer, train_with_hp_optimization
from app.utils.hp_streamlit import render_hp_optimization_config, train_with_optional_tuning


def example_basic_hp_optimization():
    """Basic HP optimization example."""
    st.markdown("## Example 1: Basic HP Optimization")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(random_state=42)
    
    best_model, search_results, predictions = train_with_hp_optimization(
        model, X_train, y_train, X_test, y_test,
        'random_forest', 'classification', n_iter=20, cv=5
    )
    
    st.write(f"Best Score: {search_results['best_score']:.4f}")
    st.write(f"Best Parameters: {search_results['best_params']}")


def example_param_distributions():
    """Show parameter distributions for models."""
    st.markdown("## Example 2: Parameter Distributions")
    
    models = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']
    
    selected_model = st.selectbox("Select Model", models)
    
    param_dist = HyperparameterOptimizer.get_param_distribution(selected_model)
    
    if param_dist:
        st.write(f"**{selected_model.replace('_', ' ').title()} Parameters:**")
        for param, values in param_dist.items():
            st.write(f"- {param}: {values}")
    else:
        st.write("No parameters to optimize")


def example_full_pipeline():
    """Full training pipeline with optional HP optimization."""
    st.markdown("## Example 3: Full Training Pipeline")
    
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
        model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])
        
        if model_choice == "Random Forest":
            model = RandomForestClassifier(random_state=42)
            model_name = "random_forest"
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "logistic_regression"
    
    with col2:
        cv_folds = st.slider("K-Fold Splits", 3, 10, 5)
    
    # HP optimization config
    enable_tuning, n_iter = render_hp_optimization_config(model_name)
    
    if st.button("🚀 Train Model", type="primary"):
        trained_model, search_results, predictions = train_with_optional_tuning(
            model, X_train, y_train, X_test, y_test,
            model_name, 'classification', cv_folds,
            enable_tuning, n_iter
        )
        
        if search_results:
            HyperparameterOptimizer.display_optimization_results(search_results)
            st.success("✅ HP optimization complete!")
        else:
            st.info("ℹ️ Standard training completed (no HP optimization)")


def example_comparison():
    """Compare default vs optimized parameters."""
    st.markdown("## Example 4: Default vs Optimized")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Default model
    default_model = RandomForestClassifier(random_state=42)
    default_model.fit(X_train, y_train)
    default_score = default_model.score(X_test, y_test)
    
    # Optimized model
    optimized_model, search_results, _ = train_with_hp_optimization(
        RandomForestClassifier(random_state=42),
        X_train, y_train, X_test, y_test,
        'random_forest', 'classification', n_iter=30, cv=5
    )
    optimized_score = optimized_model.score(X_test, y_test)
    
    comparison_data = {
        'Model': ['Default', 'Optimized'],
        'Test Score': [default_score, optimized_score],
        'Improvement': [0, optimized_score - default_score]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    st.write(f"**Best Parameters Found:**")
    for param, value in search_results['best_params'].items():
        st.write(f"- {param}: {value}")


def example_iteration_impact():
    """Show impact of n_iter on results."""
    st.markdown("## Example 5: Impact of Search Iterations")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = []
    for n_iter in [5, 10, 20, 50]:
        _, search_results, _ = train_with_hp_optimization(
            RandomForestClassifier(random_state=42),
            X_train, y_train, X_test, y_test,
            'random_forest', 'classification', n_iter=n_iter, cv=5
        )
        results.append({
            'Iterations': n_iter,
            'Best Score': search_results['best_score']
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    st.write("More iterations generally find better parameters but take longer.")


def main():
    """Main example app."""
    st.set_page_config(page_title="HP Optimization Examples", layout="wide")
    
    st.title("Hyperparameter Optimization Examples")
    st.divider()
    
    example_choice = st.selectbox(
        "Select Example",
        [
            "Basic HP Optimization",
            "Parameter Distributions",
            "Full Training Pipeline",
            "Default vs Optimized",
            "Impact of Search Iterations"
        ]
    )
    
    st.divider()
    
    if example_choice == "Basic HP Optimization":
        example_basic_hp_optimization()
    elif example_choice == "Parameter Distributions":
        example_param_distributions()
    elif example_choice == "Full Training Pipeline":
        example_full_pipeline()
    elif example_choice == "Default vs Optimized":
        example_comparison()
    elif example_choice == "Impact of Search Iterations":
        example_iteration_impact()


if __name__ == "__main__":
    main()
