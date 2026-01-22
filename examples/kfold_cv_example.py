"""
K-Fold Cross-Validation Integration Example

Demonstrates complete training pipeline with k-fold CV for ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation.kfold_validator import KFoldCrossValidator, train_ml_with_cv
from app.utils.cv_streamlit import (
    render_cv_config, train_with_cv_pipeline, display_training_results,
    get_cv_info_text
)
from models.model_config import is_deep_learning


def example_basic_cv():
    """Basic k-fold CV example."""
    st.markdown("## Example 1: Basic K-Fold CV")
    
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Compute CV scores
    cv_scores, mean_score, std_score = KFoldCrossValidator.compute_cv_scores(
        model, X_train, y_train, k=5, task_type='classification'
    )
    
    st.write(f"CV Scores: {cv_scores}")
    st.write(f"Mean: {mean_score:.4f}, Std: {std_score:.4f}")


def example_ml_vs_dl():
    """ML vs DL comparison."""
    st.markdown("## Example 2: ML vs Deep Learning")
    
    st.write("**ML Models (with k-fold CV):**")
    ml_models = ['random_forest', 'logistic_regression', 'svm']
    for model in ml_models:
        st.write(f"- {model}: CV enabled ✅")
    
    st.write("\n**Deep Learning Models (no k-fold CV):**")
    dl_models = ['sequential', 'cnn', 'rnn']
    for model in dl_models:
        st.write(f"- {model}: CV skipped ❌ (uses epochs instead)")


def example_full_pipeline():
    """Full training pipeline with CV."""
    st.markdown("## Example 3: Full Training Pipeline")
    
    # Load data
    wine = load_wine()
    X, y = wine.data, wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression"]
        )
        
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model_name = "random_forest"
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model_name = "logistic_regression"
    
    with col2:
        k, enable_cv = render_cv_config(model_name)
    
    if st.button("🚀 Train Model", type="primary"):
        if enable_cv and k:
            trained_model, cv_results, predictions, metrics = train_with_cv_pipeline(
                model, X_train, y_train, X_test, y_test, k, 'classification', model_name
            )
            
            display_training_results(cv_results, metrics, 'classification')
            
            st.success("✅ Training complete!")
        else:
            st.warning("⚠️ Cross-validation disabled")


def example_cv_comparison():
    """Compare different k values."""
    st.markdown("## Example 4: K-Fold Comparison")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    results = []
    for k in [3, 5, 10]:
        cv_scores, mean_score, std_score = KFoldCrossValidator.compute_cv_scores(
            model, X_train, y_train, k, 'classification'
        )
        results.append({
            'K': k,
            'Mean Score': mean_score,
            'Std Dev': std_score,
            'Min': cv_scores.min(),
            'Max': cv_scores.max()
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)


def example_cv_info():
    """Display CV configuration info."""
    st.markdown("## Example 5: CV Configuration")
    
    k = st.slider("Select K", 3, 10, 5)
    task_type = st.selectbox("Task Type", ["Classification", "Regression"])
    
    info = get_cv_info_text(k, task_type)
    st.info(info)


def main():
    """Main example app."""
    st.set_page_config(page_title="K-Fold CV Examples", layout="wide")
    
    st.title("K-Fold Cross-Validation Examples")
    st.divider()
    
    example_choice = st.selectbox(
        "Select Example",
        [
            "Basic K-Fold CV",
            "ML vs Deep Learning",
            "Full Training Pipeline",
            "K-Fold Comparison",
            "CV Configuration"
        ]
    )
    
    st.divider()
    
    if example_choice == "Basic K-Fold CV":
        example_basic_cv()
    elif example_choice == "ML vs Deep Learning":
        example_ml_vs_dl()
    elif example_choice == "Full Training Pipeline":
        example_full_pipeline()
    elif example_choice == "K-Fold Comparison":
        example_cv_comparison()
    elif example_choice == "CV Configuration":
        example_cv_info()


if __name__ == "__main__":
    main()
