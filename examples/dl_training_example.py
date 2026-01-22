"""
Deep Learning Training Examples

Demonstrates Sequential, CNN, RNN training with epochs.
Shows training & validation loss.
Completely isolated from ML pipeline.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from models.dl_trainer import DLTrainer, prepare_dl_data, get_dl_predictions
from app.utils.dl_streamlit import render_dl_config, train_dl_model, display_dl_training_results
from evaluation.metrics import MetricsCalculator


def example_epochs_explanation():
    """Explain epochs vs max_iter."""
    st.markdown("## Example 1: Epochs vs Max Iterations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Epochs (Deep Learning)**")
        st.write("""
        - Multiple passes through entire dataset
        - Each epoch processes all data in batches
        - Example: epochs=50 means 50 complete passes
        - Used for: Neural Networks, CNN, RNN
        - Typical range: 10-500 epochs
        """)
    
    with col2:
        st.markdown("**Max Iterations (Iterative ML)**")
        st.write("""
        - Single pass through dataset
        - Optimization iterations until convergence
        - Example: max_iter=1000 means up to 1000 iterations
        - Used for: LogisticRegression, SGD, Perceptron
        - Typical range: 100-10000 iterations
        """)


def example_sequential_training():
    """Train Sequential Neural Network."""
    st.markdown("## Example 2: Sequential Neural Network")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Epochs", 1, 100, 20)
        batch_size = st.selectbox("Batch Size", [16, 32, 64])
    
    with col2:
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001)
        early_stopping = st.checkbox("Early Stopping", True)
    
    if st.button("🚀 Train Sequential", type="primary"):
        trained_model, history, predictions = train_dl_model(
            'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
            'classification', epochs, batch_size, learning_rate, early_stopping
        )
        
        display_dl_training_results(history, predictions, y_test, 'classification')
        st.success("✅ Training complete!")


def example_loss_curves():
    """Show training & validation loss curves."""
    st.markdown("## Example 3: Loss Curves")
    
    digits = load_digits()
    X, y = digits.data, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    epochs = st.slider("Epochs", 1, 100, 30)
    
    if st.button("Train & Show Loss Curves", type="primary"):
        trained_model, history, predictions = train_dl_model(
            'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
            'classification', epochs, 32, 0.001, True
        )
        
        # Plot loss curves
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training Loss',
            mode='lines+markers'
        ))
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Validation Loss',
            mode='lines+markers'
        ))
        fig.update_layout(
            title='Training & Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


def example_batch_size_impact():
    """Show impact of batch size."""
    st.markdown("## Example 4: Batch Size Impact")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    st.write("Training with different batch sizes...")
    
    results = []
    for batch_size in [16, 32, 64]:
        trained_model, history, predictions = train_dl_model(
            'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
            'classification', 20, batch_size, 0.001, True
        )
        
        metrics = MetricsCalculator.classification_metrics(y_test, predictions)
        results.append({
            'Batch Size': batch_size,
            'Final Loss': history.history['loss'][-1],
            'Accuracy': metrics.get('accuracy', 0)
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)


def example_learning_rate_impact():
    """Show impact of learning rate."""
    st.markdown("## Example 5: Learning Rate Impact")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    st.write("Training with different learning rates...")
    
    results = []
    for lr in [0.0001, 0.001, 0.01]:
        trained_model, history, predictions = train_dl_model(
            'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
            'classification', 20, 32, lr, True
        )
        
        metrics = MetricsCalculator.classification_metrics(y_test, predictions)
        results.append({
            'Learning Rate': lr,
            'Final Loss': history.history['loss'][-1],
            'Accuracy': metrics.get('accuracy', 0)
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)


def example_early_stopping():
    """Show early stopping effect."""
    st.markdown("## Example 6: Early Stopping")
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**With Early Stopping**")
        trained_model, history, predictions = train_dl_model(
            'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
            'classification', 100, 32, 0.001, True
        )
        st.write(f"Stopped at epoch: {len(history.history['loss'])}")
    
    with col2:
        st.write("**Without Early Stopping**")
        trained_model, history, predictions = train_dl_model(
            'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
            'classification', 100, 32, 0.001, False
        )
        st.write(f"Completed epochs: {len(history.history['loss'])}")


def main():
    """Main example app."""
    st.set_page_config(page_title="DL Training Examples", layout="wide")
    
    st.title("Deep Learning Training Examples")
    st.divider()
    
    example_choice = st.selectbox(
        "Select Example",
        [
            "Epochs vs Max Iterations",
            "Sequential Training",
            "Loss Curves",
            "Batch Size Impact",
            "Learning Rate Impact",
            "Early Stopping"
        ]
    )
    
    st.divider()
    
    if example_choice == "Epochs vs Max Iterations":
        example_epochs_explanation()
    elif example_choice == "Sequential Training":
        example_sequential_training()
    elif example_choice == "Loss Curves":
        example_loss_curves()
    elif example_choice == "Batch Size Impact":
        example_batch_size_impact()
    elif example_choice == "Learning Rate Impact":
        example_learning_rate_impact()
    elif example_choice == "Early Stopping":
        example_early_stopping()


if __name__ == "__main__":
    main()
