"""
Streamlit Integration for Deep Learning

Completely isolated from ML pipeline.
Shows epochs ONLY for DL models.
Displays training & validation loss.
"""

import streamlit as st
import plotly.graph_objects as go
from models.dl_trainer import DLTrainer, prepare_dl_data, get_dl_predictions
from models.model_config import is_deep_learning
from evaluation.metrics import MetricsCalculator


def render_dl_config(model_name):
    """
    Render DL configuration UI.
    
    Returns:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        early_stopping: Whether to use early stopping
    """
    
    if not is_deep_learning(model_name):
        return None, None, None, None
    
    st.markdown("**Deep Learning Parameters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider(
            "Epochs",
            min_value=1,
            max_value=500,
            value=50,
            help="Number of complete passes through training data"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            [16, 32, 64, 128, 256],
            index=1,
            help="Number of samples per gradient update"
        )
    
    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            help="Optimizer step size"
        )
        
        early_stopping = st.checkbox(
            "Early Stopping",
            value=True,
            help="Stop training if validation loss plateaus"
        )
    
    st.info(
        f"**Epochs**: {epochs} complete passes through {len(st.session_state.get('X_train', []))} training samples\n\n"
        f"**Batch Size**: {batch_size} samples per update\n\n"
        f"**Learning Rate**: {learning_rate}"
    )
    
    return epochs, batch_size, learning_rate, early_stopping


def train_dl_model(model_name, X_train, y_train, X_val, y_val, X_test, y_test,
                   task_type, epochs, batch_size, learning_rate, early_stopping):
    """
    Train DL model with Streamlit integration.
    
    Returns:
        trained_model: Trained Keras model
        history: Training history
        predictions: Test predictions
    """
    
    with st.spinner(f"🧠 Training {model_name} ({epochs} epochs)..."):
        # Prepare data
        X_train_prep, y_train_prep, X_val_prep, y_val_prep, X_test_prep, y_test_prep = prepare_dl_data(
            X_train, y_train, X_val, y_val, X_test, y_test, task_type
        )
        
        # Build model
        if model_name == 'sequential':
            output_dim = len(np.unique(y_train)) if task_type == 'classification' else 1
            model = DLTrainer.build_sequential_model(X_train_prep.shape[1], output_dim, task_type)
        elif model_name == 'cnn':
            # Reshape for CNN
            X_train_prep = X_train_prep.reshape(-1, 28, 28, 1)
            X_val_prep = X_val_prep.reshape(-1, 28, 28, 1)
            X_test_prep = X_test_prep.reshape(-1, 28, 28, 1)
            output_dim = len(np.unique(y_train)) if task_type == 'classification' else 1
            model = DLTrainer.build_cnn_model((28, 28, 1), output_dim, task_type)
        elif model_name == 'rnn':
            # Reshape for RNN
            X_train_prep = X_train_prep.reshape(X_train_prep.shape[0], -1, 1)
            X_val_prep = X_val_prep.reshape(X_val_prep.shape[0], -1, 1)
            X_test_prep = X_test_prep.reshape(X_test_prep.shape[0], -1, 1)
            output_dim = len(np.unique(y_train)) if task_type == 'classification' else 1
            model = DLTrainer.build_rnn_model((X_train_prep.shape[1], 1), output_dim, task_type)
        
        # Train
        history, trained_model = DLTrainer.train_dl_model(
            model, X_train_prep, y_train_prep, X_val_prep, y_val_prep,
            epochs, batch_size, learning_rate, early_stopping
        )
        
        # Get predictions
        predictions = get_dl_predictions(trained_model, X_test_prep, task_type)
    
    return trained_model, history, predictions


def display_dl_training_results(history, predictions, y_test, task_type):
    """Display DL training results with loss curves."""
    
    st.divider()
    st.markdown("### Training Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training & Validation Loss**")
        
        # Plot loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history.history['loss'],
            name='Training Loss',
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            y=history.history['val_loss'],
            name='Validation Loss',
            mode='lines'
        ))
        fig.update_layout(
            title='Loss Over Epochs',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Performance Metrics**")
        
        if task_type == 'classification':
            metrics = MetricsCalculator.classification_metrics(y_test, predictions)
        else:
            metrics = MetricsCalculator.regression_metrics(y_test, predictions)
        
        metric_count = 0
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and metric_count < 4:
                st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                metric_count += 1
    
    # Training summary
    st.markdown("**Training Summary**")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Final Training Loss", f"{history.history['loss'][-1]:.4f}")
    col2.metric("Final Validation Loss", f"{history.history['val_loss'][-1]:.4f}")
    col3.metric("Total Epochs", len(history.history['loss']))
    col4.metric("Best Epoch", np.argmin(history.history['val_loss']) + 1)


import numpy as np
