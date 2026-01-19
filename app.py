"""
Streamlit Web Application

Interactive ML/DL model training and evaluation dashboard.
Provides end-to-end ML workflow with dataset upload, model selection,
training, and evaluation.

Author: Streamlit Expert
Date: 2026-01-19
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import logging
import io
import joblib
from datetime import datetime

# Import custom modules
from data_preprocessing import DataPreprocessor, preprocess_dataset
from models.model_factory import ModelFactory
from train import train_model, train_full_pipeline, TrainingHistory
from evaluate import evaluate_model, generate_evaluation_report, save_metrics_json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="ML/DL Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'dataset': None,
        'data_preprocessed': False,
        'X_train': None,
        'X_val': None,
        'X_test': None,
        'y_train': None,
        'y_val': None,
        'y_test': None,
        'preprocessor': None,
        'model': None,
        'trained_model': None,
        'training_history': None,
        'metrics': None,
        'model_trained': False,
        'last_task_type': None,
        'last_model_name': None,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_data
def load_sample_dataset():
    """Load a sample dataset for demonstration."""
    np.random.seed(42)
    n_samples = 200
    
    # Create sample classification dataset
    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    
    return df


def reset_training_state():
    """Reset training-related session state."""
    st.session_state.model = None
    st.session_state.trained_model = None
    st.session_state.training_history = None
    st.session_state.metrics = None
    st.session_state.model_trained = False


# ============================================================================
# Page: Data Upload & Preprocessing
# ============================================================================

def page_data_loading():
    """Data loading and preprocessing page."""
    st.header("📊 Data Loading & Preprocessing")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    with col2:
        st.subheader("Or Load Sample")
        if st.button("Load Sample Data", key="sample_data"):
            st.session_state.dataset = load_sample_dataset()
            st.success("✓ Sample dataset loaded")
    
    # Load dataset if uploaded
    if uploaded_file is not None:
        try:
            st.session_state.dataset = pd.read_csv(uploaded_file)
            st.success(f"✓ Dataset loaded: {st.session_state.dataset.shape[0]} rows × {st.session_state.dataset.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    # Display dataset if loaded
    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        
        # Dataset Overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", dataset.shape[0])
        col2.metric("Columns", dataset.shape[1])
        col3.metric("Memory Usage", f"{dataset.memory_usage(deep=True).sum() / 1024:.2f} KB")
        col4.metric("Missing Values", dataset.isnull().sum().sum())
        
        # Data Preview
        st.subheader("Data Preview")
        st.dataframe(dataset.head(10), use_container_width=True)
        
        # Data Statistics
        st.subheader("Data Statistics")
        st.dataframe(dataset.describe(), use_container_width=True)
        
        # Data Types
        st.subheader("Column Types")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.dataframe(pd.DataFrame({
                'Column': dataset.columns,
                'Type': dataset.dtypes.astype(str)
            }), use_container_width=True)
        
        with col2:
            # Missing values visualization
            missing_data = dataset.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(
                    x=missing_data[missing_data > 0].index,
                    y=missing_data[missing_data > 0].values,
                    labels={'x': 'Column', 'y': 'Missing Count'},
                    title='Missing Values'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Preprocessing Configuration
        st.subheader("Preprocessing Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            target_col = st.selectbox(
                "Select Target Column",
                options=dataset.columns,
                key="target_column"
            )
        
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        
        # Preprocessing button
        if st.button("🔄 Preprocess Data", key="preprocess_btn", type="primary"):
            with st.spinner("Preprocessing data..."):
                try:
                    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_dataset(
                        file_like_csv(dataset),
                        target_col=target_col,
                        test_size=test_size,
                        val_size=0.1
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_val = X_val
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_val = y_val
                    st.session_state.y_test = y_test
                    st.session_state.preprocessor = preprocessor
                    st.session_state.data_preprocessed = True
                    
                    st.success("✓ Data preprocessed successfully!")
                    
                    # Show preprocessing summary
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Training Set", f"{X_train.shape[0]} samples")
                    col2.metric("Validation Set", f"{X_val.shape[0]} samples")
                    col3.metric("Test Set", f"{X_test.shape[0]} samples")
                    
                except Exception as e:
                    st.error(f"Preprocessing error: {str(e)}")
                    logger.error(f"Preprocessing error: {str(e)}")


def file_like_csv(df):
    """Convert DataFrame to file path or pass directly (deprecated - kept for compatibility)."""
    # Now we can pass DataFrames directly, no need to convert
    return df


# ============================================================================
# Page: Model Selection & Training
# ============================================================================

def page_model_training():
    """Model selection and training page."""
    st.header("🧠 Model Training")
    
    # Check if data is preprocessed
    if not st.session_state.data_preprocessed:
        st.warning("⚠️ Please preprocess data first in the Data Loading tab")
        return
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        task_type = st.selectbox(
            "Task Type",
            options=['classification', 'regression']
        )
    
    with col2:
        available_models = ModelFactory.get_available_models(task_type)
        model_name = st.selectbox(
            "Model Type",
            options=available_models
        )
    
    with col3:
        reset_training = st.checkbox("Reset Previous Training", value=False)
        if reset_training:
            reset_training_state()
    
    # Model-Specific Hyperparameters
    st.subheader("Hyperparameters")
    
    hyperparams = {}
    
    if model_name == 'neural_network':
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hyperparams['input_dim'] = st.session_state.X_train.shape[1]
            
            if task_type == 'classification':
                n_classes = len(np.unique(st.session_state.y_train))
                hyperparams['num_classes'] = n_classes
        
        with col2:
            epochs = st.slider("Epochs", 10, 500, 50, 10, key="epochs")
            hyperparams['epochs'] = epochs
        
        with col3:
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], key="batch_size")
            hyperparams['batch_size'] = batch_size
        
        # Hidden layers configuration
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers_str = st.text_input(
                "Hidden Layers (comma-separated)",
                value="128,64",
                help="e.g., 256,128,64"
            )
            try:
                hyperparams['hidden_layers'] = [int(x.strip()) for x in hidden_layers_str.split(',')]
            except:
                st.error("Invalid hidden layers format")
                return
        
        with col2:
            dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05, key="dropout")
            hyperparams['dropout_rate'] = dropout
    
    else:
        # Scikit-learn models
        col1, col2 = st.columns(2)
        
        if model_name == 'random_forest':
            with col1:
                hyperparams['n_estimators'] = st.slider(
                    "Number of Trees",
                    10, 500, 100, 10
                )
            with col2:
                hyperparams['max_depth'] = st.slider(
                    "Max Depth",
                    5, 50, 10, 1
                )
        
        elif model_name == 'svm':
            with col1:
                hyperparams['C'] = st.slider(
                    "Regularization (C)",
                    0.1, 10.0, 1.0, 0.1
                )
            with col2:
                hyperparams['kernel'] = st.selectbox(
                    "Kernel",
                    options=['rbf', 'linear', 'poly']
                )
        
        elif model_name == 'logistic_regression':
            with col1:
                hyperparams['max_iter'] = st.slider(
                    "Max Iterations",
                    100, 1000, 1000, 100
                )
            with col2:
                hyperparams['solver'] = st.selectbox(
                    "Solver",
                    options=['lbfgs', 'liblinear', 'saga']
                )
    
    # Training button
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("🚀 Train Model", key="train_btn", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Create model
                    model = ModelFactory.create_model(task_type, model_name, **hyperparams)
                    
                    # Train model
                    if model_name == 'neural_network':
                        trained_model, history = train_model(
                            model,
                            st.session_state.X_train,
                            st.session_state.y_train,
                            X_val=st.session_state.X_val,
                            y_val=st.session_state.y_val,
                            epochs=hyperparams['epochs'],
                            batch_size=hyperparams['batch_size'],
                            verbose=0
                        )
                    else:
                        trained_model, history = train_model(
                            model,
                            st.session_state.X_train,
                            st.session_state.y_train,
                            X_val=st.session_state.X_val,
                            y_val=st.session_state.y_val
                        )
                    
                    st.session_state.trained_model = trained_model
                    st.session_state.training_history = history
                    st.session_state.model_trained = True
                    st.session_state.last_task_type = task_type
                    st.session_state.last_model_name = model_name
                    
                    st.success("✓ Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    logger.error(f"Training error: {str(e)}")
    
    # Display training results
    if st.session_state.model_trained and st.session_state.training_history:
        st.subheader("Training Results")
        
        history = st.session_state.training_history.get_summary()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Time", f"{history['total_time']:.2f}s")
        col2.metric("Total Epochs", history['epochs'])
        col3.metric("Model Type", model_name)
        
        # Show training curves for neural networks
        if model_name == 'neural_network' and 'metrics' in history:
            metrics_data = history['metrics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss curve
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=metrics_data.get('loss', []),
                    name='Training Loss',
                    mode='lines'
                ))
                if 'val_loss' in metrics_data:
                    fig_loss.add_trace(go.Scatter(
                        y=metrics_data.get('val_loss', []),
                        name='Validation Loss',
                        mode='lines'
                    ))
                fig_loss.update_layout(
                    title='Loss Curve',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_loss, use_container_width=True)
            
            with col2:
                # Accuracy curve
                if 'accuracy' in metrics_data:
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(
                        y=metrics_data.get('accuracy', []),
                        name='Training Accuracy',
                        mode='lines'
                    ))
                    if 'val_accuracy' in metrics_data:
                        fig_acc.add_trace(go.Scatter(
                            y=metrics_data.get('val_accuracy', []),
                            name='Validation Accuracy',
                            mode='lines'
                        ))
                    fig_acc.update_layout(
                        title='Accuracy Curve',
                        xaxis_title='Epoch',
                        yaxis_title='Accuracy',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)


# ============================================================================
# Page: Model Evaluation
# ============================================================================

def page_evaluation():
    """Model evaluation page."""
    st.header("📈 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the Model Training tab")
        return
    
    st.subheader("Evaluating on Test Set...")
    
    with st.spinner("Computing metrics..."):
        try:
            metrics = evaluate_model(
                st.session_state.trained_model,
                st.session_state.X_test,
                st.session_state.y_test,
                task_type=st.session_state.last_task_type
            )
            st.session_state.metrics = metrics
            
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            logger.error(f"Evaluation error: {str(e)}")
            return
    
    # Display metrics
    st.subheader("Performance Metrics")
    
    if st.session_state.last_task_type == 'classification':
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        col3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        col4.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
        
        if 'roc_auc' in metrics:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        
        # Confusion matrix visualization
        if 'confusion_matrix' in metrics:
            st.subheader("Confusion Matrix")
            cm = np.array(metrics['confusion_matrix'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:  # Regression
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{metrics.get('mae', 0):.4f}")
        col2.metric("MSE", f"{metrics.get('mse', 0):.4f}")
        col3.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
        col4.metric("R² Score", f"{metrics.get('r2', 0):.4f}")
        
        # Predictions vs Actual
        st.subheader("Predictions vs Actual")
        y_true = np.array(metrics['y_true'])
        y_pred = np.array(metrics['y_pred'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6)
        ))
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        fig.update_layout(
            title='Predictions vs Actual Values',
            xaxis_title='Actual',
            yaxis_title='Predicted',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        st.subheader("Residuals")
        residuals = y_true - y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=8, opacity=0.6)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted',
            yaxis_title='Residuals',
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Page: Model Download
# ============================================================================

def page_download():
    """Model download and export page."""
    st.header("📥 Model Export & Download")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first")
        return
    
    col1, col2 = st.columns(2)
    
    # Download trained model
    with col1:
        st.subheader("Download Trained Model")
        
        model_bytes = io.BytesIO()
        joblib.dump(st.session_state.trained_model, model_bytes)
        model_bytes.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{st.session_state.last_model_name}_{timestamp}.pkl"
        
        st.download_button(
            label="⬇️ Download Model (.pkl)",
            data=model_bytes,
            file_name=filename,
            mime="application/octet-stream"
        )
    
    # Download metrics as JSON
    with col2:
        st.subheader("Download Evaluation Metrics")
        
        if st.session_state.metrics:
            # Convert to JSON-serializable format
            def convert_to_serializable(obj):
                """Recursively convert non-JSON-serializable objects."""
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, pd.Series):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, (bool, np.bool_)):
                    return bool(obj)
                else:
                    return obj
            
            metrics_json = convert_to_serializable(st.session_state.metrics)
            json_str = json.dumps(metrics_json, indent=2)
            
            st.download_button(
                label="⬇️ Download Metrics (JSON)",
                data=json_str,
                file_name=f"metrics_{timestamp}.json",
                mime="application/json"
            )
    
    # Model Configuration Summary
    st.subheader("Model Configuration")
    
    config_summary = f"""
    **Task Type:** {st.session_state.last_task_type}
    **Model Type:** {st.session_state.last_model_name}
    **Training Timestamp:** {timestamp}
    """
    
    if st.session_state.training_history:
        history = st.session_state.training_history.get_summary()
        config_summary += f"\n**Training Time:** {history['total_time']:.2f}s\n"
        config_summary += f"**Total Epochs:** {history['epochs']}\n"
    
    if st.session_state.metrics:
        if st.session_state.last_task_type == 'classification':
            config_summary += f"\n**Accuracy:** {st.session_state.metrics.get('accuracy', 0):.4f}"
        else:
            config_summary += f"\n**R² Score:** {st.session_state.metrics.get('r2', 0):.4f}"
    
    st.markdown(config_summary)
    
    # Training History Export
    if st.session_state.training_history:
        st.subheader("Training History")
        
        history_json = json.dumps(
            st.session_state.training_history.get_summary(),
            indent=2,
            default=str
        )
        
        st.download_button(
            label="⬇️ Download Training History (JSON)",
            data=history_json,
            file_name=f"training_history_{timestamp}.json",
            mime="application/json"
        )


# ============================================================================
# Main App
# ============================================================================

def main():
    """Main application."""
    # Sidebar
    st.sidebar.title("🤖 ML/DL Trainer")
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Data Loading",
            "🧠 Model Training",
            "📈 Evaluation",
            "📥 Download"
        ]
    )
    
    # Page content
    if page == "📊 Data Loading":
        page_data_loading()
    elif page == "🧠 Model Training":
        page_model_training()
    elif page == "📈 Evaluation":
        page_evaluation()
    elif page == "📥 Download":
        page_download()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**ML/DL Trainer v1.0**\n\n"
        "End-to-end machine learning training and evaluation platform.\n\n"
        "📚 [Documentation](https://github.com)\n"
        "💬 [Support](https://github.com/issues)"
    )


if __name__ == "__main__":
    main()
