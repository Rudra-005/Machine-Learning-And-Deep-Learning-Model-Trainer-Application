"""
ML/DL Trainer - Production Ready Application

Complete machine learning and deep learning training platform with AutoML mode.
Supports data upload, preprocessing, model training, evaluation, and deployment.

Run with: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from models.automl import AutoMLConfig, detect_model_category
from models.automl_trainer import train_with_automl
from app.utils.automl_ui import (
    render_automl_mode,
    render_automl_summary,
    render_automl_comparison,
    display_automl_results
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="ML/DL Trainer - Production Ready",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem; font-weight: bold;
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
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.task_type = None
    st.session_state.trained_model = None
    st.session_state.training_results = None
    st.session_state.model_trained = False


def is_data_ready():
    """Single source of truth: data is ready if X_train is not None."""
    return st.session_state.X_train is not None

# Model registry
ML_MODELS = {
    'Classification': {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier()
    },
    'Regression': {
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
}

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar
    st.sidebar.title("🤖 ML/DL Trainer")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "🏠 Home",
            "📊 Data Loading",
            "🧠 AutoML Training",
            "📈 Results & Evaluation",
            "📚 Documentation",
            "ℹ️ About"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status**")
    if is_data_ready():
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.warning("⚠️ No Data Loaded")
    
    if st.session_state.model_trained:
        st.sidebar.success("✅ Model Trained")
    else:
        st.sidebar.info("ℹ️ No Model Trained")
    
    # Route to pages
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Data Loading":
        page_data_loading()
    elif page == "🧠 AutoML Training":
        page_automl_training()
    elif page == "📈 Results & Evaluation":
        page_results()
    elif page == "📚 Documentation":
        page_documentation()
    elif page == "ℹ️ About":
        page_about()


def page_home():
    """Home page."""
    st.title("🤖 ML/DL Trainer - Production Ready")
    
    st.markdown("""
    ## Welcome to ML/DL Trainer
    
    A production-ready platform for training, evaluating, and deploying machine learning and deep learning models.
    
    ### ✨ Key Features
    
    - **📤 Data Upload** - Upload CSV files or use sample datasets
    - **🔍 EDA** - Exploratory data analysis with visualizations
    - **🤖 AutoML Mode** - Automatic model detection and strategy selection
    - **🎯 Model Selection** - 9+ ML algorithms and 3 DL architectures
    - **⚙️ Hyperparameter Tuning** - Automatic hyperparameter optimization
    - **📊 Evaluation** - Comprehensive metrics and visualizations
    - **💾 Model Export** - Download trained models
    - **🚀 Production Ready** - Error handling, logging, monitoring
    
    ### 🚀 Quick Start
    
    1. **Load Data** - Go to "📊 Data Loading" tab
    2. **Train Model** - Go to "🧠 AutoML Training" tab
    3. **View Results** - Go to "📈 Results & Evaluation" tab
    
    ### 🎯 What is AutoML Mode?
    
    AutoML Mode automatically:
    - Detects your model type (tree-based, iterative, SVM, deep learning)
    - Selects the optimal training strategy (K-Fold CV, epochs, tuning)
    - Shows only relevant parameters
    - Trains with best practices
    
    **No configuration needed!**
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Supported Models", "15+")
    col2.metric("ML Algorithms", "9")
    col3.metric("DL Architectures", "3")
    col4.metric("Evaluation Metrics", "10+")


def page_data_loading():
    """Data loading page."""
    st.title("📊 Data Loading & Preprocessing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Load Data")
        
        data_source = st.radio(
            "Choose data source:",
            options=["Sample Dataset", "Upload CSV"],
            horizontal=True
        )
        
        if data_source == "Sample Dataset":
            dataset_choice = st.selectbox(
                "Select sample dataset:",
                options=["Iris (Classification)", "Wine (Classification)", "Diabetes (Regression)"]
            )
            
            if st.button("Load Sample Dataset", type="primary", use_container_width=True):
                try:
                    if "Iris" in dataset_choice:
                        data = load_iris()
                        X, y = data.data, data.target
                        st.session_state.task_type = "Classification"
                        dataset_name = "Iris"
                    elif "Wine" in dataset_choice:
                        data = load_wine()
                        X, y = data.data, data.target
                        st.session_state.task_type = "Classification"
                        dataset_name = "Wine"
                    else:
                        data = load_diabetes()
                        X, y = data.data, data.target
                        st.session_state.task_type = "Regression"
                        dataset_name = "Diabetes"
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success(f"✅ {dataset_name} dataset loaded successfully!")
                    logger.info(f"Loaded {dataset_name} dataset: {X_train.shape[0]} training samples")
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
                    logger.error(f"Error loading dataset: {str(e)}")
        
        else:
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("✅ File uploaded successfully!")
                    logger.info(f"Uploaded file: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error uploading file: {str(e)}")
                    logger.error(f"Error uploading file: {str(e)}")
    
    with col2:
        st.subheader("Data Info")
        if is_data_ready():
            col1, col2 = st.columns(2)
            col1.metric("Training Samples", st.session_state.X_train.shape[0])
            col2.metric("Test Samples", st.session_state.X_test.shape[0])
            col1.metric("Features", st.session_state.X_train.shape[1])
            col2.metric("Task Type", st.session_state.task_type or "Unknown")
    
    # Display data info
    if is_data_ready():
        st.subheader("📋 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Set", f"{st.session_state.X_train.shape[0]} samples")
        col2.metric("Test Set", f"{st.session_state.X_test.shape[0]} samples")
        col3.metric("Features", st.session_state.X_train.shape[1])
        
        # Data statistics
        st.subheader("📊 Data Statistics")
        stats_df = pd.DataFrame({
            'Mean': st.session_state.X_train.mean(axis=0),
            'Std': st.session_state.X_train.std(axis=0),
            'Min': st.session_state.X_train.min(axis=0),
            'Max': st.session_state.X_train.max(axis=0)
        })
        st.dataframe(stats_df, use_container_width=True)


def page_automl_training():
    """AutoML training page."""
    st.title("🧠 AutoML Training")
    
    if not is_data_ready():
        st.warning("⚠️ Please load data first in the Data Loading tab")
        return
    
    st.markdown("""
    **AutoML Mode** automatically detects your model type and applies the optimal training strategy.
    """)
    
    # Step 1: Task Type
    st.subheader("1️⃣ Task Type")
    task_type = st.radio(
        "Select task type:",
        options=['Classification', 'Regression'],
        horizontal=True
    )
    st.session_state.task_type = task_type
    
    # Step 2: Model Selection
    st.subheader("2️⃣ Model Selection")
    model_name = st.selectbox(
        "Choose a model (strategy will be auto-detected):",
        options=list(ML_MODELS[task_type].keys())
    )
    
    model = ML_MODELS[task_type][model_name]
    
    # Step 3: AutoML Configuration
    st.subheader("3️⃣ AutoML Configuration")
    
    automl = AutoMLConfig(model)
    render_automl_summary(model, {})
    
    # Render parameters
    params = render_automl_mode(model)
    
    # Step 4: Strategy Explanation
    with st.expander("📖 Why this strategy?"):
        render_automl_comparison(model)
    
    # Step 5: Training
    st.subheader("4️⃣ Train Model")
    
    if st.button("🚀 Start AutoML Training", type="primary", use_container_width=True):
        with st.spinner("Training model with optimal strategy..."):
            try:
                # Get data
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                # Display training info
                config = automl.config
                st.info(
                    f"🤖 Training **{config['model_name']}** with **{config['description']}**"
                )
                
                # Train
                results = train_with_automl(
                    model, X_train, y_train, X_test, y_test, params
                )
                
                # Store results
                st.session_state.trained_model = results.get('best_estimator', model)
                st.session_state.training_results = results
                st.session_state.model_trained = True
                
                # Display results
                display_automl_results(model, results)
                
                st.success("✅ Training completed successfully!")
                logger.info(f"Model trained: {config['model_name']}")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                logger.error(f"Training failed: {str(e)}")


def page_results():
    """Results and evaluation page."""
    st.title("📈 Results & Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the AutoML Training tab")
        return
    
    st.subheader("📊 Training Results")
    
    results = st.session_state.training_results
    
    if results['strategy'] == 'k_fold_cv':
        col1, col2, col3 = st.columns(3)
        col1.metric("CV Mean Score", f"{results['cv_mean']:.4f}")
        col2.metric("CV Std Dev", f"{results['cv_std']:.4f}")
        col3.metric("Test Score", f"{results['test_score']:.4f}")
        
        # Confidence interval
        cv_mean = results['cv_mean']
        cv_std = results['cv_std']
        ci_lower = cv_mean - 1.96 * cv_std
        ci_upper = cv_mean + 1.96 * cv_std
        st.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Best params
        if results.get('best_params'):
            st.subheader("🎯 Best Hyperparameters")
            for param, value in results['best_params'].items():
                st.write(f"- **{param}**: {value}")
    
    elif results['strategy'] == 'epochs_with_early_stopping':
        col1, col2, col3 = st.columns(3)
        col1.metric("Train Loss", f"{results['train_loss']:.4f}")
        col2.metric("Val Loss", f"{results['val_loss']:.4f}")
        col3.metric("Test Accuracy", f"{results['test_accuracy']:.4f}")
    
    # Download model
    st.subheader("💾 Export Model")
    if st.button("Download Trained Model", use_container_width=True):
        import pickle
        model_bytes = pickle.dumps(st.session_state.trained_model)
        st.download_button(
            label="Download Model (PKL)",
            data=model_bytes,
            file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            mime="application/octet-stream"
        )


def page_documentation():
    """Documentation page."""
    st.title("📚 Documentation")
    
    st.markdown("""
    ## AutoML Mode Documentation
    
    ### How AutoML Works
    
    AutoML automatically detects your model type and applies the optimal training strategy:
    
    #### 🌳 Tree-Based Models
    - **Models**: Random Forest, Gradient Boosting, Decision Trees
    - **Strategy**: K-Fold Cross-Validation
    - **Why**: Single-pass convergence, robust overfitting detection
    - **Visible Parameters**: CV Folds, HP Tuning
    
    #### 🔄 Iterative Models
    - **Models**: Logistic Regression, SGD, Perceptron
    - **Strategy**: K-Fold CV + Max Iterations
    - **Why**: Need convergence control, robust evaluation
    - **Visible Parameters**: CV Folds, Max Iter, HP Tuning
    
    #### 🎯 SVM Models
    - **Models**: SVC, SVR, LinearSVC, LinearSVR
    - **Strategy**: K-Fold CV with Kernel Tuning
    - **Why**: Kernel selection critical, single-pass convergence
    - **Visible Parameters**: CV Folds, HP Tuning
    
    #### 🧠 Deep Learning Models
    - **Models**: Sequential, CNN, LSTM, RNN
    - **Strategy**: Epochs with Early Stopping
    - **Why**: Multiple passes needed, automatic overfitting prevention
    - **Visible Parameters**: Epochs, Batch Size, Learning Rate
    
    ### Parameter Visibility Matrix
    
    | Parameter | Tree | Iterative | SVM | DL |
    |-----------|------|-----------|-----|-----|
    | CV Folds | ✓ | ✓ | ✓ | ✗ |
    | Max Iter | ✗ | ✓ | ✗ | ✗ |
    | Epochs | ✗ | ✗ | ✗ | ✓ |
    | Batch Size | ✗ | ✗ | ✗ | ✓ |
    | Learning Rate | ✗ | ✓ | ✗ | ✓ |
    | HP Tuning | ✓ | ✓ | ✓ | ✗ |
    
    ### Workflow
    
    1. **Load Data** - Upload CSV or use sample dataset
    2. **Select Task** - Choose Classification or Regression
    3. **Select Model** - Choose from available models
    4. **AutoML Detects** - Model category identified automatically
    5. **Configure** - Set parameters (optional)
    6. **Train** - Click to train with optimal strategy
    7. **Evaluate** - View results and metrics
    8. **Export** - Download trained model
    """)


def page_about():
    """About page."""
    st.title("ℹ️ About ML/DL Trainer")
    
    st.markdown("""
    ## ML/DL Trainer - Production Ready Platform
    
    A comprehensive machine learning and deep learning training platform with AutoML capabilities.
    
    ### Features
    
    ✅ **Automatic Model Detection** - Detects model category instantly  
    ✅ **Intelligent Strategy Selection** - Applies optimal approach per model  
    ✅ **Clean UI** - Only relevant parameters shown  
    ✅ **Robust Evaluation** - K-Fold CV for ML, epochs for DL  
    ✅ **Optional Tuning** - Hyperparameter optimization available  
    ✅ **Production Ready** - Error handling, logging, monitoring  
    ✅ **Model Export** - Download trained models  
    ✅ **Comprehensive Metrics** - Detailed evaluation results  
    
    ### Technology Stack
    
    - **Frontend**: Streamlit
    - **ML/DL**: Scikit-learn, TensorFlow/Keras
    - **Data**: Pandas, NumPy
    - **Visualization**: Plotly
    - **DevOps**: Docker, Logging
    
    ### Supported Models
    
    **Classification** (5 models):
    - Random Forest
    - Gradient Boosting
    - Logistic Regression
    - SVM
    - KNN
    
    **Regression** (6 models):
    - Ridge
    - Lasso
    - Random Forest
    - Gradient Boosting
    - SVR
    - KNN
    
    ### Version
    
    **Version**: 1.0  
    **Status**: Production Ready  
    **Last Updated**: 2026-01-19  
    
    ### Support
    
    For documentation, see the "📚 Documentation" tab.
    """)


if __name__ == "__main__":
    main()
