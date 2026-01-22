"""
ML/DL Trainer - AutoML Mode Demo

Minimal runnable Streamlit application demonstrating AutoML functionality.
Run with: streamlit run app_demo.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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

# Configure Streamlit
st.set_page_config(
    page_title="ML/DL Trainer - AutoML Mode",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None

# Model registry
ML_MODELS = {
    'Classification': {
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    },
    'Regression': {
        'Ridge': Ridge(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
}

# ============================================================================
# Main App
# ============================================================================

def main():
    st.title("🤖 ML/DL Trainer - AutoML Mode")
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Select Page",
        options=["📊 Data Loading", "🧠 AutoML Training", "📈 Strategy Guide", "ℹ️ About"]
    )
    
    if page == "📊 Data Loading":
        page_data_loading()
    elif page == "🧠 AutoML Training":
        page_automl_training()
    elif page == "📈 Strategy Guide":
        page_strategy_guide()
    elif page == "ℹ️ About":
        page_about()


def page_data_loading():
    """Data loading page."""
    st.header("📊 Data Loading")
    
    st.markdown("""
    Load sample datasets or upload your own CSV file.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Datasets")
        dataset_choice = st.radio(
            "Choose a sample dataset:",
            options=["Iris (Classification)", "Diabetes (Regression)"]
        )
        
        if st.button("Load Sample Dataset", type="primary"):
            if "Iris" in dataset_choice:
                iris = load_iris()
                X = iris.data
                y = iris.target
                st.session_state.task_type = "Classification"
            else:
                diabetes = load_diabetes()
                X = diabetes.data
                y = diabetes.target
                st.session_state.task_type = "Regression"
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.data_loaded = True
            
            st.success("✅ Dataset loaded successfully!")
    
    with col2:
        st.subheader("Upload CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            st.success("✅ File uploaded successfully!")
    
    # Display loaded data info
    if st.session_state.data_loaded:
        st.subheader("📋 Loaded Data Info")
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Samples", st.session_state.X_train.shape[0])
        col2.metric("Test Samples", st.session_state.X_test.shape[0])
        col3.metric("Features", st.session_state.X_train.shape[1])


def page_automl_training():
    """AutoML training page."""
    st.header("🧠 AutoML Training")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first in the Data Loading tab")
        return
    
    st.markdown("""
    **AutoML Mode** automatically detects your model type and applies the optimal training strategy.
    """)
    
    # Step 1: Task Type
    st.subheader("1️⃣ Select Task Type")
    task_type = st.radio(
        "What type of problem are you solving?",
        options=['Classification', 'Regression'],
        horizontal=True
    )
    
    # Step 2: Model Selection
    st.subheader("2️⃣ Select Model")
    model_name = st.selectbox(
        "Choose a model (strategy will be auto-detected)",
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
                
                # Display results
                display_automl_results(model, results)
                
                st.success("✅ Training completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


def page_strategy_guide():
    """Strategy guide page."""
    st.header("📈 AutoML Strategy Guide")
    
    st.markdown("""
    ## How AutoML Selects Strategies
    
    AutoML automatically detects your model type and applies the optimal training approach:
    
    ### 🌳 Tree-Based Models
    **Models**: Random Forest, Gradient Boosting, Decision Trees
    
    **Strategy**: K-Fold Cross-Validation
    - Single-pass convergence (no epochs needed)
    - Robust overfitting detection
    - All data used for training
    
    **Visible Parameters**: CV Folds, HP Tuning
    
    ---
    
    ### 🔄 Iterative Models
    **Models**: Logistic Regression, SGD, Perceptron
    
    **Strategy**: K-Fold CV + Max Iterations
    - Need convergence control (max_iter)
    - Robust evaluation across folds
    - Prevents infinite training loops
    
    **Visible Parameters**: CV Folds, Max Iter, HP Tuning
    
    ---
    
    ### 🎯 SVM Models
    **Models**: SVC, SVR, LinearSVC, LinearSVR
    
    **Strategy**: K-Fold CV with Kernel Tuning
    - Kernel selection critical for performance
    - K-Fold CV validates choices
    - Hyperparameter tuning essential
    
    **Visible Parameters**: CV Folds, HP Tuning
    
    ---
    
    ### 🧠 Deep Learning Models
    **Models**: Sequential, CNN, LSTM, RNN
    
    **Strategy**: Epochs with Early Stopping
    - Multiple passes through data needed
    - Epochs track training progress
    - Early stopping prevents overfitting
    
    **Visible Parameters**: Epochs, Batch Size, Learning Rate, Early Stopping
    
    ---
    
    ## Parameter Visibility Matrix
    
    | Parameter | Tree | Iterative | SVM | DL |
    |-----------|------|-----------|-----|-----|
    | CV Folds | ✓ | ✓ | ✓ | ✗ |
    | Max Iter | ✗ | ✓ | ✗ | ✗ |
    | Epochs | ✗ | ✗ | ✗ | ✓ |
    | Batch Size | ✗ | ✗ | ✗ | ✓ |
    | Learning Rate | ✗ | ✓ | ✗ | ✓ |
    | HP Tuning | ✓ | ✓ | ✓ | ✗ |
    """)


def page_about():
    """About page."""
    st.header("ℹ️ About AutoML Mode")
    
    st.markdown("""
    ## What is AutoML Mode?
    
    AutoML Mode automatically detects your model type and applies the optimal training strategy.
    You don't need to understand the differences between CV, epochs, and convergence—the system
    handles it for you.
    
    ## Key Features
    
    ✅ **Automatic Model Detection** - Detects model category instantly  
    ✅ **Intelligent Strategy Selection** - Applies optimal approach per model  
    ✅ **Clean UI** - Only relevant parameters shown  
    ✅ **Robust Evaluation** - K-Fold CV for ML, epochs for DL  
    ✅ **Optional Tuning** - Hyperparameter optimization available  
    ✅ **Production Ready** - Error handling, logging, testing  
    
    ## How It Works
    
    1. **Select Model** - Choose from available models
    2. **AutoML Detects** - Model category identified automatically
    3. **Strategy Selected** - Optimal training approach chosen
    4. **Parameters Shown** - Only relevant controls displayed
    5. **Train** - Click to train with optimal strategy
    6. **View Results** - Metrics displayed appropriately
    
    ## Example Workflows
    
    ### Random Forest Classification
    - AutoML detects: Tree-based model
    - AutoML selects: K-Fold CV (5 folds)
    - UI shows: CV folds slider, HP tuning checkbox
    - UI hides: Epochs, max_iter, batch_size
    
    ### Logistic Regression Classification
    - AutoML detects: Iterative model
    - AutoML selects: K-Fold CV + max_iter
    - UI shows: CV folds, max_iter, HP tuning
    - UI hides: Epochs, batch_size
    
    ### Neural Network Classification
    - AutoML detects: Deep learning model
    - AutoML selects: Epochs + Early Stopping
    - UI shows: Epochs, batch_size, learning_rate
    - UI hides: CV folds, max_iter
    
    ## Benefits
    
    ✅ No confusion about parameters  
    ✅ Optimal strategy for each model  
    ✅ Faster training  
    ✅ Better results  
    ✅ Production ready  
    
    ## Documentation
    
    - **AUTOML_DOCUMENTATION.md** - Comprehensive guide
    - **AUTOML_QUICK_REFERENCE.md** - Quick reference
    - **AUTOML_VISUAL_REFERENCE.md** - Visual diagrams
    - **TRAINING_STRATEGY.md** - Strategy explanations
    """)


if __name__ == "__main__":
    main()
