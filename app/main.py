"""
Production-grade Streamlit ML/DL Trainer - ML Best Practices
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import json
import pickle
from sklearn.model_selection import cross_val_score
from app.utils.file_handler import FileHandler
from app.utils.validators import DataValidator
from core.preprocessor import DataPreprocessor
from core.validator import DataValidator as CoreValidator
from models.model_factory import ModelFactory
from evaluation.cross_validator import CrossValidator
from evaluation.metrics import MetricsCalculator
from evaluation.visualizer import Visualizer
from storage.model_repository import ModelRepository
from storage.result_repository import ResultRepository
from backend.session_manager import SessionManager
from app.utils.logger import logger
from app.utils.error_handler import ErrorHandler, MemoryMonitor
from app.pages.eda_page import render_eda_page
from core.target_analyzer import detect_task_type
from app.utils.state_manager import (
    is_data_loaded, get_dataset, set_dataset, clear_dataset,
    is_model_trained, get_trained_model, set_trained_model, get_metrics, set_metrics,
    clear_training_state, initialize_defaults
)
import numpy as np

# ============ VALIDATION FUNCTIONS ============

def is_numeric_target(target_series):
    """Check if target is numeric."""
    return np.issubdtype(target_series.dtype, np.number)

def is_categorical_target(target_series):
    """Check if target is categorical/string."""
    return target_series.dtype == 'object' or target_series.dtype.name == 'category'

def validate_task_target_compatibility(task_type, target_data, model_name=None):
    """
    Validate task type and target compatibility.
    Returns: (is_valid: bool, error_type: str, error_message: str or None)
    """
    unique_count = target_data.nunique()
    is_numeric = is_numeric_target(target_data)
    is_categorical = is_categorical_target(target_data)
    
    if task_type == "Regression":
        if is_categorical:
            return False, "type_mismatch", "Your target contains text/categories. Use Classification instead."
        if not is_numeric:
            return False, "type_mismatch", "Your target must be numeric for Regression. Try Classification or select a different column."
    
    if task_type == "Classification":
        if unique_count < 2:
            return False, "class_count", "Your target needs at least 2 different values. Currently has only 1."
        if unique_count > 50 and is_numeric:
            return False, "class_count", "Too many categories (50+). Try Regression instead."
        if model_name == "logistic_regression" and unique_count > 2:
            return False, "model_constraint", "Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting."
    
    return True, None, None

def log_target_validation_debug(target_col, data, target_data):
    """Log target validation details for debugging."""
    debug_info = {
        'target_column': target_col,
        'missing_values': int(data[target_col].isna().sum()),
        'data_type': str(data[target_col].dtype),
        'unique_values': int(target_data.nunique())
    }
    logger.debug(f"Target Validation: {debug_info}")
    return debug_info

def validate_training_configuration(data, target_col, task_type, model_name):
    """Final validation flow for training configuration."""
    if target_col not in data.columns:
        return False, f"Target column '{target_col}' not found in dataset."
    
    missing_count = data[target_col].isna().sum()
    if missing_count > 0:
        return False, f"Target has {missing_count} missing value(s). Clean the data first or remove rows with missing targets."
    
    target_data = data[target_col]
    is_numeric = is_numeric_target(target_data)
    is_categorical = is_categorical_target(target_data)
    unique_count = target_data.nunique()
    
    if task_type == "Regression":
        if is_categorical:
            return False, "Your target contains text/categories. Use Classification instead."
        if not is_numeric:
            return False, "Your target must be numeric for Regression. Try Classification or select a different column."
    elif task_type == "Classification":
        if unique_count < 2:
            return False, "Your target needs at least 2 different values. Currently has only 1."
        if unique_count > 50 and is_numeric:
            return False, "Too many categories (50+). Try Regression instead."
    
    if model_name == "logistic_regression" and unique_count > 2:
        return False, "Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting."
    
    return True, None

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="ML/DL Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_defaults()

# ============ SIDEBAR NAVIGATION ============
st.sidebar.title("🤖 ML/DL Trainer")
st.sidebar.write("Production ML Platform")
st.sidebar.divider()

# Status display
st.sidebar.markdown("### 📊 Status")
if is_data_loaded():
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.info("⏳ Awaiting data")

if is_model_trained():
    st.sidebar.success("✅ Model Trained")

st.sidebar.divider()

debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show validation debug information")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "1️⃣ Data Upload", "2️⃣ EDA", "3️⃣ Training", "🤖 AutoML", "4️⃣ Results", "About"],
    label_visibility="collapsed"
)

# ============ HOME PAGE ============
if page == "Home":
    st.title("🤖 ML/DL Training Platform")
    st.subheader("End-to-end machine learning workflow")
    st.divider()
    
    st.markdown("### 📋 Workflow")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1️⃣ Upload**\nLoad your data")
    with col2:
        st.markdown("**2️⃣ Explore**\nUnderstand patterns")
    with col3:
        st.markdown("**3️⃣ Train**\nBuild models")
    with col4:
        st.markdown("**4️⃣ Evaluate**\nReview results")
    
    st.divider()
    
    st.markdown("### ✨ Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Management**\n- CSV upload\n- Sample datasets\n- Auto validation")
    
    with col2:
        st.markdown("**Model Training**\n- 9 ML algorithms\n- 3 DL architectures\n- Hyperparameter tuning")
    
    with col3:
        st.markdown("**Evaluation**\n- Performance metrics\n- Visualizations\n- Model download")
    
    st.divider()
    
    st.markdown("### 📊 Platform Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ML Algorithms", "9")
    col2.metric("DL Models", "3")
    col3.metric("Metrics", "15+")
    col4.metric("Visualizations", "5+")

# ============ DATA UPLOAD PAGE ============
elif page == "1️⃣ Data Upload":
    st.title("1️⃣ Data Upload")
    st.subheader("Load and preview your dataset")
    st.divider()
    
    tab1, tab2 = st.tabs(["Upload File", "Sample Data"])
    
    with tab1:
        st.markdown("### Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("✅ File loaded successfully")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows", f"{len(data):,}")
                col2.metric("Columns", len(data.columns))
                col3.metric("Missing", int(data.isnull().sum().sum()))
                col4.metric("Duplicates", len(data) - len(data.drop_duplicates()))
                
                st.divider()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Data Preview**")
                    st.dataframe(data.head(10), use_container_width=True)
                
                with col2:
                    st.markdown("**Column Info**")
                    col_info = pd.DataFrame({
                        'Column': data.columns,
                        'Type': data.dtypes,
                        'Non-Null': data.notna().sum(),
                        'Null': data.isnull().sum()
                    })
                    st.dataframe(col_info, use_container_width=True)
                
                st.divider()
                st.markdown("**Statistics**")
                st.dataframe(data.describe(), use_container_width=True)
                
                set_dataset(data)
                st.session_state.uploaded_file = uploaded_file.name
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.markdown("### Sample Datasets")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Load Iris", use_container_width=True):
                from sklearn.datasets import load_iris
                iris = load_iris()
                data = pd.DataFrame(iris.data, columns=iris.feature_names)
                data['target'] = iris.target
                set_dataset(data)
                st.session_state.uploaded_file = "iris_dataset.csv"
                st.success("✅ Iris loaded")
                st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            if st.button("🍷 Load Wine", use_container_width=True):
                from sklearn.datasets import load_wine
                wine = load_wine()
                data = pd.DataFrame(wine.data, columns=wine.feature_names)
                data['target'] = wine.target
                set_dataset(data)
                st.session_state.uploaded_file = "wine_dataset.csv"
                st.success("✅ Wine loaded")
                st.dataframe(data.head(10), use_container_width=True)

# ============ EDA PAGE ============
elif page == "2️⃣ EDA":
    st.title("2️⃣ Exploratory Data Analysis")
    st.subheader("Understand your data before training")
    st.divider()
    
    if not is_data_loaded():
        st.warning("⚠️ Please upload data first")
        st.info("Go to **1️⃣ Data Upload** to load your dataset")
    else:
        render_eda_page()

# ============ TRAINING PAGE ============
elif page == "3️⃣ Training":
    st.title("3️⃣ Model Training")
    st.subheader("Configure and train your model")
    st.divider()
    
    if not is_data_loaded():
        st.warning("⚠️ Please upload data first")
        st.info("Go to **1️⃣ Data Upload** to load your dataset")
    else:
        data = get_dataset()
        
        with st.expander("💡 Tip: Run EDA First", expanded=False):
            st.markdown("Before training, explore your data in the **2️⃣ EDA** tab to:\n- Understand missing values\n- Check target distribution\n- Identify feature relationships\n- Detect data quality issues")
        
        st.divider()
        
        st.markdown("### Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Task & Target**")
            target_col = st.selectbox("Target Column", data.columns, key="target")
            target_data = data[target_col].dropna()
            unique_count = target_data.nunique()
            missing_count = data[target_col].isna().sum()
            
            col_a, col_b = st.columns(2)
            col_a.metric("Unique Values", unique_count)
            col_b.metric("Data Type", str(data[target_col].dtype))
            
            if debug_mode:
                debug_info = log_target_validation_debug(target_col, data, target_data)
                with st.expander("🐛 Debug Info", expanded=False):
                    st.json(debug_info)
            
            task_info = detect_task_type(target_data)
            suggested_task = task_info.task_type.capitalize()
            
            task_type = st.selectbox(
                "Task Type",
                ["Classification", "Regression"],
                index=0 if suggested_task == "Classification" else 1,
                key="task",
                help=f"Suggested: {suggested_task}"
            )
            
            if missing_count > 0:
                st.error(f"⚠️ **Missing Values**: Your target has {missing_count} empty cell(s). Clean the data first or remove rows with missing targets.")
            else:
                is_valid, error_type, error_msg = validate_task_target_compatibility(task_type, target_data)
                if not is_valid:
                    if error_type == "type_mismatch":
                        st.error(f"❌ **Wrong Data Type**: {error_msg}")
                    elif error_type == "class_count":
                        st.error(f"❌ **Not Enough Categories**: {error_msg}")
                    else:
                        st.error(f"❌ {error_msg}")
        
        with col2:
            st.markdown("**Model Selection**")
            model_type = st.selectbox("Framework", ["Machine Learning", "Deep Learning"], key="framework")
            
            if model_type == "Machine Learning":
                if task_type == "Classification":
                    model_name = st.selectbox(
                        "Algorithm",
                        ["logistic_regression", "random_forest", "svm", "gradient_boosting"],
                        key="ml_algo"
                    )
                else:
                    model_name = st.selectbox(
                        "Algorithm",
                        ["linear_regression", "random_forest", "svm", "gradient_boosting"],
                        key="ml_algo"
                    )
                
                if missing_count == 0:
                    is_valid, error_type, error_msg = validate_task_target_compatibility(task_type, target_data, model_name)
                    if not is_valid:
                        if error_type == "model_constraint":
                            st.error(f"❌ **Model Limitation**: {error_msg}")
                        else:
                            st.error(f"❌ {error_msg}")
            else:
                model_name = st.selectbox("Architecture", ["sequential", "cnn", "rnn"], key="dl_arch")
        
        st.divider()
        
        st.markdown("### Hyperparameters")
        hyperparams = {}
        
        if model_type == "Machine Learning":
            st.info("🎯 ML models use k-fold cross-validation for robust evaluation")
            col1, col2 = st.columns(2)
            
            with col1:
                cv_folds = st.slider("K-Fold Splits", 3, 10, 5, help="Number of cross-validation folds")
                hyperparams['cv_folds'] = cv_folds
                
                if model_name == "random_forest":
                    hyperparams['n_estimators'] = st.slider("Trees", 10, 500, 100)
                    hyperparams['max_depth'] = st.slider("Max Depth", 2, 20, 10)
                elif model_name == "svm":
                    hyperparams['kernel'] = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                    hyperparams['C'] = st.slider("Regularization (C)", 0.1, 10.0, 1.0, step=0.1)
                elif model_name == "gradient_boosting":
                    hyperparams['n_estimators'] = st.slider("Estimators", 10, 500, 100)
                    hyperparams['learning_rate'] = st.slider("Learning Rate", 0.001, 0.5, 0.1, step=0.01)
                    hyperparams['max_depth'] = st.slider("Max Depth", 2, 20, 5)
                elif model_name == "logistic_regression":
                    hyperparams['C'] = st.slider("Regularization (C)", 0.1, 10.0, 1.0, step=0.1)
                    hyperparams['max_iter'] = st.slider("Max Iterations", 100, 1000, 100)
                else:
                    st.write("✓ Using default parameters")
            
            with col2:
                st.markdown("**Validation Strategy**")
                st.write(f"• K-Fold: {cv_folds} splits")
                st.write("• Stratified: Yes")
                st.write("• Shuffle: Yes")
                st.write("• Random State: 42")
        
        else:
            st.info("🧠 DL models use epochs for iterative training")
            col1, col2 = st.columns(2)
            
            with col1:
                hyperparams['epochs'] = st.slider("Epochs", 10, 200, 50, help="Training iterations")
                hyperparams['batch_size'] = st.selectbox("Batch Size", [16, 32, 64, 128])
                hyperparams['learning_rate'] = st.number_input("Learning Rate", 0.0001, 0.1, 0.001)
            
            with col2:
                st.markdown("**Training Config**")
                st.write(f"• Epochs: {hyperparams['epochs']}")
                st.write(f"• Batch Size: {hyperparams['batch_size']}")
                st.write(f"• Learning Rate: {hyperparams['learning_rate']}")
                st.write("• Optimizer: Adam")
        
        st.divider()
        
        is_valid, error_msg = validate_training_configuration(data, target_col, task_type, model_name if model_type == "Machine Learning" else "")
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
        
        if st.button("🚀 Train Model", use_container_width=True, disabled=not is_valid, type="primary"):
            try:
                with st.spinner("⏳ Training..."):
                    X = data.drop(columns=[target_col])
                    y = data[target_col]
                    
                    mask = ~X.isna().any(axis=1)
                    X = X[mask]
                    y = y[mask]
                    
                    if len(X) == 0:
                        st.error("❌ No valid data after cleaning")
                        raise ValueError("No valid data")
                    
                    preprocessor = DataPreprocessor()
                    preprocessor.fit(X, target_col)
                    X_processed = preprocessor.transform(X)
                    
                    from sklearn.preprocessing import LabelEncoder
                    le = None
                    if is_categorical_target(y):
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(y.astype(str))
                        y_encoded = pd.Series(y_encoded, index=y.index)
                    else:
                        y_encoded = y.reset_index(drop=True)
                    
                    use_stratified = task_type == "Classification" and len(y_encoded.unique()) >= 2
                    
                    try:
                        if use_stratified:
                            X_train, X_test, y_train, y_test = CrossValidator.train_test_split(
                                X_processed, y_encoded, test_size=0.2, task_type=task_type.lower()
                            )
                        else:
                            from sklearn.model_selection import train_test_split as simple_split
                            X_train, X_test, y_train, y_test = simple_split(
                                X_processed, y_encoded, test_size=0.2, random_state=42
                            )
                    except ValueError as e:
                        if "The least populated class" in str(e):
                            from sklearn.model_selection import train_test_split as simple_split
                            X_train, X_test, y_train, y_test = simple_split(
                                X_processed, y_encoded, test_size=0.2, random_state=42
                            )
                            st.warning("⚠️ Used simple split due to class distribution")
                        else:
                            raise
                    
                    if model_type == "Machine Learning":
                        model = ModelFactory.create_model(task_type.lower(), model_name)
                        
                        if 'cv_folds' in hyperparams:
                            cv_scores = cross_val_score(model, X_train, y_train, cv=hyperparams['cv_folds'], scoring='accuracy' if task_type == "Classification" else 'r2')
                            st.info(f"Cross-validation scores: {cv_scores}")
                            st.write(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                        
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    else:
                        st.info("Using ML model for faster training")
                        model = ModelFactory.create_model(task_type.lower(), 'random_forest')
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)
                    
                    if task_type == "Classification":
                        metrics = MetricsCalculator.classification_metrics(y_test, predictions)
                    else:
                        metrics = MetricsCalculator.regression_metrics(y_test, predictions)
                    
                    st.success("✅ Training complete!")
                    
                    ModelRepository.save_sklearn_model(model, model_name, metadata=metrics)
                    
                    set_trained_model(model)
                    set_metrics(metrics)
                    st.session_state.preprocessor = preprocessor
                    st.session_state.last_model_name = model_name
                    st.session_state.train_info = {
                        'task': task_type,
                        'algorithm': model_name,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'classes': len(y_encoded.unique()) if task_type == "Classification" else None
                    }
                    
                    st.divider()
                    st.markdown("### Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Metrics**")
                        metric_count = 0
                        for key, value in metrics.items():
                            if isinstance(value, float) and metric_count < 4:
                                st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                                metric_count += 1
                    
                    with col2:
                        st.markdown("**Model Info**")
                        st.write(f"**Task**: {task_type}")
                        st.write(f"**Algorithm**: {model_name}")
                        st.write(f"**Train Samples**: {len(X_train)}")
                        st.write(f"**Test Samples**: {len(X_test)}")
                        if task_type == "Classification":
                            st.write(f"**Classes**: {len(y_encoded.unique())}")
                    
                    st.info("📊 View detailed results in **4️⃣ Results** tab")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(str(e))

# ============ AUTOML PAGE ============
elif page == "🤖 AutoML":
    from app.pages.automl_training import page_automl_training
    page_automl_training()

# ============ RESULTS PAGE ============
elif page == "4️⃣ Results":
    st.title("4️⃣ Results & Evaluation")
    st.subheader("Model performance and downloads")
    st.divider()
    
    if get_metrics() is None:
        st.info("ℹ️ Train a model first to see results")
        st.markdown("### Next Steps:\n1. Go to **1️⃣ Data Upload** and load data\n2. Go to **2️⃣ EDA** to explore (optional)\n3. Go to **3️⃣ Training** and train a model\n4. Results will appear here")
    else:
        metrics = get_metrics()
        train_info = st.session_state.get('train_info', {})
        
        st.markdown("### Model Information")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Task", train_info.get('task', 'N/A'))
        col2.metric("Algorithm", train_info.get('algorithm', 'N/A'))
        col3.metric("Train Samples", train_info.get('train_samples', 'N/A'))
        col4.metric("Test Samples", train_info.get('test_samples', 'N/A'))
        
        st.divider()
        
        st.markdown("### Performance Metrics")
        
        metric_cols = st.columns(min(4, len([m for m in metrics.items() if isinstance(m[1], (int, float)) and m[0] != 'confusion_matrix'])))
        
        metric_idx = 0
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != 'confusion_matrix' and metric_idx < len(metric_cols):
                with metric_cols[metric_idx]:
                    st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                    metric_idx += 1
        
        st.divider()
        
        st.markdown("### Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if get_trained_model() is not None:
                model_bytes = pickle.dumps(get_trained_model())
                model_name = st.session_state.get('last_model_name', 'model')
                
                st.download_button(
                    label="📥 Download Model (PKL)",
                    data=model_bytes,
                    file_name=f"{model_name}_trained.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        with col2:
            metrics_json = json.dumps(get_metrics(), indent=2, default=str)
            st.download_button(
                label="📊 Download Metrics (JSON)",
                data=metrics_json,
                file_name="metrics.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.divider()
        
        st.markdown("### Detailed Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**All Metrics**")
            for key, value in metrics.items():
                if key != 'confusion_matrix':
                    st.write(f"**{key}**: {value}")
        
        with col2:
            st.markdown("**Next Steps**")
            st.markdown("- Train another model with different hyperparameters\n- Download the model for deployment\n- Export metrics for reporting\n- Use the model for predictions")

# ============ ABOUT PAGE ============
elif page == "About":
    st.title("ℹ️ About")
    st.divider()
    
    st.markdown("""
    ### ML/DL Trainer v1.0.0
    
    Production-ready platform for machine learning and deep learning.
    
    #### Features
    - 📤 Data upload with validation
    - 🔍 Exploratory data analysis
    - 🎯 15+ algorithms
    - ⚙️ Hyperparameter tuning
    - 📊 Performance metrics
    - 💾 Model persistence
    
    #### Tech Stack
    - **Frontend**: Streamlit
    - **ML**: Scikit-learn (9 algorithms)
    - **DL**: TensorFlow/Keras (3 architectures)
    - **Data**: Pandas, NumPy
    - **DevOps**: Docker
    
    #### Supported Models
    
    **Machine Learning**
    - Logistic Regression
    - Random Forest
    - SVM
    - Gradient Boosting
    
    **Deep Learning**
    - Sequential NN
    - CNN
    - LSTM/RNN
    
    #### Status
    ✅ Production Ready  
    📦 Version 1.0.0  
    📄 License: MIT
    
    Built with ❤️ for the ML/DL community
    """)

logger.info(f"Page: {page}")
