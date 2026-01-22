"""
ML/DL Trainer - Production Ready Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import json
import io
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import logging
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from models.automl import AutoMLConfig
from models.automl_trainer import train_with_automl
from app.utils.automl_ui import render_automl_mode, render_automl_summary, render_automl_comparison, display_automl_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ML/DL Trainer", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

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
    return st.session_state.X_train is not None

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

def encode_categorical(df):
    """Encode categorical columns to numeric"""
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def main():
    st.sidebar.title("🤖 ML/DL Trainer")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", options=["🏠 Home", "📊 Data Loading", "🧠 AutoML Training", "📈 Results", "📚 Docs", "ℹ️ About"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status**")
    if is_data_ready():
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.warning("⚠️ No Data")
    
    if st.session_state.model_trained:
        st.sidebar.success("✅ Model Trained")
    
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Data Loading":
        page_data_loading()
    elif page == "🧠 AutoML Training":
        page_automl_training()
    elif page == "📈 Results":
        page_results()
    elif page == "📚 Docs":
        page_documentation()
    elif page == "ℹ️ About":
        page_about()

def page_home():
    st.title("🤖 ML/DL Trainer")
    st.markdown("Production-ready ML/DL platform with AutoML mode")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", "15+")
    col2.metric("ML Algos", "9")
    col3.metric("DL Archs", "3")
    col4.metric("Metrics", "10+")

def page_data_loading():
    st.title("📊 Data Loading")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Load Data")
        data_source = st.radio("Choose source:", options=["Sample", "Upload CSV"], horizontal=True)
        
        if data_source == "Sample":
            dataset_choice = st.selectbox("Dataset:", options=["Iris", "Wine", "Diabetes"])
            
            if st.button("Load", type="primary", use_container_width=True):
                try:
                    if dataset_choice == "Iris":
                        data = load_iris()
                        X, y = data.data, data.target
                        st.session_state.task_type = "Classification"
                    elif dataset_choice == "Wine":
                        data = load_wine()
                        X, y = data.data, data.target
                        st.session_state.task_type = "Classification"
                    else:
                        data = load_diabetes()
                        X, y = data.data, data.target
                        st.session_state.task_type = "Regression"
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success(f"✅ {dataset_choice} loaded!")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        else:
            uploaded_file = st.file_uploader("CSV file:", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("✅ File uploaded!")
                    
                    st.dataframe(df.head(), use_container_width=True)
                    
                    target_col = st.selectbox("Target column:", df.columns)
                    task_type_auto = st.selectbox("Task type:", ["Classification", "Regression"])
                    
                    if st.button("Process", type="primary", use_container_width=True):
                        X_df = df.drop(columns=[target_col])
                        X_encoded = encode_categorical(X_df)
                        X = X_encoded.values.astype(float)
                        y = df[target_col].values
                        
                        try:
                            y = y.astype(float)
                        except:
                            y_encoded = encode_categorical(pd.DataFrame({target_col: y}))
                            y = y_encoded.values.flatten()
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.task_type = task_type_auto
                        
                        st.success("✅ Data processed!")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Info")
        if is_data_ready():
            st.metric("Train", st.session_state.X_train.shape[0])
            st.metric("Test", st.session_state.X_test.shape[0])
            st.metric("Features", st.session_state.X_train.shape[1])
    
    if is_data_ready():
        st.subheader("Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Train Set", f"{st.session_state.X_train.shape[0]} samples")
        col2.metric("Test Set", f"{st.session_state.X_test.shape[0]} samples")
        col3.metric("Features", st.session_state.X_train.shape[1])

def page_automl_training():
    st.title("🧠 AutoML Training")
    
    if not is_data_ready():
        st.warning("⚠️ Load data first")
        return
    
    st.markdown("AutoML detects model type and applies optimal strategy")
    
    st.subheader("1️⃣ Task Type")
    task_type = st.radio("Select:", options=['Classification', 'Regression'], horizontal=True)
    st.session_state.task_type = task_type
    
    st.subheader("2️⃣ Model")
    model_name = st.selectbox("Choose:", options=list(ML_MODELS[task_type].keys()))
    model = ML_MODELS[task_type][model_name]
    
    st.subheader("3️⃣ Config")
    automl = AutoMLConfig(model)
    render_automl_summary(model, {})
    params = render_automl_mode(model)
    
    st.subheader("4️⃣ Train")
    
    if st.button("🚀 Train", type="primary", use_container_width=True):
        with st.spinner("Training..."):
            try:
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                
                config = automl.config
                st.info(f"Training {config['model_name']}")
                
                results = train_with_automl(model, X_train, y_train, X_test, y_test, params)
                
                st.session_state.trained_model = results.get('best_estimator', model)
                st.session_state.training_results = results
                st.session_state.model_trained = True
                
                display_automl_results(model, results)
                st.success("✅ Done!")
                
            except Exception as e:
                st.error(f"❌ Failed: {str(e)}")

def page_results():
    st.title("📈 Results")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Train a model first")
        return
    
    results = st.session_state.training_results
    
    if results['strategy'] == 'k_fold_cv':
        col1, col2, col3 = st.columns(3)
        col1.metric("CV Mean", f"{results['cv_mean']:.4f}")
        col2.metric("CV Std", f"{results['cv_std']:.4f}")
        col3.metric("Test", f"{results['test_score']:.4f}")
    
    st.subheader("💾 Download Model & Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_pkl = pickle.dumps(st.session_state.trained_model)
        st.download_button(
            label="📥 Download PKL",
            data=model_pkl,
            file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
            mime="application/octet-stream"
        )
    
    with col2:
        buffer = io.BytesIO()
        joblib.dump(st.session_state.trained_model, buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Download JOBLIB",
            data=buffer.getvalue(),
            file_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
            mime="application/octet-stream"
        )
    
    with col3:
        results_json = json.dumps(results, default=str, indent=2)
        st.download_button(
            label="📊 Download Metrics",
            data=results_json,
            file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def page_documentation():
    st.title("📚 Documentation")
    st.markdown("AutoML automatically selects optimal training strategy based on model type")

def page_about():
    st.title("ℹ️ About")
    st.markdown("ML/DL Trainer v1.0 - Production Ready")

if __name__ == "__main__":
    main()
