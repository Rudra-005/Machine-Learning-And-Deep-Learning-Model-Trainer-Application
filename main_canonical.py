"""
ML/DL Trainer - Production Ready Application with Canonical Session State

Uses st.session_state["dataset"] as single source of truth for data availability.
All pages check data status using is_data_loaded() helper.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import logging
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from session_state_contract import is_data_loaded, initialize_session_state, clear_dataset
from models.automl import AutoMLConfig
from models.automl_trainer import train_with_automl
from app.utils.automl_ui import render_automl_mode, render_automl_summary, render_automl_comparison, display_automl_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="ML/DL Trainer", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

initialize_session_state()

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


def main():
    st.sidebar.title("🤖 ML/DL Trainer")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio("Navigation", options=["🏠 Home", "📊 Data Loading", "🧠 AutoML Training", "📈 Results", "📚 Docs", "ℹ️ About"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status**")
    if is_data_loaded():
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.warning("⚠️ No Data")
    
    if st.session_state.model_trained:
        st.sidebar.success("✅ Model Trained")
    else:
        st.sidebar.info("ℹ️ No Model")
    
    if page == "🏠 Home":
        page_home()
    elif page == "📊 Data Loading":
        page_data_loading()
    elif page == "🧠 AutoML Training":
        page_automl_training()
    elif page == "📈 Results":
        page_results()
    elif page == "📚 Docs":
        page_docs()
    elif page == "ℹ️ About":
        page_about()


def page_home():
    st.title("🤖 ML/DL Trainer")
    st.markdown("Production-ready ML/DL platform with AutoML capabilities.")
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
        data_source = st.radio("Source:", options=["Sample", "Upload"], horizontal=True)
        
        if data_source == "Sample":
            dataset_choice = st.selectbox("Dataset:", options=["Iris", "Wine", "Diabetes"])
            if st.button("Load", type="primary", use_container_width=True):
                try:
                    if dataset_choice == "Iris":
                        data = load_iris()
                        X, y = data.data, data.target
                    elif dataset_choice == "Wine":
                        data = load_wine()
                        X, y = data.data, data.target
                    else:
                        data = load_diabetes()
                        X, y = data.data, data.target
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    st.session_state.dataset = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    
                    st.success(f"✅ {dataset_choice} loaded!")
                    logger.info(f"Loaded {dataset_choice}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            uploaded_file = st.file_uploader("CSV:", type=['csv'])
            if uploaded_file is not None:
                try:
                    st.session_state.dataset = pd.read_csv(uploaded_file)
                    st.success("✅ File loaded!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.subheader("Info")
        if is_data_loaded():
            st.metric("Rows", st.session_state.dataset.shape[0])
            st.metric("Cols", st.session_state.dataset.shape[1])
    
    if is_data_loaded():
        st.subheader("📋 Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", st.session_state.dataset.shape[0])
        col2.metric("Cols", st.session_state.dataset.shape[1])
        col3.metric("Memory", f"{st.session_state.dataset.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        st.subheader("Preview")
        st.dataframe(st.session_state.dataset.head(), use_container_width=True)


def page_automl_training():
    st.title("🧠 AutoML Training")
    
    if not is_data_loaded():
        st.warning("⚠️ Load data first")
        return
    
    st.markdown("AutoML detects model type and applies optimal strategy.")
    
    st.subheader("1️⃣ Task")
    task_type = st.radio("Type:", options=['Classification', 'Regression'], horizontal=True)
    
    st.subheader("2️⃣ Model")
    model_name = st.selectbox("Model:", options=list(ML_MODELS[task_type].keys()))
    model = ML_MODELS[task_type][model_name]
    
    st.subheader("3️⃣ Config")
    automl = AutoMLConfig(model)
    render_automl_summary(model, {})
    params = render_automl_mode(model)
    
    with st.expander("📖 Strategy"):
        render_automl_comparison(model)
    
    st.subheader("4️⃣ Train")
    if st.button("🚀 Train", type="primary", use_container_width=True):
        with st.spinner("Training..."):
            try:
                results = train_with_automl(
                    model, st.session_state.X_train, st.session_state.y_train,
                    st.session_state.X_test, st.session_state.y_test, params
                )
                st.session_state.trained_model = results.get('best_estimator', model)
                st.session_state.training_results = results
                st.session_state.model_trained = True
                display_automl_results(model, results)
                st.success("✅ Done!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def page_results():
    st.title("📈 Results")
    if not st.session_state.model_trained:
        st.warning("⚠️ Train a model first")
        return
    
    results = st.session_state.training_results
    col1, col2, col3 = st.columns(3)
    col1.metric("CV Score", f"{results.get('cv_mean', 0):.4f}")
    col2.metric("Std Dev", f"{results.get('cv_std', 0):.4f}")
    col3.metric("Test Score", f"{results.get('test_score', 0):.4f}")


def page_docs():
    st.title("📚 Documentation")
    st.markdown("See AUTOML_DOCUMENTATION.md for details.")


def page_about():
    st.title("ℹ️ About")
    st.markdown("ML/DL Trainer v1.0 - Production Ready")


if __name__ == "__main__":
    main()
