"""
Refactored Data Loading Page

Stores dataframe in st.session_state["dataset"] as single source of truth.
Metadata stored in dataset_shape and dataset_columns.
No boolean flags - only actual data.
Dataset persists across page navigation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging

logger = logging.getLogger(__name__)


def is_data_loaded():
    """Canonical check: data is loaded if dataset exists."""
    return st.session_state.get("dataset") is not None


def set_dataset(df):
    """Set dataset and metadata. Single point for dataset storage."""
    st.session_state.dataset = df
    st.session_state.dataset_shape = df.shape
    st.session_state.dataset_columns = list(df.columns)


def initialize_session_state():
    """Initialize session state with dataset as source of truth."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_shape" not in st.session_state:
        st.session_state.dataset_shape = None
    if "dataset_columns" not in st.session_state:
        st.session_state.dataset_columns = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "X_test" not in st.session_state:
        st.session_state.X_test = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False


@st.cache_data
def load_sample_dataset():
    """Load sample dataset."""
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


def page_data_loading():
    """Refactored Data Loading page."""
    st.title("📊 Data Loading")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Choose CSV", type=['csv'])
    
    with col2:
        st.subheader("Or Load Sample")
        if st.button("Load Sample", key="sample_data", type="primary"):
            set_dataset(load_sample_dataset())
            st.success("✓ Sample loaded")
    
    # Handle CSV upload
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            set_dataset(df)
            st.success(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
            logger.info(f"Uploaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Upload error: {str(e)}")
            return
    
    # Display if data loaded
    if is_data_loaded():
        dataset = st.session_state.dataset
        
        st.subheader("📋 Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", st.session_state.dataset_shape[0])
        col2.metric("Columns", st.session_state.dataset_shape[1])
        col3.metric("Memory", f"{dataset.memory_usage(deep=True).sum() / 1024:.1f} KB")
        col4.metric("Missing", dataset.isnull().sum().sum())
        
        st.subheader("Preview")
        st.dataframe(dataset.head(10), use_container_width=True)
        
        st.subheader("Statistics")
        st.dataframe(dataset.describe(), use_container_width=True)
        
        st.subheader("Columns")
        col1, col2 = st.columns([2, 2])
        with col1:
            st.dataframe(pd.DataFrame({
                'Column': st.session_state.dataset_columns,
                'Type': [str(dataset[col].dtype) for col in st.session_state.dataset_columns]
            }), use_container_width=True)
        
        with col2:
            missing = dataset.isnull().sum()
            if missing.sum() > 0:
                fig = px.bar(
                    x=missing[missing > 0].index,
                    y=missing[missing > 0].values,
                    labels={'x': 'Column', 'y': 'Missing'},
                    title='Missing Values'
                )
                st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    initialize_session_state()
    page_data_loading()
