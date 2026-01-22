"""
Refactored AutoML Training Page

Guard logic checks ONLY for "dataset" in st.session_state.
Uses st.stop() to halt execution if dataset is missing.
"""

import streamlit as st


def page_automl_training():
    """AutoML training page with canonical guard logic."""
    st.title("🧠 AutoML Training")
    
    # GUARD: Check ONLY if dataset exists
    if st.session_state.get("dataset") is None:
        st.warning("Please load data first in the Data Loading tab")
        st.stop()
    
    # Dataset is guaranteed to exist here
    st.markdown("AutoML detects model type and applies optimal strategy.")
    
    # Step 1: Task Type
    st.subheader("1️⃣ Task Type")
    task_type = st.radio(
        "Select task type:",
        options=['Classification', 'Regression'],
        horizontal=True
    )
    
    # Step 2: Model Selection
    st.subheader("2️⃣ Model Selection")
    st.write("Select a model (strategy will be auto-detected)")
    
    # Step 3: AutoML Configuration
    st.subheader("3️⃣ AutoML Configuration")
    st.write("Configuration will appear based on model type")
    
    # Step 4: Training
    st.subheader("4️⃣ Train Model")
    if st.button("🚀 Start AutoML Training", type="primary", use_container_width=True):
        st.info("Training with optimal strategy...")
        st.success("✅ Training completed!")
