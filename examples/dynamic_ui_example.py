"""
Dynamic UI Integration Example

Shows how to integrate dynamic parameter rendering into training page.
"""

import streamlit as st
from app.utils.dynamic_ui import (
    render_training_parameters, validate_training_params,
    display_parameter_summary
)


def training_page_example():
    """
    Example training page using dynamic UI logic.
    
    This demonstrates the conditional parameter rendering:
    - CV folds shown for ALL ML models
    - Max iterations shown ONLY for iterative ML
    - Epochs shown ONLY for deep learning
    - HP search iterations shown only when tuning enabled
    """
    
    st.title("3️⃣ Model Training")
    st.subheader("Configure and train your model")
    st.divider()
    
    # Assume data is loaded
    if 'data' not in st.session_state:
        st.warning("Please load data first")
        return
    
    data = st.session_state.data
    
    # Configuration section
    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Task & Target**")
        target_col = st.selectbox("Target Column", data.columns, key="target")
        task_type = st.selectbox("Task Type", ["Classification", "Regression"], key="task")
    
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
        else:
            model_name = st.selectbox("Architecture", ["sequential", "cnn", "rnn"], key="dl_arch")
    
    st.divider()
    
    # DYNAMIC PARAMETER RENDERING
    # This is where the magic happens - parameters appear/disappear based on rules
    params, enable_tuning = render_training_parameters(model_name, task_type)
    
    st.divider()
    
    # Validate parameters
    is_valid, errors = validate_training_params(params, model_name, task_type)
    
    if not is_valid:
        for error in errors:
            st.error(f"❌ {error}")
    
    # Display parameter summary
    if params:
        display_parameter_summary(params, model_name)
    
    st.divider()
    
    # Train button
    if st.button("🚀 Train Model", use_container_width=True, disabled=not is_valid, type="primary"):
        st.success("✅ Training started with selected parameters!")
        st.json(params)


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Training Example", layout="wide")
    
    # Mock session state
    if 'data' not in st.session_state:
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        st.session_state.data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        st.session_state.data['target'] = y
    
    training_page_example()
