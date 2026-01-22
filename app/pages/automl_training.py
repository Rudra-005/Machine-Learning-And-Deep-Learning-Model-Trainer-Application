"""
Streamlit AutoML Training Page

Complete AutoML workflow: model selection → auto-detection → strategy selection → training.
"""

import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from models.automl import AutoMLConfig, detect_model_category
from models.automl_trainer import train_with_automl
from app.utils.automl_ui import (
    render_automl_mode,
    render_automl_summary,
    render_automl_comparison,
    get_automl_training_info,
    display_automl_results
)


# Model registry for UI
ML_MODELS = {
    'Classification': {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    },
    'Regression': {
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
}


def page_automl_training():
    """AutoML training page with automatic strategy selection."""
    st.header("🤖 AutoML Training Mode")
    
    # Check if data is loaded
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please load data first in the Data Upload tab")
        st.info("Go to **1️⃣ Data Upload** to load your dataset")
        return
    
    data = st.session_state.data
    
    st.markdown("""
    **AutoML Mode** automatically detects your model type and applies the optimal training strategy:
    - **Tree-based models** → K-Fold Cross-Validation
    - **Iterative models** → K-Fold CV with convergence control
    - **SVM models** → K-Fold CV with kernel tuning
    - **Deep Learning** → Epochs with Early Stopping
    """)
    
    # Step 1: Task Type Selection
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
    
    # Create model instance
    model = ML_MODELS[task_type][model_name]
    
    # Step 3: AutoML Configuration
    st.subheader("3️⃣ AutoML Configuration")
    
    # Display auto-detected strategy
    automl = AutoMLConfig(model)
    render_automl_summary(model, {})
    
    # Render only relevant parameters
    params = render_automl_mode(model)
    
    # Step 4: Strategy Explanation
    with st.expander("📖 Why this strategy?"):
        render_automl_comparison(model)
    
    # Training button
    st.subheader("4️⃣ Train Model")
    
    if st.button("🚀 Start AutoML Training", type="primary", use_container_width=True):
        with st.spinner("Training model with optimal strategy..."):
            try:
                # Auto-preprocess if not already done
                if st.session_state.get('X_train') is None:
                    st.info("Preprocessing data...")
                    from data_preprocessing import preprocess_dataset
                    
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    target_col = numeric_cols[-1] if numeric_cols else data.columns[-1]
                    
                    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_dataset(
                        data,
                        target_col=target_col,
                        test_size=0.2,
                        val_size=0.1
                    )
                    
                    st.session_state.X_train = X_train
                    st.session_state.X_val = X_val
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_val = y_val
                    st.session_state.y_test = y_test
                    st.session_state.preprocessor = preprocessor
                else:
                    X_train = st.session_state.X_train
                    y_train = st.session_state.y_train
                    X_test = st.session_state.X_test
                    y_test = st.session_state.y_test
                
                # Convert to numpy arrays if needed
                if X_train is not None and not isinstance(X_train, np.ndarray):
                    X_train = np.asarray(X_train)
                if y_train is not None and not isinstance(y_train, np.ndarray):
                    y_train = np.asarray(y_train)
                if X_test is not None and not isinstance(X_test, np.ndarray):
                    X_test = np.asarray(X_test)
                if y_test is not None and not isinstance(y_test, np.ndarray):
                    y_test = np.asarray(y_test)
                
                # Display training info
                st.info(get_automl_training_info(model))
                
                # Train with AutoML
                results = train_with_automl(
                    model, X_train, y_train, X_test, y_test, params
                )
                
                # Store results
                st.session_state.trained_model = results.get('best_estimator', model)
                st.session_state.model_trained = True
                
                # Display results
                display_automl_results(model, results)
                
                st.success("✅ Training completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


def page_automl_comparison():
    """Page comparing AutoML strategies across models."""
    st.header("📊 AutoML Strategy Comparison")
    
    st.markdown("""
    This page shows how AutoML automatically selects the best strategy for different models.
    """)
    
    # Create sample models
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    # Display comparison table
    st.subheader("Strategy Selection Matrix")
    
    comparison_data = []
    for model_name, model in models.items():
        automl = AutoMLConfig(model)
        config = automl.config
        visible = automl.visible_params
        
        comparison_data.append({
            'Model': model_name,
            'Category': config['category'],
            'Strategy': config['description'].split('(')[0].strip(),
            'CV': '✓' if visible['cv_folds'] else '✗',
            'Max Iter': '✓' if visible['max_iter'] else '✗',
            'Epochs': '✓' if visible['epochs'] else '✗',
            'HP Tuning': '✓' if visible['hp_tuning'] else '✗'
        })
    
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True)
    
    # Detailed explanations
    st.subheader("Strategy Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **K-Fold Cross-Validation**
        - Used for: Tree-based, SVM, KNN
        - Why: Single-pass convergence, robust overfitting detection
        - Benefit: All data used for training
        """)
    
    with col2:
        st.markdown("""
        **K-Fold CV + Convergence**
        - Used for: Iterative models (LR, SGD)
        - Why: Need convergence control (max_iter)
        - Benefit: Prevents infinite training loops
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Epochs + Early Stopping**
        - Used for: Deep Learning (NN, CNN, LSTM)
        - Why: Multiple passes through data needed
        - Benefit: Automatic overfitting prevention
        """)
    
    with col2:
        st.markdown("""
        **Hyperparameter Tuning**
        - Optional for: All ML models
        - Why: Find optimal hyperparameters
        - Benefit: Better generalization
        """)


def page_automl_guide():
    """AutoML mode user guide."""
    st.header("📚 AutoML Mode Guide")
    
    st.markdown("""
    ## What is AutoML Mode?
    
    AutoML Mode automatically detects your model type and applies the optimal training strategy.
    You don't need to understand the differences between CV, epochs, and convergence—the system
    handles it for you.
    
    ## How It Works
    
    ### 1. Model Detection
    When you select a model, AutoML automatically detects its category:
    - **Tree-Based**: Random Forest, Gradient Boosting, Decision Trees
    - **Iterative**: Logistic Regression, SGD, Perceptron
    - **SVM**: Support Vector Machines
    - **Deep Learning**: Neural Networks, CNN, LSTM
    
    ### 2. Strategy Selection
    Based on the category, AutoML selects the optimal training strategy:
    
    | Category | Strategy | Why |
    |----------|----------|-----|
    | Tree-Based | K-Fold CV | Single-pass convergence, robust evaluation |
    | Iterative | K-Fold CV + max_iter | Need convergence control |
    | SVM | K-Fold CV | Kernel optimization via CV |
    | Deep Learning | Epochs + Early Stop | Multiple passes needed |
    
    ### 3. Parameter Visibility
    Only relevant parameters are shown:
    - **CV Folds**: For all ML models
    - **Max Iter**: Only for iterative models
    - **Epochs**: Only for deep learning
    - **Batch Size**: Only for deep learning
    - **HP Tuning**: Optional for all ML models
    
    ### 4. Intelligent Training
    The system applies the right strategy automatically:
    - ML models: Cross-validation with optional hyperparameter tuning
    - DL models: Epochs with early stopping
    
    ## Example Workflows
    
    ### Workflow 1: Random Forest Classification
    1. Select "Classification" task
    2. Select "Random Forest" model
    3. AutoML detects: Tree-based model
    4. AutoML selects: K-Fold CV (5 folds)
    5. UI shows: CV folds slider, HP tuning checkbox
    6. UI hides: Epochs, max_iter, batch_size
    7. Training: 5-fold cross-validation with optional tuning
    
    ### Workflow 2: Logistic Regression Classification
    1. Select "Classification" task
    2. Select "Logistic Regression" model
    3. AutoML detects: Iterative model
    4. AutoML selects: K-Fold CV + max_iter
    5. UI shows: CV folds slider, max_iter slider, HP tuning checkbox
    6. UI hides: Epochs, batch_size
    7. Training: 5-fold CV with convergence control
    
    ### Workflow 3: Neural Network Classification
    1. Select "Classification" task
    2. Select "Neural Network" model
    3. AutoML detects: Deep learning model
    4. AutoML selects: Epochs + Early Stopping
    5. UI shows: Epochs slider, batch_size slider, early stopping checkbox
    6. UI hides: CV folds, max_iter
    7. Training: Multiple epochs with automatic early stopping
    
    ## Benefits
    
    ✅ **No Confusion**: Only see relevant parameters  
    ✅ **Optimal Strategy**: Best approach for each model type  
    ✅ **Faster Training**: No wasted iterations  
    ✅ **Better Results**: Robust evaluation and tuning  
    ✅ **Production Ready**: Follows ML best practices  
    
    ## When to Use AutoML Mode
    
    - ✅ You're new to ML and want best practices
    - ✅ You want to quickly try different models
    - ✅ You want optimal hyperparameters automatically
    - ✅ You want robust cross-validation
    - ✅ You don't want to worry about training strategy
    
    ## When to Use Manual Mode
    
    - ✅ You have specific training requirements
    - ✅ You want full control over parameters
    - ✅ You're experimenting with custom strategies
    - ✅ You need to match a specific paper/benchmark
    """)


if __name__ == "__main__":
    # Initialize session state
    if 'data_preprocessed' not in st.session_state:
        st.session_state.data_preprocessed = False
    
    # Sidebar navigation
    page = st.sidebar.radio(
        "Select Page",
        options=["AutoML Training", "Strategy Comparison", "User Guide"]
    )
    
    if page == "AutoML Training":
        page_automl_training()
    elif page == "Strategy Comparison":
        page_automl_comparison()
    elif page == "User Guide":
        page_automl_guide()
