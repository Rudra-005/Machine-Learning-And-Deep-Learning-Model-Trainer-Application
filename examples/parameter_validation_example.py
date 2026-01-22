"""
Parameter Validation Integration Examples

Demonstrates validation layer preventing parameter mismatches.
Shows user-friendly warnings instead of errors.
"""

import streamlit as st
from app.utils.parameter_validator import ParameterValidator


def example_epochs_validation():
    """Show epochs validation."""
    st.markdown("## Example 1: Epochs Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Valid: DL Model with Epochs**")
        params = {'epochs': 50, 'batch_size': 32}
        is_valid, warnings = ParameterValidator.validate_all_parameters('sequential', params)
        
        if is_valid:
            st.success("✅ Valid parameters for Sequential model")
        else:
            ParameterValidator.display_warnings(warnings)
    
    with col2:
        st.markdown("**Invalid: ML Model with Epochs**")
        params = {'epochs': 50, 'cv_folds': 5}
        is_valid, warnings = ParameterValidator.validate_all_parameters('random_forest', params)
        
        if not is_valid:
            ParameterValidator.display_warnings(warnings)


def example_max_iter_validation():
    """Show max_iter validation."""
    st.markdown("## Example 2: Max Iterations Validation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Valid: Iterative Model**")
        params = {'max_iter': 1000, 'cv_folds': 5}
        is_valid, warnings = ParameterValidator.validate_all_parameters('logistic_regression', params)
        
        if is_valid:
            st.success("✅ Valid for LogisticRegression")
        else:
            ParameterValidator.display_warnings(warnings)
    
    with col2:
        st.markdown("**Invalid: Tree-Based Model**")
        params = {'max_iter': 1000}
        is_valid, warnings = ParameterValidator.validate_all_parameters('random_forest', params)
        
        if not is_valid:
            ParameterValidator.display_warnings(warnings)
    
    with col3:
        st.markdown("**Invalid: DL Model**")
        params = {'max_iter': 1000}
        is_valid, warnings = ParameterValidator.validate_all_parameters('cnn', params)
        
        if not is_valid:
            ParameterValidator.display_warnings(warnings)


def example_cv_validation():
    """Show CV validation."""
    st.markdown("## Example 3: Cross-Validation Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Valid: ML Model with CV**")
        params = {'cv_folds': 5}
        is_valid, warnings = ParameterValidator.validate_all_parameters('random_forest', params)
        
        if is_valid:
            st.success("✅ Valid for Random Forest")
        else:
            ParameterValidator.display_warnings(warnings)
    
    with col2:
        st.markdown("**Invalid: DL Model with CV**")
        params = {'cv_folds': 5}
        is_valid, warnings = ParameterValidator.validate_all_parameters('sequential', params)
        
        if not is_valid:
            ParameterValidator.display_warnings(warnings)


def example_batch_size_validation():
    """Show batch_size validation."""
    st.markdown("## Example 4: Batch Size Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Valid: DL Model**")
        params = {'batch_size': 32, 'epochs': 50}
        is_valid, warnings = ParameterValidator.validate_all_parameters('cnn', params)
        
        if is_valid:
            st.success("✅ Valid for CNN")
        else:
            ParameterValidator.display_warnings(warnings)
    
    with col2:
        st.markdown("**Invalid: ML Model**")
        params = {'batch_size': 32}
        is_valid, warnings = ParameterValidator.validate_all_parameters('svm', params)
        
        if not is_valid:
            ParameterValidator.display_warnings(warnings)


def example_filter_parameters():
    """Show parameter filtering."""
    st.markdown("## Example 5: Parameter Filtering")
    
    st.write("**Input Parameters:**")
    params = {
        'epochs': 50,
        'max_iter': 1000,
        'cv_folds': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'n_estimators': 100
    }
    st.json(params)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Random Forest (Tree-Based)**")
        filtered = ParameterValidator.get_valid_parameters('random_forest', params)
        st.json(filtered)
    
    with col2:
        st.write("**Logistic Regression (Iterative)**")
        filtered = ParameterValidator.get_valid_parameters('logistic_regression', params)
        st.json(filtered)
    
    with col3:
        st.write("**Sequential (DL)**")
        filtered = ParameterValidator.get_valid_parameters('sequential', params)
        st.json(filtered)


def example_comprehensive_validation():
    """Comprehensive validation example."""
    st.markdown("## Example 6: Comprehensive Validation")
    
    model_choice = st.selectbox(
        "Select Model",
        ['random_forest', 'logistic_regression', 'sequential', 'cnn', 'rnn']
    )
    
    st.divider()
    
    # Show what parameters are valid for this model
    st.markdown(f"**Valid Parameters for {model_choice}:**")
    
    if model_choice == 'random_forest':
        st.write("""
        ✅ n_estimators
        ✅ max_depth
        ✅ cv_folds
        ❌ epochs
        ❌ max_iter
        ❌ batch_size
        """)
    elif model_choice == 'logistic_regression':
        st.write("""
        ✅ max_iter
        ✅ cv_folds
        ✅ learning_rate
        ✅ C
        ❌ epochs
        ❌ batch_size
        """)
    elif model_choice in ['sequential', 'cnn', 'rnn']:
        st.write("""
        ✅ epochs
        ✅ batch_size
        ✅ learning_rate
        ❌ max_iter
        ❌ cv_folds
        """)
    
    st.divider()
    
    # Test validation
    st.markdown("**Test Validation:**")
    
    test_params = {
        'epochs': 50,
        'max_iter': 1000,
        'cv_folds': 5,
        'batch_size': 32
    }
    
    is_valid, warnings = ParameterValidator.validate_all_parameters(model_choice, test_params)
    
    if is_valid:
        st.success("✅ All parameters valid!")
    else:
        st.warning(f"⚠️ Found {len(warnings)} warning(s):")
        ParameterValidator.display_warnings(warnings)


def main():
    """Main example app."""
    st.set_page_config(page_title="Parameter Validation Examples", layout="wide")
    
    st.title("Parameter Validation Examples")
    st.divider()
    
    example_choice = st.selectbox(
        "Select Example",
        [
            "Epochs Validation",
            "Max Iterations Validation",
            "Cross-Validation Validation",
            "Batch Size Validation",
            "Parameter Filtering",
            "Comprehensive Validation"
        ]
    )
    
    st.divider()
    
    if example_choice == "Epochs Validation":
        example_epochs_validation()
    elif example_choice == "Max Iterations Validation":
        example_max_iter_validation()
    elif example_choice == "Cross-Validation Validation":
        example_cv_validation()
    elif example_choice == "Batch Size Validation":
        example_batch_size_validation()
    elif example_choice == "Parameter Filtering":
        example_filter_parameters()
    elif example_choice == "Comprehensive Validation":
        example_comprehensive_validation()


if __name__ == "__main__":
    main()
