"""
Preprocessing Validation Module

Validates user selections and provides user-friendly error messages.
"""

import pandas as pd
from typing import Dict, List, Tuple, Any


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class PreprocessingValidator:
    """Validates preprocessing configurations and selections."""
    
    @staticmethod
    def validate_selections(
        df: pd.DataFrame,
        target_col: str,
        user_selections: Dict[str, str],
        recommendations: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate user preprocessing selections.
        
        Args:
            df: Original DataFrame
            target_col: Target column name
            user_selections: User-selected strategies per column
            recommendations: Original recommendations
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check 1: Target column not dropped
        if target_col in user_selections:
            if user_selections[target_col] == 'drop_column':
                errors.append(
                    "❌ Cannot drop target column\n\n"
                    "The target column is needed to train your model. "
                    "Please select a different strategy for this column."
                )
        
        # Check 2: Not all features dropped
        features_to_drop = [
            col for col, strategy in user_selections.items()
            if strategy == 'drop_column' and col != target_col
        ]
        remaining_features = len(df.columns) - len(features_to_drop) - 1  # -1 for target
        
        if remaining_features <= 0:
            errors.append(
                "❌ Cannot drop all features\n\n"
                "You need at least one feature to train a model. "
                "Please keep at least one column (other than the target)."
            )
        
        # Check 3: Incompatible strategies
        for rec in recommendations:
            col = rec['column']
            if col not in user_selections:
                continue
            
            data_type = rec['data_type']
            selected_strategy = user_selections[col]
            
            incompatible = PreprocessingValidator._check_incompatible_strategy(
                data_type, selected_strategy
            )
            if incompatible:
                errors.append(incompatible)
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _check_incompatible_strategy(data_type: str, strategy: str) -> str:
        """
        Check if strategy is incompatible with data type.
        
        Args:
            data_type: 'numeric' or 'categorical'
            strategy: Selected strategy
            
        Returns:
            Error message if incompatible, empty string otherwise
        """
        incompatibilities = {
            'numeric': {
                'most_frequent': (
                    "⚠️ Strategy mismatch: 'Most Frequent' for numeric data\n\n"
                    "This strategy works better with categorical data. "
                    "Consider using 'Median' or 'Mean' instead."
                )
            },
            'categorical': {
                'median': (
                    "⚠️ Strategy mismatch: 'Median' for categorical data\n\n"
                    "This strategy only works with numbers. "
                    "Use 'Most Frequent' or 'Constant' instead."
                ),
                'mean': (
                    "⚠️ Strategy mismatch: 'Mean' for categorical data\n\n"
                    "This strategy only works with numbers. "
                    "Use 'Most Frequent' or 'Constant' instead."
                )
            }
        }
        
        if data_type in incompatibilities:
            if strategy in incompatibilities[data_type]:
                return incompatibilities[data_type][strategy]
        
        return ""
    
    @staticmethod
    def validate_config(config: Dict[str, Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate preprocessing configuration.
        
        Args:
            config: Preprocessing configuration
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        valid_strategies = {'mean', 'median', 'mode', 'most_frequent', 'constant', 'drop_column'}
        
        for col, cfg in config.items():
            strategy = cfg.get('strategy')
            
            if strategy not in valid_strategies:
                errors.append(
                    f"❌ Invalid strategy for '{col}'\n\n"
                    f"'{strategy}' is not a valid strategy. "
                    f"Valid options: {', '.join(valid_strategies)}"
                )
            
            if strategy == 'constant' and 'value' not in cfg:
                errors.append(
                    f"❌ Missing value for '{col}'\n\n"
                    f"When using 'Constant' strategy, you must specify a fill value."
                )
        
        return len(errors) == 0, errors


def display_validation_errors(errors: List[str]) -> None:
    """
    Display validation errors in Streamlit.
    
    Args:
        errors: List of error messages
    """
    import streamlit as st
    
    if not errors:
        return
    
    st.error("⚠️ Configuration Issues Found")
    
    for i, error in enumerate(errors, 1):
        st.markdown(f"**Issue {i}:**\n{error}")
        st.divider()


def display_validation_warnings(warnings: List[str]) -> None:
    """
    Display validation warnings in Streamlit.
    
    Args:
        warnings: List of warning messages
    """
    import streamlit as st
    
    if not warnings:
        return
    
    st.warning("⚠️ Recommendations")
    
    for warning in warnings:
        st.markdown(f"• {warning}")


def get_validation_summary(
    df: pd.DataFrame,
    target_col: str,
    user_selections: Dict[str, str]
) -> Dict[str, Any]:
    """
    Get summary of validation results.
    
    Args:
        df: Original DataFrame
        target_col: Target column name
        user_selections: User-selected strategies
        
    Returns:
        Summary dictionary
    """
    features_to_drop = [
        col for col, strategy in user_selections.items()
        if strategy == 'drop_column' and col != target_col
    ]
    features_to_impute = [
        col for col, strategy in user_selections.items()
        if strategy != 'drop_column'
    ]
    
    return {
        'total_columns': len(df.columns),
        'target_column': target_col,
        'features_to_drop': len(features_to_drop),
        'features_to_impute': len(features_to_impute),
        'remaining_features': len(df.columns) - len(features_to_drop) - 1,
        'dropped_columns': features_to_drop,
        'imputed_columns': features_to_impute
    }


def validate_and_display(
    df: pd.DataFrame,
    target_col: str,
    user_selections: Dict[str, str],
    recommendations: List[Dict[str, Any]]
) -> bool:
    """
    Validate selections and display results in Streamlit.
    
    Args:
        df: Original DataFrame
        target_col: Target column name
        user_selections: User-selected strategies
        recommendations: Original recommendations
        
    Returns:
        True if valid, False otherwise
    """
    import streamlit as st
    
    is_valid, errors = PreprocessingValidator.validate_selections(
        df, target_col, user_selections, recommendations
    )
    
    if not is_valid:
        display_validation_errors(errors)
        return False
    
    # Show summary
    summary = get_validation_summary(df, target_col, user_selections)
    
    st.success("✅ Configuration Valid")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Columns", summary['total_columns'])
    col2.metric("Columns to Drop", summary['features_to_drop'])
    col3.metric("Columns to Impute", summary['features_to_impute'])
    col4.metric("Remaining Features", summary['remaining_features'])
    
    return True
