"""
Missing Value Handling Pipeline

Applies various imputation strategies to pandas DataFrames using scikit-learn.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Optional


def handle_missing_values(
    df: pd.DataFrame,
    config: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Apply missing value handling strategies to DataFrame columns.
    
    Strategies:
    - 'mean': Fill with mean (numeric only)
    - 'median': Fill with median (numeric only)
    - 'mode': Fill with mode (numeric or categorical)
    - 'most_frequent': Fill with most frequent value
    - 'constant': Fill with constant value (specify 'value' in config)
    - 'drop_rows': Drop rows with missing values
    - 'drop_column': Drop the column entirely
    
    Args:
        df: Input DataFrame
        config: Dictionary mapping column names to strategy configs
                Example: {
                    'age': {'strategy': 'median'},
                    'city': {'strategy': 'constant', 'value': 'Unknown'},
                    'sparse_col': {'strategy': 'drop_column'},
                    'incomplete_rows': {'strategy': 'drop_rows'}
                }
        
    Returns:
        DataFrame with missing values handled
        
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, None, 45],
        ...     'city': ['NYC', None, 'LA', 'NYC'],
        ...     'salary': [50000, None, 60000, 70000]
        ... })
        >>> config = {
        ...     'age': {'strategy': 'median'},
        ...     'city': {'strategy': 'constant', 'value': 'Unknown'},
        ...     'salary': {'strategy': 'mean'}
        ... }
        >>> result = handle_missing_values(df, config)
    """
    df_processed = df.copy()
    
    # Handle drop_rows first (affects all columns)
    if any(cfg.get('strategy') == 'drop_rows' for cfg in config.values()):
        df_processed = df_processed.dropna()
    
    # Handle drop_column
    cols_to_drop = [col for col, cfg in config.items() if cfg.get('strategy') == 'drop_column']
    df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle imputation strategies
    for col, cfg in config.items():
        if col not in df_processed.columns:
            continue
        
        strategy = cfg.get('strategy')
        
        if strategy == 'drop_column':
            continue
        
        elif strategy == 'drop_rows':
            continue
        
        elif strategy == 'constant':
            value = cfg.get('value', 'Unknown')
            df_processed[col].fillna(value, inplace=True)
        
        elif strategy in ['mean', 'median', 'mode', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            df_processed[[col]] = imputer.fit_transform(df_processed[[col]])
    
    return df_processed


def handle_missing_values_advanced(
    df: pd.DataFrame,
    config: Dict[str, Dict[str, Any]],
    fit_on: Optional[pd.DataFrame] = None
) -> tuple:
    """
    Apply missing value handling with fitted imputers for train/test split.
    
    Useful for ML pipelines where imputers are fit on training data
    and applied to test data.
    
    Args:
        df: DataFrame to transform
        config: Strategy configuration
        fit_on: DataFrame to fit imputers on (if None, fit on df)
        
    Returns:
        Tuple of (transformed_df, imputers_dict)
    """
    df_processed = df.copy()
    fit_df = fit_on if fit_on is not None else df
    imputers = {}
    
    # Handle drop_rows
    if any(cfg.get('strategy') == 'drop_rows' for cfg in config.values()):
        df_processed = df_processed.dropna()
    
    # Handle drop_column
    cols_to_drop = [col for col, cfg in config.items() if cfg.get('strategy') == 'drop_column']
    df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle imputation with fitted imputers
    for col, cfg in config.items():
        if col not in df_processed.columns:
            continue
        
        strategy = cfg.get('strategy')
        
        if strategy in ['mean', 'median', 'mode', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            imputer.fit(fit_df[[col]])
            df_processed[[col]] = imputer.transform(df_processed[[col]])
            imputers[col] = imputer
        
        elif strategy == 'constant':
            value = cfg.get('value', 'Unknown')
            df_processed[col].fillna(value, inplace=True)
    
    return df_processed, imputers


def get_missing_value_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary of missing values before and after handling.
    
    Args:
        df: DataFrame
        
    Returns:
        Summary statistics
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isnull().sum().sum()
    
    return {
        'total_missing': int(total_missing),
        'total_cells': total_cells,
        'missing_percentage': round((total_missing / total_cells) * 100, 2),
        'affected_columns': len(df.columns[df.isnull().any()]),
        'columns_with_missing': df.columns[df.isnull().any()].tolist()
    }


def validate_config(config: Dict[str, Dict[str, Any]]) -> tuple:
    """
    Validate missing value handling configuration.
    
    Args:
        config: Strategy configuration
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    valid_strategies = {'mean', 'median', 'mode', 'most_frequent', 'constant', 'drop_rows', 'drop_column'}
    errors = []
    
    for col, cfg in config.items():
        strategy = cfg.get('strategy')
        
        if strategy not in valid_strategies:
            errors.append(f"Column '{col}': Invalid strategy '{strategy}'")
        
        if strategy == 'constant' and 'value' not in cfg:
            errors.append(f"Column '{col}': 'constant' strategy requires 'value' parameter")
    
    return len(errors) == 0, errors
