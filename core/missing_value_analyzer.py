"""
Missing Value Analysis & Preprocessing Recommendation Module

Provides rule-based recommendations for handling missing values in pandas DataFrames.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def analyze_missing_values(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Analyze missing values in a pandas DataFrame.
    
    For each column, returns:
    - Column name
    - Data type (numeric or categorical)
    - Percentage of missing values
    - Total missing count
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        List of dictionaries with missing value analysis for each column
    """
    if df.empty:
        return []
    
    results = []
    total_rows = len(df)
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        data_type = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
        
        results.append({
            'column': col,
            'data_type': data_type,
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_pct, 2)
        })
    
    return results


def recommend_missing_value_strategy(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Recommend preprocessing strategy for handling missing values.
    
    Rules:
    - If missing > 40% → recommend "drop_column"
    - Numeric → recommend "median"
    - Categorical → recommend "most_frequent"
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        List of dictionaries with recommendations and explanations
        
    Example:
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, None, 4],
        ...     'B': ['x', None, 'y', 'z'],
        ...     'C': [None] * 100 + [1] * 50  # 66% missing
        ... })
        >>> recommendations = recommend_missing_value_strategy(df)
        >>> recommendations[0]
        {
            'column': 'A',
            'data_type': 'numeric',
            'missing_percentage': 25.0,
            'strategy': 'median',
            'explanation': 'Impute numeric column with median value'
        }
    """
    analysis = analyze_missing_values(df)
    recommendations = []
    
    for item in analysis:
        col = item['column']
        data_type = item['data_type']
        missing_pct = item['missing_percentage']
        
        # Rule 1: Drop if > 40% missing
        if missing_pct > 40:
            strategy = 'drop_column'
            explanation = f'Column has {missing_pct}% missing values - too sparse to impute'
        
        # Rule 2: Numeric → median
        elif data_type == 'numeric':
            strategy = 'median'
            explanation = 'Impute numeric column with median (robust to outliers)'
        
        # Rule 3: Categorical → most_frequent
        else:
            strategy = 'most_frequent'
            explanation = 'Impute categorical column with most frequent value'
        
        recommendations.append({
            'column': col,
            'data_type': data_type,
            'missing_percentage': missing_pct,
            'strategy': strategy,
            'explanation': explanation
        })
    
    return recommendations


def get_preprocessing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary of preprocessing recommendations.
    
    Args:
        df: Input pandas DataFrame
        
    Returns:
        Dictionary with summary statistics and action counts
    """
    recommendations = recommend_missing_value_strategy(df)
    
    strategies = {}
    for rec in recommendations:
        strategy = rec['strategy']
        strategies[strategy] = strategies.get(strategy, 0) + 1
    
    return {
        'total_columns': len(recommendations),
        'columns_to_drop': strategies.get('drop_column', 0),
        'columns_to_impute': len(recommendations) - strategies.get('drop_column', 0),
        'strategy_breakdown': strategies
    }


def apply_recommendations(df: pd.DataFrame, recommendations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply preprocessing recommendations to DataFrame.
    
    Args:
        df: Input pandas DataFrame
        recommendations: List of recommendations from recommend_missing_value_strategy()
        
    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    
    for rec in recommendations:
        col = rec['column']
        strategy = rec['strategy']
        
        if strategy == 'drop_column':
            df_processed = df_processed.drop(columns=[col])
        
        elif strategy == 'median':
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        elif strategy == 'most_frequent':
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    
    return df_processed


def compute_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value statistics for each column."""
    stats = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_percentage': round((missing_count / len(df)) * 100, 2)
        })
    return pd.DataFrame(stats)


def get_columns_above_threshold(df: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    """Get columns with missing values above threshold."""
    cols = []
    for col in df.columns:
        missing_pct = (df[col].isnull().sum() / len(df))
        if missing_pct > threshold:
            cols.append(col)
    return {'count': len(cols), 'columns': cols}


def get_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Get missing value patterns."""
    rows_with_missing = (df.isnull().any(axis=1)).sum()
    return {
        'rows_with_missing': rows_with_missing,
        'rows_with_missing_pct': round((rows_with_missing / len(df)) * 100, 2)
    }


def create_missing_bar_chart(df: pd.DataFrame, backend: str = 'plotly'):
    """Create bar chart of missing values."""
    try:
        import plotly.graph_objects as go
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        fig = go.Figure(data=[go.Bar(x=missing.index, y=missing.values)])
        fig.update_layout(title='Missing Values', xaxis_title='Column', yaxis_title='Count')
        return fig
    except:
        return None


def create_missing_heatmap(df: pd.DataFrame, backend: str = 'plotly'):
    """Create heatmap of missing values."""
    try:
        import plotly.graph_objects as go
        missing_matrix = df.isnull().astype(int)
        fig = go.Figure(data=go.Heatmap(z=missing_matrix.T))
        fig.update_layout(title='Missing Values Heatmap')
        return fig
    except:
        return None
