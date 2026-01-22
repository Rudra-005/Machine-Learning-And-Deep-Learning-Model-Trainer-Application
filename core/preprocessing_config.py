"""
Human-in-the-Loop Preprocessing Configuration Module

Validates user-selected strategies and generates final preprocessing configuration.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple


VALID_STRATEGIES = {'drop_column', 'median', 'most_frequent', 'forward_fill', 'backward_fill'}

STRATEGY_COMPATIBILITY = {
    'numeric': {'drop_column', 'median', 'forward_fill', 'backward_fill'},
    'categorical': {'drop_column', 'most_frequent', 'forward_fill', 'backward_fill'}
}


def validate_user_selection(
    column: str,
    data_type: str,
    user_strategy: str,
    recommended_strategy: str
) -> Tuple[bool, str, str]:
    """
    Validate user-selected strategy for a column.
    
    Args:
        column: Column name
        data_type: 'numeric' or 'categorical'
        user_strategy: User-selected strategy
        recommended_strategy: Recommended strategy from analysis
        
    Returns:
        Tuple of (is_valid, message, final_strategy)
        - is_valid: Whether selection is valid
        - message: Validation message
        - final_strategy: Final strategy to use (user's or fallback to recommended)
    """
    # Check if strategy is valid
    if user_strategy not in VALID_STRATEGIES:
        return False, f"Invalid strategy '{user_strategy}'", recommended_strategy
    
    # Check if strategy is compatible with data type
    if user_strategy not in STRATEGY_COMPATIBILITY[data_type]:
        msg = f"Strategy '{user_strategy}' incompatible with {data_type} data"
        return False, msg, recommended_strategy
    
    return True, "Valid", user_strategy


def build_preprocessing_config(
    analysis_report: List[Dict[str, Any]],
    user_selections: Dict[str, str]
) -> Dict[str, Any]:
    """
    Build final preprocessing configuration from analysis and user selections.
    
    Validates each user selection and falls back to recommended strategy if invalid.
    
    Args:
        analysis_report: Output from recommend_missing_value_strategy()
        user_selections: Dict mapping column names to user-selected strategies
                        Example: {'age': 'median', 'city': 'most_frequent'}
        
    Returns:
        Dictionary with final configuration and validation results
        
    Example:
        >>> analysis = [
        ...     {
        ...         'column': 'age',
        ...         'data_type': 'numeric',
        ...         'missing_percentage': 25.0,
        ...         'strategy': 'median',
        ...         'explanation': '...'
        ...     }
        ... ]
        >>> user_selections = {'age': 'median'}
        >>> config = build_preprocessing_config(analysis, user_selections)
        >>> config['final_config'][0]['strategy']
        'median'
    """
    final_config = []
    validation_results = []
    
    for item in analysis_report:
        col = item['column']
        data_type = item['data_type']
        recommended = item['strategy']
        
        # Get user selection or use recommended
        user_strategy = user_selections.get(col, recommended)
        
        # Validate selection
        is_valid, message, final_strategy = validate_user_selection(
            col, data_type, user_strategy, recommended
        )
        
        # Build final config entry
        final_config.append({
            'column': col,
            'data_type': data_type,
            'missing_percentage': item['missing_percentage'],
            'strategy': final_strategy,
            'explanation': item['explanation'],
            'user_selected': user_strategy,
            'recommended': recommended,
            'is_valid': is_valid
        })
        
        # Track validation
        validation_results.append({
            'column': col,
            'user_selection': user_strategy,
            'is_valid': is_valid,
            'message': message,
            'final_strategy': final_strategy
        })
    
    return {
        'final_config': final_config,
        'validation_results': validation_results,
        'all_valid': all(v['is_valid'] for v in validation_results),
        'invalid_count': sum(1 for v in validation_results if not v['is_valid']),
        'fallback_count': sum(1 for v in validation_results if v['user_selection'] != v['final_strategy'])
    }


def get_preprocessing_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get summary of preprocessing configuration.
    
    Args:
        config: Output from build_preprocessing_config()
        
    Returns:
        Summary statistics
    """
    final_config = config['final_config']
    
    strategies = {}
    for item in final_config:
        strategy = item['strategy']
        strategies[strategy] = strategies.get(strategy, 0) + 1
    
    return {
        'total_columns': len(final_config),
        'all_valid': config['all_valid'],
        'invalid_selections': config['invalid_count'],
        'fallback_applied': config['fallback_count'],
        'strategy_breakdown': strategies,
        'columns_to_drop': sum(1 for item in final_config if item['strategy'] == 'drop_column'),
        'columns_to_impute': sum(1 for item in final_config if item['strategy'] != 'drop_column')
    }


def apply_preprocessing_config(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply final preprocessing configuration to DataFrame.
    
    Args:
        df: Input DataFrame
        config: Output from build_preprocessing_config()
        
    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    
    for item in config['final_config']:
        col = item['column']
        strategy = item['strategy']
        
        if strategy == 'drop_column':
            df_processed = df_processed.drop(columns=[col])
        
        elif strategy == 'median':
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        elif strategy == 'most_frequent':
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        elif strategy == 'forward_fill':
            df_processed[col].fillna(method='ffill', inplace=True)
        
        elif strategy == 'backward_fill':
            df_processed[col].fillna(method='bfill', inplace=True)
    
    return df_processed
