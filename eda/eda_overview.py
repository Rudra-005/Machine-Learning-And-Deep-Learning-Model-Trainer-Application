"""
EDA Overview Module

Provides comprehensive overview of dataset structure, data types, memory usage,
and detects potential data quality issues.

Author: Data Science Team
Date: 2026-01-19
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Basic Dataset Information
# ============================================================================

def get_dataset_shape(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get dataset dimensions.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict[str, int]: Dictionary with 'rows' and 'columns' keys
        
    Example:
        >>> shape = get_dataset_shape(df)
        >>> print(f"Dataset: {shape['rows']} rows, {shape['columns']} columns")
        Dataset: 10000 rows, 25 columns
    """
    return {
        'rows': df.shape[0],
        'columns': df.shape[1]
    }


def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate memory usage of dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict with 'total_mb', 'by_column', and 'largest_columns' keys
        
    Example:
        >>> mem = get_memory_usage(df)
        >>> print(f"Total: {mem['total_mb']:.2f} MB")
        Total: 125.45 MB
    """
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    
    # Get top 5 memory consumers
    column_memory = df.memory_usage(deep=True).sort_values(ascending=False) / 1024**2
    
    return {
        'total_mb': round(memory_usage, 2),
        'by_column': {col: round(mem, 3) for col, mem in column_memory.items()},
        'largest_columns': list(column_memory.head(5).index),
        'largest_memory_mb': round(column_memory.head(5).values, 3)
    }


def get_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data types in dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict with type distribution and column mappings
        
    Example:
        >>> dtypes = get_data_types(df)
        >>> print(f"Numeric: {dtypes['count']['numeric']}")
        Numeric: 15
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    return {
        'count': {
            'numeric': len(numeric_cols),
            'categorical': len(categorical_cols),
            'datetime': len(datetime_cols),
            'boolean': len(bool_cols)
        },
        'columns': {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols,
            'boolean': bool_cols
        },
        'distribution': df.dtypes.value_counts().to_dict()
    }


def get_duplicate_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect duplicate rows in dataset.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict with duplicate statistics
        
    Example:
        >>> dupes = get_duplicate_info(df)
        >>> print(f"Duplicates: {dupes['count']} ({dupes['percentage']:.2f}%)")
        Duplicates: 42 (0.42%)
    """
    total_duplicates = df.duplicated().sum()
    duplicate_pct = (total_duplicates / len(df)) * 100 if len(df) > 0 else 0
    
    # Check duplicates with/without index
    duplicates_with_index = df.duplicated(keep=False).sum()
    
    # Find columns that make rows duplicate
    duplicate_cols = df.columns[df.duplicated(subset=df.columns, keep=False)].tolist()
    
    return {
        'count': int(total_duplicates),
        'percentage': round(duplicate_pct, 2),
        'duplicate_rows_with_index': int(duplicates_with_index),
        'severity': _categorize_duplication(duplicate_pct),
        'recommendation': _get_duplicate_recommendation(duplicate_pct)
    }


# ============================================================================
# Data Quality Issue Detection
# ============================================================================

def detect_high_cardinality(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Detect high cardinality columns (many unique values relative to rows).
    
    High cardinality can:
    - Increase model complexity
    - Cause overfitting
    - Require special encoding strategies
    
    Args:
        df (pd.DataFrame): Input dataset
        threshold (float): Uniqueness ratio threshold (default 0.5 = 50%)
        
    Returns:
        Dict with high cardinality columns and severity
        
    Example:
        >>> high_card = detect_high_cardinality(df, threshold=0.5)
        >>> for col in high_card['columns']:
        ...     print(f"{col}: {col['unique_count']} unique values")
    """
    high_cardinality_cols = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        
        if unique_ratio >= threshold:
            high_cardinality_cols.append({
                'column': col,
                'unique_count': int(unique_count),
                'unique_ratio': round(unique_ratio, 3),
                'percentage_unique': round(unique_ratio * 100, 2),
                'severity': _categorize_cardinality(unique_ratio)
            })
    
    # Sort by uniqueness
    high_cardinality_cols = sorted(
        high_cardinality_cols,
        key=lambda x: x['unique_ratio'],
        reverse=True
    )
    
    return {
        'count': len(high_cardinality_cols),
        'columns': high_cardinality_cols,
        'threshold': threshold,
        'recommendation': _get_cardinality_recommendation(high_cardinality_cols)
    }


def detect_constant_features(df: pd.DataFrame, variance_threshold: float = 0.01) -> Dict[str, Any]:
    """
    Detect constant or near-constant features.
    
    Constant features:
    - Have zero variance (only one unique value)
    - Provide no predictive information
    - Should be removed before modeling
    
    Near-constant features:
    - Have very low variance
    - Dominated by one value (>99%)
    - May still be noise
    
    Args:
        df (pd.DataFrame): Input dataset
        variance_threshold (float): Variance ratio threshold (default 0.01 = 1%)
        
    Returns:
        Dict with constant and near-constant features
        
    Example:
        >>> constants = detect_constant_features(df)
        >>> print(f"Constant: {len(constants['constant'])}")
        >>> print(f"Near-constant: {len(constants['near_constant'])}")
    """
    constant_cols = []
    near_constant_cols = []
    
    for col in df.columns:
        unique_count = df[col].nunique()
        
        # Constant feature (only 1 unique value)
        if unique_count == 1:
            constant_cols.append({
                'column': col,
                'unique_count': 1,
                'value': df[col].iloc[0],
                'impact': 'CRITICAL - Remove immediately'
            })
        
        # Near-constant feature (variance of top value)
        elif unique_count > 1:
            value_counts = df[col].value_counts()
            top_value_ratio = value_counts.iloc[0] / len(df)
            
            if top_value_ratio >= (1 - variance_threshold):
                near_constant_cols.append({
                    'column': col,
                    'unique_count': int(unique_count),
                    'top_value': value_counts.index[0],
                    'top_value_ratio': round(top_value_ratio, 3),
                    'top_value_percentage': round(top_value_ratio * 100, 2),
                    'impact': 'HIGH - Consider removal'
                })
    
    return {
        'constant': constant_cols,
        'near_constant': near_constant_cols,
        'total_problematic': len(constant_cols) + len(near_constant_cols),
        'variance_threshold': variance_threshold,
        'recommendation': _get_constant_recommendation(constant_cols, near_constant_cols)
    }


def detect_id_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect ID-like columns that should not be used as features.
    
    ID columns are identified by:
    - Column name contains 'id', 'index', 'key'
    - All or nearly all values are unique
    - Numeric or string with no pattern
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict with suspected ID columns and confidence scores
        
    Example:
        >>> ids = detect_id_columns(df)
        >>> for col in ids['suspected_ids']:
        ...     print(f"{col['column']}: {col['confidence']:.0%}")
    """
    suspected_ids = []
    id_keywords = ['id', 'index', 'key', 'pk', 'primarykey', 'rowid']
    
    for col in df.columns:
        confidence_score = 0
        reasons = []
        
        # Check column name
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords):
            confidence_score += 0.4
            reasons.append("Column name suggests ID")
        
        # Check uniqueness
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95:  # 95%+ unique
            confidence_score += 0.35
            reasons.append("Nearly all values unique")
        
        # Check for sequential pattern (1, 2, 3, ...) or UUID-like
        if _is_sequential_or_uuid(df[col]):
            confidence_score += 0.25
            reasons.append("Sequential or UUID-like pattern")
        
        # Add to results if confidence > 30%
        if confidence_score > 0.3:
            suspected_ids.append({
                'column': col,
                'confidence': round(confidence_score, 2),
                'unique_count': int(df[col].nunique()),
                'unique_ratio': round(unique_ratio, 3),
                'reasons': reasons
            })
    
    # Sort by confidence
    suspected_ids = sorted(suspected_ids, key=lambda x: x['confidence'], reverse=True)
    
    return {
        'count': len(suspected_ids),
        'suspected_ids': suspected_ids,
        'recommendation': 'Remove these columns before training' if suspected_ids else 'No ID columns detected'
    }


def detect_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive missing value analysis.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        Dict with missing value statistics
        
    Example:
        >>> missing = detect_missing_values(df)
        >>> print(f"Total missing: {missing['total_count']}")
    """
    missing_stats = []
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        
        if missing_count > 0:
            missing_stats.append({
                'column': col,
                'count': int(missing_count),
                'percentage': round(missing_pct, 2),
                'severity': _categorize_missing(missing_pct),
                'recommendation': _get_missing_recommendation(missing_pct)
            })
    
    # Sort by percentage descending
    missing_stats = sorted(missing_stats, key=lambda x: x['percentage'], reverse=True)
    
    total_missing = sum(df.isna().sum())
    total_cells = df.shape[0] * df.shape[1]
    overall_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    return {
        'total_count': int(total_missing),
        'total_percentage': round(overall_pct, 2),
        'columns_with_missing': len(missing_stats),
        'columns': missing_stats,
        'overall_severity': _categorize_missing(overall_pct)
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _is_sequential_or_uuid(series: pd.Series) -> bool:
    """
    Check if series contains sequential numbers or UUID-like values.
    
    Args:
        series (pd.Series): Column to check
        
    Returns:
        bool: True if sequential or UUID-like pattern detected
    """
    try:
        # Check for sequential integers
        if series.dtype in ['int64', 'int32']:
            sorted_vals = sorted(series.dropna().unique())
            if len(sorted_vals) > 10:
                # Check if first 10 are sequential
                is_seq = all(
                    sorted_vals[i] == sorted_vals[i-1] + 1
                    for i in range(1, min(10, len(sorted_vals)))
                )
                return is_seq
        
        # Check for UUID-like pattern (long strings with dashes)
        if series.dtype == 'object':
            sample = series.dropna().head(5).astype(str)
            if all('-' in str(val) and len(str(val)) > 20 for val in sample):
                return True
    
    except Exception:
        pass
    
    return False


def _categorize_missing(percentage: float) -> str:
    """Categorize missing value severity."""
    if percentage == 0:
        return 'None'
    elif percentage < 5:
        return 'Low'
    elif percentage < 20:
        return 'Medium'
    elif percentage < 50:
        return 'High'
    else:
        return 'Critical'


def _categorize_duplication(percentage: float) -> str:
    """Categorize duplication severity."""
    if percentage < 0.1:
        return 'None'
    elif percentage < 1:
        return 'Low'
    elif percentage < 5:
        return 'Medium'
    elif percentage < 10:
        return 'High'
    else:
        return 'Critical'


def _categorize_cardinality(unique_ratio: float) -> str:
    """Categorize cardinality severity."""
    if unique_ratio < 0.5:
        return 'Low'
    elif unique_ratio < 0.8:
        return 'Medium'
    else:
        return 'High'


def _get_missing_recommendation(percentage: float) -> str:
    """Get recommendation based on missing percentage."""
    if percentage < 5:
        return 'Use mean/median imputation or drop'
    elif percentage < 20:
        return 'Consider KNN imputation or advanced methods'
    elif percentage < 50:
        return 'Consider feature deletion or advanced imputation'
    else:
        return 'CRITICAL: Consider removing feature entirely'


def _get_duplicate_recommendation(percentage: float) -> str:
    """Get duplicate removal recommendation."""
    if percentage < 0.1:
        return 'Minimal impact - optional to remove'
    elif percentage < 1:
        return 'Low impact - consider removing'
    elif percentage < 5:
        return 'Moderate impact - should remove duplicates'
    else:
        return 'CRITICAL: Remove duplicates before modeling'


def _get_cardinality_recommendation(high_card_cols: List[Dict]) -> str:
    """Get high cardinality recommendation."""
    if not high_card_cols:
        return 'All columns have reasonable cardinality'
    
    critical = [c for c in high_card_cols if c['severity'] == 'High']
    
    if len(critical) > 0:
        return (
            'CRITICAL: High cardinality columns may cause overfitting. '
            'Consider target encoding, frequency encoding, or feature selection.'
        )
    return 'Consider grouping categories or using target encoding'


def _get_constant_recommendation(constant: List, near_constant: List) -> str:
    """Get constant feature recommendation."""
    if constant:
        return f'CRITICAL: Remove {len(constant)} constant feature(s) immediately'
    elif near_constant:
        return f'Recommended: Remove or investigate {len(near_constant)} near-constant feature(s)'
    return 'No constant features detected'


# ============================================================================
# Main Overview Function
# ============================================================================

def generate_overview_summary(
    df: pd.DataFrame,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive dataset overview and issue detection.
    
    This is the main function that orchestrates all overview analysis.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_column (Optional[str]): Name of target column (for context)
        
    Returns:
        Dict with complete overview including:
        - Dataset shape and memory
        - Data types
        - Quality metrics
        - Detected issues
        - Recommendations
        
    Example:
        >>> summary = generate_overview_summary(df, target_column='price')
        >>> print(f"Shape: {summary['shape']}")
        >>> print(f"Issues Found: {summary['total_issues']}")
        >>> for issue in summary['issues']:
        ...     print(f"  - {issue['type']}: {issue['description']}")
    """
    logger.info(f"Generating overview for dataset with shape {df.shape}")
    
    # Collect all analyses
    shape = get_dataset_shape(df)
    memory = get_memory_usage(df)
    dtypes = get_data_types(df)
    duplicates = get_duplicate_info(df)
    high_cardinality = detect_high_cardinality(df)
    constants = detect_constant_features(df)
    ids = detect_id_columns(df)
    missing = detect_missing_values(df)
    
    # Build issues list
    issues = []
    
    if constants['constant']:
        issues.append({
            'type': 'Constant Features',
            'severity': 'CRITICAL',
            'count': len(constants['constant']),
            'description': f"{len(constants['constant'])} columns with only 1 unique value",
            'recommendation': constants['recommendation']
        })
    
    if constants['near_constant']:
        issues.append({
            'type': 'Near-Constant Features',
            'severity': 'HIGH',
            'count': len(constants['near_constant']),
            'description': f"{len(constants['near_constant'])} columns with >99% single value",
            'recommendation': constants['recommendation']
        })
    
    if duplicates['count'] > 0:
        issues.append({
            'type': 'Duplicate Rows',
            'severity': duplicates['severity'],
            'count': duplicates['count'],
            'description': f"{duplicates['count']} duplicate rows ({duplicates['percentage']}%)",
            'recommendation': duplicates['recommendation']
        })
    
    if high_cardinality['count'] > 0:
        issues.append({
            'type': 'High Cardinality',
            'severity': 'MEDIUM',
            'count': high_cardinality['count'],
            'description': f"{high_cardinality['count']} columns with >50% unique values",
            'recommendation': high_cardinality['recommendation']
        })
    
    if ids['count'] > 0:
        issues.append({
            'type': 'ID-like Columns',
            'severity': 'MEDIUM',
            'count': ids['count'],
            'description': f"{ids['count']} suspected ID columns detected",
            'recommendation': ids['recommendation']
        })
    
    if missing['total_count'] > 0:
        issues.append({
            'type': 'Missing Values',
            'severity': missing['overall_severity'],
            'count': missing['total_count'],
            'description': f"{missing['total_count']} missing values ({missing['total_percentage']}%)",
            'recommendation': 'See detailed missing value analysis'
        })
    
    # Calculate data quality score
    quality_score = _calculate_quality_score(
        duplicates['percentage'],
        constants['total_problematic'],
        shape['rows'],
        missing['total_percentage']
    )
    
    # Build comprehensive summary
    summary = {
        'dataset_info': {
            'shape': shape,
            'memory_usage': memory,
            'data_types': dtypes,
            'quality_score': round(quality_score, 1)
        },
        'duplicates': duplicates,
        'missing_values': missing,
        'high_cardinality': high_cardinality,
        'constant_features': constants,
        'id_columns': ids,
        'issues': issues,
        'total_issues': len(issues),
        'target_column': target_column,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    # Log summary
    logger.info(
        f"Overview complete: {len(issues)} issues found, "
        f"Quality score: {quality_score:.1f}%"
    )
    
    return summary


def _calculate_quality_score(
    duplicate_pct: float,
    problematic_features: int,
    total_rows: int,
    missing_pct: float
) -> float:
    """
    Calculate overall data quality score (0-100).
    
    Weighted formula:
    - Missing values: 40% weight
    - Duplicates: 30% weight
    - Constant features: 20% weight
    - Other issues: 10% weight
    
    Args:
        duplicate_pct (float): Percentage of duplicate rows
        problematic_features (int): Count of constant/near-constant features
        total_rows (int): Total rows in dataset
        missing_pct (float): Overall missing percentage
        
    Returns:
        float: Quality score 0-100
    """
    # Convert to impact scores (0-1, where 1 = worst)
    missing_impact = min(missing_pct / 100, 1.0)
    duplicate_impact = min(duplicate_pct / 100, 1.0)
    feature_impact = min(problematic_features / max(total_rows, 1) * 100 / 100, 1.0)
    
    # Apply weights
    quality = (
        (1 - missing_impact) * 0.4 +
        (1 - duplicate_impact) * 0.3 +
        (1 - feature_impact) * 0.2 +
        0.1  # Other issues fixed
    )
    
    return quality * 100


# ============================================================================
# Utility Functions for Streamlit Integration
# ============================================================================

def print_overview_summary(summary: Dict) -> None:
    """
    Pretty print overview summary (console output).
    
    Args:
        summary (Dict): Summary dictionary from generate_overview_summary()
    """
    print("\n" + "="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    
    # Basic info
    info = summary['dataset_info']
    print(f"\n📊 Shape: {info['shape']['rows']:,} rows × {info['shape']['columns']} columns")
    print(f"💾 Memory: {info['memory_usage']['total_mb']:.2f} MB")
    print(f"📈 Quality Score: {info['quality_score']}/100")
    
    # Data types
    print(f"\n📋 Data Types:")
    dtypes = info['data_types']['count']
    print(f"  • Numeric: {dtypes['numeric']}")
    print(f"  • Categorical: {dtypes['categorical']}")
    print(f"  • DateTime: {dtypes['datetime']}")
    print(f"  • Boolean: {dtypes['boolean']}")
    
    # Issues
    if summary['issues']:
        print(f"\n⚠️  Issues Detected: {summary['total_issues']}")
        for issue in summary['issues']:
            print(f"  • [{issue['severity']}] {issue['type']}: {issue['description']}")
    else:
        print(f"\n✅ No issues detected!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Example usage
    # Create sample dataset
    sample_df = pd.DataFrame({
        'id': range(1000),
        'name': ['User_' + str(i) for i in range(1000)],
        'age': np.random.randint(18, 80, 1000),
        'constant_col': 5,
        'high_card': np.random.choice(500, 1000),
        'duplicated': np.random.choice([1, 2, 3, 4, 5], 1000),
        'missing_col': [np.nan if i % 20 == 0 else i for i in range(1000)]
    })
    
    # Generate summary
    summary = generate_overview_summary(sample_df, target_column='age')
    
    # Print results
    print_overview_summary(summary)
    
    # Access specific sections
    print("\nDetailed Missing Values:")
    for col in summary['missing_values']['columns'][:3]:
        print(f"  {col['column']}: {col['percentage']:.2f}% missing")
    
    print("\nDetected ID Columns:")
    for id_col in summary['id_columns']['suspected_ids']:
        print(f"  {id_col['column']}: {id_col['confidence']:.0%} confidence")
