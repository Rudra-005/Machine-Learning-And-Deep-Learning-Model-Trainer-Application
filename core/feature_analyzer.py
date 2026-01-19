"""
Feature Analysis Utilities

User-selected feature analysis with numerical and categorical support.
Optimized for Streamlit interactivity.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import logging

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


def detect_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect numerical and categorical features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dict with:
            - numerical: List of numerical column names
            - categorical: List of categorical column names
            
    Example:
        >>> features = detect_feature_types(df)
        >>> print(f"Numerical: {features['numerical']}")
        >>> print(f"Categorical: {features['categorical']}")
    """
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Detected {len(numerical)} numerical, {len(categorical)} categorical features")
    
    return {
        'numerical': numerical,
        'categorical': categorical
    }


def get_feature_stats(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Get statistics for a single feature.
    
    Args:
        df: Input DataFrame
        column: Column name
        
    Returns:
        Dict with feature statistics
        
    Example:
        >>> stats = get_feature_stats(df, 'age')
        >>> print(stats)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    series = df[column]
    
    if pd.api.types.is_numeric_dtype(series):
        return {
            'type': 'numerical',
            'count': len(series),
            'missing': series.isna().sum(),
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'median': float(series.median()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75))
        }
    else:
        value_counts = series.value_counts()
        return {
            'type': 'categorical',
            'count': len(series),
            'missing': series.isna().sum(),
            'unique': series.nunique(),
            'top_value': value_counts.index[0] if len(value_counts) > 0 else None,
            'top_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
        }


def plot_numerical_histogram(
    df: pd.DataFrame,
    column: str,
    backend: str = 'plotly',
    bins: int = 30
) -> Optional[Any]:
    """
    Create histogram for numerical feature.
    
    Args:
        df: Input DataFrame
        column: Column name
        backend: 'plotly' or 'matplotlib'
        bins: Number of bins
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = plot_numerical_histogram(df, 'age', backend='plotly')
        >>> fig.show()
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    data = df[column].dropna()
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Histogram(
                x=data,
                nbinsx=bins,
                marker=dict(color='steelblue', line=dict(color='darkblue', width=1)),
                hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'<b>{column} Distribution</b>',
            xaxis_title=f'<b>{column}</b>',
            yaxis_title='<b>Frequency</b>',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=80, l=80, r=40, t=80)
        )
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        ax.hist(data, bins=bins, color='steelblue', edgecolor='darkblue', alpha=0.7)
        
        ax.set_xlabel(column, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{column} Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


def plot_numerical_boxplot(
    df: pd.DataFrame,
    column: str,
    backend: str = 'plotly'
) -> Optional[Any]:
    """
    Create boxplot for numerical feature.
    
    Args:
        df: Input DataFrame
        column: Column name
        backend: 'plotly' or 'matplotlib'
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = plot_numerical_boxplot(df, 'age', backend='plotly')
        >>> fig.show()
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    data = df[column].dropna()
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Box(
                y=data,
                name=column,
                marker=dict(color='steelblue'),
                boxmean='sd',
                hovertemplate='Value: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'<b>{column} Boxplot</b>',
            yaxis_title=f'<b>{column}</b>',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=80, l=80, r=40, t=80)
        )
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        bp = ax.boxplot(data, vert=True, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        
        for whisker in bp['whiskers']:
            whisker.set(color='darkblue', linewidth=1.5)
        
        for cap in bp['caps']:
            cap.set(color='darkblue', linewidth=1.5)
        
        ax.set_ylabel(column, fontsize=12, fontweight='bold')
        ax.set_title(f'{column} Boxplot', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


def plot_categorical_bar(
    df: pd.DataFrame,
    column: str,
    backend: str = 'plotly',
    top_n: int = 20
) -> Optional[Any]:
    """
    Create bar chart for categorical feature.
    
    Args:
        df: Input DataFrame
        column: Column name
        backend: 'plotly' or 'matplotlib'
        top_n: Show top N categories
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = plot_categorical_bar(df, 'category', backend='plotly')
        >>> fig.show()
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    value_counts = df[column].value_counts().head(top_n)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=value_counts.index.astype(str),
                y=value_counts.values,
                text=value_counts.values,
                textposition='outside',
                marker=dict(color='steelblue', line=dict(color='darkblue', width=1)),
                hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'<b>{column} Value Counts</b>',
            xaxis_title=f'<b>{column}</b>',
            yaxis_title='<b>Count</b>',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=150, l=80, r=40, t=80)
        )
        
        fig.update_xaxes(tickangle=-45)
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        bars = ax.bar(range(len(value_counts)), value_counts.values, 
                     color='steelblue', edgecolor='darkblue', linewidth=1.2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel(column, fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{column} Value Counts', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index.astype(str), rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


def analyze_feature(
    df: pd.DataFrame,
    column: str,
    backend: str = 'plotly'
) -> Dict[str, Any]:
    """
    Comprehensive analysis for a single feature.
    
    Args:
        df: Input DataFrame
        column: Column name
        backend: 'plotly' or 'matplotlib'
        
    Returns:
        Dict with:
            - stats: Feature statistics
            - plot: Figure object
            - feature_type: 'numerical' or 'categorical'
            
    Example:
        >>> analysis = analyze_feature(df, 'age')
        >>> print(analysis['stats'])
        >>> analysis['plot'].show()
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    stats = get_feature_stats(df, column)
    feature_type = stats['type']
    
    if feature_type == 'numerical':
        plot = plot_numerical_histogram(df, column, backend=backend)
    else:
        plot = plot_categorical_bar(df, column, backend=backend)
    
    return {
        'column': column,
        'feature_type': feature_type,
        'stats': stats,
        'plot': plot
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000)
    })
    
    # Detect features
    features = detect_feature_types(df)
    print(f"Numerical: {features['numerical']}")
    print(f"Categorical: {features['categorical']}")
    
    # Analyze single feature
    analysis = analyze_feature(df, 'age')
    print(f"\nFeature: {analysis['column']}")
    print(f"Type: {analysis['feature_type']}")
    print(f"Stats: {analysis['stats']}")
