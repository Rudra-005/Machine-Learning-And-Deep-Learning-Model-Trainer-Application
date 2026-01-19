"""
Feature-Target Relationship Analysis

Analyze relationships between features and target variable.
Supports both regression and classification tasks.
Optimized for large datasets.
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


def compute_correlation_matrix(
    df: pd.DataFrame,
    target: str,
    method: str = 'pearson',
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Compute correlation matrix for numerical features with target.
    
    Args:
        df: Input DataFrame
        target: Target column name
        method: 'pearson' or 'spearman'
        sample_size: Sample size for large datasets (optional)
        
    Returns:
        Correlation matrix (features x target)
        
    Example:
        >>> corr = compute_correlation_matrix(df, 'price')
        >>> print(corr.sort_values(ascending=False))
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target not in numerical_cols:
        raise ValueError(f"Target '{target}' must be numerical")
    
    # Sample for large datasets
    if sample_size and len(df) > sample_size:
        df_sample = df[numerical_cols].sample(n=sample_size, random_state=42)
        logger.info(f"Computing correlation on sample of {sample_size} rows")
    else:
        df_sample = df[numerical_cols]
    
    # Compute correlation
    corr_matrix = df_sample.corr(method=method)
    
    # Return correlations with target
    target_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)
    
    logger.info(f"Computed {method} correlation for {len(target_corr)} features")
    return target_corr


def get_top_correlated_features(
    df: pd.DataFrame,
    target: str,
    method: str = 'pearson',
    top_n: int = 10,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get top N features most correlated with target.
    
    Args:
        df: Input DataFrame
        target: Target column name
        method: 'pearson' or 'spearman'
        top_n: Number of top features to return
        sample_size: Sample size for large datasets
        
    Returns:
        Dict with:
            - features: List of top feature names
            - correlations: List of correlation values
            - method: Correlation method used
            
    Example:
        >>> top = get_top_correlated_features(df, 'price', top_n=5)
        >>> print(top['features'])
    """
    corr = compute_correlation_matrix(df, target, method=method, sample_size=sample_size)
    
    top_features = corr.head(top_n)
    
    return {
        'features': top_features.index.tolist(),
        'correlations': top_features.values.tolist(),
        'method': method,
        'count': len(top_features)
    }


def analyze_categorical_regression(
    df: pd.DataFrame,
    feature: str,
    target: str,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze target mean per category (for regression).
    
    Args:
        df: Input DataFrame
        feature: Categorical feature column name
        target: Numerical target column name
        sample_size: Sample size for large datasets
        
    Returns:
        Dict with:
            - categories: List of category values
            - means: List of target means per category
            - counts: List of sample counts per category
            - overall_mean: Overall target mean
            
    Example:
        >>> analysis = analyze_categorical_regression(df, 'city', 'price')
        >>> print(analysis['means'])
    """
    # Sample for large datasets
    if sample_size and len(df) > sample_size:
        df_sample = df[[feature, target]].sample(n=sample_size, random_state=42)
    else:
        df_sample = df[[feature, target]]
    
    # Group by category
    grouped = df_sample.groupby(feature)[target].agg(['mean', 'count']).reset_index()
    grouped = grouped.sort_values('mean', ascending=False)
    
    return {
        'categories': grouped[feature].tolist(),
        'means': grouped['mean'].tolist(),
        'counts': grouped['count'].tolist(),
        'overall_mean': float(df_sample[target].mean()),
        'n_categories': len(grouped)
    }


def analyze_categorical_classification(
    df: pd.DataFrame,
    feature: str,
    target: str,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze class proportion per category (for classification).
    
    Args:
        df: Input DataFrame
        feature: Categorical feature column name
        target: Categorical target column name
        sample_size: Sample size for large datasets
        
    Returns:
        Dict with:
            - categories: List of category values
            - class_proportions: Dict of class proportions per category
            - class_counts: Dict of class counts per category
            
    Example:
        >>> analysis = analyze_categorical_classification(df, 'city', 'purchased')
        >>> print(analysis['class_proportions'])
    """
    # Sample for large datasets
    if sample_size and len(df) > sample_size:
        df_sample = df[[feature, target]].sample(n=sample_size, random_state=42)
    else:
        df_sample = df[[feature, target]]
    
    # Get unique classes
    classes = df_sample[target].unique()
    
    # Compute proportions per category
    result = {
        'categories': [],
        'class_proportions': {},
        'class_counts': {}
    }
    
    for cat in df_sample[feature].unique():
        cat_data = df_sample[df_sample[feature] == cat][target]
        result['categories'].append(cat)
        
        # Proportions
        props = {}
        counts = {}
        for cls in classes:
            count = (cat_data == cls).sum()
            props[str(cls)] = float(count / len(cat_data)) if len(cat_data) > 0 else 0
            counts[str(cls)] = int(count)
        
        result['class_proportions'][str(cat)] = props
        result['class_counts'][str(cat)] = counts
    
    return result


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target: str,
    method: str = 'pearson',
    backend: str = 'plotly',
    sample_size: Optional[int] = None
) -> Optional[Any]:
    """
    Create heatmap of correlations with target.
    
    Args:
        df: Input DataFrame
        target: Target column name
        method: 'pearson' or 'spearman'
        backend: 'plotly' or 'matplotlib'
        sample_size: Sample size for large datasets
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = plot_correlation_heatmap(df, 'price', backend='plotly')
        >>> fig.show()
    """
    corr = compute_correlation_matrix(df, target, method=method, sample_size=sample_size)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=go.Bar(
            x=corr.values,
            y=corr.index,
            orientation='h',
            marker=dict(
                color=corr.values,
                colorscale='RdBu',
                cmin=-1,
                cmax=1,
                showscale=True
            ),
            hovertemplate='<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'<b>Feature Correlation with {target}</b> ({method})',
            xaxis_title='<b>Correlation</b>',
            yaxis_title='<b>Feature</b>',
            height=max(400, len(corr) * 20),
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(l=150, r=40, t=80, b=80)
        )
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, max(6, len(corr) * 0.3)), dpi=100)
        
        colors = ['red' if x < 0 else 'blue' for x in corr.values]
        ax.barh(range(len(corr)), corr.values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.index, fontsize=10)
        ax.set_xlabel('Correlation', fontsize=12, fontweight='bold')
        ax.set_title(f'{target} Correlation ({method})', fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


def plot_categorical_regression(
    df: pd.DataFrame,
    feature: str,
    target: str,
    backend: str = 'plotly',
    sample_size: Optional[int] = None
) -> Optional[Any]:
    """
    Create bar plot of target mean per category.
    
    Args:
        df: Input DataFrame
        feature: Categorical feature column name
        target: Numerical target column name
        backend: 'plotly' or 'matplotlib'
        sample_size: Sample size for large datasets
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = plot_categorical_regression(df, 'city', 'price')
        >>> fig.show()
    """
    analysis = analyze_categorical_regression(df, feature, target, sample_size=sample_size)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=analysis['categories'],
                y=analysis['means'],
                text=[f"{m:.2f}" for m in analysis['means']],
                textposition='outside',
                marker=dict(color='steelblue', line=dict(color='darkblue', width=1)),
                hovertemplate='<b>%{x}</b><br>Mean: %{y:.2f}<extra></extra>'
            )
        ])
        
        fig.add_hline(
            y=analysis['overall_mean'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Overall Mean: {analysis['overall_mean']:.2f}",
            annotation_position="right"
        )
        
        fig.update_layout(
            title=f'<b>{target} Mean by {feature}</b>',
            xaxis_title=f'<b>{feature}</b>',
            yaxis_title=f'<b>{target} Mean</b>',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=150, l=80, r=40, t=80)
        )
        
        fig.update_xaxes(tickangle=-45)
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        bars = ax.bar(range(len(analysis['categories'])), analysis['means'],
                     color='steelblue', edgecolor='darkblue', linewidth=1.2)
        
        for bar, mean in zip(bars, analysis['means']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.axhline(y=analysis['overall_mean'], color='red', linestyle='--', linewidth=2, label='Overall Mean')
        
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{target} Mean', fontsize=12, fontweight='bold')
        ax.set_title(f'{target} Mean by {feature}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(range(len(analysis['categories'])))
        ax.set_xticklabels(analysis['categories'], rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    return None


def plot_categorical_classification(
    df: pd.DataFrame,
    feature: str,
    target: str,
    backend: str = 'plotly',
    sample_size: Optional[int] = None
) -> Optional[Any]:
    """
    Create stacked bar plot of class proportions per category.
    
    Args:
        df: Input DataFrame
        feature: Categorical feature column name
        target: Categorical target column name
        backend: 'plotly' or 'matplotlib'
        sample_size: Sample size for large datasets
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = plot_categorical_classification(df, 'city', 'purchased')
        >>> fig.show()
    """
    analysis = analyze_categorical_classification(df, feature, target, sample_size=sample_size)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        # Get all classes
        classes = list(next(iter(analysis['class_proportions'].values())).keys())
        
        for cls in classes:
            values = [analysis['class_proportions'][str(cat)][cls] for cat in analysis['categories']]
            fig.add_trace(go.Bar(
                x=analysis['categories'],
                y=values,
                name=str(cls),
                hovertemplate='<b>%{x}</b><br>' + f'{cls}: ' + '%{y:.2%}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'<b>{target} Proportion by {feature}</b>',
            xaxis_title=f'<b>{feature}</b>',
            yaxis_title='<b>Proportion</b>',
            barmode='stack',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=150, l=80, r=40, t=80)
        )
        
        fig.update_xaxes(tickangle=-45)
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        
        classes = list(next(iter(analysis['class_proportions'].values())).keys())
        x = np.arange(len(analysis['categories']))
        width = 0.6
        
        bottom = np.zeros(len(analysis['categories']))
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        
        for i, cls in enumerate(classes):
            values = [analysis['class_proportions'][str(cat)][cls] for cat in analysis['categories']]
            ax.bar(x, values, width, label=str(cls), bottom=bottom, color=colors[i], edgecolor='black', linewidth=0.5)
            bottom += np.array(values)
        
        ax.set_xlabel(feature, fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion', fontsize=12, fontweight='bold')
        ax.set_title(f'{target} Proportion by {feature}', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(analysis['categories'], rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    # Correlation analysis
    corr = compute_correlation_matrix(df, 'price')
    print("Correlations with price:")
    print(corr)
    
    # Top correlated features
    top = get_top_correlated_features(df, 'price', top_n=3)
    print(f"\nTop features: {top['features']}")
    
    # Categorical regression
    analysis = analyze_categorical_regression(df, 'city', 'price')
    print(f"\nPrice by city: {analysis['means']}")
