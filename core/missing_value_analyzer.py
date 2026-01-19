"""
Missing Value Analysis Module

Production-ready functions for comprehensive missing value analysis:
- Compute missing statistics per column
- Identify columns exceeding thresholds
- Generate efficient visualizations
- Handle large datasets with sampling

Optimized for performance and memory efficiency.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MissingStats:
    """Missing value statistics for a column."""
    column: str
    missing_count: int
    missing_pct: float
    dtype: str
    severity: str


def compute_missing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missing value count and percentage per column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with columns: column, missing_count, missing_pct, dtype, severity
        Sorted by missing_pct descending
        
    Example:
        >>> stats = compute_missing_stats(df)
        >>> print(stats[stats['missing_pct'] > 10])
    """
    total_rows = len(df)
    
    stats_list = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
        
        # Categorize severity
        if missing_pct == 0:
            severity = 'OK'
        elif missing_pct < 5:
            severity = 'Low'
        elif missing_pct < 20:
            severity = 'Medium'
        elif missing_pct < 50:
            severity = 'High'
        else:
            severity = 'Critical'
        
        stats_list.append({
            'column': col,
            'missing_count': int(missing_count),
            'missing_pct': round(missing_pct, 2),
            'dtype': str(df[col].dtype),
            'severity': severity
        })
    
    stats_df = pd.DataFrame(stats_list).sort_values(
        'missing_pct', ascending=False
    ).reset_index(drop=True)
    
    logger.info(f"Computed missing stats for {len(df.columns)} columns")
    return stats_df


def get_columns_above_threshold(
    df: pd.DataFrame,
    threshold: float = 20.0
) -> Dict[str, Any]:
    """
    Identify columns exceeding missing value threshold.
    
    Args:
        df: Input DataFrame
        threshold: Missing percentage threshold (default 20%)
        
    Returns:
        Dict with:
            - columns: List of column names exceeding threshold
            - count: Number of columns
            - details: List of dicts with column stats
            
    Example:
        >>> result = get_columns_above_threshold(df, threshold=15)
        >>> print(f"Columns above 15%: {result['count']}")
    """
    stats_df = compute_missing_stats(df)
    exceeding = stats_df[stats_df['missing_pct'] >= threshold]
    
    return {
        'columns': exceeding['column'].tolist(),
        'count': len(exceeding),
        'threshold': threshold,
        'details': exceeding.to_dict('records')
    }


def get_missing_patterns(
    df: pd.DataFrame,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Identify missing value patterns and co-occurrence.
    
    Detects if missing values appear together systematically,
    suggesting data collection issues or patterns.
    
    Args:
        df: Input DataFrame
        sample_size: Sample size for large datasets (optional)
        
    Returns:
        Dict with:
            - total_patterns: Number of distinct patterns
            - rows_with_missing: Count of rows with any missing
            - rows_with_missing_pct: Percentage of rows with missing
            - completely_missing_rows: Count of all-missing rows
            - top_patterns: List of most common patterns
            
    Example:
        >>> patterns = get_missing_patterns(df)
        >>> print(f"Rows with missing: {patterns['rows_with_missing']}")
    """
    missing_matrix = df.isna().astype(int)
    
    # Sample for large datasets
    if sample_size and len(missing_matrix) > sample_size:
        missing_matrix = missing_matrix.sample(n=sample_size, random_state=42)
        logger.info(f"Analyzing patterns on sample of {sample_size} rows")
    
    # Identify patterns
    patterns = missing_matrix.apply(tuple, axis=1).value_counts()
    
    # Get top patterns
    top_patterns = []
    for pattern, freq in patterns.head(5).items():
        missing_cols = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
        top_patterns.append({
            'columns': missing_cols,
            'frequency': int(freq),
            'percentage': round(freq / len(missing_matrix) * 100, 2)
        })
    
    rows_with_missing = missing_matrix.any(axis=1).sum()
    completely_missing = missing_matrix.all(axis=1).sum()
    
    return {
        'total_patterns': len(patterns),
        'rows_with_missing': int(rows_with_missing),
        'rows_with_missing_pct': round(rows_with_missing / len(df) * 100, 2),
        'completely_missing_rows': int(completely_missing),
        'top_patterns': top_patterns
    }


def create_missing_bar_chart(
    df: pd.DataFrame,
    backend: str = 'plotly',
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100
) -> Optional[Any]:
    """
    Create bar chart of missing percentages.
    
    Args:
        df: Input DataFrame
        backend: 'plotly' or 'matplotlib' (default 'plotly')
        figsize: Figure size for matplotlib (width, height)
        dpi: Resolution for matplotlib
        
    Returns:
        Figure object (go.Figure for plotly, plt.Figure for matplotlib)
        
    Example:
        >>> fig = create_missing_bar_chart(df, backend='plotly')
        >>> fig.show()
    """
    stats_df = compute_missing_stats(df)
    stats_df = stats_df[stats_df['missing_pct'] > 0]
    
    if len(stats_df) == 0:
        logger.info("No missing values found")
        return None
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        return _create_bar_plotly(stats_df)
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        return _create_bar_matplotlib(stats_df, figsize, dpi)
    else:
        logger.warning(f"{backend} not available")
        return None


def create_missing_heatmap(
    df: pd.DataFrame,
    backend: str = 'plotly',
    sample_rows: int = 500,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100
) -> Optional[Any]:
    """
    Create heatmap of missing values.
    
    Shows which cells contain missing values across rows and columns.
    Automatically samples large datasets for performance.
    
    Args:
        df: Input DataFrame
        backend: 'plotly' or 'matplotlib' (default 'plotly')
        sample_rows: Max rows to display (default 500)
        figsize: Figure size for matplotlib
        dpi: Resolution for matplotlib
        
    Returns:
        Figure object (go.Figure for plotly, plt.Figure for matplotlib)
        
    Example:
        >>> fig = create_missing_heatmap(df, backend='matplotlib', sample_rows=1000)
        >>> fig.savefig('heatmap.png', dpi=150, bbox_inches='tight')
    """
    # Sample for large datasets
    if len(df) > sample_rows:
        df_display = df.sample(n=sample_rows, random_state=42)
    else:
        df_display = df
    
    missing_matrix = df_display.isna().astype(int)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        return _create_heatmap_plotly(missing_matrix, len(df_display))
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        return _create_heatmap_matplotlib(missing_matrix, figsize, dpi)
    else:
        logger.warning(f"{backend} not available")
        return None


def _create_bar_plotly(stats_df: pd.DataFrame) -> go.Figure:
    """Create Plotly bar chart."""
    colors = [_get_color(sev) for sev in stats_df['severity']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=stats_df['column'],
            y=stats_df['missing_pct'],
            text=stats_df['missing_pct'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            marker=dict(color=colors, line=dict(color='darkgray', width=1)),
            hovertemplate='<b>%{x}</b><br>Missing: %{y:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='<b>Missing Values by Column</b>',
        xaxis_title='<b>Column</b>',
        yaxis_title='<b>Missing %</b>',
        height=600,
        hovermode='closest',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        margin=dict(b=150, l=80, r=40, t=80)
    )
    
    # Add threshold line
    if stats_df['missing_pct'].max() > 20:
        fig.add_hline(y=20, line_dash="dash", line_color="orange",
                     annotation_text="20% threshold", annotation_position="right")
    
    return fig


def _create_bar_matplotlib(
    stats_df: pd.DataFrame,
    figsize: Tuple[int, int],
    dpi: int
) -> plt.Figure:
    """Create Matplotlib bar chart."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    colors = [_get_color_rgb(sev) for sev in stats_df['severity']]
    
    bars = ax.bar(range(len(stats_df)), stats_df['missing_pct'],
                  color=colors, edgecolor='darkgray', linewidth=1.2)
    
    # Add percentage labels
    for bar, pct in zip(bars, stats_df['missing_pct']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Column', fontsize=12, fontweight='bold')
    ax.set_ylabel('Missing %', fontsize=12, fontweight='bold')
    ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df['column'], rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add threshold line
    if stats_df['missing_pct'].max() > 20:
        ax.axhline(y=20, color='orange', linestyle='--', linewidth=2,
                  label='20% Threshold', alpha=0.7)
    
    plt.tight_layout()
    return fig


def _create_heatmap_plotly(missing_matrix: pd.DataFrame, total_rows: int) -> go.Figure:
    """Create Plotly heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix.values,
        x=missing_matrix.columns,
        y=[f'Row {i}' for i in range(len(missing_matrix))],
        colorscale=[[0, '#2ecc71'], [1, '#e74c3c']],
        colorbar=dict(title="Missing", tickvals=[0, 1], ticktext=['Present', 'Missing']),
        hovertemplate='<b>%{y}</b><br>Column: %{x}<br>Status: %{text}<extra></extra>',
        text=[[('Missing' if val == 1 else 'Present') for val in row]
              for row in missing_matrix.values]
    ))
    
    fig.update_layout(
        title=f'<b>Missing Values Heatmap</b> ({total_rows} rows)',
        xaxis_title='<b>Columns</b>',
        yaxis_title='<b>Rows</b>',
        height=400,
        margin=dict(b=150, l=100, r=40, t=80)
    )
    
    return fig


def _create_heatmap_matplotlib(
    missing_matrix: pd.DataFrame,
    figsize: Tuple[int, int],
    dpi: int
) -> plt.Figure:
    """Create Matplotlib heatmap."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    cmap = LinearSegmentedColormap.from_list('missing', ['#2ecc71', '#e74c3c'])
    im = ax.imshow(missing_matrix.values, aspect='auto', cmap=cmap, interpolation='nearest')
    
    ax.set_xticks(range(len(missing_matrix.columns)))
    ax.set_yticks(range(0, len(missing_matrix), max(1, len(missing_matrix)//10)))
    ax.set_xticklabels(missing_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels([f'Row {i}' for i in ax.get_yticks()], fontsize=9)
    
    ax.set_xlabel('Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rows', fontsize=12, fontweight='bold')
    ax.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Missing Status', fontsize=10, fontweight='bold')
    cbar.ax.set_yticklabels(['Present', 'Missing'])
    
    plt.tight_layout()
    return fig


def _get_color(severity: str) -> str:
    """Get Plotly color for severity."""
    colors = {
        'OK': '#2ecc71', 'Low': '#27ae60', 'Medium': '#f39c12',
        'High': '#e74c3c', 'Critical': '#c0392b'
    }
    return colors.get(severity, '#95a5a6')


def _get_color_rgb(severity: str) -> Tuple[float, float, float]:
    """Get Matplotlib RGB color for severity."""
    colors = {
        'OK': (46/255, 204/255, 113/255),
        'Low': (39/255, 174/255, 96/255),
        'Medium': (243/255, 156/255, 18/255),
        'High': (231/255, 76/255, 60/255),
        'Critical': (192/255, 57/255, 43/255)
    }
    return colors.get(severity, (149/255, 165/255, 166/255))


def analyze_missing_values(
    df: pd.DataFrame,
    threshold: float = 20.0,
    create_plots: bool = True,
    backend: str = 'plotly'
) -> Dict[str, Any]:
    """
    Comprehensive missing value analysis.
    
    Main orchestration function for complete missing value analysis
    with statistics, threshold detection, patterns, and visualizations.
    
    Args:
        df: Input DataFrame
        threshold: Missing percentage threshold (default 20%)
        create_plots: Whether to create visualizations (default True)
        backend: 'plotly' or 'matplotlib' (default 'plotly')
        
    Returns:
        Dict with:
            - stats: DataFrame of missing statistics
            - above_threshold: Dict of columns exceeding threshold
            - patterns: Dict of missing patterns
            - plots: Dict of figure objects (if create_plots=True)
            - summary: Text summary
            
    Example:
        >>> analysis = analyze_missing_values(df, threshold=15)
        >>> print(analysis['summary'])
        >>> analysis['plots']['bar_chart'].show()
    """
    logger.info("Starting missing value analysis")
    
    stats_df = compute_missing_stats(df)
    above_threshold = get_columns_above_threshold(df, threshold)
    patterns = get_missing_patterns(df)
    
    result = {
        'stats': stats_df,
        'above_threshold': above_threshold,
        'patterns': patterns,
        'threshold': threshold
    }
    
    if create_plots:
        result['plots'] = {
            'bar_chart': create_missing_bar_chart(df, backend=backend),
            'heatmap': create_missing_heatmap(df, backend=backend)
        }
    
    result['summary'] = _generate_summary(stats_df, above_threshold, patterns, threshold)
    
    logger.info("Missing value analysis complete")
    return result


def _generate_summary(
    stats_df: pd.DataFrame,
    above_threshold: Dict,
    patterns: Dict,
    threshold: float
) -> str:
    """Generate text summary."""
    lines = [
        "=" * 70,
        "MISSING VALUE ANALYSIS SUMMARY",
        "=" * 70,
        ""
    ]
    
    total_missing = stats_df['missing_count'].sum()
    cols_with_missing = (stats_df['missing_pct'] > 0).sum()
    
    lines.extend([
        f"📊 Total Missing Cells: {total_missing:,}",
        f"📋 Columns with Missing: {cols_with_missing} / {len(stats_df)}",
        ""
    ])
    
    if above_threshold['count'] > 0:
        lines.append(f"⚠️  ABOVE {threshold}% THRESHOLD: {above_threshold['count']} columns")
        for detail in above_threshold['details'][:3]:
            lines.append(f"   • {detail['column']}: {detail['missing_pct']:.2f}%")
        lines.append("")
    
    lines.extend([
        f"📍 Rows with Missing: {patterns['rows_with_missing']:,} ({patterns['rows_with_missing_pct']:.1f}%)",
        f"📍 Distinct Patterns: {patterns['total_patterns']}",
        "=" * 70
    ])
    
    return '\n'.join(lines)
