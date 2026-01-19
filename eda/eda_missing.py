"""
EDA Missing Values Module

Comprehensive missing value analysis including detection, visualization,
and recommendations for handling missing data.

Features:
- Missing value statistics per column
- Threshold-based identification
- Interactive and static visualizations
- Missing value patterns
- Efficient handling of large datasets

Author: Data Quality Engineering Team
Date: 2026-01-19
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

# Visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive visualizations disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib not available. Static visualizations disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MissingValueStats:
    """Container for missing value statistics of a column."""
    column_name: str
    missing_count: int
    missing_percentage: float
    data_type: str
    severity: str
    
    def __repr__(self) -> str:
        return (
            f"MissingValueStats(column='{self.column_name}', "
            f"missing={self.missing_percentage:.2f}%, "
            f"severity='{self.severity}')"
        )


# ============================================================================
# Missing Value Statistics
# ============================================================================

def compute_missing_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive missing value statistics for all columns.
    
    Args:
        df (pd.DataFrame): Input dataset
        
    Returns:
        pd.DataFrame: Statistics with columns:
            - column: Column name
            - missing_count: Number of missing values
            - missing_percentage: % of missing values
            - data_type: Column data type
            - non_missing_count: Count of non-missing values
            - severity: Categorized severity level
            
    Example:
        >>> stats = compute_missing_statistics(df)
        >>> stats[stats['missing_percentage'] > 20]
    """
    missing_stats = []
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        total_count = len(df)
        missing_pct = (missing_count / total_count * 100) if total_count > 0 else 0
        severity = _categorize_missing_severity(missing_pct)
        
        missing_stats.append({
            'column': col,
            'missing_count': int(missing_count),
            'missing_percentage': round(missing_pct, 2),
            'data_type': str(df[col].dtype),
            'non_missing_count': int(total_count - missing_count),
            'severity': severity
        })
    
    # Create DataFrame and sort by missing percentage descending
    stats_df = pd.DataFrame(missing_stats).sort_values(
        'missing_percentage',
        ascending=False
    ).reset_index(drop=True)
    
    logger.info(f"Computed missing statistics for {len(df.columns)} columns")
    return stats_df


def get_columns_above_threshold(
    df: pd.DataFrame,
    threshold: float = 20.0
) -> Dict[str, Any]:
    """
    Identify columns exceeding missing value threshold.
    
    Args:
        df (pd.DataFrame): Input dataset
        threshold (float): Missing percentage threshold (default 20%)
        
    Returns:
        Dict with:
            - columns: List of columns exceeding threshold
            - count: Number of columns exceeding threshold
            - threshold: The threshold used
            - details: Detailed statistics for each column
            
    Example:
        >>> above_threshold = get_columns_above_threshold(df, threshold=15)
        >>> print(f"Columns with >15% missing: {above_threshold['count']}")
    """
    missing_stats = compute_missing_statistics(df)
    
    # Filter columns exceeding threshold
    exceeding = missing_stats[missing_stats['missing_percentage'] >= threshold]
    
    columns_list = exceeding['column'].tolist()
    
    return {
        'columns': columns_list,
        'count': len(columns_list),
        'threshold': threshold,
        'details': exceeding.to_dict('records'),
        'recommendation': _get_missing_threshold_recommendation(
            len(columns_list),
            threshold
        )
    }


def get_missing_patterns(df: pd.DataFrame, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analyze patterns in missing values (co-occurrence and correlations).
    
    Detects if missing values appear together in specific rows,
    suggesting systematic patterns or data collection issues.
    
    Args:
        df (pd.DataFrame): Input dataset
        sample_size (Optional[int]): Sample size for pattern analysis (for large datasets)
        
    Returns:
        Dict with:
            - total_patterns: Number of distinct missing patterns
            - pattern_list: Top patterns with frequency
            - rows_with_any_missing: Count of rows with any missing values
            - completely_missing_rows: Count of rows all missing
            - correlation: Correlation between missing indicators
            
    Example:
        >>> patterns = get_missing_patterns(df)
        >>> print(f"Distinct patterns: {patterns['total_patterns']}")
    """
    # Create missing indicator matrix
    missing_matrix = df.isna().astype(int)
    
    # Sample for large datasets
    if sample_size and len(missing_matrix) > sample_size:
        missing_matrix_sample = missing_matrix.sample(
            n=sample_size,
            random_state=42
        )
        logger.info(f"Analyzing missing patterns on sample of {sample_size} rows")
    else:
        missing_matrix_sample = missing_matrix
    
    # Convert rows to tuples to identify patterns
    patterns = missing_matrix_sample.apply(tuple, axis=1).value_counts()
    
    # Identify specific patterns
    pattern_details = []
    for pattern, frequency in patterns.head(10).items():
        missing_cols = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
        pattern_details.append({
            'pattern': missing_cols,
            'frequency': int(frequency),
            'percentage': round(frequency / len(missing_matrix_sample) * 100, 2)
        })
    
    # Count rows with any missing
    rows_with_missing = missing_matrix.any(axis=1).sum()
    completely_missing_rows = missing_matrix.all(axis=1).sum()
    
    return {
        'total_patterns': len(patterns),
        'top_patterns': pattern_details,
        'rows_with_any_missing': int(rows_with_missing),
        'rows_with_any_missing_percentage': round(
            rows_with_missing / len(df) * 100, 2
        ),
        'completely_missing_rows': int(completely_missing_rows),
        'completely_missing_percentage': round(
            completely_missing_rows / len(df) * 100, 2
        ) if len(df) > 0 else 0
    }


# ============================================================================
# Missing Value Visualizations - Plotly (Interactive)
# ============================================================================

def create_missing_bar_chart_plotly(
    df: pd.DataFrame,
    height: int = 600,
    show_all: bool = False
) -> Optional[go.Figure]:
    """
    Create interactive bar chart of missing percentages using Plotly.
    
    Shows columns with missing values sorted by percentage.
    
    Args:
        df (pd.DataFrame): Input dataset
        height (int): Chart height in pixels (default 600)
        show_all (bool): Show all columns or only those with missing (default False)
        
    Returns:
        go.Figure: Plotly figure object or None if Plotly unavailable
        
    Example:
        >>> fig = create_missing_bar_chart_plotly(df)
        >>> fig.show()
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Returning None.")
        return None
    
    missing_stats = compute_missing_statistics(df)
    
    # Filter columns with missing values if not showing all
    if not show_all:
        missing_stats = missing_stats[missing_stats['missing_percentage'] > 0]
    
    if len(missing_stats) == 0:
        logger.info("No missing values found")
        missing_stats = pd.DataFrame({
            'column': ['No missing data'],
            'missing_percentage': [0],
            'severity': ['OK']
        })
    
    # Color by severity
    colors = [_get_severity_color(sev) for sev in missing_stats['severity']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=missing_stats['column'],
            y=missing_stats['missing_percentage'],
            text=missing_stats['missing_percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            marker=dict(color=colors, line=dict(color='darkgray', width=1)),
            hovertemplate=(
                '<b>%{x}</b><br>' +
                'Missing: %{y:.2f}%<br>' +
                '<extra></extra>'
            )
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='<b>Missing Values by Column</b>',
            font=dict(size=18)
        ),
        xaxis=dict(
            title='<b>Column Name</b>',
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title='<b>Missing Percentage (%)</b>',
            tickfont=dict(size=11)
        ),
        height=height,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        margin=dict(b=150, l=80, r=40, t=80)
    )
    
    # Add threshold line if any values above 20%
    max_missing = missing_stats['missing_percentage'].max()
    if max_missing > 20:
        fig.add_hline(
            y=20,
            line_dash="dash",
            line_color="orange",
            annotation_text="20% threshold",
            annotation_position="right"
        )
    
    logger.info(f"Created Plotly missing bar chart")
    return fig


def create_missing_heatmap_plotly(
    df: pd.DataFrame,
    sample_rows: int = 500,
    height: int = 400
) -> Optional[go.Figure]:
    """
    Create interactive heatmap of missing values using Plotly.
    
    Shows which cells contain missing values across rows and columns.
    Useful for identifying patterns and co-occurrence.
    
    Args:
        df (pd.DataFrame): Input dataset
        sample_rows (int): Max rows to display (default 500)
        height (int): Chart height in pixels (default 400)
        
    Returns:
        go.Figure: Plotly figure object or None if Plotly unavailable
        
    Example:
        >>> fig = create_missing_heatmap_plotly(df, sample_rows=1000)
        >>> fig.show()
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Returning None.")
        return None
    
    # Sample rows for large datasets
    if len(df) > sample_rows:
        df_display = df.sample(n=sample_rows, random_state=42)
        note = f" (showing random sample of {sample_rows} rows)"
    else:
        df_display = df
        note = ""
    
    # Create missing value matrix (1 = missing, 0 = present)
    missing_matrix = df_display.isna().astype(int)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=missing_matrix.values,
        x=missing_matrix.columns,
        y=[f'Row {i}' for i in range(len(missing_matrix))],
        colorscale=[[0, 'green'], [1, 'red']],
        colorbar=dict(
            title="Missing",
            tickvals=[0, 1],
            ticktext=['Present', 'Missing']
        ),
        hovertemplate='<b>%{y}</b><br>Column: %{x}<br>Status: %{text}<extra></extra>',
        text=[[('Missing' if val == 1 else 'Present') 
              for val in row] for row in missing_matrix.values]
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>Missing Values Heatmap</b>{note}',
            font=dict(size=16)
        ),
        xaxis=dict(
            title='<b>Columns</b>',
            tickangle=-45,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title='<b>Rows</b>',
            tickfont=dict(size=9)
        ),
        height=height,
        margin=dict(b=150, l=100, r=40, t=80)
    )
    
    logger.info(f"Created Plotly missing heatmap for {len(missing_matrix)} rows")
    return fig


# ============================================================================
# Missing Value Visualizations - Matplotlib (Static)
# ============================================================================

def create_missing_bar_chart_matplotlib(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100
) -> Optional[plt.Figure]:
    """
    Create static bar chart of missing percentages using Matplotlib.
    
    Args:
        df (pd.DataFrame): Input dataset
        figsize (Tuple[int, int]): Figure size (width, height) in inches
        dpi (int): Resolution in dots per inch
        
    Returns:
        plt.Figure: Matplotlib figure object or None if Matplotlib unavailable
        
    Example:
        >>> fig = create_missing_bar_chart_matplotlib(df, figsize=(16, 8))
        >>> fig.savefig('missing_values.png', dpi=150, bbox_inches='tight')
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Returning None.")
        return None
    
    missing_stats = compute_missing_statistics(df)
    
    # Filter columns with missing values
    missing_stats = missing_stats[missing_stats['missing_percentage'] > 0]
    
    if len(missing_stats) == 0:
        logger.info("No missing values found")
        return None
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Color mapping for severity
    colors = [_get_severity_color_rgb(sev) for sev in missing_stats['severity']]
    
    # Create bar chart
    bars = ax.bar(
        range(len(missing_stats)),
        missing_stats['missing_percentage'],
        color=colors,
        edgecolor='darkgray',
        linewidth=1.2
    )
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, missing_stats['missing_percentage'])):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 1,
            f'{pct:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    # Customize axes
    ax.set_xlabel('Column Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Missing Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Missing Values by Column', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(missing_stats)))
    ax.set_xticklabels(
        missing_stats['column'],
        rotation=45,
        ha='right',
        fontsize=10
    )
    
    # Add threshold line
    max_missing = missing_stats['missing_percentage'].max()
    if max_missing > 20:
        ax.axhline(
            y=20,
            color='orange',
            linestyle='--',
            linewidth=2,
            label='20% Threshold',
            alpha=0.7
        )
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    ax.set_ylim(0, max(missing_stats['missing_percentage'].max() * 1.15, 25))
    
    # Add legend with severity colors
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', edgecolor='darkgray', label='Low (<5%)'),
        mpatches.Patch(facecolor='#f39c12', edgecolor='darkgray', label='Medium (5-20%)'),
        mpatches.Patch(facecolor='#e74c3c', edgecolor='darkgray', label='High (>20%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    logger.info(f"Created Matplotlib missing bar chart")
    return fig


def create_missing_heatmap_matplotlib(
    df: pd.DataFrame,
    sample_rows: int = 500,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 100
) -> Optional[plt.Figure]:
    """
    Create static heatmap of missing values using Matplotlib.
    
    Args:
        df (pd.DataFrame): Input dataset
        sample_rows (int): Max rows to display (default 500)
        figsize (Tuple[int, int]): Figure size (width, height) in inches
        dpi (int): Resolution in dots per inch
        
    Returns:
        plt.Figure: Matplotlib figure object or None if Matplotlib unavailable
        
    Example:
        >>> fig = create_missing_heatmap_matplotlib(df, sample_rows=1000)
        >>> fig.savefig('missing_heatmap.png', dpi=150, bbox_inches='tight')
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available. Returning None.")
        return None
    
    # Sample rows for large datasets
    if len(df) > sample_rows:
        df_display = df.sample(n=sample_rows, random_state=42)
        title_suffix = f" (Random Sample of {sample_rows} Rows)"
    else:
        df_display = df
        title_suffix = ""
    
    # Create missing value matrix
    missing_matrix = df_display.isna().astype(int)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Custom colormap: green for present, red for missing
    cmap = LinearSegmentedColormap.from_list('missing', ['#2ecc71', '#e74c3c'])
    
    # Create heatmap
    im = ax.imshow(missing_matrix.values, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(range(len(missing_matrix.columns)))
    ax.set_yticks(range(0, len(missing_matrix), max(1, len(missing_matrix)//10)))
    
    # Set labels
    ax.set_xticklabels(missing_matrix.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels([f'Row {i}' for i in ax.get_yticks()], fontsize=9)
    
    ax.set_xlabel('Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rows', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Missing Values Heatmap{title_suffix}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Missing Status', fontsize=10, fontweight='bold')
    cbar.ax.set_yticklabels(['Present', 'Missing'])
    
    plt.tight_layout()
    
    logger.info(f"Created Matplotlib missing heatmap for {len(missing_matrix)} rows")
    return fig


# ============================================================================
# Helper Functions
# ============================================================================

def _categorize_missing_severity(percentage: float) -> str:
    """
    Categorize missing value severity.
    
    Args:
        percentage (float): Percentage of missing values
        
    Returns:
        str: Severity level
    """
    if percentage == 0:
        return 'OK'
    elif percentage < 5:
        return 'Low'
    elif percentage < 20:
        return 'Medium'
    elif percentage < 50:
        return 'High'
    else:
        return 'Critical'


def _get_severity_color(severity: str) -> str:
    """Get Plotly color for severity level."""
    color_map = {
        'OK': '#2ecc71',          # Green
        'Low': '#27ae60',         # Dark green
        'Medium': '#f39c12',      # Orange
        'High': '#e74c3c',        # Red
        'Critical': '#c0392b'     # Dark red
    }
    return color_map.get(severity, '#95a5a6')


def _get_severity_color_rgb(severity: str) -> Tuple[float, float, float]:
    """Get Matplotlib RGB color for severity level."""
    color_map = {
        'OK': (46/255, 204/255, 113/255),       # Green
        'Low': (39/255, 174/255, 96/255),       # Dark green
        'Medium': (243/255, 156/255, 18/255),   # Orange
        'High': (231/255, 76/255, 60/255),      # Red
        'Critical': (192/255, 57/255, 43/255)   # Dark red
    }
    return color_map.get(severity, (149/255, 165/255, 166/255))


def _get_missing_threshold_recommendation(count: int, threshold: float) -> str:
    """Generate recommendation based on threshold exceedances."""
    if count == 0:
        return 'All columns below threshold'
    elif count == 1:
        return f'1 column exceeds {threshold}% threshold - consider imputation or removal'
    elif count <= 3:
        return (
            f'{count} columns exceed {threshold}% threshold - '
            'recommend advanced imputation (KNN, multiple imputation)'
        )
    else:
        return (
            f'{count} columns exceed {threshold}% threshold - '
            'significant missing data detected. Consider feature selection '
            'or collecting more data'
        )


# ============================================================================
# Comprehensive Analysis Function
# ============================================================================

def analyze_missing_values(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    threshold: float = 20.0,
    create_plots: bool = True,
    plot_backend: str = 'plotly'
) -> Dict[str, Any]:
    """
    Comprehensive missing value analysis with visualizations.
    
    Main function that orchestrates all missing value analyses.
    
    Args:
        df (pd.DataFrame): Input dataset
        target_column (Optional[str]): Target column for context
        threshold (float): Missing percentage threshold for flagging (default 20%)
        create_plots (bool): Whether to create visualizations (default True)
        plot_backend (str): 'plotly' or 'matplotlib' (default 'plotly')
        
    Returns:
        Dict with:
            - statistics: Missing value statistics per column
            - above_threshold: Columns exceeding threshold
            - patterns: Missing value patterns
            - plots: Figure objects (if create_plots=True)
            - summary: Text summary of findings
            
    Example:
        >>> analysis = analyze_missing_values(df, target_column='price', threshold=15)
        >>> print(analysis['summary'])
        >>> analysis['plots']['bar_chart'].show()
    """
    logger.info("Starting comprehensive missing value analysis")
    
    # Compute statistics
    stats_df = compute_missing_statistics(df)
    above_threshold = get_columns_above_threshold(df, threshold)
    patterns = get_missing_patterns(df)
    
    result = {
        'statistics': stats_df.to_dict('records'),
        'statistics_df': stats_df,
        'above_threshold': above_threshold,
        'patterns': patterns,
        'target_column': target_column,
        'threshold': threshold
    }
    
    # Create visualizations
    if create_plots:
        plots = {}
        
        if plot_backend == 'plotly':
            plots['bar_chart'] = create_missing_bar_chart_plotly(df)
            plots['heatmap'] = create_missing_heatmap_plotly(df)
        elif plot_backend == 'matplotlib':
            plots['bar_chart'] = create_missing_bar_chart_matplotlib(df)
            plots['heatmap'] = create_missing_heatmap_matplotlib(df)
        
        result['plots'] = plots
    
    # Generate summary text
    summary = _generate_missing_summary(
        stats_df,
        above_threshold,
        patterns,
        threshold
    )
    result['summary'] = summary
    
    logger.info("Missing value analysis complete")
    return result


def _generate_missing_summary(
    stats_df: pd.DataFrame,
    above_threshold: Dict,
    patterns: Dict,
    threshold: float
) -> str:
    """Generate text summary of missing value analysis."""
    summary_lines = [
        "=" * 70,
        "MISSING VALUE ANALYSIS SUMMARY",
        "=" * 70,
        ""
    ]
    
    # Overall stats
    total_missing = stats_df['missing_count'].sum()
    total_cells = len(stats_df) * stats_df['non_missing_count'].sum()
    overall_pct = (total_missing / (total_missing + total_cells) * 100) if (total_missing + total_cells) > 0 else 0
    
    summary_lines.extend([
        f"📊 Overall Missing Data: {total_missing:,} cells ({overall_pct:.2f}%)",
        f"📋 Columns with Missing: {(stats_df['missing_count'] > 0).sum()} / {len(stats_df)}",
        ""
    ])
    
    # Above threshold
    if above_threshold['count'] > 0:
        summary_lines.extend([
            f"⚠️  ABOVE {threshold}% THRESHOLD:",
            f"   Count: {above_threshold['count']} columns"
        ])
        for col_detail in above_threshold['details'][:5]:
            summary_lines.append(
                f"   • {col_detail['column']}: {col_detail['missing_percentage']:.2f}%"
            )
        if above_threshold['count'] > 5:
            summary_lines.append(f"   ... and {above_threshold['count'] - 5} more")
        summary_lines.append("")
    
    # Patterns
    summary_lines.extend([
        "📍 Missing Value Patterns:",
        f"   Distinct patterns: {patterns['total_patterns']}",
        f"   Rows with any missing: {patterns['rows_with_any_missing']:,} "
        f"({patterns['rows_with_any_missing_percentage']:.2f}%)",
        f"   Completely missing rows: {patterns['completely_missing_rows']}"
    ])
    
    if patterns['top_patterns']:
        summary_lines.append("\n   Top patterns:")
        for pattern in patterns['top_patterns'][:3]:
            cols_str = ', '.join(pattern['pattern'][:3])
            if len(pattern['pattern']) > 3:
                cols_str += f", ... (+{len(pattern['pattern']) - 3})"
            summary_lines.append(
                f"   • {cols_str}: {pattern['frequency']} rows ({pattern['percentage']:.1f}%)"
            )
    
    summary_lines.extend([
        "",
        "=" * 70
    ])
    
    return '\n'.join(summary_lines)


if __name__ == "__main__":
    # Example usage
    sample_df = pd.DataFrame({
        'id': range(100),
        'col_no_missing': np.random.randn(100),
        'col_low_missing': [np.nan if i % 20 == 0 else i for i in range(100)],
        'col_med_missing': [np.nan if i % 5 == 0 else i for i in range(100)],
        'col_high_missing': [np.nan if i % 2 == 0 else i for i in range(100)],
    })
    
    # Run analysis
    analysis = analyze_missing_values(
        sample_df,
        target_column='col_low_missing',
        threshold=15,
        create_plots=True,
        plot_backend='plotly'
    )
    
    # Print summary
    print(analysis['summary'])
    
    # Show statistics
    print("\nDetailed Statistics:")
    print(analysis['statistics_df'].to_string())
    
    # Display plots (if available)
    if 'plots' in analysis and analysis['plots']['bar_chart']:
        print("\n✅ Plotly visualizations created successfully")
    else:
        print("\n⚠️  Plotly not available for visualization")
