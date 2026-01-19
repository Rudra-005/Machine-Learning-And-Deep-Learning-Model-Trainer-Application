"""
Target Analysis Utilities

Automatically detect task type and provide appropriate analysis:
- Classification: class distribution, imbalance ratio, bar plots
- Regression: distribution histogram, boxplot, statistics

Clean separation from model evaluation logic.
"""

from typing import Dict, Tuple, Optional, Any, Union
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
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TaskType:
    """Task type classification result."""
    task_type: str  # 'classification' or 'regression'
    confidence: float
    n_unique: int
    n_samples: int


def detect_task_type(y: Union[pd.Series, np.ndarray]) -> TaskType:
    """
    Automatically detect if task is classification or regression.
    
    Args:
        y: Target variable
        
    Returns:
        TaskType with task_type, confidence, n_unique, n_samples
        
    Example:
        >>> task = detect_task_type(y)
        >>> print(f"Task: {task.task_type}, Confidence: {task.confidence:.2f}")
    """
    y_array = np.asarray(y)
    n_samples = len(y_array)
    n_unique = len(np.unique(y_array))
    
    # Classification heuristics
    is_integer = np.all(y_array == y_array.astype(int))
    is_binary = n_unique == 2
    is_small_unique = n_unique <= np.sqrt(n_samples)
    
    if is_binary or (is_integer and is_small_unique):
        task_type = 'classification'
        confidence = 0.95 if is_binary else 0.85
    else:
        task_type = 'regression'
        confidence = 0.95 if not is_integer else 0.75
    
    logger.info(f"Detected task: {task_type} (confidence: {confidence:.2f})")
    
    return TaskType(
        task_type=task_type,
        confidence=confidence,
        n_unique=n_unique,
        n_samples=n_samples
    )


# ============================================================================
# Classification Analysis
# ============================================================================

def analyze_classification(y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze classification target variable.
    
    Args:
        y: Target variable
        
    Returns:
        Dict with:
            - class_counts: Count per class
            - class_distribution: Percentage per class
            - imbalance_ratio: Max/min class ratio
            - n_classes: Number of classes
            - majority_class: Most common class
            - minority_class: Least common class
            
    Example:
        >>> metrics = analyze_classification(y)
        >>> print(f"Classes: {metrics['n_classes']}")
        >>> print(f"Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    """
    y_array = np.asarray(y)
    unique, counts = np.unique(y_array, return_counts=True)
    
    class_counts = dict(zip(unique, counts))
    total = len(y_array)
    class_dist = {cls: count/total*100 for cls, count in class_counts.items()}
    
    min_count = min(counts)
    max_count = max(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'class_counts': class_counts,
        'class_distribution': class_dist,
        'imbalance_ratio': round(imbalance_ratio, 2),
        'n_classes': len(unique),
        'majority_class': unique[np.argmax(counts)],
        'minority_class': unique[np.argmin(counts)],
        'is_imbalanced': imbalance_ratio > 1.5
    }


def create_class_distribution_plot(
    y: Union[pd.Series, np.ndarray],
    backend: str = 'plotly'
) -> Optional[Any]:
    """
    Create bar plot of class distribution.
    
    Args:
        y: Target variable
        backend: 'plotly' or 'matplotlib'
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = create_class_distribution_plot(y, backend='plotly')
        >>> fig.show()
    """
    y_array = np.asarray(y)
    unique, counts = np.unique(y_array, return_counts=True)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=[str(c) for c in unique],
                y=counts,
                text=counts,
                textposition='outside',
                marker=dict(color='steelblue', line=dict(color='darkblue', width=1)),
                hovertemplate='<b>Class %{x}</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='<b>Class Distribution</b>',
            xaxis_title='<b>Class</b>',
            yaxis_title='<b>Count</b>',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=80, l=80, r=40, t=80)
        )
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        bars = ax.bar([str(c) for c in unique], counts, color='steelblue', edgecolor='darkblue', linewidth=1.2)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


# ============================================================================
# Regression Analysis
# ============================================================================

def analyze_regression(y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """
    Analyze regression target variable.
    
    Args:
        y: Target variable
        
    Returns:
        Dict with:
            - mean: Mean value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - median: Median value
            - q25: 25th percentile
            - q75: 75th percentile
            - skewness: Distribution skewness
            - kurtosis: Distribution kurtosis
            - n_outliers: Number of outliers (IQR method)
            
    Example:
        >>> metrics = analyze_regression(y)
        >>> print(f"Mean: {metrics['mean']:.2f}")
        >>> print(f"Skewness: {metrics['skewness']:.2f}")
    """
    y_array = np.asarray(y, dtype=float)
    
    q25 = np.percentile(y_array, 25)
    q75 = np.percentile(y_array, 75)
    iqr = q75 - q25
    
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    n_outliers = np.sum((y_array < lower_bound) | (y_array > upper_bound))
    
    # Skewness and kurtosis
    mean = np.mean(y_array)
    std = np.std(y_array)
    skewness = np.mean(((y_array - mean) / std) ** 3) if std > 0 else 0
    kurtosis = np.mean(((y_array - mean) / std) ** 4) - 3 if std > 0 else 0
    
    return {
        'mean': round(float(np.mean(y_array)), 4),
        'std': round(float(np.std(y_array)), 4),
        'min': round(float(np.min(y_array)), 4),
        'max': round(float(np.max(y_array)), 4),
        'median': round(float(np.median(y_array)), 4),
        'q25': round(float(q25), 4),
        'q75': round(float(q75), 4),
        'iqr': round(float(iqr), 4),
        'skewness': round(float(skewness), 4),
        'kurtosis': round(float(kurtosis), 4),
        'n_outliers': int(n_outliers),
        'outlier_percentage': round(n_outliers / len(y_array) * 100, 2)
    }


def create_regression_histogram(
    y: Union[pd.Series, np.ndarray],
    backend: str = 'plotly',
    bins: int = 30
) -> Optional[Any]:
    """
    Create histogram of target distribution.
    
    Args:
        y: Target variable
        backend: 'plotly' or 'matplotlib'
        bins: Number of bins
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = create_regression_histogram(y, backend='plotly')
        >>> fig.show()
    """
    y_array = np.asarray(y, dtype=float)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Histogram(
                x=y_array,
                nbinsx=bins,
                marker=dict(color='steelblue', line=dict(color='darkblue', width=1)),
                hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='<b>Target Distribution</b>',
            xaxis_title='<b>Value</b>',
            yaxis_title='<b>Frequency</b>',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=80, l=80, r=40, t=80)
        )
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        ax.hist(y_array, bins=bins, color='steelblue', edgecolor='darkblue', alpha=0.7)
        
        ax.set_xlabel('Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Target Distribution', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


def create_regression_boxplot(
    y: Union[pd.Series, np.ndarray],
    backend: str = 'plotly'
) -> Optional[Any]:
    """
    Create boxplot to visualize outliers.
    
    Args:
        y: Target variable
        backend: 'plotly' or 'matplotlib'
        
    Returns:
        Figure object or None
        
    Example:
        >>> fig = create_regression_boxplot(y, backend='plotly')
        >>> fig.show()
    """
    y_array = np.asarray(y, dtype=float)
    
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Box(
                y=y_array,
                name='Target',
                marker=dict(color='steelblue'),
                boxmean='sd',
                hovertemplate='Value: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='<b>Target Distribution (Boxplot)</b>',
            yaxis_title='<b>Value</b>',
            height=500,
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(b=80, l=80, r=40, t=80)
        )
        return fig
    
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
        
        bp = ax.boxplot(y_array, vert=True, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        
        for whisker in bp['whiskers']:
            whisker.set(color='darkblue', linewidth=1.5)
        
        for cap in bp['caps']:
            cap.set(color='darkblue', linewidth=1.5)
        
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title('Target Distribution (Boxplot)', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig
    
    return None


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_target(
    y: Union[pd.Series, np.ndarray],
    task_type: Optional[str] = None,
    create_plots: bool = True,
    backend: str = 'plotly'
) -> Dict[str, Any]:
    """
    Comprehensive target variable analysis.
    
    Automatically detects task type and provides appropriate metrics and plots.
    
    Args:
        y: Target variable
        task_type: 'classification' or 'regression' (auto-detected if None)
        create_plots: Whether to create visualizations
        backend: 'plotly' or 'matplotlib'
        
    Returns:
        Dict with:
            - task_type: Detected task type
            - metrics: Task-specific metrics
            - plots: Figure objects (if create_plots=True)
            
    Example:
        >>> analysis = analyze_target(y)
        >>> print(f"Task: {analysis['task_type']}")
        >>> print(analysis['metrics'])
        >>> analysis['plots']['distribution'].show()
    """
    # Detect task type
    if task_type is None:
        task_info = detect_task_type(y)
        task_type = task_info.task_type
    
    logger.info(f"Analyzing target for {task_type} task")
    
    result = {'task_type': task_type}
    
    # Get metrics
    if task_type == 'classification':
        result['metrics'] = analyze_classification(y)
    else:
        result['metrics'] = analyze_regression(y)
    
    # Create plots
    if create_plots:
        plots = {}
        
        if task_type == 'classification':
            plots['distribution'] = create_class_distribution_plot(y, backend=backend)
        else:
            plots['histogram'] = create_regression_histogram(y, backend=backend)
            plots['boxplot'] = create_regression_boxplot(y, backend=backend)
        
        result['plots'] = plots
    
    logger.info(f"Target analysis complete")
    return result


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Classification example
    y_class = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
    analysis_class = analyze_target(y_class)
    print("Classification Analysis:")
    print(f"  Task: {analysis_class['task_type']}")
    print(f"  Metrics: {analysis_class['metrics']}")
    
    # Regression example
    y_reg = np.random.normal(100, 20, 1000)
    analysis_reg = analyze_target(y_reg)
    print("\nRegression Analysis:")
    print(f"  Task: {analysis_reg['task_type']}")
    print(f"  Metrics: {analysis_reg['metrics']}")
