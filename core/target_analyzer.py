"""
Target Analysis Utilities - FIXED VERSION

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
    Handles both numeric and string/categorical data.
    
    Args:
        y: Target variable
        
    Returns:
        TaskType with task_type, confidence, n_unique, n_samples
    """
    y_array = np.asarray(y)
    n_samples = len(y_array)
    n_unique = len(np.unique(y_array))
    
    # Check if numeric or string
    is_numeric = np.issubdtype(y_array.dtype, np.number)
    
    # String or categorical data is always classification
    if not is_numeric:
        task_type = 'classification'
        confidence = 0.95
    else:
        # Try to check if integer
        try:
            is_integer = np.all(y_array == y_array.astype(int))
        except (ValueError, TypeError):
            is_integer = False
        
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


def analyze_classification(y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Analyze classification target variable."""
    y_array = np.asarray(y)
    unique, counts = np.unique(y_array, return_counts=True)
    
    class_counts = dict(zip(unique, counts))
    total = len(y_array)
    class_dist = {str(cls): count/total*100 for cls, count in class_counts.items()}
    
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
    """Create bar plot of class distribution."""
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
    
    return None


def analyze_regression(y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
    """Analyze regression target variable."""
    y_array = np.asarray(y, dtype=float)
    
    q25 = np.percentile(y_array, 25)
    q75 = np.percentile(y_array, 75)
    iqr = q75 - q25
    
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    n_outliers = np.sum((y_array < lower_bound) | (y_array > upper_bound))
    
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
    """Create histogram of target distribution."""
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
    
    return None


def create_regression_boxplot(
    y: Union[pd.Series, np.ndarray],
    backend: str = 'plotly'
) -> Optional[Any]:
    """Create boxplot to visualize outliers."""
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
    
    return None


def analyze_target(
    y: Union[pd.Series, np.ndarray],
    task_type: Optional[str] = None,
    create_plots: bool = True,
    backend: str = 'plotly'
) -> Dict[str, Any]:
    """Comprehensive target variable analysis."""
    if task_type is None:
        task_info = detect_task_type(y)
        task_type = task_info.task_type
    
    logger.info(f"Analyzing target for {task_type} task")
    
    result = {'task_type': task_type}
    
    if task_type == 'classification':
        result['metrics'] = analyze_classification(y)
    else:
        result['metrics'] = analyze_regression(y)
    
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
