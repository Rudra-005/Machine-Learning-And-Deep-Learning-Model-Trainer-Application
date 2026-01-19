"""
Model Evaluation Module

Comprehensive evaluation framework for ML/DL models.
Automatically computes appropriate metrics based on task type and provides visualizations.

Author: ML Evaluation Specialist
Date: 2026-01-19
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve,
        mean_absolute_error, mean_squared_error, r2_score,
        precision_recall_curve, auc
    )
    SKLEARN_METRICS_AVAILABLE = True
except ImportError:
    SKLEARN_METRICS_AVAILABLE = False
    logging.warning("Scikit-learn metrics not available.")

try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


# ============================================================================
# Classification Metrics
# ============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (Optional[np.ndarray]): Predicted probabilities (for AUC)
        average (str): Averaging method for multi-class ('weighted', 'macro', 'micro').
                      Default: 'weighted'
        
    Returns:
        Dict containing classification metrics
        
    Raises:
        ValueError: If inputs invalid or sklearn not available
    """
    if not SKLEARN_METRICS_AVAILABLE:
        raise ImportError("Scikit-learn required for classification metrics")
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    logger.info("Computing classification metrics...")
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1']:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    # ROC-AUC for binary/multi-class classification
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 2 and y_pred_proba is not None:
        try:
            # Binary classification
            if y_pred_proba.ndim == 1:
                proba = y_pred_proba
            else:
                proba = y_pred_proba[:, 1]
            
            metrics['roc_auc'] = float(roc_auc_score(y_true, proba))
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {str(e)}")
    elif len(unique_classes) > 2 and y_pred_proba is not None:
        try:
            # Multi-class classification
            metrics['roc_auc'] = float(roc_auc_score(
                y_true, y_pred_proba, multi_class='ovr', average='weighted'
            ))
            logger.info(f"  ROC-AUC (One-vs-Rest): {metrics['roc_auc']:.4f}")
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {str(e)}")
    
    # Classification report
    try:
        class_report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        metrics['classification_report'] = class_report
    except Exception as e:
        logger.warning(f"Could not compute classification report: {str(e)}")
    
    logger.info("✓ Classification metrics computed successfully")
    return metrics


# ============================================================================
# Regression Metrics
# ============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict containing regression metrics
        
    Raises:
        ValueError: If inputs invalid or sklearn not available
    """
    if not SKLEARN_METRICS_AVAILABLE:
        raise ImportError("Scikit-learn required for regression metrics")
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    logger.info("Computing regression metrics...")
    
    metrics = {}
    
    # Mean Absolute Error
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    
    # Mean Squared Error
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    logger.info(f"  MSE: {metrics['mse']:.4f}")
    
    # Root Mean Squared Error
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    
    # R² Score
    metrics['r2'] = float(r2_score(y_true, y_pred))
    logger.info(f"  R² Score: {metrics['r2']:.4f}")
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics['mape'] = float(mape)
        logger.info(f"  MAPE: {metrics['mape']:.4f}%")
    
    logger.info("✓ Regression metrics computed successfully")
    return metrics


# ============================================================================
# Unified Evaluation Interface
# ============================================================================

def evaluate_model(
    model: Union[Any, keras.Model],
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    threshold: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified evaluation interface for both sklearn and Keras models.
    
    Args:
        model: Trained sklearn model or Keras model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels/values
        task_type (str): 'classification' or 'regression'
        threshold (Optional[float]): Classification threshold (for binary classification).
                                    Default: 0.5
        **kwargs: Additional parameters
        
    Returns:
        Dict containing computed metrics
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np
        
        >>> clf = RandomForestClassifier()
        >>> clf.fit(X_train, y_train)
        
        >>> metrics = evaluate_model(clf, X_test, y_test, 'classification')
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    
    logger.info("\n" + "="*70)
    logger.info(f"EVALUATING MODEL - Task Type: {task_type.upper()}")
    logger.info("="*70)
    
    if task_type not in ['classification', 'regression']:
        raise ValueError("task_type must be 'classification' or 'regression'")
    
    # Determine if Keras model
    is_keras = KERAS_AVAILABLE and isinstance(model, keras.Model)
    
    # Get predictions
    if is_keras:
        y_pred_raw = model.predict(X_test, verbose=0)
        
        if task_type == 'classification':
            # For classification
            if y_pred_raw.shape[1] == 1:  # Binary classification
                y_pred = (y_pred_raw.flatten() > (threshold or 0.5)).astype(int)
                y_pred_proba = y_pred_raw.flatten()
            else:  # Multi-class
                y_pred = np.argmax(y_pred_raw, axis=1)
                y_pred_proba = y_pred_raw
        else:
            # For regression
            y_pred = y_pred_raw.flatten()
            y_pred_proba = None
    else:
        # Sklearn model
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            # Try to get probabilities
            try:
                y_pred_proba = model.predict_proba(X_test)
            except (AttributeError, NotImplementedError):
                y_pred_proba = None
        else:
            y_pred_proba = None
    
    # Compute metrics based on task type
    if task_type == 'classification':
        metrics = compute_classification_metrics(
            y_test, y_pred,
            y_pred_proba=y_pred_proba,
            **kwargs
        )
    else:
        metrics = compute_regression_metrics(y_test, y_pred)
    
    # Add predictions to metrics for visualization
    metrics['y_true'] = y_test.tolist() if isinstance(y_test, np.ndarray) else y_test
    metrics['y_pred'] = y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
    
    logger.info("\n" + "="*70)
    return metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[list] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        classes (Optional[list]): Class labels
        save_path (Optional[str]): Path to save figure
        figsize (Tuple): Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not SKLEARN_METRICS_AVAILABLE:
        raise ImportError("Scikit-learn required for confusion matrix")
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true (np.ndarray): True labels (binary)
        y_pred_proba (np.ndarray): Predicted probabilities for positive class
        save_path (Optional[str]): Path to save figure
        figsize (Tuple): Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not SKLEARN_METRICS_AVAILABLE:
        raise ImportError("Scikit-learn required for ROC curve")
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve plot saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true (np.ndarray): True labels (binary)
        y_pred_proba (np.ndarray): Predicted probabilities for positive class
        save_path (Optional[str]): Path to save figure
        figsize (Tuple): Figure size
        
    Returns:
        matplotlib Figure object
    """
    if not SKLEARN_METRICS_AVAILABLE:
        raise ImportError("Scikit-learn required for Precision-Recall curve")
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'Precision-Recall curve (AUC = {pr_auc:.4f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve plot saved to {save_path}")
    
    return fig


def plot_regression_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot regression results (predictions vs actual, residuals).
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        save_path (Optional[str]): Path to save figure
        figsize (Tuple): Figure size
        
    Returns:
        matplotlib Figure object
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Predictions vs Actual
    axes[0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predictions vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    # Residuals histogram
    axes[2].hist(residuals, bins=20, edgecolor='k', alpha=0.7)
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    axes[2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Regression results plot saved to {save_path}")
    
    return fig


def plot_metrics_summary(
    metrics: Dict[str, Any],
    task_type: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot summary of key metrics.
    
    Args:
        metrics (Dict): Metrics dictionary from evaluate_model()
        task_type (str): 'classification' or 'regression'
        save_path (Optional[str]): Path to save figure
        figsize (Tuple): Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if task_type == 'classification':
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = [metrics.get(m, 0) for m in metric_names]
    else:
        metric_names = ['r2', 'mae', 'mape']
        metric_values = [metrics.get(m, 0) for m in metric_names]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='k', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Metrics - {task_type.capitalize()}', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1] if task_type == 'classification' else [None, None])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics summary plot saved to {save_path}")
    
    return fig


# ============================================================================
# Evaluation Report
# ============================================================================

def generate_evaluation_report(
    metrics: Dict[str, Any],
    task_type: str,
    model_name: str = "Model",
    save_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive text report of model evaluation.
    
    Args:
        metrics (Dict): Metrics dictionary from evaluate_model()
        task_type (str): 'classification' or 'regression'
        model_name (str): Name of the model
        save_path (Optional[str]): Path to save report
        
    Returns:
        str: Formatted evaluation report
    """
    report = []
    report.append("\n" + "="*70)
    report.append(f"MODEL EVALUATION REPORT - {model_name}")
    report.append("="*70)
    report.append(f"Task Type: {task_type.upper()}")
    report.append("")
    
    if task_type == 'classification':
        report.append("-"*70)
        report.append("CLASSIFICATION METRICS")
        report.append("-"*70)
        report.append(f"Accuracy:  {metrics.get('accuracy', 'N/A'):>10.4f}")
        report.append(f"Precision: {metrics.get('precision', 'N/A'):>10.4f}")
        report.append(f"Recall:    {metrics.get('recall', 'N/A'):>10.4f}")
        report.append(f"F1-Score:  {metrics.get('f1', 'N/A'):>10.4f}")
        
        if 'roc_auc' in metrics:
            report.append(f"ROC-AUC:   {metrics['roc_auc']:>10.4f}")
        
        if 'confusion_matrix' in metrics:
            report.append("\nConfusion Matrix:")
            cm = np.array(metrics['confusion_matrix'])
            report.append(str(cm))
    
    else:
        report.append("-"*70)
        report.append("REGRESSION METRICS")
        report.append("-"*70)
        report.append(f"MAE:  {metrics.get('mae', 'N/A'):>10.4f}")
        report.append(f"MSE:  {metrics.get('mse', 'N/A'):>10.4f}")
        report.append(f"RMSE: {metrics.get('rmse', 'N/A'):>10.4f}")
        report.append(f"R²:   {metrics.get('r2', 'N/A'):>10.4f}")
        
        if 'mape' in metrics:
            report.append(f"MAPE: {metrics['mape']:>10.2f}%")
    
    report.append("\n" + "="*70)
    
    report_text = "\n".join(report)
    
    if save_path:
        try:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
    
    print(report_text)
    return report_text


def save_metrics_json(
    metrics: Dict[str, Any],
    save_path: str
) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics (Dict): Metrics dictionary
        save_path (str): Path to save JSON file
    """
    try:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metrics_json[key] = float(value)
            else:
                metrics_json[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Metrics saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Model Evaluation Module - ML Model Evaluation System")
    print("=" * 70)
    
    print("\nUsage Examples:")
    print("""
    from evaluate import evaluate_model, plot_confusion_matrix, plot_regression_results
    from evaluate import generate_evaluation_report, save_metrics_json
    import numpy as np
    
    # Example 1: Classification
    from sklearn.ensemble import RandomForestClassifier
    
    X_test = np.random.randn(50, 10)
    y_test = np.random.randint(0, 2, 50)
    
    clf = RandomForestClassifier(n_estimators=100)
    # ... train model ...
    
    metrics = evaluate_model(clf, X_test, y_test, task_type='classification')
    
    # Visualizations
    plot_confusion_matrix(y_test, np.array(metrics['y_pred']),
                         save_path='confusion_matrix.png')
    plot_roc_curve(y_test, clf.predict_proba(X_test)[:, 1],
                   save_path='roc_curve.png')
    
    # Report
    generate_evaluation_report(metrics, 'classification', 'RandomForest',
                             save_path='evaluation_report.txt')
    save_metrics_json(metrics, 'metrics.json')
    
    # Example 2: Regression
    from sklearn.ensemble import RandomForestRegressor
    
    X_test = np.random.randn(50, 10)
    y_test = np.random.randn(50)
    
    reg = RandomForestRegressor(n_estimators=100)
    # ... train model ...
    
    metrics = evaluate_model(reg, X_test, y_test, task_type='regression')
    
    # Visualizations
    plot_regression_results(y_test, np.array(metrics['y_pred']),
                           save_path='regression_results.png')
    
    # Report
    generate_evaluation_report(metrics, 'regression', 'RandomForest',
                             save_path='evaluation_report.txt')
    """)
    
    print("\n" + "=" * 70)
