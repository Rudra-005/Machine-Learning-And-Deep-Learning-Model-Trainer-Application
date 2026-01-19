"""
Metrics calculation for classification and regression
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, 
    mean_absolute_error, r2_score
)
import numpy as np
from app.utils.logger import logger

class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def classification_metrics(y_true, y_pred, y_pred_proba=None, average='weighted'):
        """
        Calculate classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for ROC-AUC)
            average: Averaging method for multi-class
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        # ROC-AUC for binary or one-vs-rest
        try:
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
        
        logger.info(f"Classification metrics: Accuracy={metrics['accuracy']:.4f}")
        return metrics
    
    @staticmethod
    def regression_metrics(y_true, y_pred):
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
        }
        
        logger.info(f"Regression metrics: RMSE={rmse:.4f}, R²={r2:.4f}")
        return metrics
