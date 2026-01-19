"""
Visualization module for plots and charts
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from app.utils.logger import logger

class Visualizer:
    """Generate visualizations"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    @staticmethod
    def plot_feature_importance(feature_names, importances, save_path=None):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        sorted_idx = np.argsort(importances)[-20:]  # Top 20
        
        plt.figure(figsize=(10, 6))
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importance')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return plt.gcf()
    
    @staticmethod
    def plot_regression_residuals(y_true, y_pred, save_path=None):
        """
        Plot regression residuals
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_ylabel('Residuals')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_title('Residuals vs Predicted')
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_title('Residuals Distribution')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
        
        return fig
