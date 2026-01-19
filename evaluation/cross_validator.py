"""
Cross-validation utilities
"""
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import pandas as pd
from app.utils.logger import logger

class CrossValidator:
    """Handle cross-validation and data splitting"""
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, task_type='classification', random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            task_type: 'classification' or 'regression'
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if task_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=random_state
            )
            logger.info(f"Stratified split: train={len(X_train)}, test={len(X_test)}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            logger.info(f"Random split: train={len(X_train)}, test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def kfold_split(X, y, n_splits=5, task_type='classification', random_state=42):
        """
        Generate k-fold splits
        
        Args:
            X: Features
            y: Target
            n_splits: Number of folds
            task_type: 'classification' or 'regression'
            random_state: Random seed
            
        Returns:
            Generator of (train_idx, test_idx)
        """
        if task_type == 'classification':
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        return kf.split(X, y)
