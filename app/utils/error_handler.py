"""
Error handling and custom exceptions for ML/DL Trainer
"""
import logging
import traceback
from typing import Optional, Callable, Any
import streamlit as st
from functools import wraps

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class MLTrainerException(Exception):
    """Base exception for ML Trainer."""
    pass


class DataValidationError(MLTrainerException):
    """Raised when data validation fails."""
    pass


class TargetSelectionError(MLTrainerException):
    """Raised when target selection is invalid."""
    pass


class ModelTrainingError(MLTrainerException):
    """Raised when model training fails."""
    pass


class MemoryError(MLTrainerException):
    """Raised when memory limit exceeded."""
    pass


class PreprocessingError(MLTrainerException):
    """Raised when preprocessing fails."""
    pass


# ============================================================================
# Error Handlers
# ============================================================================

class ErrorHandler:
    """Centralized error handling for the application."""
    
    @staticmethod
    def handle_data_validation(data) -> bool:
        """Validate dataset and return True if valid."""
        try:
            if data is None or len(data) == 0:
                raise DataValidationError("Dataset is empty")
            
            if len(data.columns) < 2:
                raise DataValidationError("Dataset must have at least 2 columns")
            
            if len(data) < 10:
                raise DataValidationError("Dataset must have at least 10 rows")
            
            logger.info(f"Data validation passed: {len(data)} rows, {len(data.columns)} columns")
            return True
            
        except DataValidationError as e:
            logger.error(f"Data validation error: {str(e)}")
            st.error(f"❌ **Data Validation Error**\n\n{str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during data validation: {str(e)}\n{traceback.format_exc()}")
            st.error(f"❌ **Unexpected Error**\n\nFailed to validate data: {str(e)}")
            return False
    
    @staticmethod
    def handle_target_selection(data, target_col: str, task_type: str) -> bool:
        """Validate target column selection."""
        try:
            if target_col not in data.columns:
                raise TargetSelectionError(f"Target column '{target_col}' not found in dataset")
            
            target = data[target_col]
            
            # Check for missing values
            if target.isna().all():
                raise TargetSelectionError(f"Target column '{target_col}' is completely empty")
            
            missing_pct = (target.isna().sum() / len(target)) * 100
            if missing_pct > 50:
                raise TargetSelectionError(f"Target column has {missing_pct:.1f}% missing values")
            
            # Task-specific validation
            unique_count = target.nunique()
            
            if task_type == "Classification":
                if unique_count < 2:
                    raise TargetSelectionError(
                        f"Classification requires at least 2 unique values. Found: {unique_count}"
                    )
                if unique_count > 100:
                    raise TargetSelectionError(
                        f"Too many classes ({unique_count}). Consider regression instead."
                    )
            else:  # Regression
                try:
                    target_numeric = pd.to_numeric(target.dropna(), errors='coerce')
                    if target_numeric.isna().all():
                        raise TargetSelectionError("Target column contains non-numeric values for regression")
                except Exception as e:
                    raise TargetSelectionError(f"Cannot convert target to numeric: {str(e)}")
            
            logger.info(f"Target validation passed: {target_col} ({task_type})")
            return True
            
        except TargetSelectionError as e:
            logger.error(f"Target selection error: {str(e)}")
            st.error(f"❌ **Target Selection Error**\n\n{str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during target validation: {str(e)}\n{traceback.format_exc()}")
            st.error(f"❌ **Unexpected Error**\n\nFailed to validate target: {str(e)}")
            return False
    
    @staticmethod
    def handle_model_training(train_func: Callable) -> Callable:
        """Decorator for model training with error handling."""
        @wraps(train_func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Starting model training: {train_func.__name__}")
                result = train_func(*args, **kwargs)
                logger.info(f"Model training completed successfully")
                return result
                
            except MemoryError as e:
                logger.error(f"Memory error during training: {str(e)}")
                st.error(
                    f"❌ **Memory Error**\n\n"
                    f"Insufficient memory for training.\n\n"
                    f"**Solutions:**\n"
                    f"1. Use a smaller dataset\n"
                    f"2. Reduce batch size\n"
                    f"3. Close other applications"
                )
                return None
                
            except ModelTrainingError as e:
                logger.error(f"Model training error: {str(e)}")
                st.error(f"❌ **Training Error**\n\n{str(e)}")
                return None
                
            except ValueError as e:
                error_msg = str(e)
                logger.error(f"Value error during training: {error_msg}")
                
                if "class" in error_msg.lower():
                    st.error(
                        f"❌ **Class Distribution Error**\n\n"
                        f"Issue with class distribution in data.\n\n"
                        f"**Solutions:**\n"
                        f"1. Check for missing values in target\n"
                        f"2. Ensure at least 2 classes exist\n"
                        f"3. Try a different train-test split"
                    )
                else:
                    st.error(f"❌ **Training Error**\n\n{error_msg}")
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error during training: {str(e)}\n{traceback.format_exc()}")
                st.error(
                    f"❌ **Unexpected Training Error**\n\n"
                    f"An unexpected error occurred during training.\n\n"
                    f"**Troubleshooting:**\n"
                    f"1. Check data quality in EDA tab\n"
                    f"2. Verify target column selection\n"
                    f"3. Try with sample dataset\n"
                    f"4. Check application logs"
                )
                return None
        
        return wrapper
    
    @staticmethod
    def handle_preprocessing(preprocess_func: Callable) -> Callable:
        """Decorator for preprocessing with error handling."""
        @wraps(preprocess_func)
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"Starting preprocessing: {preprocess_func.__name__}")
                result = preprocess_func(*args, **kwargs)
                logger.info(f"Preprocessing completed successfully")
                return result
                
            except PreprocessingError as e:
                logger.error(f"Preprocessing error: {str(e)}")
                st.error(f"❌ **Preprocessing Error**\n\n{str(e)}")
                return None
                
            except Exception as e:
                logger.error(f"Unexpected error during preprocessing: {str(e)}\n{traceback.format_exc()}")
                st.error(f"❌ **Preprocessing Error**\n\n{str(e)}")
                return None
        
        return wrapper
    
    @staticmethod
    def handle_eda_operation(eda_func: Callable) -> Callable:
        """Decorator for EDA operations with error handling."""
        @wraps(eda_func)
        def wrapper(*args, **kwargs):
            try:
                return eda_func(*args, **kwargs)
                
            except Exception as e:
                logger.warning(f"EDA operation warning: {str(e)}")
                # Don't show error to user for EDA, just log it
                return None
        
        return wrapper


# ============================================================================
# Safe Execution Wrapper
# ============================================================================

def safe_execute(func: Callable, *args, **kwargs) -> Optional[Any]:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Function result or None if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
        return None


# ============================================================================
# Memory Monitoring
# ============================================================================

import psutil
import os

class MemoryMonitor:
    """Monitor memory usage and prevent out-of-memory errors."""
    
    MEMORY_THRESHOLD = 0.9  # 90% of available memory
    
    @staticmethod
    def check_memory() -> bool:
        """Check if memory usage is within limits."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            
            if memory_percent > MemoryMonitor.MEMORY_THRESHOLD:
                logger.warning(f"High memory usage: {memory_percent*100:.1f}%")
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Could not check memory: {str(e)}")
            return True
    
    @staticmethod
    def get_memory_info() -> dict:
        """Get current memory information."""
        try:
            vm = psutil.virtual_memory()
            return {
                'total': vm.total / (1024**3),  # GB
                'available': vm.available / (1024**3),
                'percent': vm.percent,
                'used': vm.used / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Could not get memory info: {str(e)}")
            return {}


# ============================================================================
# Import pandas for type checking
# ============================================================================

import pandas as pd
