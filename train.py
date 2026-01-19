"""
Training Module

Comprehensive training pipeline for both scikit-learn and TensorFlow/Keras models.
Handles model training, validation, and training history tracking.

Author: ML Systems Engineer
Date: 2026-01-19
"""

import time
import logging
from typing import Tuple, Dict, Any, Optional, Union
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    logging.warning("TensorFlow/Keras not available. Deep learning models disabled.")

try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Training History Tracking
# ============================================================================

class TrainingHistory:
    """
    Tracks and manages training history for both sklearn and Keras models.
    
    Attributes:
        model_type (str): 'sklearn' or 'keras'
        total_time (float): Total training time in seconds
        history (Dict): Training metrics history
    """
    
    def __init__(self, model_type: str = 'sklearn'):
        """Initialize training history tracker."""
        self.model_type = model_type
        self.history = {
            'model_type': model_type,
            'start_time': None,
            'end_time': None,
            'total_time': 0.0,
            'epochs': 0,
            'metrics': {}
        }
    
    def start(self) -> None:
        """Mark the start of training."""
        self.history['start_time'] = datetime.now().isoformat()
        self.start_timestamp = time.time()
    
    def end(self) -> None:
        """Mark the end of training and calculate total time."""
        self.history['end_time'] = datetime.now().isoformat()
        self.history['total_time'] = time.time() - self.start_timestamp
    
    def add_epoch_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Add metrics for a specific epoch.
        
        Args:
            epoch (int): Epoch number (0-indexed)
            metrics (Dict): Dictionary of metric names and values
        """
        if 'epoch_metrics' not in self.history:
            self.history['epoch_metrics'] = {}
        
        self.history['epoch_metrics'][epoch] = metrics
        self.history['epochs'] = max(self.history['epochs'], epoch + 1)
    
    def add_keras_history(self, keras_history: Dict[str, list]) -> None:
        """
        Add Keras training history.
        
        Args:
            keras_history (Dict): History dict from Keras model.fit()
        """
        self.history['metrics'] = keras_history
        self.history['epochs'] = len(keras_history.get('loss', []))
    
    def add_sklearn_metrics(self, train_score: float, 
                           val_score: Optional[float] = None) -> None:
        """
        Add scikit-learn training metrics.
        
        Args:
            train_score (float): Training score
            val_score (Optional[float]): Validation score
        """
        self.history['metrics']['train_score'] = train_score
        if val_score is not None:
            self.history['metrics']['val_score'] = val_score
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training history summary."""
        return self.history
    
    def save(self, filepath: str) -> None:
        """
        Save training history to JSON file.
        
        Args:
            filepath (str): Path to save the history
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            logger.info(f"Training history saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving training history: {str(e)}")
    
    def __repr__(self) -> str:
        """String representation of training history."""
        return (
            f"TrainingHistory(model_type={self.model_type}, "
            f"total_time={self.history['total_time']:.2f}s, "
            f"epochs={self.history['epochs']})"
        )


# ============================================================================
# Scikit-learn Model Training
# ============================================================================

def train_sklearn_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[BaseEstimator, TrainingHistory]:
    """
    Train a scikit-learn model.
    
    Args:
        model (BaseEstimator): Scikit-learn model instance
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (Optional[np.ndarray]): Validation features
        y_val (Optional[np.ndarray]): Validation targets
        **kwargs: Additional parameters (unused for sklearn)
        
    Returns:
        Tuple containing:
            - Trained model
            - TrainingHistory object
            
    Raises:
        ValueError: If model is not a scikit-learn estimator
    """
    if not isinstance(model, BaseEstimator):
        raise ValueError("Model must be a scikit-learn BaseEstimator")
    
    logger.info(f"Starting scikit-learn model training...")
    logger.info(f"  Training set: {X_train.shape}")
    if X_val is not None:
        logger.info(f"  Validation set: {X_val.shape}")
    
    history = TrainingHistory(model_type='sklearn')
    history.start()
    
    try:
        # Train model
        model.fit(X_train, y_train)
        logger.info("✓ Model training completed")
        
        # Compute scores
        train_score = model.score(X_train, y_train)
        logger.info(f"  Training score: {train_score:.4f}")
        
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"  Validation score: {val_score:.4f}")
        
        history.add_sklearn_metrics(train_score, val_score)
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
    
    finally:
        history.end()
    
    logger.info(f"Total training time: {history.history['total_time']:.2f} seconds")
    return model, history


# ============================================================================
# Keras Model Training
# ============================================================================

def train_keras_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: Optional[float] = None,
    verbose: int = 1,
    **kwargs
) -> Tuple[keras.Model, TrainingHistory]:
    """
    Train a TensorFlow/Keras model.
    
    Args:
        model (keras.Model): Compiled Keras model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (Optional[np.ndarray]): Validation features
        y_val (Optional[np.ndarray]): Validation targets
        epochs (int): Number of epochs. Default: 50
        batch_size (int): Batch size. Default: 32
        validation_split (Optional[float]): Fraction of training data for validation.
                                           Default: None (uses X_val, y_val)
        verbose (int): Verbosity level (0=silent, 1=progress, 2=one line per epoch).
                      Default: 1
        **kwargs: Additional parameters for model.fit()
                 (e.g., callbacks, class_weight)
        
    Returns:
        Tuple containing:
            - Trained model
            - TrainingHistory object
            
    Raises:
        ImportError: If TensorFlow/Keras not available
        ValueError: If model is not a Keras model
    """
    if not KERAS_AVAILABLE:
        raise ImportError("TensorFlow/Keras required for neural network training")
    
    if not isinstance(model, keras.Model):
        raise ValueError("Model must be a Keras Model instance")
    
    logger.info(f"Starting Keras model training...")
    logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}")
    logger.info(f"  Training set: {X_train.shape}")
    
    if X_val is not None and y_val is not None:
        logger.info(f"  Validation set: {X_val.shape}")
    elif validation_split is not None:
        logger.info(f"  Validation split: {validation_split*100:.1f}%")
    
    history = TrainingHistory(model_type='keras')
    history.start()
    
    try:
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train model
        fit_history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split if validation_data is None else None,
            verbose=verbose,
            **kwargs
        )
        
        logger.info("✓ Model training completed")
        
        # Store history
        history.add_keras_history(fit_history.history)
        
        # Log final metrics
        final_loss = fit_history.history['loss'][-1]
        logger.info(f"  Final training loss: {final_loss:.4f}")
        
        if 'val_loss' in fit_history.history:
            final_val_loss = fit_history.history['val_loss'][-1]
            logger.info(f"  Final validation loss: {final_val_loss:.4f}")
        
        if 'accuracy' in fit_history.history:
            final_acc = fit_history.history['accuracy'][-1]
            logger.info(f"  Final training accuracy: {final_acc:.4f}")
            
            if 'val_accuracy' in fit_history.history:
                final_val_acc = fit_history.history['val_accuracy'][-1]
                logger.info(f"  Final validation accuracy: {final_val_acc:.4f}")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise
    
    finally:
        history.end()
    
    logger.info(f"Total training time: {history.history['total_time']:.2f} seconds")
    return model, history


# ============================================================================
# Unified Training Interface
# ============================================================================

def train_model(
    model: Union[BaseEstimator, keras.Model],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 50,
    batch_size: int = 32,
    validation_split: Optional[float] = None,
    verbose: int = 1,
    **kwargs
) -> Tuple[Union[BaseEstimator, keras.Model], TrainingHistory]:
    """
    Train either a scikit-learn or Keras model (auto-detection).
    
    Automatically detects model type and applies appropriate training strategy.
    
    Args:
        model: Scikit-learn estimator or Keras model
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (Optional[np.ndarray]): Validation features
        y_val (Optional[np.ndarray]): Validation targets
        epochs (int): Number of epochs (Keras only). Default: 50
        batch_size (int): Batch size (Keras only). Default: 32
        validation_split (Optional[float]): Validation split fraction (Keras only).
                                           Default: None
        verbose (int): Verbosity level (Keras only). Default: 1
        **kwargs: Model-specific parameters
        
    Returns:
        Tuple containing:
            - Trained model
            - TrainingHistory object
            
    Example:
        >>> # Scikit-learn
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> clf = RandomForestClassifier()
        >>> trained_clf, history = train_model(clf, X_train, y_train, X_val, y_val)
        
        >>> # Keras
        >>> from tensorflow.keras.models import Sequential
        >>> nn = Sequential([...])
        >>> trained_nn, history = train_model(
        ...     nn, X_train, y_train, X_val, y_val,
        ...     epochs=100, batch_size=16
        ... )
    """
    
    # Detect model type
    is_keras = False
    if KERAS_AVAILABLE and isinstance(model, keras.Model):
        is_keras = True
    elif SKLEARN_AVAILABLE and isinstance(model, BaseEstimator):
        is_keras = False
    else:
        raise ValueError(
            "Model must be either a scikit-learn BaseEstimator or Keras Model. "
            "Ensure scikit-learn and/or TensorFlow are installed."
        )
    
    # Train based on model type
    if is_keras:
        logger.info("Detected Keras model. Using neural network training pipeline.")
        return train_keras_model(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            **kwargs
        )
    else:
        logger.info("Detected scikit-learn model. Using traditional ML training pipeline.")
        return train_sklearn_model(
            model, X_train, y_train,
            X_val=X_val, y_val=y_val,
            **kwargs
        )


# ============================================================================
# Training Pipeline (End-to-End)
# ============================================================================

def train_full_pipeline(
    model: Union[BaseEstimator, keras.Model],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_model: Optional[str] = None,
    save_history: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
    **kwargs
) -> Dict[str, Any]:
    """
    Complete training pipeline with train/val/test evaluation.
    
    Args:
        model: Model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        save_model (Optional[str]): Path to save trained model
        save_history (Optional[str]): Path to save training history
        epochs (int): Number of epochs (Keras). Default: 50
        batch_size (int): Batch size (Keras). Default: 32
        **kwargs: Additional training parameters
        
    Returns:
        Dict containing:
            - 'model': Trained model
            - 'history': TrainingHistory object
            - 'train_score': Training score
            - 'val_score': Validation score
            - 'test_score': Test score
            
    Example:
        >>> results = train_full_pipeline(
        ...     model, X_train, y_train, X_val, y_val, X_test, y_test,
        ...     save_model='trained_model.pkl',
        ...     save_history='training_history.json'
        ... )
    """
    logger.info("\n" + "="*70)
    logger.info("STARTING FULL TRAINING PIPELINE")
    logger.info("="*70)
    
    # Train model
    trained_model, history = train_model(
        model, X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        **kwargs
    )
    
    # Evaluate on all sets
    logger.info("\n" + "-"*70)
    logger.info("EVALUATING MODEL ON ALL DATASETS")
    logger.info("-"*70)
    
    is_keras = KERAS_AVAILABLE and isinstance(trained_model, keras.Model)
    
    if is_keras:
        train_score = trained_model.evaluate(X_train, y_train, verbose=0)[1]  # accuracy
        val_score = trained_model.evaluate(X_val, y_val, verbose=0)[1]
        test_score = trained_model.evaluate(X_test, y_test, verbose=0)[1]
    else:
        train_score = trained_model.score(X_train, y_train)
        val_score = trained_model.score(X_val, y_val)
        test_score = trained_model.score(X_test, y_test)
    
    logger.info(f"Train Score: {train_score:.4f}")
    logger.info(f"Validation Score: {val_score:.4f}")
    logger.info(f"Test Score: {test_score:.4f}")
    
    # Save model if requested
    if save_model:
        try:
            save_path = Path(save_model)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            if is_keras:
                trained_model.save(save_model)
            else:
                import joblib
                joblib.dump(trained_model, save_model)
            
            logger.info(f"\n✓ Model saved to {save_model}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    # Save history if requested
    if save_history:
        history.save(save_history)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*70 + "\n")
    
    return {
        'model': trained_model,
        'history': history,
        'train_score': train_score,
        'val_score': val_score,
        'test_score': test_score,
    }


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Training Module - ML/DL Model Training System")
    print("=" * 70)
    
    print("\nUsage Examples:")
    print("""
    from train import train_model, train_full_pipeline
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    # Generate dummy data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(30, 10)
    y_val = np.random.randint(0, 2, 30)
    X_test = np.random.randn(30, 10)
    y_test = np.random.randint(0, 2, 30)
    
    # Train scikit-learn model
    clf = RandomForestClassifier(n_estimators=100)
    trained_model, history = train_model(
        clf, X_train, y_train,
        X_val=X_val, y_val=y_val
    )
    
    # Full pipeline with evaluation
    results = train_full_pipeline(
        clf, X_train, y_train, X_val, y_val, X_test, y_test,
        save_model='models/trained_model.pkl',
        save_history='logs/training_history.json'
    )
    
    print(f"Test Score: {results['test_score']:.4f}")
    
    # For Keras models (if TensorFlow installed)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    nn = Sequential([
        Dense(64, activation='relu', input_dim=10),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    trained_nn, nn_history = train_model(
        nn, X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=50,
        batch_size=16
    )
    """)
    
    print("\n" + "=" * 70)
