"""
AutoML Training Orchestrator: Execute optimal training strategy for any model.

Automatically applies K-Fold CV, epochs, or hyperparameter tuning based on model type.
"""

from typing import Any, Dict, Tuple
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import logging

from models.automl import AutoMLConfig, should_use_cv, should_use_epochs

logger = logging.getLogger(__name__)


class AutoMLTrainer:
    """Orchestrates training with automatic strategy selection."""
    
    def __init__(self, model: Any):
        """Initialize with model."""
        self.model = model
        self.automl = AutoMLConfig(model)
        self.config = self.automl.config
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Train model with optimal strategy.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            params: User parameters
            
        Returns:
            Training results dictionary
        """
        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train cannot be None")
        
        final_params = self.automl.get_training_params(params)
        
        if should_use_cv(self.model):
            return self._train_with_cv(X_train, y_train, X_test, y_test, final_params)
        elif should_use_epochs(self.model):
            return self._train_with_epochs(X_train, y_train, X_test, y_test, final_params)
        else:
            raise ValueError(f"Unknown training strategy for {self.config['model_name']}")
    
    def _train_with_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train with K-Fold cross-validation."""
        cv_folds = params.get('cv_folds', 5)
        
        # Determine if classification or regression
        is_classification = len(np.unique(y_train)) < 20
        
        # Setup cross-validator
        if is_classification:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Apply max_iter if needed
        if params.get('use_max_iter') and hasattr(self.model, 'max_iter'):
            self.model.max_iter = params.get('max_iter', 1000)
        
        # Hyperparameter tuning if enabled
        if params.get('enable_hp_tuning'):
            return self._tune_hyperparameters(
                X_train, y_train, X_test, y_test, cv, params, is_classification
            )
        
        # Standard CV
        scoring = 'accuracy' if is_classification else 'r2'
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
        
        # Train on full training set
        self.model.fit(X_train, y_train)
        
        # Test score
        test_score = self.model.score(X_test, y_test) if X_test is not None else None
        
        return {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'test_score': test_score,
            'strategy': 'k_fold_cv',
            'hp_tuning_enabled': False
        }
    
    def _train_with_epochs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train with epochs (deep learning)."""
        try:
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            raise ImportError("TensorFlow/Keras required for deep learning models")
        
        epochs = params.get('epochs', 50)
        batch_size = params.get('batch_size', 32)
        early_stopping = params.get('early_stopping', True)
        
        # Setup early stopping
        callbacks = []
        if early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ))
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        test_accuracy = None
        if X_test is not None:
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_accuracy': test_accuracy,
            'history': history.history,
            'strategy': 'epochs_with_early_stopping',
            'hp_tuning_enabled': False
        }
    
    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv,
        params: Dict[str, Any],
        is_classification: bool
    ) -> Dict[str, Any]:
        """Tune hyperparameters with RandomizedSearchCV."""
        hp_iterations = params.get('hp_iterations', 30)
        
        # Get parameter distributions for this model
        param_dist = self._get_param_distributions()
        
        if not param_dist:
            logger.warning(f"No hyperparameters to tune for {self.config['model_name']}")
            # Fall back to standard CV
            scoring = 'accuracy' if is_classification else 'r2'
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring=scoring)
            self.model.fit(X_train, y_train)
            return {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_score': self.model.score(X_test, y_test) if X_test is not None else None,
                'strategy': 'k_fold_cv',
                'hp_tuning_enabled': False
            }
        
        # Randomized search
        scoring = 'accuracy' if is_classification else 'r2'
        searcher = RandomizedSearchCV(
            self.model,
            param_dist,
            n_iter=hp_iterations,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1
        )
        
        searcher.fit(X_train, y_train)
        
        # Test score
        test_score = searcher.best_estimator_.score(X_test, y_test) if X_test is not None else None
        
        return {
            'cv_mean': searcher.best_score_,
            'cv_std': searcher.cv_results_['std_test_score'][searcher.best_index_],
            'test_score': test_score,
            'best_params': searcher.best_params_,
            'best_estimator': searcher.best_estimator_,
            'strategy': 'k_fold_cv_with_tuning',
            'hp_tuning_enabled': True
        }
    
    def _get_param_distributions(self) -> Dict[str, list]:
        """Get hyperparameter distributions for this model."""
        model_name = self.config['model_name']
        
        distributions = {
            'RandomForestClassifier': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'RandomForestRegressor': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoostingClassifier': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            'SVC': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'SVR': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        return distributions.get(model_name, {})


def train_with_automl(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Train model with AutoML strategy selection.
    
    Args:
        model: Model instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        params: User parameters
        
    Returns:
        Training results
    """
    trainer = AutoMLTrainer(model)
    return trainer.train(X_train, y_train, X_test, y_test, params)
