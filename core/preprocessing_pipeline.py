"""
Integrated Data Preprocessing Pipeline

Combines missing value handling, categorical encoding, scaling, and data splitting
using scikit-learn Pipelines and ColumnTransformer.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Any, List


class PreprocessingPipeline:
    """
    Modular preprocessing pipeline for ML workflows.
    
    Handles:
    - Missing value imputation
    - Categorical encoding
    - Numerical scaling
    - Train/validation/test splitting
    """
    
    def __init__(self, missing_config: Dict[str, Dict[str, Any]]):
        """
        Initialize preprocessing pipeline.
        
        Args:
            missing_config: Missing value handling configuration
                           Example: {'age': {'strategy': 'median'}, ...}
        """
        self.missing_config = missing_config
        self.preprocessor = None
        self.numeric_cols = []
        self.categorical_cols = []
    
    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build scikit-learn ColumnTransformer pipeline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Fitted ColumnTransformer
        """
        # Identify column types
        self.numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Numeric pipeline: impute + scale
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute + encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, self.numeric_cols),
            ('categorical', categorical_pipeline, self.categorical_cols)
        ], remainder='drop')
        
        return preprocessor
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (transformed_X, y)
        """
        self.preprocessor = self._build_preprocessor(X)
        X_transformed = self.preprocessor.fit_transform(X)
        return X_transformed, y.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return self.preprocessor.transform(X)


def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str,
    missing_config: Dict[str, Dict[str, Any]],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete preprocessing workflow with train/val/test split.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        missing_config: Missing value handling configuration
        test_size: Test set proportion (0.0-1.0)
        val_size: Validation set proportion of remaining data (0.0-1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with:
        - X_train, X_val, X_test: Feature arrays
        - y_train, y_val, y_test: Target arrays
        - preprocessor: Fitted PreprocessingPipeline
        - feature_names: Names of transformed features
        
    Example:
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, None, 45],
        ...     'city': ['NYC', None, 'LA', 'NYC'],
        ...     'salary': [50000, 60000, 70000, 80000],
        ...     'target': [0, 1, 0, 1]
        ... })
        >>> config = {
        ...     'age': {'strategy': 'median'},
        ...     'city': {'strategy': 'most_frequent'},
        ...     'salary': {'strategy': 'median'}
        ... }
        >>> result = preprocess_and_split(df, 'target', config)
        >>> X_train, y_train = result['X_train'], result['y_train']
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle missing values first
    X_clean = _handle_missing_values(X, missing_config)
    
    # Split into train+val and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_clean, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Split train+val into train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state
    )
    
    # Build and fit preprocessor on training data
    preprocessor = PreprocessingPipeline(missing_config)
    X_train_transformed, y_train_transformed = preprocessor.fit_transform(X_train, y_train)
    
    # Transform validation and test sets
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    
    return {
        'X_train': X_train_transformed,
        'X_val': X_val_transformed,
        'X_test': X_test_transformed,
        'y_train': y_train_transformed,
        'y_val': y_val.values,
        'y_test': y_test.values,
        'preprocessor': preprocessor,
        'feature_names': _get_feature_names(preprocessor),
        'shapes': {
            'train': X_train_transformed.shape,
            'val': X_val_transformed.shape,
            'test': X_test_transformed.shape
        }
    }


def _handle_missing_values(
    X: pd.DataFrame,
    config: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Apply missing value handling strategies.
    
    Args:
        X: Feature DataFrame
        config: Strategy configuration
        
    Returns:
        DataFrame with missing values handled
    """
    X_clean = X.copy()
    
    for col, cfg in config.items():
        if col not in X_clean.columns:
            continue
        
        strategy = cfg.get('strategy', 'median')
        
        if strategy == 'drop_column':
            X_clean = X_clean.drop(columns=[col])
        elif strategy == 'constant':
            value = cfg.get('value', 'Unknown')
            X_clean[col].fillna(value, inplace=True)
        elif strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
            X_clean[[col]] = imputer.fit_transform(X_clean[[col]])
    
    return X_clean


def _get_feature_names(preprocessor: PreprocessingPipeline) -> List[str]:
    """
    Extract feature names from fitted preprocessor.
    
    Args:
        preprocessor: Fitted PreprocessingPipeline
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    # Numeric features
    feature_names.extend(preprocessor.numeric_cols)
    
    # Categorical features (one-hot encoded)
    if preprocessor.categorical_cols:
        encoder = preprocessor.preprocessor.named_transformers_['categorical'].named_steps['encoder']
        cat_features = encoder.get_feature_names_out(preprocessor.categorical_cols)
        feature_names.extend(cat_features)
    
    return feature_names


def get_preprocessing_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get summary of preprocessing results.
    
    Args:
        result: Output from preprocess_and_split()
        
    Returns:
        Summary statistics
    """
    return {
        'train_shape': result['shapes']['train'],
        'val_shape': result['shapes']['val'],
        'test_shape': result['shapes']['test'],
        'total_features': result['shapes']['train'][1],
        'feature_names': result['feature_names'],
        'train_samples': result['shapes']['train'][0],
        'val_samples': result['shapes']['val'][0],
        'test_samples': result['shapes']['test'][0]
    }
