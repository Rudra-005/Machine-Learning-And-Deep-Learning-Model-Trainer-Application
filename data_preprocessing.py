"""
Data Preprocessing Module

A comprehensive data preprocessing pipeline using scikit-learn best practices.
Handles CSV loading, missing values, feature scaling, and categorical encoding.

Author: ML Engineer
Date: 2026-01-19
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any, Union
from pathlib import Path
import logging

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class that handles data loading, exploration,
    and transformation using scikit-learn pipelines.
    
    Attributes:
        df (pd.DataFrame): The loaded dataset
        numerical_cols (List[str]): Detected numerical columns
        categorical_cols (List[str]): Detected categorical columns
        target_col (str): Target column name
        preprocessor (ColumnTransformer): Fitted preprocessing pipeline
        X_transformed (np.ndarray): Transformed feature data
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.df = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.target_col = None
        self.preprocessor = None
        self.X_transformed = None
        self.feature_names = []
        
    def load_data(self, filepath: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Load CSV dataset from file or accept DataFrame directly.
        
        Args:
            filepath (Union[str, pd.DataFrame]): Path to CSV file or DataFrame object
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If CSV is malformed
            TypeError: If invalid input type
        """
        try:
            if isinstance(filepath, pd.DataFrame):
                self.df = filepath.copy()
                logger.info(f"Loaded DataFrame: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
                return self.df
            
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            self.df = pd.read_csv(filepath)
            logger.info(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def detect_column_types(self, numerical_threshold: int = 10) -> Dict[str, List[str]]:
        """
        Automatically detect numerical and categorical columns.
        
        Uses heuristics:
        - Numerical: numeric dtypes (int, float)
        - Categorical: object dtypes or numeric with few unique values
        
        Args:
            numerical_threshold (int): Max unique values for numeric column to be
                                      considered categorical. Default: 10
            
        Returns:
            Dict containing 'numerical' and 'categorical' column lists
            
        Raises:
            ValueError: If DataFrame not loaded
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        numerical = []
        categorical = []
        
        for col in self.df.columns:
            # Skip target column
            if col == self.target_col:
                continue
            
            # Check data type
            if self.df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                # Numeric type
                unique_count = self.df[col].nunique()
                if unique_count <= numerical_threshold:
                    categorical.append(col)
                else:
                    numerical.append(col)
            else:
                # Object or other types
                categorical.append(col)
        
        self.numerical_cols = numerical
        self.categorical_cols = categorical
        
        logger.info(f"Detected {len(numerical)} numerical columns: {numerical}")
        logger.info(f"Detected {len(categorical)} categorical columns: {categorical}")
        
        return {
            'numerical': numerical,
            'categorical': categorical
        }
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Returns:
            pd.DataFrame: Summary statistics of missing values
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum().values,
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).values
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        if len(missing_data) > 0:
            logger.info("\nMissing Values Summary:")
            logger.info(missing_data.to_string(index=False))
        else:
            logger.info("No missing values detected in dataset.")
        
        return missing_data
    
    def build_preprocessing_pipeline(
        self,
        numerical_strategy: str = 'mean',
        categorical_strategy: str = 'most_frequent',
        handle_unknown: str = 'ignore'
    ) -> ColumnTransformer:
        """
        Build a scikit-learn ColumnTransformer pipeline for preprocessing.
        
        Args:
            numerical_strategy (str): Imputation strategy for numerical features.
                                     Options: 'mean', 'median', 'most_frequent'
            categorical_strategy (str): Imputation strategy for categorical features.
                                       Options: 'most_frequent', 'constant'
            handle_unknown (str): How to handle unknown categories in OneHotEncoder.
                                 Options: 'ignore', 'use_encoded_value'
            
        Returns:
            ColumnTransformer: Fitted preprocessing pipeline
            
        Raises:
            ValueError: If column types not detected or empty
        """
        if not self.numerical_cols and not self.categorical_cols:
            raise ValueError(
                "No columns detected. Call detect_column_types() first."
            )
        
        transformers = []
        
        # Numerical pipeline
        if self.numerical_cols:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=numerical_strategy)),
                ('scaler', StandardScaler())
            ])
            transformers.append((
                'num',
                numerical_transformer,
                self.numerical_cols
            ))
            logger.info(f"Created numerical pipeline with {numerical_strategy} imputation")
        
        # Categorical pipeline
        if self.categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=categorical_strategy)),
                ('onehot', OneHotEncoder(
                    handle_unknown=handle_unknown,
                    sparse_output=False,
                    drop='first'  # Drop first category to avoid multicollinearity
                ))
            ])
            transformers.append((
                'cat',
                categorical_transformer,
                self.categorical_cols
            ))
            logger.info(f"Created categorical pipeline with OneHotEncoder")
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        logger.info("ColumnTransformer pipeline created successfully")
        return self.preprocessor
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform feature data.
        
        Args:
            X (pd.DataFrame): Feature data
            
        Returns:
            np.ndarray: Transformed features
            
        Raises:
            ValueError: If preprocessor not built
        """
        if self.preprocessor is None:
            raise ValueError(
                "Preprocessor not built. Call build_preprocessing_pipeline() first."
            )
        
        X_transformed = self.preprocessor.fit_transform(X)
        self.X_transformed = X_transformed
        
        # Get feature names after transformation
        self._update_feature_names()
        
        logger.info(f"Transformed features shape: {X_transformed.shape}")
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform feature data using fitted preprocessor.
        
        Args:
            X (pd.DataFrame): Feature data
            
        Returns:
            np.ndarray: Transformed features
            
        Raises:
            ValueError: If preprocessor not fitted
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
        
        return self.preprocessor.transform(X)
    
    def _update_feature_names(self) -> None:
        """Update feature names after transformation."""
        feature_names = []
        
        # Numerical features (unchanged names)
        feature_names.extend(self.numerical_cols)
        
        # Categorical features (encoded names)
        if self.categorical_cols:
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_names = cat_encoder.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_names)
        
        self.feature_names = feature_names
        logger.info(f"Updated feature names: {len(feature_names)} features")
    
    def split_data(
        self,
        X: np.ndarray,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
               np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Performs stratified split for classification tasks and ensures
        reproducibility with random_state.
        
        Args:
            X (np.ndarray): Feature data (transformed)
            y (pd.Series): Target data
            test_size (float): Proportion of test set. Default: 0.2
            val_size (float): Proportion of validation set (from training data).
                             Default: 0.1
            random_state (int): Random seed for reproducibility. Default: 42
            stratify (bool): Use stratified split for classification. Default: True
            
        Returns:
            Tuple containing:
                - X_train: Training features
                - X_val: Validation features
                - X_test: Test features
                - y_train: Training target
                - y_val: Validation target
                - y_test: Test target
        """
        # Determine stratification
        stratify_y = y if stratify and len(y.unique()) < 20 else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_y_temp = y_temp if stratify and len(y_temp.unique()) < 20 else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_y_temp
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        logger.info(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the preprocessing configuration.
        
        Returns:
            Dict: Summary information
        """
        summary = {
            'dataset_shape': self.df.shape if self.df is not None else None,
            'numerical_columns': self.numerical_cols,
            'categorical_columns': self.categorical_cols,
            'target_column': self.target_col,
            'transformed_shape': self.X_transformed.shape if self.X_transformed is not None else None,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names
        }
        return summary


def preprocess_dataset(
    filepath: Union[str, pd.DataFrame],
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    numerical_strategy: str = 'mean',
    categorical_strategy: str = 'most_frequent'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           DataPreprocessor]:
    """
    End-to-end data preprocessing pipeline.
    
    Loads data, detects column types, builds preprocessing pipeline,
    and splits into train/validation/test sets.
    
    Args:
        filepath (Union[str, pd.DataFrame]): Path to CSV file or DataFrame object
        target_col (str): Name of target column
        test_size (float): Proportion of test set. Default: 0.2
        val_size (float): Proportion of validation set. Default: 0.1
        random_state (int): Random seed. Default: 42
        numerical_strategy (str): Imputation strategy for numerical features
        categorical_strategy (str): Imputation strategy for categorical features
        
    Returns:
        Tuple containing:
            - X_train, X_val, X_test: Feature arrays
            - y_train, y_val, y_test: Target arrays
            - preprocessor: Fitted DataPreprocessor instance for future use
            
    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test, prep = preprocess_dataset(
        ...     'data.csv',
        ...     target_col='price'
        ... )
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(filepath)
    preprocessor.target_col = target_col
    
    # Detect column types
    preprocessor.detect_column_types()
    
    # Analyze missing values
    preprocessor.analyze_missing_values()
    
    # Build pipeline
    preprocessor.build_preprocessing_pipeline(
        numerical_strategy=numerical_strategy,
        categorical_strategy=categorical_strategy
    )
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X_transformed,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    logger.info("\n✓ Preprocessing pipeline completed successfully!")
    logger.info(f"\nSummary:")
    logger.info(preprocessor.get_summary())
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


if __name__ == "__main__":
    # Example usage
    print("Data Preprocessing Module")
    print("=" * 50)
    print("\nUsage Example:")
    print("""
    from data_preprocessing import preprocess_dataset
    
    # End-to-end preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_dataset(
        'data.csv',
        target_col='target_column_name',
        test_size=0.2,
        val_size=0.1
    )
    
    # Or use the class for more control
    from data_preprocessing import DataPreprocessor
    
    prep = DataPreprocessor()
    prep.load_data('data.csv')
    prep.target_col = 'target_column_name'
    prep.detect_column_types()
    prep.build_preprocessing_pipeline()
    
    X = prep.df.drop(columns=['target_column_name'])
    y = prep.df['target_column_name']
    
    X_transformed = prep.fit_transform(X)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(
        X_transformed, y
    )
    """)
