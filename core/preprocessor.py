"""
Data preprocessing pipeline
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from app.utils.logger import logger

class DataPreprocessor:
    """Handle data preprocessing operations"""
    
    def __init__(self):
        """Initialize preprocessor with transformers"""
        self.imputer = None
        self.scaler = None
        self.label_encoders = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.target_col = None
    
    def fit(self, df: pd.DataFrame, target_col: str, scaling: str = 'standard'):
        """
        Fit preprocessor on training data
        
        Args:
            df: Training DataFrame
            target_col: Target column name
            scaling: 'standard' or 'minmax'
        """
        self.target_col = target_col
        
        # Identify column types
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature columns
        if target_col in self.numeric_cols:
            self.numeric_cols.remove(target_col)
        if target_col in self.categorical_cols:
            self.categorical_cols.remove(target_col)
        
        # Fit imputer
        if self.numeric_cols:
            self.imputer = SimpleImputer(strategy='mean')
            self.imputer.fit(df[self.numeric_cols])
        
        # Fit scaler
        if self.numeric_cols:
            if scaling == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()
            self.scaler.fit(df[self.numeric_cols])
        
        # Fit label encoders for categorical
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            self.label_encoders[col] = le
        
        logger.info(f"Preprocessor fitted on {len(df)} samples")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Preprocessed DataFrame
        """
        df_copy = df.copy()
        
        # Handle missing values
        if self.numeric_cols and self.imputer:
            df_copy[self.numeric_cols] = self.imputer.transform(df_copy[self.numeric_cols])
        
        # Scale numeric features
        if self.numeric_cols and self.scaler:
            df_copy[self.numeric_cols] = self.scaler.transform(df_copy[self.numeric_cols])
        
        # Encode categorical features
        for col in self.categorical_cols:
            if col in self.label_encoders:
                df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
        
        return df_copy
    
    def fit_transform(self, df: pd.DataFrame, target_col: str, scaling: str = 'standard') -> pd.DataFrame:
        """
        Fit and transform data in one step
        
        Args:
            df: DataFrame to fit and transform
            target_col: Target column name
            scaling: Scaling method
            
        Returns:
            Preprocessed DataFrame
        """
        self.fit(df, target_col, scaling)
        return self.transform(df)
