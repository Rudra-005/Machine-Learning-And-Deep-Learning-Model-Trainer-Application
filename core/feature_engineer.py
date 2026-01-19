"""
Feature engineering module
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from app.utils.logger import logger

class FeatureEngineer:
    """Feature engineering operations"""
    
    @staticmethod
    def create_polynomial_features(X: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            X: Feature matrix
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return X
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X[numeric_cols])
        X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(numeric_cols))
        
        # Add original categorical columns
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            X_poly_df[col] = X[col].values
        
        logger.info(f"Polynomial features created: {X_poly_df.shape[1]} features")
        return X_poly_df
    
    @staticmethod
    def create_interaction_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between numeric columns
        
        Args:
            X: Feature matrix
            
        Returns:
            DataFrame with interaction features
        """
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        X_copy = X.copy()
        
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                X_copy[f"{col1}_x_{col2}"] = X[col1] * X[col2]
        
        logger.info(f"Interaction features created: {X_copy.shape[1]} features")
        return X_copy
    
    @staticmethod
    def drop_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Drop highly correlated features
        
        Args:
            X: Feature matrix
            threshold: Correlation threshold
            
        Returns:
            DataFrame with reduced features
        """
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            return X
        
        corr_matrix = X[numeric_cols].corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [col for col in upper_triangle.columns if (upper_triangle[col] > threshold).any()]
        
        X_reduced = X.drop(columns=to_drop)
        logger.info(f"Dropped {len(to_drop)} highly correlated features")
        return X_reduced
