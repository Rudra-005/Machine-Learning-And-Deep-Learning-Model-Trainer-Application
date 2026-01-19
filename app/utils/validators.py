"""
Input validation utilities
"""
import pandas as pd
from app.utils.logger import logger

class DataValidator:
    """Validate data before processing"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate DataFrame structure
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (is_valid, message)
        """
        if df is None or df.empty:
            return False, "DataFrame is empty"
        
        if df.shape[0] < 10:
            return False, "Dataset must have at least 10 rows"
        
        if df.shape[1] < 2:
            return False, "Dataset must have at least 2 columns"
        
        return True, "DataFrame is valid"
    
    @staticmethod
    def validate_target_column(df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
        """
        Validate target column
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (is_valid, message)
        """
        if target_col not in df.columns:
            return False, f"Target column '{target_col}' not found"
        
        if df[target_col].isnull().any():
            return False, f"Target column has missing values"
        
        return True, "Target column is valid"
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> dict:
        """
        Get dataset summary statistics
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary info
        """
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_cols": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object']).columns.tolist(),
        }
