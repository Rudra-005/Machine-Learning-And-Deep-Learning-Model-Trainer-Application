"""
Data validation and quality checks
"""
import pandas as pd
from app.utils.logger import logger

class DataValidator:
    """Validate data quality"""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> dict:
        """Check for missing values"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        return {
            "missing_count": missing.to_dict(),
            "missing_percentage": missing_pct.to_dict(),
            "has_missing": missing.sum() > 0
        }
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> dict:
        """Check for duplicate rows"""
        duplicates = df.duplicated().sum()
        return {
            "duplicate_rows": int(duplicates),
            "has_duplicates": duplicates > 0
        }
    
    @staticmethod
    def check_class_imbalance(y: pd.Series) -> dict:
        """Check for class imbalance in target"""
        value_counts = y.value_counts()
        distribution = (value_counts / len(y)).to_dict()
        min_class_pct = (value_counts.min() / len(y)) * 100
        
        return {
            "class_distribution": value_counts.to_dict(),
            "class_percentages": distribution,
            "min_class_percentage": min_class_pct,
            "is_imbalanced": min_class_pct < 20
        }
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame, target_col: str = None) -> dict:
        """
        Generate comprehensive data quality report
        
        Args:
            df: DataFrame to validate
            target_col: Target column name (for classification)
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            "shape": df.shape,
            "dtypes": df.dtypes.to_dict(),
            "missing_values": DataValidator.check_missing_values(df),
            "duplicates": DataValidator.check_duplicates(df),
        }
        
        if target_col and target_col in df.columns:
            report["class_imbalance"] = DataValidator.check_class_imbalance(df[target_col])
        
        logger.info("Data quality report generated")
        return report
