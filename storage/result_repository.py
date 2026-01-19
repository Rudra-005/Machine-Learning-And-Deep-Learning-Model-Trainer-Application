"""
Result repository for saving experiment results
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from app.config import RESULTS_DIR
from app.utils.logger import logger

class ResultRepository:
    """Repository for storing experiment results"""
    
    @staticmethod
    def save_results(results: dict, experiment_name: str) -> str:
        """
        Save experiment results to JSON
        
        Args:
            results: Dictionary of results/metrics
            experiment_name: Name of the experiment
            
        Returns:
            Path to saved results
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{experiment_name}_{timestamp}.json"
            filepath = RESULTS_DIR / filename
            
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            
            logger.info(f"Results saved: {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    @staticmethod
    def save_metrics_csv(metrics_df: pd.DataFrame, experiment_name: str) -> str:
        """
        Save metrics as CSV
        
        Args:
            metrics_df: DataFrame of metrics
            experiment_name: Name of the experiment
            
        Returns:
            Path to saved CSV
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{experiment_name}_metrics_{timestamp}.csv"
            filepath = RESULTS_DIR / filename
            
            metrics_df.to_csv(filepath, index=False)
            logger.info(f"Metrics CSV saved: {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Error saving metrics CSV: {str(e)}")
            raise
    
    @staticmethod
    def load_results(filepath: str) -> dict:
        """
        Load results from JSON
        
        Args:
            filepath: Path to results file
            
        Returns:
            Dictionary of results
        """
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            logger.info(f"Results loaded: {filepath}")
            return results
        
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            raise
    
    @staticmethod
    def get_all_results() -> list:
        """
        Get list of all saved results
        
        Returns:
            List of result files
        """
        results_files = list(RESULTS_DIR.glob("*.json"))
        logger.info(f"Found {len(results_files)} result files")
        return results_files
