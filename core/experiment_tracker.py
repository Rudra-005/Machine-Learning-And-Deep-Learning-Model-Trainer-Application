"""
Experiment Tracking Module

Logs ML experiments to SQLite database with minimal overhead.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Lightweight experiment tracking using SQLite."""
    
    def __init__(self, db_path: str = "experiments.db"):
        """
        Initialize experiment tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS experiments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        dataset_name TEXT,
                        dataset_rows INTEGER,
                        dataset_cols INTEGER,
                        missing_strategies TEXT,
                        model_type TEXT,
                        model_name TEXT,
                        metrics TEXT,
                        status TEXT,
                        error_message TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def log_experiment(
        self,
        dataset_name: str,
        dataset_rows: int,
        dataset_cols: int,
        missing_strategies: Dict[str, str],
        model_type: str,
        model_name: str,
        metrics: Dict[str, Any],
        status: str = "success",
        error_message: Optional[str] = None
    ) -> int:
        """
        Log an experiment run.
        
        Args:
            dataset_name: Name of dataset
            dataset_rows: Number of rows
            dataset_cols: Number of columns
            missing_strategies: Dict of column -> strategy mappings
            model_type: 'classification' or 'regression'
            model_name: Name of model (e.g., 'random_forest')
            metrics: Dictionary of evaluation metrics
            status: 'success' or 'failed'
            error_message: Error message if failed
            
        Returns:
            Experiment ID
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO experiments (
                        timestamp, dataset_name, dataset_rows, dataset_cols,
                        missing_strategies, model_type, model_name, metrics,
                        status, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    dataset_name,
                    dataset_rows,
                    dataset_cols,
                    json.dumps(missing_strategies),
                    model_type,
                    model_name,
                    json.dumps(metrics, default=str),
                    status,
                    error_message
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to log experiment: {str(e)}")
            return -1
    
    def get_experiments(self, limit: int = 100) -> list:
        """
        Retrieve recent experiments.
        
        Args:
            limit: Maximum number of experiments to retrieve
            
        Returns:
            List of experiment records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM experiments
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to retrieve experiments: {str(e)}")
            return []
    
    def get_experiment_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of all experiments.
        
        Returns:
            Dictionary with stats
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total experiments
                cursor.execute("SELECT COUNT(*) FROM experiments")
                total = cursor.fetchone()[0]
                
                # Successful experiments
                cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'success'")
                successful = cursor.fetchone()[0]
                
                # Failed experiments
                cursor.execute("SELECT COUNT(*) FROM experiments WHERE status = 'failed'")
                failed = cursor.fetchone()[0]
                
                # Most used models
                cursor.execute("""
                    SELECT model_name, COUNT(*) as count
                    FROM experiments
                    WHERE status = 'success'
                    GROUP BY model_name
                    ORDER BY count DESC
                    LIMIT 5
                """)
                top_models = [dict(zip(['model', 'count'], row)) for row in cursor.fetchall()]
                
                return {
                    'total_experiments': total,
                    'successful': successful,
                    'failed': failed,
                    'success_rate': round(successful / total * 100, 2) if total > 0 else 0,
                    'top_models': top_models
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {}
    
    def export_to_csv(self, output_path: str = "experiments.csv") -> bool:
        """
        Export experiments to CSV.
        
        Args:
            output_path: Path to output CSV file
            
        Returns:
            True if successful
        """
        try:
            experiments = self.get_experiments(limit=10000)
            if not experiments:
                logger.warning("No experiments to export")
                return False
            
            import pandas as pd
            df = pd.DataFrame(experiments)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(experiments)} experiments to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export experiments: {str(e)}")
            return False


def log_training_run(
    tracker: ExperimentTracker,
    dataset_name: str,
    dataset_rows: int,
    dataset_cols: int,
    missing_strategies: Dict[str, str],
    model_type: str,
    model_name: str,
    metrics: Dict[str, Any]
) -> None:
    """
    Convenience function to log a successful training run.
    
    Args:
        tracker: ExperimentTracker instance
        dataset_name: Name of dataset
        dataset_rows: Number of rows
        dataset_cols: Number of columns
        missing_strategies: Missing value strategies used
        model_type: 'classification' or 'regression'
        model_name: Name of model
        metrics: Evaluation metrics
    """
    try:
        tracker.log_experiment(
            dataset_name=dataset_name,
            dataset_rows=dataset_rows,
            dataset_cols=dataset_cols,
            missing_strategies=missing_strategies,
            model_type=model_type,
            model_name=model_name,
            metrics=metrics,
            status="success"
        )
    except Exception as e:
        logger.error(f"Failed to log training run: {str(e)}")


def log_training_error(
    tracker: ExperimentTracker,
    dataset_name: str,
    dataset_rows: int,
    dataset_cols: int,
    missing_strategies: Dict[str, str],
    model_type: str,
    model_name: str,
    error_message: str
) -> None:
    """
    Convenience function to log a failed training run.
    
    Args:
        tracker: ExperimentTracker instance
        dataset_name: Name of dataset
        dataset_rows: Number of rows
        dataset_cols: Number of columns
        missing_strategies: Missing value strategies used
        model_type: 'classification' or 'regression'
        model_name: Name of model
        error_message: Error message
    """
    try:
        tracker.log_experiment(
            dataset_name=dataset_name,
            dataset_rows=dataset_rows,
            dataset_cols=dataset_cols,
            missing_strategies=missing_strategies,
            model_type=model_type,
            model_name=model_name,
            metrics={},
            status="failed",
            error_message=error_message
        )
    except Exception as e:
        logger.error(f"Failed to log training error: {str(e)}")
