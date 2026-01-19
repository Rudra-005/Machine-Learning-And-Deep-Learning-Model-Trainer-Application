"""
Test suite for ML/DL Trainer
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.preprocessor import DataPreprocessor
from core.validator import DataValidator as CoreValidator
from app.utils.validators import DataValidator
from models.model_factory import ModelFactory
from evaluation.cross_validator import CrossValidator
from evaluation.metrics import MetricsCalculator

@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    return df

class TestDataValidator:
    """Test data validation"""
    
    def test_validate_dataframe(self, sample_data):
        """Test DataFrame validation"""
        is_valid, msg = DataValidator.validate_dataframe(sample_data)
        assert is_valid is True
    
    def test_validate_empty_dataframe(self):
        """Test empty DataFrame validation"""
        df = pd.DataFrame()
        is_valid, msg = DataValidator.validate_dataframe(df)
        assert is_valid is False

class TestPreprocessor:
    """Test data preprocessing"""
    
    def test_preprocessor_fit_transform(self, sample_data):
        """Test fit and transform"""
        preprocessor = DataPreprocessor()
        X = sample_data.drop('target', axis=1)
        result = preprocessor.fit_transform(X, 'target')
        assert result is not None
        assert len(result) == len(X)

class TestModelFactory:
    """Test model creation"""
    
    def test_create_classifier(self):
        """Test creating classifier"""
        model = ModelFactory.create_ml_model('classification', 'logistic_regression')
        assert model is not None
    
    def test_create_regressor(self):
        """Test creating regressor"""
        model = ModelFactory.create_ml_model('regression', 'linear_regression')
        assert model is not None

class TestMetrics:
    """Test metric calculations"""
    
    def test_classification_metrics(self):
        """Test classification metrics"""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        metrics = MetricsCalculator.classification_metrics(y_true, y_pred)
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
    
    def test_regression_metrics(self):
        """Test regression metrics"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1])
        metrics = MetricsCalculator.regression_metrics(y_true, y_pred)
        assert 'mse' in metrics
        assert 'r2_score' in metrics

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
