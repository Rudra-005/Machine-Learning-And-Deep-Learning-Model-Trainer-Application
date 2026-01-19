"""
Unit Tests for Target Analyzer

Comprehensive test suite for target analysis functions.
Run with: pytest tests/test_target_analyzer_unit.py -v
"""

import pytest
import pandas as pd
import numpy as np
from core.target_analyzer import (
    detect_task_type,
    analyze_classification,
    analyze_regression,
    create_class_distribution_plot,
    create_regression_histogram,
    create_regression_boxplot,
    analyze_target
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def y_binary():
    """Binary classification target."""
    return np.array([0, 1, 0, 1, 0, 1, 0, 1])


@pytest.fixture
def y_multiclass():
    """Multi-class classification target."""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])


@pytest.fixture
def y_imbalanced():
    """Imbalanced classification target."""
    return np.array([0]*95 + [1]*5)


@pytest.fixture
def y_regression():
    """Regression target."""
    np.random.seed(42)
    return np.random.normal(100, 20, 1000)


@pytest.fixture
def y_regression_with_outliers():
    """Regression target with outliers."""
    np.random.seed(42)
    y = np.concatenate([
        np.random.normal(100, 20, 950),
        np.array([500, 600, 700, 800, 900])
    ])
    return y


# ============================================================================
# Tests: detect_task_type
# ============================================================================

class TestDetectTaskType:
    """Tests for detect_task_type function."""
    
    def test_binary_classification(self, y_binary):
        """Test binary classification detection."""
        task = detect_task_type(y_binary)
        assert task.task_type == 'classification'
        assert task.confidence >= 0.9
        assert task.n_unique == 2
    
    def test_multiclass_classification(self, y_multiclass):
        """Test multi-class classification detection."""
        task = detect_task_type(y_multiclass)
        assert task.task_type == 'classification'
        assert task.n_unique == 3
    
    def test_regression(self, y_regression):
        """Test regression detection."""
        task = detect_task_type(y_regression)
        assert task.task_type == 'regression'
        assert task.confidence >= 0.9
    
    def test_n_samples(self, y_binary):
        """Test n_samples attribute."""
        task = detect_task_type(y_binary)
        assert task.n_samples == len(y_binary)
    
    def test_single_class(self):
        """Test single class (edge case)."""
        y = np.array([0, 0, 0, 0])
        task = detect_task_type(y)
        assert task.n_unique == 1


# ============================================================================
# Tests: analyze_classification
# ============================================================================

class TestAnalyzeClassification:
    """Tests for analyze_classification function."""
    
    def test_balanced_classification(self, y_binary):
        """Test balanced classification."""
        metrics = analyze_classification(y_binary)
        assert metrics['n_classes'] == 2
        assert metrics['imbalance_ratio'] == 1.0
        assert not metrics['is_imbalanced']
    
    def test_imbalanced_classification(self, y_imbalanced):
        """Test imbalanced classification."""
        metrics = analyze_classification(y_imbalanced)
        assert metrics['n_classes'] == 2
        assert metrics['imbalance_ratio'] == 19.0
        assert metrics['is_imbalanced']
    
    def test_class_counts(self, y_multiclass):
        """Test class counts accuracy."""
        metrics = analyze_classification(y_multiclass)
        assert len(metrics['class_counts']) == 3
        assert sum(metrics['class_counts'].values()) == len(y_multiclass)
    
    def test_class_distribution(self, y_binary):
        """Test class distribution percentages."""
        metrics = analyze_classification(y_binary)
        total_pct = sum(metrics['class_distribution'].values())
        assert abs(total_pct - 100.0) < 0.01
    
    def test_majority_minority_classes(self, y_imbalanced):
        """Test majority and minority class identification."""
        metrics = analyze_classification(y_imbalanced)
        assert metrics['majority_class'] == 0
        assert metrics['minority_class'] == 1


# ============================================================================
# Tests: analyze_regression
# ============================================================================

class TestAnalyzeRegression:
    """Tests for analyze_regression function."""
    
    def test_basic_statistics(self, y_regression):
        """Test basic statistics computation."""
        metrics = analyze_regression(y_regression)
        
        assert 'mean' in metrics
        assert 'std' in metrics
        assert 'min' in metrics
        assert 'max' in metrics
        assert 'median' in metrics
    
    def test_statistics_accuracy(self, y_regression):
        """Test accuracy of statistics."""
        metrics = analyze_regression(y_regression)
        
        assert abs(metrics['mean'] - np.mean(y_regression)) < 0.01
        assert abs(metrics['std'] - np.std(y_regression)) < 0.01
        assert abs(metrics['median'] - np.median(y_regression)) < 0.01
    
    def test_percentiles(self, y_regression):
        """Test percentile calculations."""
        metrics = analyze_regression(y_regression)
        
        assert metrics['q25'] <= metrics['median'] <= metrics['q75']
        assert metrics['q25'] >= metrics['min']
        assert metrics['q75'] <= metrics['max']
    
    def test_outlier_detection(self, y_regression_with_outliers):
        """Test outlier detection."""
        metrics = analyze_regression(y_regression_with_outliers)
        
        assert metrics['n_outliers'] > 0
        assert metrics['outlier_percentage'] > 0
    
    def test_skewness_calculation(self, y_regression):
        """Test skewness calculation."""
        metrics = analyze_regression(y_regression)
        
        # Normal distribution should have skewness close to 0
        assert abs(metrics['skewness']) < 0.5


# ============================================================================
# Tests: Visualization Functions
# ============================================================================

class TestVisualizationFunctions:
    """Tests for visualization functions."""
    
    def test_class_distribution_plot_plotly(self, y_multiclass):
        """Test Plotly class distribution plot."""
        fig = create_class_distribution_plot(y_multiclass, backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_class_distribution_plot_matplotlib(self, y_multiclass):
        """Test Matplotlib class distribution plot."""
        fig = create_class_distribution_plot(y_multiclass, backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_regression_histogram_plotly(self, y_regression):
        """Test Plotly histogram."""
        fig = create_regression_histogram(y_regression, backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_regression_histogram_matplotlib(self, y_regression):
        """Test Matplotlib histogram."""
        fig = create_regression_histogram(y_regression, backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_regression_boxplot_plotly(self, y_regression):
        """Test Plotly boxplot."""
        fig = create_regression_boxplot(y_regression, backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_regression_boxplot_matplotlib(self, y_regression):
        """Test Matplotlib boxplot."""
        fig = create_regression_boxplot(y_regression, backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')


# ============================================================================
# Tests: analyze_target
# ============================================================================

class TestAnalyzeTarget:
    """Tests for analyze_target function."""
    
    def test_classification_analysis(self, y_binary):
        """Test classification analysis."""
        analysis = analyze_target(y_binary, create_plots=False)
        
        assert analysis['task_type'] == 'classification'
        assert 'metrics' in analysis
        assert 'n_classes' in analysis['metrics']
    
    def test_regression_analysis(self, y_regression):
        """Test regression analysis."""
        analysis = analyze_target(y_regression, create_plots=False)
        
        assert analysis['task_type'] == 'regression'
        assert 'metrics' in analysis
        assert 'mean' in analysis['metrics']
    
    def test_explicit_task_type(self, y_binary):
        """Test explicit task type specification."""
        analysis = analyze_target(y_binary, task_type='classification', create_plots=False)
        assert analysis['task_type'] == 'classification'
    
    def test_with_plots(self, y_binary):
        """Test analysis with plots."""
        analysis = analyze_target(y_binary, create_plots=True, backend='plotly')
        
        assert 'plots' in analysis
        assert len(analysis['plots']) > 0
    
    def test_without_plots(self, y_binary):
        """Test analysis without plots."""
        analysis = analyze_target(y_binary, create_plots=False)
        
        assert 'plots' not in analysis or len(analysis['plots']) == 0


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_sample(self):
        """Test with single sample."""
        y = np.array([1])
        task = detect_task_type(y)
        assert task.n_samples == 1
    
    def test_all_same_value(self):
        """Test with all same values."""
        y = np.array([5, 5, 5, 5, 5])
        metrics = analyze_regression(y)
        assert metrics['std'] == 0.0
    
    def test_two_values(self):
        """Test with only two unique values."""
        y = np.array([1, 2, 1, 2, 1, 2])
        task = detect_task_type(y)
        assert task.task_type == 'classification'
    
    def test_pandas_series(self):
        """Test with pandas Series."""
        y = pd.Series([0, 1, 0, 1, 0, 1])
        task = detect_task_type(y)
        assert task.task_type == 'classification'
    
    def test_negative_values(self):
        """Test with negative values."""
        y = np.array([-10, -5, 0, 5, 10])
        metrics = analyze_regression(y)
        assert metrics['mean'] == 0.0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests."""
    
    def test_large_classification(self):
        """Test with large classification dataset."""
        y = np.random.choice([0, 1], size=100000)
        metrics = analyze_classification(y)
        assert metrics['n_classes'] == 2
    
    def test_large_regression(self):
        """Test with large regression dataset."""
        y = np.random.normal(100, 20, 100000)
        metrics = analyze_regression(y)
        assert 'mean' in metrics
    
    def test_many_classes(self):
        """Test with many classes."""
        y = np.random.choice(range(100), size=10000)
        metrics = analyze_classification(y)
        assert metrics['n_classes'] == 100


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_classification_pipeline(self):
        """Test classification analysis pipeline."""
        np.random.seed(42)
        y = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
        
        # Step 1: Detect task
        task = detect_task_type(y)
        assert task.task_type == 'classification'
        
        # Step 2: Analyze
        metrics = analyze_classification(y)
        assert metrics['is_imbalanced']
        
        # Step 3: Create plot
        fig = create_class_distribution_plot(y, backend='plotly')
        assert fig is None or hasattr(fig, 'show')
    
    def test_regression_pipeline(self):
        """Test regression analysis pipeline."""
        np.random.seed(42)
        y = np.random.normal(100, 20, 1000)
        
        # Step 1: Detect task
        task = detect_task_type(y)
        assert task.task_type == 'regression'
        
        # Step 2: Analyze
        metrics = analyze_regression(y)
        assert 'mean' in metrics
        
        # Step 3: Create plots
        fig1 = create_regression_histogram(y, backend='plotly')
        fig2 = create_regression_boxplot(y, backend='plotly')
        assert fig1 is None or hasattr(fig1, 'show')
        assert fig2 is None or hasattr(fig2, 'show')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
