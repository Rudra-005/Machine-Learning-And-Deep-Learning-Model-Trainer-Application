"""
Unit Tests for Relationship Analyzer

Comprehensive test suite for feature-target relationship analysis.
Run with: pytest tests/test_relationship_analyzer_unit.py -v
"""

import pytest
import pandas as pd
import numpy as np
from core.relationship_analyzer import (
    compute_correlation_matrix,
    get_top_correlated_features,
    analyze_categorical_regression,
    analyze_categorical_classification,
    plot_correlation_heatmap,
    plot_categorical_regression,
    plot_categorical_classification
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def df_regression():
    """DataFrame for regression analysis."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'experience': np.random.randint(0, 40, 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })


@pytest.fixture
def df_classification():
    """DataFrame for classification analysis."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'purchased': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })


@pytest.fixture
def df_categorical_regression():
    """DataFrame with categorical feature for regression."""
    np.random.seed(42)
    return pd.DataFrame({
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })


# ============================================================================
# Tests: compute_correlation_matrix
# ============================================================================

class TestComputeCorrelationMatrix:
    """Tests for compute_correlation_matrix function."""
    
    def test_pearson_correlation(self, df_regression):
        """Test Pearson correlation computation."""
        corr = compute_correlation_matrix(df_regression, 'price', method='pearson')
        
        assert len(corr) == 3  # 4 columns - 1 target
        assert all(-1 <= c <= 1 for c in corr.values)
    
    def test_spearman_correlation(self, df_regression):
        """Test Spearman correlation computation."""
        corr = compute_correlation_matrix(df_regression, 'price', method='spearman')
        
        assert len(corr) == 3
        assert all(-1 <= c <= 1 for c in corr.values)
    
    def test_invalid_target(self, df_regression):
        """Test with invalid target column."""
        with pytest.raises(ValueError):
            compute_correlation_matrix(df_regression, 'nonexistent')
    
    def test_non_numerical_target(self, df_classification):
        """Test with non-numerical target."""
        with pytest.raises(ValueError):
            compute_correlation_matrix(df_classification, 'city')
    
    def test_sampling(self, df_regression):
        """Test sampling for large datasets."""
        corr = compute_correlation_matrix(df_regression, 'price', sample_size=500)
        assert len(corr) == 3


# ============================================================================
# Tests: get_top_correlated_features
# ============================================================================

class TestGetTopCorrelatedFeatures:
    """Tests for get_top_correlated_features function."""
    
    def test_top_features(self, df_regression):
        """Test getting top correlated features."""
        top = get_top_correlated_features(df_regression, 'price', top_n=2)
        
        assert len(top['features']) == 2
        assert len(top['correlations']) == 2
        assert top['method'] == 'pearson'
    
    def test_top_n_parameter(self, df_regression):
        """Test top_n parameter."""
        top = get_top_correlated_features(df_regression, 'price', top_n=1)
        assert len(top['features']) == 1
    
    def test_output_structure(self, df_regression):
        """Test output structure."""
        top = get_top_correlated_features(df_regression, 'price')
        
        assert 'features' in top
        assert 'correlations' in top
        assert 'method' in top
        assert 'count' in top


# ============================================================================
# Tests: analyze_categorical_regression
# ============================================================================

class TestAnalyzeCategoricalRegression:
    """Tests for analyze_categorical_regression function."""
    
    def test_basic_analysis(self, df_categorical_regression):
        """Test basic categorical regression analysis."""
        analysis = analyze_categorical_regression(df_categorical_regression, 'city', 'price')
        
        assert 'categories' in analysis
        assert 'means' in analysis
        assert 'counts' in analysis
        assert 'overall_mean' in analysis
    
    def test_categories_count(self, df_categorical_regression):
        """Test number of categories."""
        analysis = analyze_categorical_regression(df_categorical_regression, 'city', 'price')
        
        assert len(analysis['categories']) == 3
        assert len(analysis['means']) == 3
        assert len(analysis['counts']) == 3
    
    def test_means_validity(self, df_categorical_regression):
        """Test validity of computed means."""
        analysis = analyze_categorical_regression(df_categorical_regression, 'city', 'price')
        
        # All means should be close to overall mean
        overall = analysis['overall_mean']
        for mean in analysis['means']:
            assert 0 < mean < 200000  # Reasonable range


# ============================================================================
# Tests: analyze_categorical_classification
# ============================================================================

class TestAnalyzeCategoricalClassification:
    """Tests for analyze_categorical_classification function."""
    
    def test_basic_analysis(self, df_classification):
        """Test basic categorical classification analysis."""
        analysis = analyze_categorical_classification(df_classification, 'city', 'purchased')
        
        assert 'categories' in analysis
        assert 'class_proportions' in analysis
        assert 'class_counts' in analysis
    
    def test_proportions_sum_to_one(self, df_classification):
        """Test that proportions sum to 1."""
        analysis = analyze_categorical_classification(df_classification, 'city', 'purchased')
        
        for cat in analysis['categories']:
            props = analysis['class_proportions'][str(cat)]
            total = sum(props.values())
            assert abs(total - 1.0) < 0.01
    
    def test_class_values(self, df_classification):
        """Test class values in output."""
        analysis = analyze_categorical_classification(df_classification, 'city', 'purchased')
        
        # Should have classes 0 and 1
        first_cat = analysis['class_proportions'][str(analysis['categories'][0])]
        assert '0' in first_cat
        assert '1' in first_cat


# ============================================================================
# Tests: Plotting Functions
# ============================================================================

class TestPlottingFunctions:
    """Tests for plotting functions."""
    
    def test_correlation_heatmap_plotly(self, df_regression):
        """Test Plotly correlation heatmap."""
        fig = plot_correlation_heatmap(df_regression, 'price', backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_correlation_heatmap_matplotlib(self, df_regression):
        """Test Matplotlib correlation heatmap."""
        fig = plot_correlation_heatmap(df_regression, 'price', backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_categorical_regression_plot_plotly(self, df_categorical_regression):
        """Test Plotly categorical regression plot."""
        fig = plot_categorical_regression(df_categorical_regression, 'city', 'price', backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_categorical_regression_plot_matplotlib(self, df_categorical_regression):
        """Test Matplotlib categorical regression plot."""
        fig = plot_categorical_regression(df_categorical_regression, 'city', 'price', backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_categorical_classification_plot_plotly(self, df_classification):
        """Test Plotly categorical classification plot."""
        fig = plot_categorical_classification(df_classification, 'city', 'purchased', backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_categorical_classification_plot_matplotlib(self, df_classification):
        """Test Matplotlib categorical classification plot."""
        fig = plot_categorical_classification(df_classification, 'city', 'purchased', backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_category(self):
        """Test with single category."""
        df = pd.DataFrame({
            'city': ['NYC'] * 100,
            'price': np.random.normal(100000, 30000, 100)
        })
        
        analysis = analyze_categorical_regression(df, 'city', 'price')
        assert len(analysis['categories']) == 1
    
    def test_two_classes(self):
        """Test with two classes."""
        df = pd.DataFrame({
            'city': np.random.choice(['NYC', 'LA'], 100),
            'purchased': np.random.choice([0, 1], 100)
        })
        
        analysis = analyze_categorical_classification(df, 'city', 'purchased')
        assert len(analysis['categories']) == 2
    
    def test_perfect_correlation(self):
        """Test with perfect correlation."""
        df = pd.DataFrame({
            'x': np.arange(100),
            'y': np.arange(100)
        })
        
        corr = compute_correlation_matrix(df, 'y')
        assert abs(corr['x'] - 1.0) < 0.01


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests."""
    
    def test_large_dataset_correlation(self):
        """Test correlation on large dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            'col1': np.random.randn(100000),
            'col2': np.random.randn(100000),
            'target': np.random.randn(100000)
        })
        
        corr = compute_correlation_matrix(df, 'target', sample_size=50000)
        assert len(corr) == 2
    
    def test_many_categories(self):
        """Test with many categories."""
        np.random.seed(42)
        df = pd.DataFrame({
            'category': np.random.choice(range(100), 10000),
            'price': np.random.normal(100000, 30000, 10000)
        })
        
        analysis = analyze_categorical_regression(df, 'category', 'price')
        assert len(analysis['categories']) == 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
