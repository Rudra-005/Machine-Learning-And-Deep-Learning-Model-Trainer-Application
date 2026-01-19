"""
Unit Tests for Feature Analyzer

Comprehensive test suite for feature analysis functions.
Run with: pytest tests/test_feature_analyzer_unit.py -v
"""

import pytest
import pandas as pd
import numpy as np
from core.feature_analyzer import (
    detect_feature_types,
    get_feature_stats,
    plot_numerical_histogram,
    plot_numerical_boxplot,
    plot_categorical_bar,
    analyze_feature
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def df_mixed():
    """DataFrame with mixed feature types."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 20000, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100)
    })


@pytest.fixture
def df_numerical():
    """DataFrame with only numerical features."""
    np.random.seed(42)
    return pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randint(0, 100, 100),
        'col3': np.random.uniform(0, 1, 100)
    })


@pytest.fixture
def df_categorical():
    """DataFrame with only categorical features."""
    np.random.seed(42)
    return pd.DataFrame({
        'col1': np.random.choice(['A', 'B', 'C'], 100),
        'col2': np.random.choice(['X', 'Y', 'Z'], 100)
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    df.loc[::10, 'age'] = np.nan
    df.loc[::15, 'category'] = np.nan
    return df


# ============================================================================
# Tests: detect_feature_types
# ============================================================================

class TestDetectFeatureTypes:
    """Tests for detect_feature_types function."""
    
    def test_mixed_features(self, df_mixed):
        """Test detection of mixed feature types."""
        features = detect_feature_types(df_mixed)
        
        assert len(features['numerical']) == 2
        assert len(features['categorical']) == 2
        assert 'age' in features['numerical']
        assert 'category' in features['categorical']
    
    def test_only_numerical(self, df_numerical):
        """Test detection with only numerical features."""
        features = detect_feature_types(df_numerical)
        
        assert len(features['numerical']) == 3
        assert len(features['categorical']) == 0
    
    def test_only_categorical(self, df_categorical):
        """Test detection with only categorical features."""
        features = detect_feature_types(df_categorical)
        
        assert len(features['numerical']) == 0
        assert len(features['categorical']) == 2
    
    def test_output_structure(self, df_mixed):
        """Test output structure."""
        features = detect_feature_types(df_mixed)
        
        assert isinstance(features, dict)
        assert 'numerical' in features
        assert 'categorical' in features
        assert isinstance(features['numerical'], list)
        assert isinstance(features['categorical'], list)


# ============================================================================
# Tests: get_feature_stats
# ============================================================================

class TestGetFeatureStats:
    """Tests for get_feature_stats function."""
    
    def test_numerical_stats(self, df_mixed):
        """Test statistics for numerical feature."""
        stats = get_feature_stats(df_mixed, 'age')
        
        assert stats['type'] == 'numerical'
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
    
    def test_categorical_stats(self, df_mixed):
        """Test statistics for categorical feature."""
        stats = get_feature_stats(df_mixed, 'category')
        
        assert stats['type'] == 'categorical'
        assert 'unique' in stats
        assert 'top_value' in stats
        assert 'top_count' in stats
    
    def test_missing_values_count(self, df_with_missing):
        """Test missing values counting."""
        stats = get_feature_stats(df_with_missing, 'age')
        
        assert stats['missing'] > 0
        assert stats['missing'] == df_with_missing['age'].isna().sum()
    
    def test_invalid_column(self, df_mixed):
        """Test with invalid column name."""
        with pytest.raises(ValueError):
            get_feature_stats(df_mixed, 'nonexistent')
    
    def test_stats_accuracy(self, df_mixed):
        """Test accuracy of computed statistics."""
        stats = get_feature_stats(df_mixed, 'age')
        
        assert abs(stats['mean'] - df_mixed['age'].mean()) < 0.01
        assert abs(stats['median'] - df_mixed['age'].median()) < 0.01


# ============================================================================
# Tests: Plotting Functions
# ============================================================================

class TestPlottingFunctions:
    """Tests for plotting functions."""
    
    def test_histogram_plotly(self, df_mixed):
        """Test Plotly histogram."""
        fig = plot_numerical_histogram(df_mixed, 'age', backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_histogram_matplotlib(self, df_mixed):
        """Test Matplotlib histogram."""
        fig = plot_numerical_histogram(df_mixed, 'age', backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_boxplot_plotly(self, df_mixed):
        """Test Plotly boxplot."""
        fig = plot_numerical_boxplot(df_mixed, 'age', backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_boxplot_matplotlib(self, df_mixed):
        """Test Matplotlib boxplot."""
        fig = plot_numerical_boxplot(df_mixed, 'age', backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_bar_chart_plotly(self, df_mixed):
        """Test Plotly bar chart."""
        fig = plot_categorical_bar(df_mixed, 'category', backend='plotly')
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_bar_chart_matplotlib(self, df_mixed):
        """Test Matplotlib bar chart."""
        fig = plot_categorical_bar(df_mixed, 'category', backend='matplotlib')
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_invalid_column_plot(self, df_mixed):
        """Test plotting with invalid column."""
        with pytest.raises(ValueError):
            plot_numerical_histogram(df_mixed, 'nonexistent')
    
    def test_top_n_parameter(self, df_mixed):
        """Test top_n parameter in bar chart."""
        fig = plot_categorical_bar(df_mixed, 'category', backend='plotly', top_n=2)
        # Should not raise error
        assert fig is None or hasattr(fig, 'show')


# ============================================================================
# Tests: analyze_feature
# ============================================================================

class TestAnalyzeFeature:
    """Tests for analyze_feature function."""
    
    def test_numerical_analysis(self, df_mixed):
        """Test analysis of numerical feature."""
        analysis = analyze_feature(df_mixed, 'age')
        
        assert analysis['column'] == 'age'
        assert analysis['feature_type'] == 'numerical'
        assert 'stats' in analysis
        assert 'plot' in analysis
    
    def test_categorical_analysis(self, df_mixed):
        """Test analysis of categorical feature."""
        analysis = analyze_feature(df_mixed, 'category')
        
        assert analysis['column'] == 'category'
        assert analysis['feature_type'] == 'categorical'
        assert 'stats' in analysis
        assert 'plot' in analysis
    
    def test_invalid_column_analysis(self, df_mixed):
        """Test analysis with invalid column."""
        with pytest.raises(ValueError):
            analyze_feature(df_mixed, 'nonexistent')
    
    def test_analysis_structure(self, df_mixed):
        """Test analysis output structure."""
        analysis = analyze_feature(df_mixed, 'age')
        
        assert isinstance(analysis, dict)
        assert 'column' in analysis
        assert 'feature_type' in analysis
        assert 'stats' in analysis
        assert 'plot' in analysis


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({'col': [1]})
        features = detect_feature_types(df)
        assert len(features['numerical']) == 1
    
    def test_single_column(self):
        """Test with single column."""
        df = pd.DataFrame({'col': [1, 2, 3]})
        features = detect_feature_types(df)
        assert len(features['numerical']) == 1
    
    def test_all_missing(self):
        """Test with all missing values."""
        df = pd.DataFrame({'col': [np.nan, np.nan, np.nan]})
        stats = get_feature_stats(df, 'col')
        assert stats['missing'] == 3
    
    def test_single_unique_value(self):
        """Test with single unique value."""
        df = pd.DataFrame({'col': [1, 1, 1, 1]})
        stats = get_feature_stats(df, 'col')
        assert stats['std'] == 0.0
    
    def test_pandas_series(self):
        """Test with pandas Series."""
        df = pd.DataFrame({'col': pd.Series([1, 2, 3])})
        stats = get_feature_stats(df, 'col')
        assert stats['type'] == 'numerical'


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests."""
    
    def test_large_dataset_detection(self):
        """Test feature detection on large dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            'col1': np.random.randn(100000),
            'col2': np.random.choice(['A', 'B'], 100000)
        })
        
        features = detect_feature_types(df)
        assert len(features['numerical']) == 1
        assert len(features['categorical']) == 1
    
    def test_large_dataset_stats(self):
        """Test statistics on large dataset."""
        np.random.seed(42)
        df = pd.DataFrame({
            'col': np.random.randn(100000)
        })
        
        stats = get_feature_stats(df, 'col')
        assert 'mean' in stats


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_streamlit_workflow(self, df_mixed):
        """Test typical Streamlit workflow."""
        # Step 1: Detect features
        features = detect_feature_types(df_mixed)
        assert len(features['numerical']) > 0
        
        # Step 2: User selects feature
        selected = features['numerical'][0]
        
        # Step 3: Analyze selected feature
        analysis = analyze_feature(df_mixed, selected)
        assert analysis['column'] == selected
        assert analysis['feature_type'] == 'numerical'
    
    def test_multiple_feature_analysis(self, df_mixed):
        """Test analyzing multiple features."""
        features = detect_feature_types(df_mixed)
        
        analyses = []
        for col in features['numerical']:
            analysis = analyze_feature(df_mixed, col)
            analyses.append(analysis)
        
        assert len(analyses) == len(features['numerical'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
