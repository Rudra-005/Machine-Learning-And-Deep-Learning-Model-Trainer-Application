"""
Unit Tests for Missing Value Analyzer

Comprehensive test suite for missing value analysis functions.
Run with: pytest tests/test_missing_value_analyzer_unit.py -v
"""

import pytest
import pandas as pd
import numpy as np
from core.missing_value_analyzer import (
    compute_missing_stats,
    get_columns_above_threshold,
    get_missing_patterns,
    create_missing_bar_chart,
    create_missing_heatmap,
    analyze_missing_values
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def df_no_missing():
    """DataFrame with no missing values."""
    return pd.DataFrame({
        'a': range(100),
        'b': np.random.randn(100),
        'c': ['x'] * 100
    })


@pytest.fixture
def df_with_missing():
    """DataFrame with various missing patterns."""
    df = pd.DataFrame({
        'col_no_missing': range(100),
        'col_low_missing': [np.nan if i % 20 == 0 else i for i in range(100)],
        'col_med_missing': [np.nan if i % 5 == 0 else i for i in range(100)],
        'col_high_missing': [np.nan if i % 2 == 0 else i for i in range(100)],
    })
    return df


@pytest.fixture
def df_large():
    """Large DataFrame for performance testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'col1': np.random.randn(10000),
        'col2': [np.nan if i % 100 == 0 else i for i in range(10000)],
        'col3': [np.nan if i % 50 == 0 else i for i in range(10000)],
    })


# ============================================================================
# Tests: compute_missing_stats
# ============================================================================

class TestComputeMissingStats:
    """Tests for compute_missing_stats function."""
    
    def test_no_missing_values(self, df_no_missing):
        """Test with DataFrame containing no missing values."""
        stats = compute_missing_stats(df_no_missing)
        
        assert len(stats) == 3
        assert (stats['missing_pct'] == 0).all()
        assert (stats['severity'] == 'OK').all()
    
    def test_with_missing_values(self, df_with_missing):
        """Test with DataFrame containing missing values."""
        stats = compute_missing_stats(df_with_missing)
        
        assert len(stats) == 4
        assert stats.loc[0, 'column'] == 'col_high_missing'  # Sorted descending
        assert stats.loc[0, 'missing_pct'] == 50.0
        assert stats.loc[0, 'severity'] == 'High'
    
    def test_severity_classification(self, df_with_missing):
        """Test severity level classification."""
        stats = compute_missing_stats(df_with_missing)
        
        severity_map = {
            'col_no_missing': 'OK',
            'col_low_missing': 'Low',
            'col_med_missing': 'Medium',
            'col_high_missing': 'High'
        }
        
        for col, expected_severity in severity_map.items():
            actual_severity = stats[stats['column'] == col]['severity'].values[0]
            assert actual_severity == expected_severity
    
    def test_missing_count_accuracy(self, df_with_missing):
        """Test accuracy of missing count calculation."""
        stats = compute_missing_stats(df_with_missing)
        
        for _, row in stats.iterrows():
            col = row['column']
            expected_count = df_with_missing[col].isna().sum()
            assert row['missing_count'] == expected_count
    
    def test_output_columns(self, df_no_missing):
        """Test that output has required columns."""
        stats = compute_missing_stats(df_no_missing)
        
        required_cols = ['column', 'missing_count', 'missing_pct', 'dtype', 'severity']
        assert all(col in stats.columns for col in required_cols)
    
    def test_sorted_by_missing_pct(self, df_with_missing):
        """Test that output is sorted by missing_pct descending."""
        stats = compute_missing_stats(df_with_missing)
        
        assert (stats['missing_pct'].iloc[:-1] >= stats['missing_pct'].iloc[1:]).all()


# ============================================================================
# Tests: get_columns_above_threshold
# ============================================================================

class TestGetColumnsAboveThreshold:
    """Tests for get_columns_above_threshold function."""
    
    def test_no_columns_above_threshold(self, df_no_missing):
        """Test when no columns exceed threshold."""
        result = get_columns_above_threshold(df_no_missing, threshold=20)
        
        assert result['count'] == 0
        assert result['columns'] == []
        assert result['threshold'] == 20
    
    def test_columns_above_threshold(self, df_with_missing):
        """Test identification of columns above threshold."""
        result = get_columns_above_threshold(df_with_missing, threshold=20)
        
        assert result['count'] == 2  # col_med_missing (20%) and col_high_missing (50%)
        assert 'col_high_missing' in result['columns']
        assert 'col_med_missing' in result['columns']
    
    def test_threshold_boundary(self, df_with_missing):
        """Test threshold boundary conditions."""
        # Exactly at threshold
        result = get_columns_above_threshold(df_with_missing, threshold=20)
        assert 'col_med_missing' in result['columns']  # 20% >= 20%
        
        # Just below threshold
        result = get_columns_above_threshold(df_with_missing, threshold=20.1)
        assert 'col_med_missing' not in result['columns']
    
    def test_output_structure(self, df_with_missing):
        """Test output dictionary structure."""
        result = get_columns_above_threshold(df_with_missing, threshold=15)
        
        assert 'columns' in result
        assert 'count' in result
        assert 'threshold' in result
        assert 'details' in result
        assert isinstance(result['columns'], list)
        assert isinstance(result['details'], list)
    
    def test_details_accuracy(self, df_with_missing):
        """Test accuracy of details in output."""
        result = get_columns_above_threshold(df_with_missing, threshold=15)
        
        for detail in result['details']:
            assert 'column' in detail
            assert 'missing_count' in detail
            assert 'missing_pct' in detail
            assert detail['missing_pct'] >= 15


# ============================================================================
# Tests: get_missing_patterns
# ============================================================================

class TestGetMissingPatterns:
    """Tests for get_missing_patterns function."""
    
    def test_no_missing_patterns(self, df_no_missing):
        """Test with DataFrame containing no missing values."""
        patterns = get_missing_patterns(df_no_missing)
        
        assert patterns['total_patterns'] == 1
        assert patterns['rows_with_missing'] == 0
        assert patterns['rows_with_missing_pct'] == 0.0
        assert patterns['completely_missing_rows'] == 0
    
    def test_with_missing_patterns(self, df_with_missing):
        """Test pattern detection with missing values."""
        patterns = get_missing_patterns(df_with_missing)
        
        assert patterns['total_patterns'] > 0
        assert patterns['rows_with_missing'] > 0
        assert patterns['rows_with_missing_pct'] > 0
    
    def test_top_patterns_structure(self, df_with_missing):
        """Test structure of top patterns."""
        patterns = get_missing_patterns(df_with_missing)
        
        for pattern in patterns['top_patterns']:
            assert 'columns' in pattern
            assert 'frequency' in pattern
            assert 'percentage' in pattern
            assert isinstance(pattern['columns'], list)
    
    def test_sampling_for_large_dataset(self, df_large):
        """Test that sampling works for large datasets."""
        patterns = get_missing_patterns(df_large, sample_size=1000)
        
        # Should complete without memory issues
        assert patterns['total_patterns'] > 0
    
    def test_rows_with_missing_count(self, df_with_missing):
        """Test accuracy of rows_with_missing count."""
        patterns = get_missing_patterns(df_with_missing)
        
        # Manually count rows with any missing
        expected_count = df_with_missing.isna().any(axis=1).sum()
        assert patterns['rows_with_missing'] == expected_count


# ============================================================================
# Tests: create_missing_bar_chart
# ============================================================================

class TestCreateMissingBarChart:
    """Tests for create_missing_bar_chart function."""
    
    def test_plotly_backend(self, df_with_missing):
        """Test Plotly backend."""
        fig = create_missing_bar_chart(df_with_missing, backend='plotly')
        
        if fig is not None:
            assert hasattr(fig, 'show')  # Plotly figure
            assert hasattr(fig, 'update_layout')
    
    def test_matplotlib_backend(self, df_with_missing):
        """Test Matplotlib backend."""
        fig = create_missing_bar_chart(df_with_missing, backend='matplotlib')
        
        if fig is not None:
            assert hasattr(fig, 'savefig')  # Matplotlib figure
    
    def test_no_missing_values(self, df_no_missing):
        """Test with DataFrame containing no missing values."""
        fig = create_missing_bar_chart(df_no_missing, backend='plotly')
        
        # Should return None or handle gracefully
        assert fig is None or hasattr(fig, 'show')
    
    def test_figsize_parameter(self, df_with_missing):
        """Test figsize parameter for matplotlib."""
        fig = create_missing_bar_chart(
            df_with_missing,
            backend='matplotlib',
            figsize=(16, 10)
        )
        
        if fig is not None:
            assert fig.get_figwidth() == 16
            assert fig.get_figheight() == 10


# ============================================================================
# Tests: create_missing_heatmap
# ============================================================================

class TestCreateMissingHeatmap:
    """Tests for create_missing_heatmap function."""
    
    def test_plotly_backend(self, df_with_missing):
        """Test Plotly backend."""
        fig = create_missing_heatmap(df_with_missing, backend='plotly')
        
        if fig is not None:
            assert hasattr(fig, 'show')
    
    def test_matplotlib_backend(self, df_with_missing):
        """Test Matplotlib backend."""
        fig = create_missing_heatmap(df_with_missing, backend='matplotlib')
        
        if fig is not None:
            assert hasattr(fig, 'savefig')
    
    def test_sampling(self, df_large):
        """Test sampling for large datasets."""
        fig = create_missing_heatmap(df_large, backend='plotly', sample_rows=100)
        
        # Should complete without memory issues
        assert fig is None or hasattr(fig, 'show')
    
    def test_sample_rows_parameter(self, df_with_missing):
        """Test sample_rows parameter."""
        fig = create_missing_heatmap(
            df_with_missing,
            backend='matplotlib',
            sample_rows=50
        )
        
        # Should not raise error
        assert fig is None or hasattr(fig, 'savefig')


# ============================================================================
# Tests: analyze_missing_values
# ============================================================================

class TestAnalyzeMissingValues:
    """Tests for analyze_missing_values function."""
    
    def test_comprehensive_analysis(self, df_with_missing):
        """Test comprehensive analysis."""
        analysis = analyze_missing_values(df_with_missing, create_plots=False)
        
        assert 'stats' in analysis
        assert 'above_threshold' in analysis
        assert 'patterns' in analysis
        assert 'summary' in analysis
        assert 'threshold' in analysis
    
    def test_with_plots(self, df_with_missing):
        """Test analysis with plots."""
        analysis = analyze_missing_values(
            df_with_missing,
            create_plots=True,
            backend='plotly'
        )
        
        assert 'plots' in analysis
        assert 'bar_chart' in analysis['plots']
        assert 'heatmap' in analysis['plots']
    
    def test_threshold_parameter(self, df_with_missing):
        """Test threshold parameter."""
        analysis = analyze_missing_values(df_with_missing, threshold=15)
        
        assert analysis['threshold'] == 15
        assert analysis['above_threshold']['threshold'] == 15
    
    def test_summary_generation(self, df_with_missing):
        """Test summary text generation."""
        analysis = analyze_missing_values(df_with_missing, create_plots=False)
        
        summary = analysis['summary']
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'MISSING VALUE ANALYSIS SUMMARY' in summary
    
    def test_large_dataset_handling(self, df_large):
        """Test handling of large datasets."""
        analysis = analyze_missing_values(df_large, create_plots=True)
        
        # Should complete without memory issues
        assert 'stats' in analysis
        assert 'patterns' in analysis


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_pipeline_flow(self, df_with_missing):
        """Test typical data quality pipeline."""
        # Step 1: Analyze
        analysis = analyze_missing_values(df_with_missing, threshold=20)
        
        # Step 2: Identify columns to drop
        cols_to_drop = analysis['above_threshold']['columns']
        
        # Step 3: Drop columns
        df_cleaned = df_with_missing.drop(columns=cols_to_drop)
        
        # Step 4: Verify
        final_analysis = analyze_missing_values(df_cleaned, create_plots=False)
        
        # Verify improvement
        assert final_analysis['above_threshold']['count'] <= analysis['above_threshold']['count']
    
    def test_multiple_thresholds(self, df_with_missing):
        """Test analysis with multiple thresholds."""
        thresholds = [5, 10, 20, 50]
        results = []
        
        for threshold in thresholds:
            result = get_columns_above_threshold(df_with_missing, threshold=threshold)
            results.append(result['count'])
        
        # Results should be non-increasing
        assert all(results[i] >= results[i+1] for i in range(len(results)-1))


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        stats = compute_missing_stats(df)
        assert len(stats) == 0
    
    def test_single_column(self):
        """Test with single column."""
        df = pd.DataFrame({'col': [1, 2, np.nan, 4, 5]})
        stats = compute_missing_stats(df)
        assert len(stats) == 1
        assert stats.iloc[0]['missing_pct'] == 20.0
    
    def test_single_row(self):
        """Test with single row."""
        df = pd.DataFrame({'a': [1], 'b': [np.nan], 'c': [3]})
        stats = compute_missing_stats(df)
        assert len(stats) == 3
    
    def test_all_missing(self):
        """Test with all missing values."""
        df = pd.DataFrame({'col': [np.nan] * 10})
        stats = compute_missing_stats(df)
        assert stats.iloc[0]['missing_pct'] == 100.0
        assert stats.iloc[0]['severity'] == 'Critical'
    
    def test_mixed_dtypes(self):
        """Test with mixed data types."""
        df = pd.DataFrame({
            'int_col': [1, 2, np.nan, 4],
            'float_col': [1.1, np.nan, 3.3, 4.4],
            'str_col': ['a', 'b', np.nan, 'd'],
            'bool_col': [True, False, np.nan, True]
        })
        stats = compute_missing_stats(df)
        assert len(stats) == 4


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for large datasets."""
    
    def test_large_dataset_performance(self, df_large):
        """Test performance with large dataset."""
        import time
        
        start = time.time()
        stats = compute_missing_stats(df_large)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second
    
    def test_pattern_analysis_performance(self, df_large):
        """Test pattern analysis performance."""
        import time
        
        start = time.time()
        patterns = get_missing_patterns(df_large, sample_size=5000)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0  # Less than 5 seconds


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
