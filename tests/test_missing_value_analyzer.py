"""
Missing Value Analysis - Usage Examples and Tests

Demonstrates all functions with realistic datasets and scenarios.
"""

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


def create_sample_dataset(n_rows: int = 1000) -> pd.DataFrame:
    """Create realistic sample dataset with various missing patterns."""
    np.random.seed(42)
    
    df = pd.DataFrame({
        'id': range(n_rows),
        'age': np.random.randint(18, 80, n_rows),
        'income': np.random.normal(50000, 20000, n_rows),
        'education': np.random.choice(['HS', 'BS', 'MS', 'PhD', np.nan], n_rows, p=[0.3, 0.4, 0.2, 0.05, 0.05]),
        'employment_years': np.random.randint(0, 40, n_rows),
        'credit_score': np.random.randint(300, 850, n_rows),
        'loan_amount': np.random.normal(100000, 50000, n_rows),
        'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-employed', np.nan], n_rows, p=[0.7, 0.15, 0.1, 0.05]),
        'phone': np.random.choice([f'555-{i:04d}' for i in range(1000)] + [np.nan], n_rows, p=[0.95] + [0.05]),
        'email': np.random.choice([f'user{i}@email.com' for i in range(1000)] + [np.nan], n_rows, p=[0.92] + [0.08]),
    })
    
    # Add systematic missing patterns
    # Missing income for unemployed
    unemployed_idx = df[df['employment_status'] == 'Unemployed'].index
    df.loc[unemployed_idx[:len(unemployed_idx)//2], 'income'] = np.nan
    
    # Missing employment_years for self-employed
    self_employed_idx = df[df['employment_status'] == 'Self-employed'].index
    df.loc[self_employed_idx, 'employment_years'] = np.nan
    
    # Missing loan_amount for some
    df.loc[df.sample(frac=0.15).index, 'loan_amount'] = np.nan
    
    return df


# ============================================================================
# Example 1: Basic Missing Statistics
# ============================================================================

def example_basic_statistics():
    """Example 1: Compute and display missing statistics."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Missing Statistics")
    print("="*70)
    
    df = create_sample_dataset(n_rows=1000)
    
    # Compute statistics
    stats = compute_missing_stats(df)
    
    print("\nMissing Value Statistics:")
    print(stats.to_string())
    
    print("\n✅ Statistics computed successfully")
    print(f"   Total columns: {len(stats)}")
    print(f"   Columns with missing: {(stats['missing_pct'] > 0).sum()}")
    print(f"   Total missing cells: {stats['missing_count'].sum():,}")


# ============================================================================
# Example 2: Threshold Detection
# ============================================================================

def example_threshold_detection():
    """Example 2: Identify columns exceeding threshold."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Threshold Detection")
    print("="*70)
    
    df = create_sample_dataset(n_rows=1000)
    
    # Check different thresholds
    for threshold in [5, 10, 15, 20]:
        result = get_columns_above_threshold(df, threshold=threshold)
        print(f"\nColumns exceeding {threshold}% threshold: {result['count']}")
        for detail in result['details']:
            print(f"  • {detail['column']}: {detail['missing_pct']:.2f}%")


# ============================================================================
# Example 3: Missing Patterns
# ============================================================================

def example_missing_patterns():
    """Example 3: Analyze missing value patterns."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Missing Value Patterns")
    print("="*70)
    
    df = create_sample_dataset(n_rows=1000)
    
    patterns = get_missing_patterns(df)
    
    print(f"\nTotal distinct patterns: {patterns['total_patterns']}")
    print(f"Rows with any missing: {patterns['rows_with_missing']:,} ({patterns['rows_with_missing_pct']:.1f}%)")
    print(f"Completely missing rows: {patterns['completely_missing_rows']}")
    
    print("\nTop missing patterns:")
    for i, pattern in enumerate(patterns['top_patterns'], 1):
        cols = ', '.join(pattern['columns'][:3])
        if len(pattern['columns']) > 3:
            cols += f", +{len(pattern['columns']) - 3} more"
        print(f"  {i}. {cols}")
        print(f"     Frequency: {pattern['frequency']} rows ({pattern['percentage']:.1f}%)")


# ============================================================================
# Example 4: Visualizations - Plotly
# ============================================================================

def example_plotly_visualizations():
    """Example 4: Create Plotly visualizations."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Plotly Visualizations")
    print("="*70)
    
    df = create_sample_dataset(n_rows=1000)
    
    # Bar chart
    print("\nCreating bar chart...")
    bar_fig = create_missing_bar_chart(df, backend='plotly')
    if bar_fig:
        print("✅ Bar chart created successfully")
        # bar_fig.show()  # Uncomment to display
    else:
        print("⚠️  Plotly not available")
    
    # Heatmap
    print("Creating heatmap...")
    heatmap_fig = create_missing_heatmap(df, backend='plotly', sample_rows=500)
    if heatmap_fig:
        print("✅ Heatmap created successfully")
        # heatmap_fig.show()  # Uncomment to display
    else:
        print("⚠️  Plotly not available")


# ============================================================================
# Example 5: Visualizations - Matplotlib
# ============================================================================

def example_matplotlib_visualizations():
    """Example 5: Create Matplotlib visualizations."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Matplotlib Visualizations")
    print("="*70)
    
    df = create_sample_dataset(n_rows=1000)
    
    # Bar chart
    print("\nCreating bar chart...")
    bar_fig = create_missing_bar_chart(df, backend='matplotlib', figsize=(14, 8))
    if bar_fig:
        print("✅ Bar chart created successfully")
        # bar_fig.savefig('missing_bar_chart.png', dpi=150, bbox_inches='tight')
    else:
        print("⚠️  Matplotlib not available")
    
    # Heatmap
    print("Creating heatmap...")
    heatmap_fig = create_missing_heatmap(df, backend='matplotlib', sample_rows=500)
    if heatmap_fig:
        print("✅ Heatmap created successfully")
        # heatmap_fig.savefig('missing_heatmap.png', dpi=150, bbox_inches='tight')
    else:
        print("⚠️  Matplotlib not available")


# ============================================================================
# Example 6: Comprehensive Analysis
# ============================================================================

def example_comprehensive_analysis():
    """Example 6: Full analysis with all components."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Comprehensive Analysis")
    print("="*70)
    
    df = create_sample_dataset(n_rows=1000)
    
    # Run complete analysis
    analysis = analyze_missing_values(
        df,
        threshold=10.0,
        create_plots=True,
        backend='plotly'
    )
    
    # Print summary
    print("\n" + analysis['summary'])
    
    # Access individual components
    print("\nStatistics DataFrame:")
    print(analysis['stats'].head(10).to_string())
    
    print(f"\nColumns above threshold: {analysis['above_threshold']['count']}")
    print(f"Distinct patterns: {analysis['patterns']['total_patterns']}")
    
    if 'plots' in analysis:
        print(f"\nVisualizations created: {len(analysis['plots'])} plots")


# ============================================================================
# Example 7: Large Dataset Handling
# ============================================================================

def example_large_dataset():
    """Example 7: Efficient handling of large datasets."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Large Dataset Handling")
    print("="*70)
    
    # Create large dataset
    print("\nCreating large dataset (100K rows)...")
    df = create_sample_dataset(n_rows=100000)
    print(f"✅ Dataset created: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Compute statistics (efficient)
    print("\nComputing statistics...")
    stats = compute_missing_stats(df)
    print(f"✅ Statistics computed in memory-efficient manner")
    
    # Analyze patterns with sampling
    print("\nAnalyzing patterns (with sampling)...")
    patterns = get_missing_patterns(df, sample_size=10000)
    print(f"✅ Patterns analyzed on sample of 10,000 rows")
    print(f"   Total patterns: {patterns['total_patterns']}")
    
    # Create visualizations with sampling
    print("\nCreating heatmap (with sampling)...")
    heatmap_fig = create_missing_heatmap(df, backend='plotly', sample_rows=500)
    if heatmap_fig:
        print(f"✅ Heatmap created for 500 sampled rows")


# ============================================================================
# Example 8: Integration with Data Pipeline
# ============================================================================

def example_pipeline_integration():
    """Example 8: Integration with data preprocessing pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Pipeline Integration")
    print("="*70)
    
    df = create_sample_dataset(n_rows=1000)
    
    # Step 1: Analyze missing values
    print("\nStep 1: Analyzing missing values...")
    analysis = analyze_missing_values(df, threshold=15, create_plots=False)
    
    # Step 2: Identify columns to drop
    print("Step 2: Identifying columns to drop...")
    cols_to_drop = analysis['above_threshold']['columns']
    if cols_to_drop:
        print(f"   Columns to drop: {cols_to_drop}")
        df_cleaned = df.drop(columns=cols_to_drop)
        print(f"   ✅ Dropped {len(cols_to_drop)} columns")
    else:
        df_cleaned = df
        print("   No columns exceed threshold")
    
    # Step 3: Handle remaining missing values
    print("Step 3: Handling remaining missing values...")
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    
    # Fill numeric with median
    for col in numeric_cols:
        if df_cleaned[col].isna().any():
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    
    # Fill categorical with mode
    for col in categorical_cols:
        if df_cleaned[col].isna().any():
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    
    print(f"   ✅ Filled missing values")
    
    # Step 4: Verify
    print("Step 4: Verifying...")
    final_stats = compute_missing_stats(df_cleaned)
    missing_remaining = final_stats['missing_count'].sum()
    print(f"   Missing values remaining: {missing_remaining}")
    print(f"   ✅ Pipeline complete")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MISSING VALUE ANALYSIS - EXAMPLES AND TESTS")
    print("="*70)
    
    # Run all examples
    example_basic_statistics()
    example_threshold_detection()
    example_missing_patterns()
    example_plotly_visualizations()
    example_matplotlib_visualizations()
    example_comprehensive_analysis()
    example_large_dataset()
    example_pipeline_integration()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
