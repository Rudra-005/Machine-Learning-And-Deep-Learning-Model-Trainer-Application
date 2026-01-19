"""
Relationship Analyzer - Examples and Tests

Demonstrates feature-target relationship analysis.
"""

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
# Example 1: Correlation Analysis
# ============================================================================

def example_correlation_analysis():
    """Example 1: Compute and analyze correlations."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Correlation Analysis")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'experience': np.random.randint(0, 40, 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    # Pearson correlation
    print("\nPearson Correlation with price:")
    corr = compute_correlation_matrix(df, 'price', method='pearson')
    print(corr)
    
    # Spearman correlation
    print("\nSpearman Correlation with price:")
    corr = compute_correlation_matrix(df, 'price', method='spearman')
    print(corr)


# ============================================================================
# Example 2: Top Correlated Features
# ============================================================================

def example_top_features():
    """Example 2: Get top correlated features."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Top Correlated Features")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'experience': np.random.randint(0, 40, 1000),
        'education': np.random.randint(1, 5, 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    top = get_top_correlated_features(df, 'price', top_n=3)
    
    print(f"\nTop 3 features correlated with price:")
    for feat, corr in zip(top['features'], top['correlations']):
        print(f"  {feat}: {corr:.3f}")


# ============================================================================
# Example 3: Categorical Regression Analysis
# ============================================================================

def example_categorical_regression():
    """Example 3: Analyze target mean per category."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Categorical Regression Analysis")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston'], 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    analysis = analyze_categorical_regression(df, 'city', 'price')
    
    print(f"\nPrice by city:")
    for cat, mean, count in zip(analysis['categories'], analysis['means'], analysis['counts']):
        print(f"  {cat}: ${mean:,.0f} (n={count})")
    print(f"  Overall mean: ${analysis['overall_mean']:,.0f}")


# ============================================================================
# Example 4: Categorical Classification Analysis
# ============================================================================

def example_categorical_classification():
    """Example 4: Analyze class proportions per category."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Categorical Classification Analysis")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'purchased': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    analysis = analyze_categorical_classification(df, 'city', 'purchased')
    
    print(f"\nPurchase rate by city:")
    for cat in analysis['categories']:
        props = analysis['class_proportions'][str(cat)]
        print(f"  {cat}: No={props['0']:.1%}, Yes={props['1']:.1%}")


# ============================================================================
# Example 5: Correlation Heatmap
# ============================================================================

def example_correlation_heatmap():
    """Example 5: Create correlation heatmap."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Correlation Heatmap")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    print("\nCreating correlation heatmap...")
    fig = plot_correlation_heatmap(df, 'price', backend='plotly')
    if fig:
        print("✅ Heatmap created successfully")


# ============================================================================
# Example 6: Categorical Regression Plot
# ============================================================================

def example_categorical_regression_plot():
    """Example 6: Create categorical regression plot."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Categorical Regression Plot")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    print("\nCreating categorical regression plot...")
    fig = plot_categorical_regression(df, 'city', 'price', backend='plotly')
    if fig:
        print("✅ Plot created successfully")


# ============================================================================
# Example 7: Categorical Classification Plot
# ============================================================================

def example_categorical_classification_plot():
    """Example 7: Create categorical classification plot."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Categorical Classification Plot")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'purchased': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    })
    
    print("\nCreating categorical classification plot...")
    fig = plot_categorical_classification(df, 'city', 'purchased', backend='plotly')
    if fig:
        print("✅ Plot created successfully")


# ============================================================================
# Example 8: Large Dataset Handling
# ============================================================================

def example_large_dataset():
    """Example 8: Handle large datasets efficiently."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Large Dataset Handling")
    print("="*70)
    
    np.random.seed(42)
    
    print("\nCreating large dataset (1M rows)...")
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1_000_000),
        'income': np.random.normal(50000, 20000, 1_000_000),
        'price': np.random.normal(100000, 30000, 1_000_000)
    })
    print(f"✅ Dataset created: {len(df):,} rows")
    
    print("\nComputing correlation on sample...")
    corr = compute_correlation_matrix(df, 'price', sample_size=100000)
    print(f"✅ Correlation computed: {len(corr)} features")


# ============================================================================
# Example 9: Multiple Categorical Features
# ============================================================================

def example_multiple_categorical():
    """Example 9: Analyze multiple categorical features."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Multiple Categorical Features")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    print("\nAnalyzing multiple categorical features:")
    for feature in ['city', 'region']:
        analysis = analyze_categorical_regression(df, feature, 'price')
        print(f"\n  {feature}:")
        for cat, mean in zip(analysis['categories'], analysis['means']):
            print(f"    {cat}: ${mean:,.0f}")


# ============================================================================
# Example 10: Correlation Methods Comparison
# ============================================================================

def example_correlation_comparison():
    """Example 10: Compare Pearson and Spearman."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Correlation Methods Comparison")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'price': np.random.normal(100000, 30000, 1000)
    })
    
    print("\nPearson vs Spearman correlation:")
    pearson = compute_correlation_matrix(df, 'price', method='pearson')
    spearman = compute_correlation_matrix(df, 'price', method='spearman')
    
    for feat in pearson.index:
        p = pearson[feat]
        s = spearman[feat]
        print(f"  {feat}: Pearson={p:.3f}, Spearman={s:.3f}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RELATIONSHIP ANALYZER - EXAMPLES")
    print("="*70)
    
    example_correlation_analysis()
    example_top_features()
    example_categorical_regression()
    example_categorical_classification()
    example_correlation_heatmap()
    example_categorical_regression_plot()
    example_categorical_classification_plot()
    example_large_dataset()
    example_multiple_categorical()
    example_correlation_comparison()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
