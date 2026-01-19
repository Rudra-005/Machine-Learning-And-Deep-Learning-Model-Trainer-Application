"""
Feature Analyzer - Examples and Tests

Demonstrates feature analysis with user-selected features.
Optimized for Streamlit interactivity.
"""

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
# Example 1: Feature Type Detection
# ============================================================================

def example_feature_detection():
    """Example 1: Detect numerical and categorical features."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Feature Type Detection")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.normal(50000, 20000, 100),
        'score': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'city': np.random.choice(['NYC', 'LA', 'Chicago'], 100),
        'country': np.random.choice(['USA', 'Canada'], 100)
    })
    
    features = detect_feature_types(df)
    
    print(f"\nNumerical features ({len(features['numerical'])}):")
    for col in features['numerical']:
        print(f"  • {col}")
    
    print(f"\nCategorical features ({len(features['categorical'])}):")
    for col in features['categorical']:
        print(f"  • {col}")


# ============================================================================
# Example 2: Feature Statistics
# ============================================================================

def example_feature_stats():
    """Example 2: Get statistics for individual features."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Feature Statistics")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 1000)
    })
    
    # Numerical feature
    print("\nNumerical feature (age):")
    stats = get_feature_stats(df, 'age')
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Categorical feature
    print("\nCategorical feature (category):")
    stats = get_feature_stats(df, 'category')
    for key, value in stats.items():
        print(f"  {key}: {value}")


# ============================================================================
# Example 3: Numerical Feature Visualization
# ============================================================================

def example_numerical_plots():
    """Example 3: Create plots for numerical features."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Numerical Feature Visualization")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000)
    })
    
    print("\nCreating histogram...")
    fig = plot_numerical_histogram(df, 'age', backend='plotly')
    if fig:
        print("✅ Histogram created successfully")
    
    print("Creating boxplot...")
    fig = plot_numerical_boxplot(df, 'age', backend='plotly')
    if fig:
        print("✅ Boxplot created successfully")


# ============================================================================
# Example 4: Categorical Feature Visualization
# ============================================================================

def example_categorical_plot():
    """Example 4: Create plot for categorical feature."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Categorical Feature Visualization")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000)
    })
    
    print("\nCreating bar chart...")
    fig = plot_categorical_bar(df, 'category', backend='plotly')
    if fig:
        print("✅ Bar chart created successfully")


# ============================================================================
# Example 5: Single Feature Analysis
# ============================================================================

def example_single_feature_analysis():
    """Example 5: Comprehensive analysis for single feature."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Single Feature Analysis")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Analyze numerical feature
    print("\nAnalyzing numerical feature (age)...")
    analysis = analyze_feature(df, 'age')
    print(f"  Feature type: {analysis['feature_type']}")
    print(f"  Stats: {analysis['stats']}")
    print(f"  Plot created: {analysis['plot'] is not None}")
    
    # Analyze categorical feature
    print("\nAnalyzing categorical feature (category)...")
    analysis = analyze_feature(df, 'category')
    print(f"  Feature type: {analysis['feature_type']}")
    print(f"  Stats: {analysis['stats']}")
    print(f"  Plot created: {analysis['plot'] is not None}")


# ============================================================================
# Example 6: Streamlit Integration
# ============================================================================

def example_streamlit_integration():
    """Example 6: Streamlit integration pattern."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Streamlit Integration Pattern")
    print("="*70)
    
    print("""
import streamlit as st
from core.feature_analyzer import detect_feature_types, analyze_feature

# Load data
df = pd.read_csv('data.csv')

# Detect features
features = detect_feature_types(df)

# User selects feature
st.header("Feature Analysis")
selected_feature = st.selectbox(
    "Select a feature to analyze",
    options=features['numerical'] + features['categorical']
)

# Analyze selected feature
if selected_feature:
    analysis = analyze_feature(df, selected_feature)
    
    st.subheader(f"Analysis: {selected_feature}")
    st.json(analysis['stats'])
    
    if analysis['plot']:
        st.plotly_chart(analysis['plot'], use_container_width=True)
    """)


# ============================================================================
# Example 7: Multiple Feature Analysis
# ============================================================================

def example_multiple_features():
    """Example 7: Analyze multiple features."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Multiple Feature Analysis")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000),
        'city': np.random.choice(['NYC', 'LA'], 1000)
    })
    
    features = detect_feature_types(df)
    
    print("\nAnalyzing all features...")
    for col in features['numerical'][:2]:
        analysis = analyze_feature(df, col, backend='plotly')
        print(f"  ✓ {col} ({analysis['feature_type']})")
    
    for col in features['categorical'][:2]:
        analysis = analyze_feature(df, col, backend='plotly')
        print(f"  ✓ {col} ({analysis['feature_type']})")


# ============================================================================
# Example 8: Handling Missing Values
# ============================================================================

def example_missing_values():
    """Example 8: Handle missing values in features."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Handling Missing Values")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'category': np.random.choice(['A', 'B', 'C', None], 1000)
    })
    
    # Add missing values
    df.loc[::10, 'age'] = np.nan
    
    print("\nFeature with missing values:")
    stats = get_feature_stats(df, 'age')
    print(f"  Total: {stats['count']}")
    print(f"  Missing: {stats['missing']}")
    print(f"  Percentage: {stats['missing']/stats['count']*100:.2f}%")


# ============================================================================
# Example 9: Top N Categories
# ============================================================================

def example_top_categories():
    """Example 9: Show top N categories."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Top N Categories")
    print("="*70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'category': np.random.choice([f'Cat_{i}' for i in range(50)], 1000)
    })
    
    print("\nShowing top 10 categories...")
    fig = plot_categorical_bar(df, 'category', backend='plotly', top_n=10)
    if fig:
        print("✅ Bar chart with top 10 categories created")


# ============================================================================
# Example 10: Large Dataset Handling
# ============================================================================

def example_large_dataset():
    """Example 10: Handle large datasets efficiently."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Large Dataset Handling")
    print("="*70)
    
    np.random.seed(42)
    
    print("\nCreating large dataset (100K rows)...")
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100000),
        'income': np.random.normal(50000, 20000, 100000),
        'category': np.random.choice(['A', 'B', 'C'], 100000)
    })
    print(f"✅ Dataset created: {len(df):,} rows × {len(df.columns)} columns")
    
    print("\nDetecting features...")
    features = detect_feature_types(df)
    print(f"✅ Detected {len(features['numerical'])} numerical, {len(features['categorical'])} categorical")
    
    print("\nAnalyzing features...")
    for col in features['numerical'][:1]:
        analysis = analyze_feature(df, col, backend='plotly')
        print(f"✅ Analyzed {col}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEATURE ANALYZER - EXAMPLES")
    print("="*70)
    
    example_feature_detection()
    example_feature_stats()
    example_numerical_plots()
    example_categorical_plot()
    example_single_feature_analysis()
    example_streamlit_integration()
    example_multiple_features()
    example_missing_values()
    example_top_categories()
    example_large_dataset()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
