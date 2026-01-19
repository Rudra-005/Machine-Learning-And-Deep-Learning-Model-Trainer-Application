"""
Target Analyzer - Examples and Tests

Demonstrates all target analysis functions with realistic scenarios.
"""

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
# Example 1: Task Type Detection
# ============================================================================

def example_task_detection():
    """Example 1: Automatically detect task type."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Task Type Detection")
    print("="*70)
    
    # Binary classification
    y_binary = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    task = detect_task_type(y_binary)
    print(f"\nBinary classification:")
    print(f"  Task: {task.task_type}")
    print(f"  Confidence: {task.confidence:.2f}")
    print(f"  Unique values: {task.n_unique}")
    
    # Multi-class classification
    y_multiclass = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    task = detect_task_type(y_multiclass)
    print(f"\nMulti-class classification:")
    print(f"  Task: {task.task_type}")
    print(f"  Confidence: {task.confidence:.2f}")
    print(f"  Unique values: {task.n_unique}")
    
    # Regression
    y_regression = np.random.normal(100, 20, 100)
    task = detect_task_type(y_regression)
    print(f"\nRegression:")
    print(f"  Task: {task.task_type}")
    print(f"  Confidence: {task.confidence:.2f}")
    print(f"  Unique values: {task.n_unique}")


# ============================================================================
# Example 2: Classification Analysis
# ============================================================================

def example_classification_analysis():
    """Example 2: Analyze classification target."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Classification Analysis")
    print("="*70)
    
    # Balanced dataset
    y_balanced = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    metrics = analyze_classification(y_balanced)
    print("\nBalanced dataset:")
    print(f"  Classes: {metrics['n_classes']}")
    print(f"  Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    print(f"  Is imbalanced: {metrics['is_imbalanced']}")
    print(f"  Class distribution: {metrics['class_distribution']}")
    
    # Imbalanced dataset
    y_imbalanced = np.array([0]*95 + [1]*5)
    metrics = analyze_classification(y_imbalanced)
    print("\nImbalanced dataset (95% vs 5%):")
    print(f"  Classes: {metrics['n_classes']}")
    print(f"  Imbalance ratio: {metrics['imbalance_ratio']:.2f}")
    print(f"  Is imbalanced: {metrics['is_imbalanced']}")
    print(f"  Majority class: {metrics['majority_class']}")
    print(f"  Minority class: {metrics['minority_class']}")


# ============================================================================
# Example 3: Regression Analysis
# ============================================================================

def example_regression_analysis():
    """Example 3: Analyze regression target."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Regression Analysis")
    print("="*70)
    
    np.random.seed(42)
    
    # Normal distribution
    y_normal = np.random.normal(100, 20, 1000)
    metrics = analyze_regression(y_normal)
    print("\nNormal distribution:")
    print(f"  Mean: {metrics['mean']:.2f}")
    print(f"  Std: {metrics['std']:.2f}")
    print(f"  Skewness: {metrics['skewness']:.4f}")
    print(f"  Outliers: {metrics['n_outliers']} ({metrics['outlier_percentage']:.2f}%)")
    
    # Skewed distribution
    y_skewed = np.random.exponential(50, 1000)
    metrics = analyze_regression(y_skewed)
    print("\nSkewed distribution:")
    print(f"  Mean: {metrics['mean']:.2f}")
    print(f"  Median: {metrics['median']:.2f}")
    print(f"  Skewness: {metrics['skewness']:.4f}")
    print(f"  Outliers: {metrics['n_outliers']} ({metrics['outlier_percentage']:.2f}%)")


# ============================================================================
# Example 4: Classification Visualization
# ============================================================================

def example_classification_plot():
    """Example 4: Create classification plot."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Classification Visualization")
    print("="*70)
    
    y = np.array([0]*50 + [1]*30 + [2]*20)
    
    print("\nCreating class distribution plot...")
    fig = create_class_distribution_plot(y, backend='plotly')
    if fig:
        print("✅ Plotly plot created successfully")
        # fig.show()  # Uncomment to display
    else:
        print("⚠️  Plotly not available")
    
    fig = create_class_distribution_plot(y, backend='matplotlib')
    if fig:
        print("✅ Matplotlib plot created successfully")
        # fig.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
    else:
        print("⚠️  Matplotlib not available")


# ============================================================================
# Example 5: Regression Visualizations
# ============================================================================

def example_regression_plots():
    """Example 5: Create regression plots."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Regression Visualizations")
    print("="*70)
    
    np.random.seed(42)
    y = np.random.normal(100, 20, 1000)
    
    print("\nCreating histogram...")
    fig = create_regression_histogram(y, backend='plotly')
    if fig:
        print("✅ Histogram created successfully")
    
    print("Creating boxplot...")
    fig = create_regression_boxplot(y, backend='plotly')
    if fig:
        print("✅ Boxplot created successfully")


# ============================================================================
# Example 6: Comprehensive Analysis
# ============================================================================

def example_comprehensive_analysis():
    """Example 6: Full analysis with auto-detection."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Comprehensive Analysis")
    print("="*70)
    
    np.random.seed(42)
    
    # Classification
    y_class = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
    analysis = analyze_target(y_class, create_plots=True, backend='plotly')
    
    print("\nClassification Analysis:")
    print(f"  Task: {analysis['task_type']}")
    print(f"  Metrics: {analysis['metrics']}")
    if 'plots' in analysis:
        print(f"  Plots created: {len(analysis['plots'])}")
    
    # Regression
    y_reg = np.random.normal(100, 20, 1000)
    analysis = analyze_target(y_reg, create_plots=True, backend='plotly')
    
    print("\nRegression Analysis:")
    print(f"  Task: {analysis['task_type']}")
    print(f"  Metrics: {analysis['metrics']}")
    if 'plots' in analysis:
        print(f"  Plots created: {len(analysis['plots'])}")


# ============================================================================
# Example 7: Imbalanced Classification
# ============================================================================

def example_imbalanced_classification():
    """Example 7: Analyze imbalanced classification."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Imbalanced Classification Detection")
    print("="*70)
    
    # Highly imbalanced
    y = np.array([0]*950 + [1]*50)
    analysis = analyze_target(y, task_type='classification', create_plots=False)
    
    metrics = analysis['metrics']
    print(f"\nHighly imbalanced dataset (95% vs 5%):")
    print(f"  Imbalance ratio: {metrics['imbalance_ratio']:.2f}:1")
    print(f"  Is imbalanced: {metrics['is_imbalanced']}")
    print(f"  Class distribution: {metrics['class_distribution']}")
    print(f"  Recommendation: Use class weights!" if metrics['is_imbalanced'] else "  Balanced dataset")


# ============================================================================
# Example 8: Outlier Detection
# ============================================================================

def example_outlier_detection():
    """Example 8: Detect outliers in regression."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Outlier Detection")
    print("="*70)
    
    np.random.seed(42)
    
    # Normal data with outliers
    y = np.concatenate([
        np.random.normal(100, 20, 950),
        np.array([500, 600, 700, 800, 900])  # Outliers
    ])
    
    metrics = analyze_regression(y)
    print(f"\nData with outliers:")
    print(f"  Mean: {metrics['mean']:.2f}")
    print(f"  Median: {metrics['median']:.2f}")
    print(f"  Outliers detected: {metrics['n_outliers']} ({metrics['outlier_percentage']:.2f}%)")
    print(f"  IQR: {metrics['iqr']:.2f}")


# ============================================================================
# Example 9: Integration with Data Pipeline
# ============================================================================

def example_pipeline_integration():
    """Example 9: Integration with data pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Pipeline Integration")
    print("="*70)
    
    np.random.seed(42)
    
    # Create sample dataset
    X = np.random.randn(1000, 5)
    y = np.random.choice([0, 1], size=1000, p=[0.7, 0.3])
    
    # Step 1: Analyze target
    print("\nStep 1: Analyzing target variable...")
    analysis = analyze_target(y, create_plots=False)
    
    # Step 2: Check for imbalance
    print("Step 2: Checking for imbalance...")
    if analysis['metrics']['is_imbalanced']:
        print(f"  ⚠️  Imbalanced dataset detected!")
        print(f"  Imbalance ratio: {analysis['metrics']['imbalance_ratio']:.2f}:1")
        print(f"  Recommendation: Use class weights or resampling")
    else:
        print(f"  ✓ Balanced dataset")
    
    # Step 3: Prepare for training
    print("Step 3: Preparing for training...")
    print(f"  Task type: {analysis['task_type']}")
    print(f"  Number of classes: {analysis['metrics']['n_classes']}")
    print(f"  ✓ Ready for training")


# ============================================================================
# Example 10: Large Dataset Handling
# ============================================================================

def example_large_dataset():
    """Example 10: Handle large datasets efficiently."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Large Dataset Handling")
    print("="*70)
    
    np.random.seed(42)
    
    # Create large dataset
    print("\nCreating large dataset (1M samples)...")
    y = np.random.normal(100, 20, 1_000_000)
    print(f"✅ Dataset created: {len(y):,} samples")
    
    # Analyze
    print("Analyzing...")
    analysis = analyze_target(y, create_plots=False)
    
    print(f"✅ Analysis complete")
    print(f"  Mean: {analysis['metrics']['mean']:.2f}")
    print(f"  Std: {analysis['metrics']['std']:.2f}")
    print(f"  Outliers: {analysis['metrics']['n_outliers']:,}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TARGET ANALYZER - EXAMPLES")
    print("="*70)
    
    example_task_detection()
    example_classification_analysis()
    example_regression_analysis()
    example_classification_plot()
    example_regression_plots()
    example_comprehensive_analysis()
    example_imbalanced_classification()
    example_outlier_detection()
    example_pipeline_integration()
    example_large_dataset()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
