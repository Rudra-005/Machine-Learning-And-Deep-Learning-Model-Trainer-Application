"""
Missing Value Analyzer - Quick Reference Guide

Production-ready functions for missing value analysis with efficient
handling of large datasets and dual visualization backends.
"""

# ============================================================================
# QUICK START
# ============================================================================

"""
from core.missing_value_analyzer import analyze_missing_values

# One-line comprehensive analysis
analysis = analyze_missing_values(df, threshold=20, backend='plotly')
print(analysis['summary'])
analysis['plots']['bar_chart'].show()
"""


# ============================================================================
# API REFERENCE
# ============================================================================

"""
1. compute_missing_stats(df: pd.DataFrame) -> pd.DataFrame
   ─────────────────────────────────────────────────────────
   Compute missing value count and percentage per column.
   
   Returns DataFrame with columns:
   - column: Column name
   - missing_count: Number of missing values
   - missing_pct: Percentage of missing values
   - dtype: Data type
   - severity: 'OK', 'Low', 'Medium', 'High', 'Critical'
   
   Example:
   >>> stats = compute_missing_stats(df)
   >>> print(stats[stats['missing_pct'] > 10])


2. get_columns_above_threshold(df, threshold=20.0) -> Dict
   ────────────────────────────────────────────────────────
   Identify columns exceeding missing value threshold.
   
   Returns Dict with:
   - columns: List of column names
   - count: Number of columns
   - threshold: Threshold used
   - details: List of column statistics
   
   Example:
   >>> result = get_columns_above_threshold(df, threshold=15)
   >>> print(f"Columns above 15%: {result['count']}")
   >>> for col in result['columns']:
   ...     print(col)


3. get_missing_patterns(df, sample_size=None) -> Dict
   ──────────────────────────────────────────────────
   Identify missing value patterns and co-occurrence.
   
   Returns Dict with:
   - total_patterns: Number of distinct patterns
   - rows_with_missing: Count of rows with any missing
   - rows_with_missing_pct: Percentage of rows
   - completely_missing_rows: Count of all-missing rows
   - top_patterns: List of most common patterns
   
   Example:
   >>> patterns = get_missing_patterns(df)
   >>> print(f"Rows with missing: {patterns['rows_with_missing']}")
   >>> for pattern in patterns['top_patterns']:
   ...     print(f"Pattern: {pattern['columns']}, Frequency: {pattern['frequency']}")


4. create_missing_bar_chart(df, backend='plotly', ...) -> Figure
   ──────────────────────────────────────────────────────────────
   Create bar chart of missing percentages.
   
   Args:
   - backend: 'plotly' or 'matplotlib'
   - figsize: (width, height) for matplotlib
   - dpi: Resolution for matplotlib
   
   Returns: Figure object (go.Figure or plt.Figure)
   
   Example:
   >>> fig = create_missing_bar_chart(df, backend='plotly')
   >>> fig.show()
   
   >>> fig = create_missing_bar_chart(df, backend='matplotlib', figsize=(14, 8))
   >>> fig.savefig('chart.png', dpi=150, bbox_inches='tight')


5. create_missing_heatmap(df, backend='plotly', sample_rows=500, ...) -> Figure
   ────────────────────────────────────────────────────────────────────────────
   Create heatmap of missing values.
   
   Args:
   - backend: 'plotly' or 'matplotlib'
   - sample_rows: Max rows to display (auto-samples large datasets)
   - figsize: (width, height) for matplotlib
   - dpi: Resolution for matplotlib
   
   Returns: Figure object (go.Figure or plt.Figure)
   
   Example:
   >>> fig = create_missing_heatmap(df, backend='plotly', sample_rows=1000)
   >>> fig.show()
   
   >>> fig = create_missing_heatmap(df, backend='matplotlib', sample_rows=500)
   >>> fig.savefig('heatmap.png', dpi=150, bbox_inches='tight')


6. analyze_missing_values(df, threshold=20, create_plots=True, backend='plotly') -> Dict
   ──────────────────────────────────────────────────────────────────────────────────────
   Comprehensive missing value analysis (main orchestration function).
   
   Returns Dict with:
   - stats: DataFrame of missing statistics
   - above_threshold: Dict of columns exceeding threshold
   - patterns: Dict of missing patterns
   - plots: Dict of figure objects (if create_plots=True)
   - summary: Text summary of findings
   
   Example:
   >>> analysis = analyze_missing_values(df, threshold=15)
   >>> print(analysis['summary'])
   >>> print(analysis['stats'])
   >>> analysis['plots']['bar_chart'].show()
"""


# ============================================================================
# COMMON USAGE PATTERNS
# ============================================================================

"""
PATTERN 1: Quick Overview
─────────────────────────
from core.missing_value_analyzer import analyze_missing_values

analysis = analyze_missing_values(df)
print(analysis['summary'])


PATTERN 2: Identify Problem Columns
────────────────────────────────────
from core.missing_value_analyzer import get_columns_above_threshold

result = get_columns_above_threshold(df, threshold=20)
print(f"Columns to investigate: {result['columns']}")


PATTERN 3: Analyze Patterns
────────────────────────────
from core.missing_value_analyzer import get_missing_patterns

patterns = get_missing_patterns(df)
print(f"Rows affected: {patterns['rows_with_missing']}")
print(f"Distinct patterns: {patterns['total_patterns']}")


PATTERN 4: Create Visualizations
─────────────────────────────────
from core.missing_value_analyzer import create_missing_bar_chart, create_missing_heatmap

# Plotly (interactive)
bar_fig = create_missing_bar_chart(df, backend='plotly')
bar_fig.show()

heatmap_fig = create_missing_heatmap(df, backend='plotly')
heatmap_fig.show()

# Matplotlib (static, saveable)
bar_fig = create_missing_bar_chart(df, backend='matplotlib')
bar_fig.savefig('missing_bar.png', dpi=150, bbox_inches='tight')

heatmap_fig = create_missing_heatmap(df, backend='matplotlib')
heatmap_fig.savefig('missing_heatmap.png', dpi=150, bbox_inches='tight')


PATTERN 5: Data Cleaning Pipeline
──────────────────────────────────
from core.missing_value_analyzer import analyze_missing_values

# Step 1: Analyze
analysis = analyze_missing_values(df, threshold=30)

# Step 2: Drop high-missing columns
df_cleaned = df.drop(columns=analysis['above_threshold']['columns'])

# Step 3: Impute remaining
df_cleaned.fillna(df_cleaned.median(), inplace=True)

# Step 4: Verify
final_analysis = analyze_missing_values(df_cleaned, create_plots=False)
print(final_analysis['summary'])


PATTERN 6: Large Dataset Handling
──────────────────────────────────
from core.missing_value_analyzer import analyze_missing_values

# Automatically samples for visualization
analysis = analyze_missing_values(
    df,  # 1M+ rows
    threshold=20,
    create_plots=True,
    backend='plotly'
)

# Heatmap uses 500 rows by default
# Patterns analysis uses sampling for efficiency


PATTERN 7: Integration with Streamlit
──────────────────────────────────────
import streamlit as st
from core.missing_value_analyzer import analyze_missing_values

st.header("Missing Value Analysis")

uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    threshold = st.slider("Threshold (%)", 5, 50, 20)
    backend = st.radio("Visualization", ["plotly", "matplotlib"])
    
    analysis = analyze_missing_values(df, threshold=threshold, backend=backend)
    
    st.text(analysis['summary'])
    
    if 'plots' in analysis:
        st.plotly_chart(analysis['plots']['bar_chart'])
        st.plotly_chart(analysis['plots']['heatmap'])
"""


# ============================================================================
# SEVERITY LEVELS
# ============================================================================

"""
Severity Classification:
─────────────────────────
OK        : 0% missing
Low       : < 5% missing
Medium    : 5-20% missing
High      : 20-50% missing
Critical  : > 50% missing

Color Coding:
─────────────
OK        : Green (#2ecc71)
Low       : Dark Green (#27ae60)
Medium    : Orange (#f39c12)
High      : Red (#e74c3c)
Critical  : Dark Red (#c0392b)
"""


# ============================================================================
# PERFORMANCE CHARACTERISTICS
# ============================================================================

"""
Dataset Size    | compute_missing_stats | get_missing_patterns | Visualization
─────────────────────────────────────────────────────────────────────────────
1K rows         | < 1ms                 | < 5ms                | < 100ms
10K rows        | < 5ms                 | < 50ms               | < 200ms
100K rows       | < 50ms                | < 500ms              | < 500ms (sampled)
1M rows         | < 500ms               | < 5s (sampled)       | < 1s (sampled)
10M rows        | < 5s                  | < 10s (sampled)      | < 2s (sampled)

Notes:
- compute_missing_stats: O(n*m) where n=rows, m=columns
- get_missing_patterns: Uses sampling for datasets > 100K rows
- Visualizations: Auto-sample to 500 rows for heatmaps
- All operations are memory-efficient (streaming where possible)
"""


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Issue: "Plotly not available"
─────────────────────────────
Solution: pip install plotly
Or use backend='matplotlib' instead


Issue: "Matplotlib not available"
─────────────────────────────────
Solution: pip install matplotlib
Or use backend='plotly' instead


Issue: Memory error with large dataset
──────────────────────────────────────
Solution: Use sample_size parameter in get_missing_patterns()
          Use sample_rows parameter in create_missing_heatmap()
          
Example:
>>> patterns = get_missing_patterns(df, sample_size=50000)
>>> heatmap = create_missing_heatmap(df, sample_rows=1000)


Issue: Heatmap too crowded
──────────────────────────
Solution: Reduce sample_rows parameter
          
Example:
>>> heatmap = create_missing_heatmap(df, sample_rows=200)


Issue: Slow performance
───────────────────────
Solution: Use create_plots=False if visualizations not needed
          Use sample_size in get_missing_patterns()
          
Example:
>>> analysis = analyze_missing_values(df, create_plots=False)
"""


# ============================================================================
# INTEGRATION EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Streamlit Dashboard
───────────────────────────────
import streamlit as st
import pandas as pd
from core.missing_value_analyzer import analyze_missing_values

st.set_page_config(page_title="Missing Value Analysis", layout="wide")

st.title("📊 Missing Value Analysis Dashboard")

# Upload
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Threshold (%)", 5, 50, 20)
    with col2:
        backend = st.radio("Backend", ["plotly", "matplotlib"])
    
    # Run analysis
    analysis = analyze_missing_values(df, threshold=threshold, backend=backend)
    
    # Display summary
    st.text_area("Summary", analysis['summary'], height=300)
    
    # Display statistics
    st.subheader("Statistics")
    st.dataframe(analysis['stats'])
    
    # Display visualizations
    if 'plots' in analysis:
        col1, col2 = st.columns(2)
        with col1:
            if analysis['plots']['bar_chart']:
                st.plotly_chart(analysis['plots']['bar_chart'], use_container_width=True)
        with col2:
            if analysis['plots']['heatmap']:
                st.plotly_chart(analysis['plots']['heatmap'], use_container_width=True)


EXAMPLE 2: Data Validation Pipeline
────────────────────────────────────
from core.missing_value_analyzer import analyze_missing_values

def validate_data_quality(df, max_missing_pct=20):
    \"\"\"Validate data quality before training.\"\"\"
    analysis = analyze_missing_values(df, threshold=max_missing_pct, create_plots=False)
    
    if analysis['above_threshold']['count'] > 0:
        raise ValueError(
            f"Data quality check failed. "
            f"{analysis['above_threshold']['count']} columns exceed {max_missing_pct}% threshold"
        )
    
    return True

# Usage
try:
    validate_data_quality(df, max_missing_pct=15)
    print("✅ Data quality check passed")
except ValueError as e:
    print(f"❌ {e}")


EXAMPLE 3: Automated Report Generation
───────────────────────────────────────
from core.missing_value_analyzer import analyze_missing_values
import json

def generate_missing_value_report(df, output_file='report.json'):
    \"\"\"Generate JSON report of missing values.\"\"\"
    analysis = analyze_missing_values(df, create_plots=False)
    
    report = {
        'summary': analysis['summary'],
        'statistics': analysis['stats'].to_dict('records'),
        'above_threshold': analysis['above_threshold'],
        'patterns': analysis['patterns']
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

# Usage
report = generate_missing_value_report(df)
"""


# ============================================================================
# BEST PRACTICES
# ============================================================================

"""
1. Always run missing value analysis before training
   ─────────────────────────────────────────────────
   analysis = analyze_missing_values(df)
   print(analysis['summary'])


2. Set appropriate threshold based on domain
   ──────────────────────────────────────────
   # Strict: 5%
   # Moderate: 20% (default)
   # Lenient: 50%
   
   analysis = analyze_missing_values(df, threshold=20)


3. Investigate patterns before imputation
   ──────────────────────────────────────
   patterns = get_missing_patterns(df)
   # Check if missing is random or systematic


4. Use Plotly for interactive exploration
   ──────────────────────────────────────
   fig = create_missing_bar_chart(df, backend='plotly')
   fig.show()


5. Use Matplotlib for reports and saving
   ────────────────────────────────────────
   fig = create_missing_bar_chart(df, backend='matplotlib')
   fig.savefig('report.png', dpi=300, bbox_inches='tight')


6. Sample large datasets for visualization
   ───────────────────────────────────────
   heatmap = create_missing_heatmap(df, sample_rows=500)


7. Document missing value handling decisions
   ──────────────────────────────────────────
   # Save analysis summary
   with open('missing_value_analysis.txt', 'w') as f:
        f.write(analysis['summary'])
"""
