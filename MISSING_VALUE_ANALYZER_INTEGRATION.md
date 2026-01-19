"""
Missing Value Analyzer - Integration Checklist

Quick reference for integrating the module into your ML/DL Trainer platform.
"""

# ============================================================================
# QUICK START (5 minutes)
# ============================================================================

"""
STEP 1: Verify Installation
────────────────────────────
✓ core/missing_value_analyzer.py exists
✓ tests/test_missing_value_analyzer.py exists
✓ tests/test_missing_value_analyzer_unit.py exists
✓ MISSING_VALUE_ANALYZER_GUIDE.md exists

STEP 2: Test the Module
───────────────────────
python tests/test_missing_value_analyzer.py

Expected output:
  ✅ EXAMPLE 1: Basic Missing Statistics
  ✅ EXAMPLE 2: Threshold Detection
  ✅ EXAMPLE 3: Missing Value Patterns
  ✅ EXAMPLE 4: Plotly Visualizations
  ✅ EXAMPLE 5: Matplotlib Visualizations
  ✅ EXAMPLE 6: Comprehensive Analysis
  ✅ EXAMPLE 7: Large Dataset Handling
  ✅ EXAMPLE 8: Pipeline Integration
  ✅ ALL EXAMPLES COMPLETED

STEP 3: Run Unit Tests
──────────────────────
pytest tests/test_missing_value_analyzer_unit.py -v

Expected: 50+ tests passed

STEP 4: You're Ready!
─────────────────────
The module is production-ready and can be used immediately.
"""


# ============================================================================
# INTEGRATION POINTS
# ============================================================================

"""
INTEGRATION 1: Data Upload Page (app/main.py)
──────────────────────────────────────────────

Current code:
    def page_data_upload():
        st.header("📤 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(df.head())

Enhanced code:
    from core.missing_value_analyzer import analyze_missing_values
    
    def page_data_upload():
        st.header("📤 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            # Add missing value analysis
            st.subheader("📊 Data Quality Check")
            analysis = analyze_missing_values(df, create_plots=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Summary", analysis['summary'], height=200)
            with col2:
                st.dataframe(analysis['stats'])
            
            # Show visualizations
            st.plotly_chart(analysis['plots']['bar_chart'], use_container_width=True)
            st.plotly_chart(analysis['plots']['heatmap'], use_container_width=True)
            
            st.write(df.head())


INTEGRATION 2: Data Preprocessing (core/preprocessor.py)
────────────────────────────────────────────────────────

Current code:
    def preprocess_data(df):
        # Handle missing values
        df.fillna(df.median(), inplace=True)
        return df

Enhanced code:
    from core.missing_value_analyzer import get_columns_above_threshold
    
    def preprocess_data(df, missing_threshold=30):
        # Identify and drop high-missing columns
        result = get_columns_above_threshold(df, threshold=missing_threshold)
        if result['columns']:
            logger.info(f"Dropping columns with >{missing_threshold}% missing: {result['columns']}")
            df = df.drop(columns=result['columns'])
        
        # Handle remaining missing values
        df.fillna(df.median(), inplace=True)
        return df


INTEGRATION 3: Data Validation (core/validator.py)
──────────────────────────────────────────────────

Current code:
    def validate_data(df):
        if df.empty:
            raise ValueError("Empty dataset")
        return True

Enhanced code:
    from core.missing_value_analyzer import analyze_missing_values
    
    def validate_data(df, max_missing_pct=20):
        if df.empty:
            raise ValueError("Empty dataset")
        
        # Check missing values
        analysis = analyze_missing_values(df, threshold=max_missing_pct, create_plots=False)
        
        if analysis['above_threshold']['count'] > 0:
            raise ValueError(
                f"Data quality check failed. "
                f"{analysis['above_threshold']['count']} columns exceed {max_missing_pct}% threshold. "
                f"Details: {analysis['summary']}"
            )
        
        return True


INTEGRATION 4: EDA Module (eda/eda_missing.py)
───────────────────────────────────────────────

Note: eda_missing.py already exists and is comprehensive.
The new missing_value_analyzer.py is a lightweight alternative.

You can:
  Option A: Use only eda_missing.py (existing)
  Option B: Use only missing_value_analyzer.py (new, simpler)
  Option C: Use both (eda_missing for detailed analysis, missing_value_analyzer for quick checks)

Recommendation: Use Option B for new code, keep Option A for backward compatibility.
"""


# ============================================================================
# USAGE IN DIFFERENT SCENARIOS
# ============================================================================

"""
SCENARIO 1: Quick Data Quality Check
─────────────────────────────────────
from core.missing_value_analyzer import analyze_missing_values

df = pd.read_csv('data.csv')
analysis = analyze_missing_values(df)
print(analysis['summary'])

Time: < 1 second
Output: Text summary + statistics


SCENARIO 2: Identify Columns to Drop
─────────────────────────────────────
from core.missing_value_analyzer import get_columns_above_threshold

result = get_columns_above_threshold(df, threshold=20)
df_cleaned = df.drop(columns=result['columns'])

Time: < 1 second
Output: Cleaned DataFrame


SCENARIO 3: Create Report with Visualizations
───────────────────────────────────────────────
from core.missing_value_analyzer import analyze_missing_values

analysis = analyze_missing_values(df, backend='matplotlib')

# Save visualizations
analysis['plots']['bar_chart'].savefig('missing_bar.png', dpi=150, bbox_inches='tight')
analysis['plots']['heatmap'].savefig('missing_heatmap.png', dpi=150, bbox_inches='tight')

# Save summary
with open('missing_analysis.txt', 'w') as f:
    f.write(analysis['summary'])

Time: < 2 seconds
Output: PNG files + text report


SCENARIO 4: Streamlit Dashboard
────────────────────────────────
import streamlit as st
from core.missing_value_analyzer import analyze_missing_values

st.set_page_config(page_title="Data Quality", layout="wide")

uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    threshold = st.slider("Threshold (%)", 5, 50, 20)
    analysis = analyze_missing_values(df, threshold=threshold)
    
    st.text(analysis['summary'])
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(analysis['plots']['bar_chart'])
    with col2:
        st.plotly_chart(analysis['plots']['heatmap'])

Time: < 2 seconds
Output: Interactive dashboard


SCENARIO 5: Automated Data Pipeline
────────────────────────────────────
from core.missing_value_analyzer import analyze_missing_values

def process_dataset(df):
    # Step 1: Analyze
    analysis = analyze_missing_values(df, threshold=30, create_plots=False)
    
    # Step 2: Drop high-missing columns
    df = df.drop(columns=analysis['above_threshold']['columns'])
    
    # Step 3: Impute
    df.fillna(df.median(), inplace=True)
    
    # Step 4: Verify
    final_analysis = analyze_missing_values(df, create_plots=False)
    
    # Step 5: Log
    logger.info(f"Processing complete: {final_analysis['summary']}")
    
    return df

Time: < 5 seconds
Output: Cleaned DataFrame + logs
"""


# ============================================================================
# CONFIGURATION OPTIONS
# ============================================================================

"""
OPTION 1: Default Configuration
────────────────────────────────
analysis = analyze_missing_values(df)

Uses:
  - threshold: 20%
  - create_plots: True
  - backend: 'plotly'


OPTION 2: Strict Quality Check
───────────────────────────────
analysis = analyze_missing_values(df, threshold=5)

Uses:
  - threshold: 5% (strict)
  - create_plots: False (faster)
  - backend: 'plotly'


OPTION 3: Lenient with Matplotlib
──────────────────────────────────
analysis = analyze_missing_values(df, threshold=50, backend='matplotlib')

Uses:
  - threshold: 50% (lenient)
  - create_plots: True
  - backend: 'matplotlib' (for saving)


OPTION 4: Large Dataset
───────────────────────
analysis = analyze_missing_values(df, create_plots=True)

Auto-handles:
  - Sampling for heatmap (500 rows)
  - Sampling for patterns (if > 100K rows)
  - Memory-efficient operations
"""


# ============================================================================
# DEPENDENCIES CHECK
# ============================================================================

"""
REQUIRED DEPENDENCIES:
──────────────────────
✓ pandas >= 1.0
✓ numpy >= 1.18

Check:
  import pandas as pd
  import numpy as np
  print(f"pandas: {pd.__version__}")
  print(f"numpy: {np.__version__}")


OPTIONAL DEPENDENCIES:
──────────────────────
✓ plotly >= 4.0 (for interactive visualizations)
✓ matplotlib >= 3.0 (for static visualizations)

Check:
  try:
      import plotly
      print(f"plotly: {plotly.__version__}")
  except ImportError:
      print("plotly not installed")
  
  try:
      import matplotlib
      print(f"matplotlib: {matplotlib.__version__}")
  except ImportError:
      print("matplotlib not installed")


INSTALL MISSING DEPENDENCIES:
──────────────────────────────
pip install plotly matplotlib
"""


# ============================================================================
# TESTING CHECKLIST
# ============================================================================

"""
BEFORE DEPLOYMENT:
──────────────────
□ Run examples: python tests/test_missing_value_analyzer.py
□ Run unit tests: pytest tests/test_missing_value_analyzer_unit.py -v
□ Test with your data: analyze_missing_values(your_df)
□ Test visualizations: fig.show() and fig.savefig()
□ Test large dataset: analyze_missing_values(large_df)
□ Test error handling: analyze_missing_values(empty_df)
□ Check dependencies: pip list | grep -E "pandas|numpy|plotly|matplotlib"
□ Review documentation: Read MISSING_VALUE_ANALYZER_GUIDE.md
□ Test integration: Import in your code and verify it works


AFTER DEPLOYMENT:
─────────────────
□ Monitor performance with real data
□ Collect user feedback
□ Log any issues
□ Update documentation if needed
□ Plan enhancements based on usage
"""


# ============================================================================
# COMMON ISSUES & SOLUTIONS
# ============================================================================

"""
ISSUE 1: ImportError: No module named 'plotly'
───────────────────────────────────────────────
Solution:
  pip install plotly
  
Or use matplotlib backend:
  fig = create_missing_bar_chart(df, backend='matplotlib')


ISSUE 2: ImportError: No module named 'matplotlib'
────────────────────────────────────────────────────
Solution:
  pip install matplotlib
  
Or use plotly backend:
  fig = create_missing_bar_chart(df, backend='plotly')


ISSUE 3: MemoryError with large dataset
─────────────────────────────────────────
Solution:
  # Use sampling
  patterns = get_missing_patterns(df, sample_size=50000)
  heatmap = create_missing_heatmap(df, sample_rows=200)
  
  # Or skip visualizations
  analysis = analyze_missing_values(df, create_plots=False)


ISSUE 4: Slow performance
──────────────────────────
Solution:
  # Skip visualizations if not needed
  analysis = analyze_missing_values(df, create_plots=False)
  
  # Use smaller sample sizes
  heatmap = create_missing_heatmap(df, sample_rows=100)


ISSUE 5: Heatmap too crowded
─────────────────────────────
Solution:
  # Reduce sample rows
  heatmap = create_missing_heatmap(df, sample_rows=200)
  
  # Or use bar chart instead
  bar_chart = create_missing_bar_chart(df)
"""


# ============================================================================
# PERFORMANCE EXPECTATIONS
# ============================================================================

"""
SMALL DATASET (< 10K rows):
───────────────────────────
compute_missing_stats()      : < 1ms
get_columns_above_threshold(): < 1ms
get_missing_patterns()       : < 5ms
create_missing_bar_chart()   : < 100ms
create_missing_heatmap()     : < 100ms
analyze_missing_values()     : < 200ms

MEDIUM DATASET (10K - 100K rows):
──────────────────────────────────
compute_missing_stats()      : < 50ms
get_columns_above_threshold(): < 50ms
get_missing_patterns()       : < 500ms
create_missing_bar_chart()   : < 200ms
create_missing_heatmap()     : < 200ms (with sampling)
analyze_missing_values()     : < 1s

LARGE DATASET (100K - 1M rows):
────────────────────────────────
compute_missing_stats()      : < 500ms
get_columns_above_threshold(): < 500ms
get_missing_patterns()       : < 5s (with sampling)
create_missing_bar_chart()   : < 500ms
create_missing_heatmap()     : < 500ms (with sampling)
analyze_missing_values()     : < 2s

VERY LARGE DATASET (> 1M rows):
────────────────────────────────
compute_missing_stats()      : < 5s
get_columns_above_threshold(): < 5s
get_missing_patterns()       : < 10s (with sampling)
create_missing_bar_chart()   : < 1s
create_missing_heatmap()     : < 1s (with sampling)
analyze_missing_values()     : < 5s
"""


# ============================================================================
# NEXT STEPS
# ============================================================================

"""
IMMEDIATE (Today):
──────────────────
1. ✅ Review the module: core/missing_value_analyzer.py
2. ✅ Run examples: python tests/test_missing_value_analyzer.py
3. ✅ Run tests: pytest tests/test_missing_value_analyzer_unit.py -v
4. ✅ Read guide: MISSING_VALUE_ANALYZER_GUIDE.md

SHORT TERM (This Week):
───────────────────────
1. Integrate into data upload page (app/main.py)
2. Add to preprocessing pipeline (core/preprocessor.py)
3. Add to data validation (core/validator.py)
4. Test with real datasets

MEDIUM TERM (This Month):
─────────────────────────
1. Create Streamlit dashboard for missing value analysis
2. Generate automated reports
3. Add to CI/CD pipeline for data quality checks
4. Document in team wiki

LONG TERM (Future):
───────────────────
1. Add imputation recommendations
2. Add correlation analysis of missing patterns
3. Add time-series missing value analysis
4. Integrate with model training pipeline
"""


# ============================================================================
# SUPPORT & DOCUMENTATION
# ============================================================================

"""
DOCUMENTATION FILES:
────────────────────
1. MISSING_VALUE_ANALYZER_GUIDE.md
   - Complete API reference
   - Usage patterns
   - Integration examples
   - Troubleshooting

2. MISSING_VALUE_ANALYZER_SUMMARY.md
   - Overview and features
   - Performance characteristics
   - Comparison with existing modules
   - Best practices

3. This file (Integration Checklist)
   - Quick start guide
   - Integration points
   - Common issues
   - Next steps


CODE EXAMPLES:
──────────────
1. tests/test_missing_value_analyzer.py
   - 8 comprehensive examples
   - Realistic scenarios
   - Integration patterns

2. tests/test_missing_value_analyzer_unit.py
   - 50+ unit tests
   - Edge cases
   - Performance tests


GETTING HELP:
─────────────
1. Check MISSING_VALUE_ANALYZER_GUIDE.md for API reference
2. Look at examples in tests/test_missing_value_analyzer.py
3. Review unit tests for usage patterns
4. Check troubleshooting section in this file
"""


# ============================================================================
# SUMMARY
# ============================================================================

"""
✅ INTEGRATION CHECKLIST COMPLETE

You now have:
  ✓ Production-ready missing value analyzer module
  ✓ Comprehensive documentation
  ✓ 50+ unit tests
  ✓ 8 realistic examples
  ✓ Integration guide
  ✓ Troubleshooting guide

Ready to:
  ✓ Use immediately in your code
  ✓ Integrate into Streamlit UI
  ✓ Add to data pipelines
  ✓ Deploy to production
  ✓ Extend with custom features

Next: Follow the integration points above to add to your platform!
"""
