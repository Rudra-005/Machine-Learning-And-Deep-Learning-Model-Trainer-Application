# EDA Integration into Streamlit App

## Overview
The EDA (Exploratory Data Analysis) modules have been integrated into the Streamlit application as a new tab called **"EDA / Data Understanding"** in the main navigation.

## Files Modified/Created

### 1. **app/pages/eda_page.py** (NEW)
Main EDA page component with 5 tabs:

#### Tab 1: Dataset Overview
- Row/column counts, missing values, duplicates
- Data preview (first 10 rows)
- Column information (type, null counts, unique values)
- Descriptive statistics

#### Tab 2: Missing Values
- Missing value statistics and percentages
- Missing values by column
- Threshold detection (configurable slider)
- Missing value patterns
- Dual visualization (bar chart or heatmap)

#### Tab 3: Target Analysis
- Target column selection
- Task type detection (classification/regression)
- Classification analysis:
  - Class distribution
  - Imbalance ratio detection
  - Minority class percentage
  - Imbalance warning with recommendations
- Regression analysis:
  - Mean, std dev, skewness, kurtosis
  - Distribution statistics (min, Q1, median, Q3, max)
- Target distribution visualization

#### Tab 4: Feature Analysis
- Feature type detection (numerical/categorical)
- Feature count metrics
- Multi-select feature analysis
- Per-feature statistics:
  - Type, missing count, unique count
  - For numerical: mean, std, min, max
  - For categorical: value counts
- Feature distribution plots (expandable)

#### Tab 5: Correlation
- Correlation method selection (Pearson/Spearman)
- Top correlated features table
- Task-aware analysis:
  - **Classification**: Class distribution by categorical feature
  - **Regression**: Target mean by categorical feature
- Correlation heatmap (numerical features only)

### 2. **app/main.py** (MODIFIED)
Changes:
- Added import: `from app.pages.eda_page import render_eda_page`
- Added "EDA / Data Understanding" to sidebar navigation
- Added conditional rendering for EDA page

## Session State Management

The EDA page uses Streamlit session state to persist:
- `st.session_state.data` - Uploaded dataset
- `st.session_state.eda_target_col` - Selected target column
- `st.session_state.eda_selected_features` - Selected features for analysis

## Integration with EDA Modules

The page integrates all four EDA modules:

1. **missing_value_analyzer.py**
   - `compute_missing_statistics()`
   - `detect_columns_above_threshold()`
   - `detect_missing_patterns()`
   - `plot_missing_values()`

2. **target_analyzer.py**
   - `detect_task_type()`
   - `analyze_classification_target()`
   - `analyze_regression_target()`
   - `plot_target_distribution()`

3. **feature_analyzer.py**
   - `detect_feature_types()`
   - `get_feature_statistics()`
   - `plot_feature_distribution()`

4. **relationship_analyzer.py**
   - `compute_correlation_matrix()`
   - `get_top_correlated_features()`
   - `analyze_categorical_regression()`
   - `analyze_categorical_classification()`
   - `plot_correlation_heatmap()`
   - `plot_categorical_regression()`
   - `plot_categorical_classification()`

## Workflow

1. **Data Upload** → Upload CSV or load sample data
2. **EDA / Data Understanding** → Explore data before training
3. **Training** → Train models with informed decisions
4. **Results** → View performance metrics

## Key Features

✅ **No Preprocessing Duplication**
- EDA uses raw data from session state
- Training uses its own preprocessing pipeline
- No shared preprocessing logic

✅ **Clean UI**
- 5 organized tabs with clear sections
- Expandable feature analysis
- Responsive metrics and visualizations
- Professional styling with dividers

✅ **Session Persistence**
- Data persists across page navigation
- Target column selection remembered
- Feature selections maintained

✅ **Error Handling**
- Try-catch blocks for all operations
- User-friendly error messages
- Logging for debugging

✅ **Large Dataset Support**
- Automatic sampling in correlation analysis
- Efficient computations
- No performance degradation

## Usage

1. Navigate to "Data Upload" and upload a CSV file
2. Click "EDA / Data Understanding" in sidebar
3. Explore data through 5 tabs:
   - Overview: Basic statistics
   - Missing Values: Data quality assessment
   - Target: Task type and distribution
   - Features: Individual feature analysis
   - Correlation: Feature-target relationships
4. Use insights to inform model selection in Training tab

## Dependencies

All EDA modules are already implemented:
- `core/missing_value_analyzer.py`
- `core/target_analyzer.py`
- `core/feature_analyzer.py`
- `core/relationship_analyzer.py`

No additional dependencies required beyond existing project requirements.
