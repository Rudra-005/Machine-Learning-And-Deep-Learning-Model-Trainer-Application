# Missing Value Handling

## Overview

This system implements a **human-in-the-loop** approach to missing value handling, combining automated detection with explicit user approval. This design prevents silent data corruption while maintaining efficiency.

## Detection

Missing values are detected using pandas' `isnull()` method during data loading:

```python
missing_count = df[col].isnull().sum()
missing_percentage = (missing_count / len(df)) * 100
```

For each column, we track:
- **Count**: Total number of missing values
- **Percentage**: Proportion of missing data (0-100%)
- **Data Type**: Numeric or categorical (determines valid strategies)

## Recommendation Generation

Recommendations follow rule-based logic, not heuristics:

| Condition | Recommendation | Rationale |
|-----------|---|---|
| Missing > 40% | Drop column | Too sparse to impute reliably |
| Numeric + Missing ≤ 40% | Median | Robust to outliers, preserves distribution |
| Categorical + Missing ≤ 40% | Most Frequent | Preserves category space |

**Why these rules?**
- **Median over mean**: Resistant to outliers in numeric data
- **Most frequent over mode**: Simpler, faster, handles ties consistently
- **40% threshold**: Industry standard for "too sparse" (varies by domain)

## User Overrides

Users can override recommendations via interactive UI:

```python
# System recommends 'median' for age column
# User can select: 'median', 'mean', or 'drop_column'
user_selection = st.selectbox(
    "Strategy for age",
    options=['median', 'mean', 'drop_column'],
    index=0  # Preselect recommendation
)
```

**Override workflow:**
1. System displays recommendation with explanation
2. User reviews and optionally changes strategy
3. System validates selection (type compatibility check)
4. If invalid, falls back to recommendation with warning
5. User sees summary before proceeding

## Validation & Safety

The system prevents unsafe configurations:

### Blocked Actions
- ❌ Dropping target column (breaks training)
- ❌ Dropping all features (no data to train on)
- ❌ Incompatible strategy/type combinations (e.g., median on categorical)

### Validation Flow
```
User Selection
    ↓
[Type Compatibility Check]
    ↓
[Target Column Check]
    ↓
[Feature Count Check]
    ↓
Valid? → Proceed : Fallback to Recommendation
```

## Why This Design Avoids Unsafe Automation

### Problem with Full Automation
Fully automated preprocessing can silently corrupt data:
- Dropping important features without notification
- Applying inappropriate strategies (median on categories)
- Removing target variable by mistake

### Our Solution: Explicit Approval

1. **Transparency**: Every action is visible before execution
2. **Validation**: Invalid combinations are rejected with clear messages
3. **Auditability**: All decisions logged to experiment tracker
4. **Reversibility**: Users can review and modify before committing

### Example: Why This Matters

```python
# ❌ Unsafe: Automatic drop
df = df.drop(columns=['sparse_col'])  # Silent data loss

# ✅ Safe: Explicit approval
recommendations = analyze_missing_values(df)
# System shows: "sparse_col has 75% missing → recommend drop"
user_selections = render_missing_value_selector(recommendations)
# User explicitly selects: "drop_column" for sparse_col
# System logs: "Dropped sparse_col per user request"
```

## Implementation Details

### Detection Module
- **File**: `core/missing_value_analyzer.py`
- **Function**: `analyze_missing_values(df)`
- **Output**: List of columns with missing %, data type, count

### Recommendation Module
- **File**: `core/missing_value_analyzer.py`
- **Function**: `recommend_missing_value_strategy(df)`
- **Output**: Recommendations with explanations

### Validation Module
- **File**: `core/preprocessing_validator.py`
- **Function**: `validate_selections(df, target_col, user_selections, recommendations)`
- **Output**: Validation status + error messages

### UI Component
- **File**: `app/components/missing_value_selector.py`
- **Function**: `render_missing_value_selector(recommendations)`
- **Output**: User selections with preselected defaults

### Execution Module
- **File**: `core/missing_value_handler.py`
- **Function**: `handle_missing_values(df, config)`
- **Output**: Processed DataFrame

## Experiment Tracking

All preprocessing decisions are logged:

```python
tracker.log_experiment(
    dataset_name="iris.csv",
    missing_strategies={'age': 'median', 'city': 'most_frequent'},
    model_type="classification",
    metrics={'accuracy': 0.95}
)
```

This enables:
- Reproducibility: Exact preprocessing can be replicated
- Debugging: Trace which strategies led to good/bad results
- Compliance: Audit trail of all data transformations

## Best Practices

1. **Always review recommendations** before accepting
2. **Understand your data** before overriding defaults
3. **Document domain knowledge** when deviating from recommendations
4. **Monitor experiment logs** for patterns in preprocessing choices
5. **Test edge cases** (all missing, no missing, mixed types)

## Performance

- Detection: O(n) where n = number of rows
- Recommendation: O(m) where m = number of columns
- Validation: O(m)
- Total overhead: < 100ms for typical datasets

## References

- Rubin, D. B. (1976). "Inference and missing data"
- Little, R. J., & Rubin, D. B. (2002). "Statistical analysis with missing data"
- Scikit-learn SimpleImputer: https://scikit-learn.org/stable/modules/impute.html
