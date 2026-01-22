# Task-Target Compatibility Validation

## Overview
Robust validation system ensuring task type and target data compatibility before model training.

## Validation Rules

### 1. Regression Validation
**Requirement**: Numeric target only

**Error Messages**:
- **Categorical target**: "This target is categorical. Please use Classification instead of Regression."
- **Non-numeric target**: "Regression requires a numeric target. Please convert or select a different column."

**Implementation**:
```python
if task_type == "Regression":
    if is_categorical:
        return False, "This target is categorical. Please use Classification instead of Regression."
    if not is_numeric:
        return False, "Regression requires a numeric target. Please convert or select a different column."
```

### 2. Classification Validation
**Requirements**:
- Minimum 2 unique values
- Maximum 50 unique values (for numeric targets)
- Binary targets for Logistic Regression

**Error Messages**:
- **Too few classes**: "Classification requires at least 2 unique values in the target."
- **Too many classes**: "Too many unique values for classification. Consider Regression instead."
- **Logistic Regression multi-class**: "Logistic Regression supports binary classification only. Your target has more than 2 classes."

**Implementation**:
```python
if task_type == "Classification":
    if unique_count < 2:
        return False, "Classification requires at least 2 unique values in the target."
    if unique_count > 50 and is_numeric:
        return False, "Too many unique values for classification. Consider Regression instead."
    if model_name == "logistic_regression" and unique_count > 2:
        return False, "Logistic Regression supports binary classification only. Your target has more than 2 classes."
```

### 3. Missing Value Validation
**Requirement**: No missing values in target column

**Error Message**: "❌ Target has X missing value(s)"

**Implementation**:
```python
missing_count = data[target_col].isna().sum()
if missing_count > 0:
    st.error(f"❌ Target has {missing_count} missing value(s)")
```

## Helper Functions

### `is_numeric_target(target_series)`
Checks if target is numeric using NumPy's type checking.
```python
return np.issubdtype(target_series.dtype, np.number)
```

### `is_categorical_target(target_series)`
Checks if target is categorical/string.
```python
return target_series.dtype == 'object' or target_series.dtype.name == 'category'
```

### `validate_task_target_compatibility(task_type, target_data, model_name=None)`
Main validation function returning (is_valid: bool, error_message: str or None).

## Validation Flow

1. **User selects target column** → Extract target data (dropna)
2. **User selects task type** → Validate task-target compatibility
3. **User selects model** → Validate model-specific constraints (e.g., Logistic Regression binary only)
4. **Train button state** → Disabled if any validation fails
5. **Error display** → Clear, specific messages guide user to fix issues

## User Experience

### Valid Scenarios
✅ Numeric target + Regression  
✅ Categorical target (2+ classes) + Classification  
✅ Numeric target (2-50 unique) + Classification  
✅ Binary categorical target + Logistic Regression  
✅ Multi-class target + Random Forest/SVM/Gradient Boosting  

### Invalid Scenarios
❌ Categorical target + Regression → "This target is categorical. Please use Classification instead of Regression."  
❌ Numeric target (1 unique) + Classification → "Classification requires at least 2 unique values in the target."  
❌ Numeric target (100+ unique) + Classification → "Too many unique values for classification. Consider Regression instead."  
❌ Multi-class target + Logistic Regression → "Logistic Regression supports binary classification only. Your target has more than 2 classes."  
❌ Target with missing values → "Target has X missing value(s)"  

## Integration Points

### Training Page (app/main.py)
- Lines 27-50: Helper functions for type checking
- Lines 52-72: Main validation function
- Lines 234-240: Task-target validation display
- Lines 250-256: Model-specific validation display
- Lines 280-283: Train button disable logic

### Key Features
- **Real-time validation**: Errors display immediately as user changes selections
- **Specific guidance**: Error messages tell users exactly what's wrong and how to fix it
- **Model-aware**: Validation considers selected model (e.g., Logistic Regression constraints)
- **Production-ready**: Prevents invalid training attempts before they start

## Testing Scenarios

```python
# Test 1: Categorical target with Regression
target = pd.Series(['A', 'B', 'A', 'B'])
is_valid, msg = validate_task_target_compatibility("Regression", target)
# Expected: (False, "This target is categorical...")

# Test 2: Numeric target with Classification (valid)
target = pd.Series([0, 1, 0, 1, 2])
is_valid, msg = validate_task_target_compatibility("Classification", target)
# Expected: (True, None)

# Test 3: Multi-class with Logistic Regression
target = pd.Series([0, 1, 2, 0, 1, 2])
is_valid, msg = validate_task_target_compatibility("Classification", target, "logistic_regression")
# Expected: (False, "Logistic Regression supports binary...")

# Test 4: Binary with Logistic Regression (valid)
target = pd.Series([0, 1, 0, 1])
is_valid, msg = validate_task_target_compatibility("Classification", target, "logistic_regression")
# Expected: (True, None)
```
