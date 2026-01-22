# Improved Error Messages Guide

## Overview
User-friendly error messages that explain WHAT is wrong and HOW to fix it. Messages are separated by error type to avoid confusion.

## Error Types & Messages

### 1. Missing Values Error
**Error Type**: `missing_values`  
**Trigger**: `data[target_col].isna().sum() > 0`  
**Display**: ⚠️ **Missing Values**

#### Message Format
```
⚠️ **Missing Values**: Your target has X empty cell(s). Clean the data first or remove rows with missing targets.
```

#### Examples

**Scenario 1**: Target column with 3 missing values
```python
target = pd.Series([1, 2, np.nan, 1, 2, np.nan, 1, np.nan])
missing_count = 3
```
**Message Shown**:
```
⚠️ **Missing Values**: Your target has 3 empty cell(s). Clean the data first or remove rows with missing targets.
```

**Scenario 2**: Target column with 1 missing value
```python
target = pd.Series(['A', 'B', np.nan, 'A', 'B'])
missing_count = 1
```
**Message Shown**:
```
⚠️ **Missing Values**: Your target has 1 empty cell(s). Clean the data first or remove rows with missing targets.
```

---

### 2. Type Mismatch Error
**Error Type**: `type_mismatch`  
**Trigger**: Task-target type incompatibility  
**Display**: ❌ **Wrong Data Type**

#### Regression + Categorical Target
```python
task_type = "Regression"
target = pd.Series(['Low', 'Medium', 'High', 'Low', 'Medium'])
is_categorical = True
```
**Message Shown**:
```
❌ **Wrong Data Type**: Your target contains text/categories. Use Classification instead.
```
**How to Fix**: Switch task type to "Classification"

#### Regression + Non-numeric Target
```python
task_type = "Regression"
target = pd.Series(['A', 'B', 'C', 'A', 'B'])
is_numeric = False
```
**Message Shown**:
```
❌ **Wrong Data Type**: Your target must be numeric for Regression. Try Classification or select a different column.
```
**How to Fix**: 
- Switch to Classification, OR
- Select a different numeric column

---

### 3. Class Count Error
**Error Type**: `class_count`  
**Trigger**: Insufficient or excessive unique values  
**Display**: ❌ **Not Enough Categories**

#### Too Few Classes (Classification)
```python
task_type = "Classification"
target = pd.Series([1, 1, 1, 1, 1])
unique_count = 1
```
**Message Shown**:
```
❌ **Not Enough Categories**: Your target needs at least 2 different values. Currently has only 1.
```
**How to Fix**: 
- Select a different target column with 2+ categories, OR
- Switch to Regression if target is numeric

#### Too Many Classes (Classification)
```python
task_type = "Classification"
target = pd.Series(range(100))  # 100 unique values
unique_count = 100
is_numeric = True
```
**Message Shown**:
```
❌ **Not Enough Categories**: Too many categories (50+). Try Regression instead.
```
**How to Fix**: Switch task type to "Regression"

---

### 4. Model Constraint Error
**Error Type**: `model_constraint`  
**Trigger**: Model-specific limitations  
**Display**: ❌ **Model Limitation**

#### Logistic Regression + Multi-class
```python
task_type = "Classification"
model_name = "logistic_regression"
target = pd.Series([0, 1, 2, 0, 1, 2])
unique_count = 3
```
**Message Shown**:
```
❌ **Model Limitation**: Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting.
```
**How to Fix**: 
- Select Random Forest, SVM, or Gradient Boosting, OR
- Create binary classification (2 categories only)

#### Logistic Regression + Binary (Valid)
```python
task_type = "Classification"
model_name = "logistic_regression"
target = pd.Series([0, 1, 0, 1, 0])
unique_count = 2
```
**Result**: ✅ No error - training enabled

---

## Error Message Flow

### Validation Sequence
```
1. Check for missing values in target
   ├─ If missing → Show "Missing Values" error
   └─ If clean → Continue to step 2

2. Check task-target type compatibility
   ├─ If mismatch → Show "Wrong Data Type" error
   └─ If compatible → Continue to step 3

3. Check class count requirements
   ├─ If insufficient → Show "Not Enough Categories" error
   └─ If sufficient → Continue to step 4

4. Check model-specific constraints
   ├─ If violated → Show "Model Limitation" error
   └─ If satisfied → ✅ Training enabled
```

### Code Implementation
```python
# Step 1: Missing values check
if missing_count > 0:
    st.error(f"⚠️ **Missing Values**: Your target has {missing_count} empty cell(s)...")
    return

# Step 2-4: Type/class/model validation
is_valid, error_type, error_msg = validate_task_target_compatibility(
    task_type, target_data, model_name
)

if not is_valid:
    if error_type == "type_mismatch":
        st.error(f"❌ **Wrong Data Type**: {error_msg}")
    elif error_type == "class_count":
        st.error(f"❌ **Not Enough Categories**: {error_msg}")
    elif error_type == "model_constraint":
        st.error(f"❌ **Model Limitation**: {error_msg}")
```

---

## User Experience Improvements

### Before (Old Messages)
```
❌ Target has missing values
❌ This target is categorical. Please use Classification instead of Regression.
❌ Classification requires at least 2 unique values in the target.
❌ Logistic Regression supports binary classification only. Your target has more than 2 classes.
```

### After (New Messages)
```
⚠️ **Missing Values**: Your target has 3 empty cell(s). Clean the data first or remove rows with missing targets.
❌ **Wrong Data Type**: Your target contains text/categories. Use Classification instead.
❌ **Not Enough Categories**: Your target needs at least 2 different values. Currently has only 1.
❌ **Model Limitation**: Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting.
```

### Key Improvements
✅ **Clear problem identification**: Bold headers explain the issue type  
✅ **Non-technical language**: Avoids jargon like "categorical", "unique values"  
✅ **Actionable guidance**: Tells users exactly what to do  
✅ **Separated concerns**: Missing values ≠ type mismatches  
✅ **Specific numbers**: Shows actual counts (e.g., "3 empty cells", "3+ categories")  
✅ **Alternative suggestions**: Offers model alternatives when applicable  

---

## Testing Scenarios

### Test 1: Categorical target with Regression
```python
data = pd.DataFrame({
    'feature': [1, 2, 3, 4, 5],
    'target': ['Low', 'High', 'Low', 'High', 'Low']
})
target_col = 'target'
task_type = 'Regression'

# Expected: ❌ **Wrong Data Type**: Your target contains text/categories. Use Classification instead.
```

### Test 2: Single-value target with Classification
```python
data = pd.DataFrame({
    'feature': [1, 2, 3, 4, 5],
    'target': [1, 1, 1, 1, 1]
})
target_col = 'target'
task_type = 'Classification'

# Expected: ❌ **Not Enough Categories**: Your target needs at least 2 different values. Currently has only 1.
```

### Test 3: Multi-class with Logistic Regression
```python
data = pd.DataFrame({
    'feature': [1, 2, 3, 4, 5, 6],
    'target': [0, 1, 2, 0, 1, 2]
})
target_col = 'target'
task_type = 'Classification'
model_name = 'logistic_regression'

# Expected: ❌ **Model Limitation**: Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting.
```

### Test 4: Missing values in target
```python
data = pd.DataFrame({
    'feature': [1, 2, 3, 4, 5],
    'target': [1, 2, np.nan, 1, 2]
})
target_col = 'target'
missing_count = 1

# Expected: ⚠️ **Missing Values**: Your target has 1 empty cell(s). Clean the data first or remove rows with missing targets.
```

### Test 5: Valid binary classification with Logistic Regression
```python
data = pd.DataFrame({
    'feature': [1, 2, 3, 4, 5],
    'target': [0, 1, 0, 1, 0]
})
target_col = 'target'
task_type = 'Classification'
model_name = 'logistic_regression'

# Expected: ✅ No error - training enabled
```

---

## Integration Points

### Training Page (app/main.py)

**Lines 234-240**: Task-target validation display
```python
if missing_count > 0:
    st.error(f"⚠️ **Missing Values**: Your target has {missing_count} empty cell(s)...")
else:
    is_valid, error_type, error_msg = validate_task_target_compatibility(task_type, target_data)
    if not is_valid:
        if error_type == "type_mismatch":
            st.error(f"❌ **Wrong Data Type**: {error_msg}")
        elif error_type == "class_count":
            st.error(f"❌ **Not Enough Categories**: {error_msg}")
```

**Lines 250-256**: Model-specific validation display
```python
if missing_count == 0:
    is_valid, error_type, error_msg = validate_task_target_compatibility(task_type, target_data, model_name)
    if not is_valid:
        if error_type == "model_constraint":
            st.error(f"❌ **Model Limitation**: {error_msg}")
```

**Lines 280-283**: Train button disable logic
```python
train_disabled = missing_count > 0
if not train_disabled and 'model_name' in locals():
    is_valid, _, _ = validate_task_target_compatibility(task_type, target_data, model_name if model_type == "Machine Learning" else None)
    train_disabled = not is_valid
```

---

## Summary

| Error Type | Display | When | How to Fix |
|-----------|---------|------|-----------|
| **Missing Values** | ⚠️ | Target has NaN/empty cells | Clean data or remove rows |
| **Type Mismatch** | ❌ | Categorical target + Regression | Switch to Classification |
| **Class Count** | ❌ | <2 or >50 unique values | Select different column or task |
| **Model Constraint** | ❌ | Multi-class + Logistic Regression | Choose different model |

All messages are **non-technical**, **actionable**, and **user-friendly**.
