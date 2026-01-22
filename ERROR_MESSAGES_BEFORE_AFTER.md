# Error Messages: Before & After Comparison

## Quick Reference

### Error Type 1: Missing Values
**Before**: ❌ Target has missing values  
**After**: ⚠️ **Missing Values**: Your target has 3 empty cell(s). Clean the data first or remove rows with missing targets.

**Code Trigger**:
```python
missing_count = data[target_col].isna().sum()
if missing_count > 0:
    st.error(f"⚠️ **Missing Values**: Your target has {missing_count} empty cell(s). Clean the data first or remove rows with missing targets.")
```

---

### Error Type 2: Categorical Target + Regression
**Before**: ❌ This target is categorical. Please use Classification instead of Regression.  
**After**: ❌ **Wrong Data Type**: Your target contains text/categories. Use Classification instead.

**Code Trigger**:
```python
task_type = "Regression"
is_categorical = is_categorical_target(target_data)
if is_categorical:
    return False, "type_mismatch", "Your target contains text/categories. Use Classification instead."
```

**Display**:
```python
if error_type == "type_mismatch":
    st.error(f"❌ **Wrong Data Type**: {error_msg}")
```

---

### Error Type 3: Non-numeric Target + Regression
**Before**: ❌ Regression requires a numeric target. Please convert or select a different column.  
**After**: ❌ **Wrong Data Type**: Your target must be numeric for Regression. Try Classification or select a different column.

**Code Trigger**:
```python
task_type = "Regression"
is_numeric = is_numeric_target(target_data)
if not is_numeric:
    return False, "type_mismatch", "Your target must be numeric for Regression. Try Classification or select a different column."
```

---

### Error Type 4: Single Value Target + Classification
**Before**: ❌ Classification requires at least 2 unique values in the target.  
**After**: ❌ **Not Enough Categories**: Your target needs at least 2 different values. Currently has only 1.

**Code Trigger**:
```python
task_type = "Classification"
unique_count = target_data.nunique()
if unique_count < 2:
    return False, "class_count", "Your target needs at least 2 different values. Currently has only 1."
```

**Display**:
```python
if error_type == "class_count":
    st.error(f"❌ **Not Enough Categories**: {error_msg}")
```

---

### Error Type 5: Too Many Classes + Classification
**Before**: ❌ Too many unique values for classification. Consider Regression instead.  
**After**: ❌ **Not Enough Categories**: Too many categories (50+). Try Regression instead.

**Code Trigger**:
```python
task_type = "Classification"
unique_count = target_data.nunique()
is_numeric = is_numeric_target(target_data)
if unique_count > 50 and is_numeric:
    return False, "class_count", "Too many categories (50+). Try Regression instead."
```

---

### Error Type 6: Multi-class + Logistic Regression
**Before**: ❌ Logistic Regression supports binary classification only. Your target has more than 2 classes.  
**After**: ❌ **Model Limitation**: Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting.

**Code Trigger**:
```python
task_type = "Classification"
model_name = "logistic_regression"
unique_count = target_data.nunique()
if unique_count > 2:
    return False, "model_constraint", "Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting."
```

**Display**:
```python
if error_type == "model_constraint":
    st.error(f"❌ **Model Limitation**: {error_msg}")
```

---

## Key Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **Clarity** | Generic, technical | Specific, user-friendly |
| **Guidance** | No action items | Clear "how to fix" steps |
| **Categorization** | All mixed together | Separated by error type |
| **Language** | Jargon-heavy | Plain English |
| **Examples** | None | Specific numbers/values |
| **Alternatives** | Not mentioned | Suggested alternatives |

---

## Implementation Details

### Validation Function Returns
```python
def validate_task_target_compatibility(task_type, target_data, model_name=None):
    """
    Returns: (is_valid: bool, error_type: str, error_message: str or None)
    
    error_type values:
    - None (valid)
    - "type_mismatch" (categorical/numeric mismatch)
    - "class_count" (too few/many classes)
    - "model_constraint" (model-specific limitation)
    """
```

### Error Display Logic
```python
# Step 1: Check missing values (separate concern)
if missing_count > 0:
    st.error(f"⚠️ **Missing Values**: Your target has {missing_count} empty cell(s)...")
    return

# Step 2-4: Check type/class/model compatibility
is_valid, error_type, error_msg = validate_task_target_compatibility(task_type, target_data, model_name)

if not is_valid:
    if error_type == "type_mismatch":
        st.error(f"❌ **Wrong Data Type**: {error_msg}")
    elif error_type == "class_count":
        st.error(f"❌ **Not Enough Categories**: {error_msg}")
    elif error_type == "model_constraint":
        st.error(f"❌ **Model Limitation**: {error_msg}")
```

---

## User Experience Flow

### Scenario 1: User selects categorical target + Regression
```
1. User selects target column (text/categories)
2. User selects "Regression" task type
3. System validates compatibility
4. Error displayed: ❌ **Wrong Data Type**: Your target contains text/categories. Use Classification instead.
5. Train button disabled
6. User sees clear guidance → switches to Classification
7. Error clears → Train button enabled
```

### Scenario 2: User selects multi-class target + Logistic Regression
```
1. User selects target column (3+ categories)
2. User selects "Classification" task type
3. User selects "Logistic Regression" model
4. System validates model compatibility
5. Error displayed: ❌ **Model Limitation**: Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting.
6. Train button disabled
7. User sees alternatives → selects Random Forest
8. Error clears → Train button enabled
```

### Scenario 3: User has missing values in target
```
1. User selects target column (with NaN values)
2. System detects missing_count > 0
3. Error displayed: ⚠️ **Missing Values**: Your target has 2 empty cell(s). Clean the data first or remove rows with missing targets.
4. Train button disabled
5. User goes to EDA tab to clean data
6. Returns to Training tab
7. Error clears → Train button enabled
```

---

## Testing Checklist

- [ ] Missing values show ⚠️ **Missing Values** (not type mismatch)
- [ ] Categorical + Regression shows ❌ **Wrong Data Type**
- [ ] Single value + Classification shows ❌ **Not Enough Categories**
- [ ] 100+ values + Classification shows ❌ **Not Enough Categories** (not type mismatch)
- [ ] Multi-class + Logistic Regression shows ❌ **Model Limitation**
- [ ] Binary + Logistic Regression works (no error)
- [ ] Train button disabled when any error present
- [ ] Train button enabled when all validations pass
- [ ] Error messages are non-technical and actionable
- [ ] Error headers clearly identify the problem type

---

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Validation function | app/main.py | 27-72 |
| Missing value check | app/main.py | 234-240 |
| Type/class validation | app/main.py | 234-240 |
| Model constraint check | app/main.py | 250-256 |
| Train button logic | app/main.py | 280-283 |

---

## Summary

✅ **Clear problem identification** with bold headers  
✅ **Non-technical language** avoiding jargon  
✅ **Actionable guidance** on how to fix  
✅ **Separated concerns** (missing values ≠ type mismatches)  
✅ **Specific numbers** showing actual counts  
✅ **Alternative suggestions** when applicable  
✅ **Production-ready** implementation  
