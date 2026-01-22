# Error Messages: Visual Quick Reference Guide

## Error Type Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ERROR MESSAGE TYPES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ⚠️  MISSING VALUES                                                          │
│  └─ Your target has X empty cell(s). Clean the data first...               │
│                                                                              │
│  ❌ WRONG DATA TYPE                                                          │
│  └─ Your target contains text/categories. Use Classification instead.       │
│                                                                              │
│  ❌ NOT ENOUGH CATEGORIES                                                    │
│  └─ Your target needs at least 2 different values. Currently has only 1.   │
│                                                                              │
│  ❌ MODEL LIMITATION                                                         │
│  └─ Logistic Regression only works with 2 categories. Try Random Forest...  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Error Decision Tree

```
                    START: User selects target column
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Missing values?     │
                    │ (NaN/empty cells)   │
                    └─────────────────────┘
                         │          │
                        YES        NO
                         │          │
                         ▼          ▼
                    ⚠️ MISSING    Check task type
                    VALUES        compatibility
                    ERROR         │
                                  ▼
                         ┌──────────────────────┐
                         │ Task-Target Match?   │
                         │ (Regression=numeric) │
                         │ (Classification=2+)  │
                         └──────────────────────┘
                              │          │
                            YES        NO
                             │          │
                             ▼          ▼
                         Check model   ❌ WRONG
                         constraints   DATA TYPE
                             │         ERROR
                             ▼
                    ┌──────────────────────┐
                    │ Model-specific OK?   │
                    │ (LogReg=binary only) │
                    └──────────────────────┘
                         │          │
                        YES        NO
                         │          │
                         ▼          ▼
                    ✅ VALID    ❌ MODEL
                    TRAINING   LIMITATION
                    ENABLED    ERROR
```

---

## Error Message Catalog

### 1️⃣ Missing Values Error

**Icon**: ⚠️  
**Header**: **Missing Values**  
**Severity**: Warning (yellow)  
**Train Button**: DISABLED

**Message Template**:
```
⚠️ **Missing Values**: Your target has {count} empty cell(s). 
Clean the data first or remove rows with missing targets.
```

**Examples**:
```
⚠️ **Missing Values**: Your target has 1 empty cell(s). Clean the data first or remove rows with missing targets.
⚠️ **Missing Values**: Your target has 5 empty cell(s). Clean the data first or remove rows with missing targets.
```

**Code**:
```python
if missing_count > 0:
    st.error(f"⚠️ **Missing Values**: Your target has {missing_count} empty cell(s). Clean the data first or remove rows with missing targets.")
```

---

### 2️⃣ Wrong Data Type Error

**Icon**: ❌  
**Header**: **Wrong Data Type**  
**Severity**: Error (red)  
**Train Button**: DISABLED

**Scenarios**:

#### Categorical + Regression
```
❌ **Wrong Data Type**: Your target contains text/categories. Use Classification instead.
```

#### Non-numeric + Regression
```
❌ **Wrong Data Type**: Your target must be numeric for Regression. Try Classification or select a different column.
```

**Code**:
```python
if task_type == "Regression":
    if is_categorical:
        return False, "type_mismatch", "Your target contains text/categories. Use Classification instead."
    if not is_numeric:
        return False, "type_mismatch", "Your target must be numeric for Regression. Try Classification or select a different column."
```

---

### 3️⃣ Not Enough Categories Error

**Icon**: ❌  
**Header**: **Not Enough Categories**  
**Severity**: Error (red)  
**Train Button**: DISABLED

**Scenarios**:

#### Too Few Classes
```
❌ **Not Enough Categories**: Your target needs at least 2 different values. Currently has only 1.
```

#### Too Many Classes
```
❌ **Not Enough Categories**: Too many categories (50+). Try Regression instead.
```

**Code**:
```python
if task_type == "Classification":
    if unique_count < 2:
        return False, "class_count", "Your target needs at least 2 different values. Currently has only 1."
    if unique_count > 50 and is_numeric:
        return False, "class_count", "Too many categories (50+). Try Regression instead."
```

---

### 4️⃣ Model Limitation Error

**Icon**: ❌  
**Header**: **Model Limitation**  
**Severity**: Error (red)  
**Train Button**: DISABLED

**Scenario**: Multi-class + Logistic Regression
```
❌ **Model Limitation**: Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting.
```

**Code**:
```python
if model_name == "logistic_regression" and unique_count > 2:
    return False, "model_constraint", "Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting."
```

---

## User Action → Error Mapping

```
┌──────────────────────────────────────────────────────────────────────────┐
│ USER ACTION                          │ ERROR SHOWN                        │
├──────────────────────────────────────────────────────────────────────────┤
│ Select column with NaN values        │ ⚠️ Missing Values                  │
│ Select text column + Regression      │ ❌ Wrong Data Type                 │
│ Select single-value column + Class   │ ❌ Not Enough Categories           │
│ Select 100+ value column + Class     │ ❌ Not Enough Categories           │
│ Select 3+ class + Logistic Reg       │ ❌ Model Limitation                │
│ Select numeric + Regression          │ ✅ No error                        │
│ Select 2+ class + Classification     │ ✅ No error                        │
│ Select binary + Logistic Reg         │ ✅ No error                        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Checklist

### Validation Function
- [x] Returns (is_valid, error_type, error_message)
- [x] error_type: 'type_mismatch', 'class_count', 'model_constraint', or None
- [x] Non-technical error messages
- [x] Actionable guidance in messages
- [x] Specific numbers/values in messages

### Error Display
- [x] Missing values shown separately (⚠️)
- [x] Type mismatches shown with ❌ **Wrong Data Type**
- [x] Class count issues shown with ❌ **Not Enough Categories**
- [x] Model constraints shown with ❌ **Model Limitation**
- [x] Bold headers for clarity
- [x] Clear, non-technical language

### Train Button Logic
- [x] Disabled when missing_count > 0
- [x] Disabled when validation fails
- [x] Enabled only when all validations pass
- [x] Consistent across all error types

---

## Message Characteristics

### ✅ Good Error Messages
```
⚠️ **Missing Values**: Your target has 3 empty cell(s). Clean the data first or remove rows with missing targets.
```
- Clear problem identification (bold header)
- Specific information (3 empty cells)
- Actionable guidance (clean the data)
- Non-technical language (empty cells, not NaN)

### ❌ Bad Error Messages
```
❌ Target has missing values
```
- No problem categorization
- No specific information
- No guidance on how to fix
- Ambiguous (could be features or target)

---

## Code Integration Points

### Location 1: Validation Function Definition
**File**: `app/main.py`  
**Lines**: 27-72  
**Purpose**: Define validation logic and error messages

### Location 2: Task-Target Validation Display
**File**: `app/main.py`  
**Lines**: 234-240  
**Purpose**: Show missing value and type mismatch errors

### Location 3: Model-Specific Validation Display
**File**: `app/main.py`  
**Lines**: 250-256  
**Purpose**: Show model constraint errors

### Location 4: Train Button Logic
**File**: `app/main.py`  
**Lines**: 280-283  
**Purpose**: Disable button based on validation results

---

## Testing Scenarios

### Test 1: Categorical Target + Regression
```python
# Setup
data = pd.DataFrame({'target': ['A', 'B', 'A', 'B']})
task_type = 'Regression'

# Expected
Error: ❌ **Wrong Data Type**: Your target contains text/categories. Use Classification instead.
Train Button: DISABLED
```

### Test 2: Single Value + Classification
```python
# Setup
data = pd.DataFrame({'target': [1, 1, 1, 1]})
task_type = 'Classification'

# Expected
Error: ❌ **Not Enough Categories**: Your target needs at least 2 different values. Currently has only 1.
Train Button: DISABLED
```

### Test 3: Multi-class + Logistic Regression
```python
# Setup
data = pd.DataFrame({'target': [0, 1, 2, 0, 1, 2]})
task_type = 'Classification'
model_name = 'logistic_regression'

# Expected
Error: ❌ **Model Limitation**: Logistic Regression only works with 2 categories. Your target has 3+. Try Random Forest or Gradient Boosting.
Train Button: DISABLED
```

### Test 4: Missing Values
```python
# Setup
data = pd.DataFrame({'target': [1, 2, np.nan, 1, 2]})

# Expected
Error: ⚠️ **Missing Values**: Your target has 1 empty cell(s). Clean the data first or remove rows with missing targets.
Train Button: DISABLED
```

### Test 5: Valid Configuration
```python
# Setup
data = pd.DataFrame({'target': [0, 1, 0, 1, 0]})
task_type = 'Classification'
model_name = 'logistic_regression'

# Expected
Error: (None)
Train Button: ENABLED ✅
```

---

## Summary Table

| Error Type | Icon | Header | Severity | Train Button | How to Fix |
|-----------|------|--------|----------|--------------|-----------|
| Missing Values | ⚠️ | Missing Values | Warning | DISABLED | Clean data or remove rows |
| Type Mismatch | ❌ | Wrong Data Type | Error | DISABLED | Switch task type or column |
| Class Count | ❌ | Not Enough Categories | Error | DISABLED | Select different column |
| Model Constraint | ❌ | Model Limitation | Error | DISABLED | Choose different model |

---

## Key Takeaways

✅ **Categorized errors** enable better user understanding  
✅ **Non-technical language** makes messages accessible  
✅ **Specific information** helps users understand the issue  
✅ **Actionable guidance** tells users exactly what to do  
✅ **Alternative suggestions** provide options  
✅ **Consistent display** across all error types  
✅ **Production-ready** implementation  

All error messages follow the pattern:
```
[ICON] **[PROBLEM TYPE]**: [WHAT IS WRONG]. [HOW TO FIX IT].
```

This ensures clarity, consistency, and user satisfaction.
