# Canonical Session State Contract for Dataset Loading

## Overview

This document defines the canonical session state contract for dataset loading in the ML/DL Trainer application.

**RULE**: `st.session_state["dataset"]` is the **ONLY** indicator that data is loaded.

---

## Contract Definition

### Single Source of Truth

```python
# CANONICAL CHECK
if st.session_state.get("dataset") is not None:
    # Data is loaded
else:
    # Data is not loaded
```

### Helper Function

```python
def is_data_loaded():
    """Canonical check: data is loaded if dataset exists."""
    return st.session_state.get("dataset") is not None
```

**All pages must use this function. No exceptions.**

---

## Session State Structure

### Canonical Key
- **`st.session_state["dataset"]`** - The ONLY indicator of data availability
  - `None` = no data loaded
  - `pd.DataFrame` = data loaded and ready

### Dependent Keys
These are populated when data is loaded but are NOT used for availability checks:
- `st.session_state["X_train"]` - Training features
- `st.session_state["X_test"]` - Test features
- `st.session_state["y_train"]` - Training labels
- `st.session_state["y_test"]` - Test labels
- `st.session_state["preprocessor"]` - Preprocessing pipeline

### Other Keys
- `st.session_state["model_trained"]` - Whether model is trained
- `st.session_state["trained_model"]` - Trained model instance
- `st.session_state["training_results"]` - Training results

---

## Usage Across Pages

### ✅ Correct Usage

**Sidebar Status**
```python
if is_data_loaded():
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.warning("⚠️ No Data")
```

**Data Loading Page**
```python
if is_data_loaded():
    st.subheader("📋 Data Overview")
    st.metric("Rows", st.session_state.dataset.shape[0])
```

**AutoML Training Page**
```python
if not is_data_loaded():
    st.warning("⚠️ Load data first")
    return
```

**Results Page**
```python
if not st.session_state.model_trained:
    st.warning("⚠️ Train a model first")
    return
```

### ❌ Incorrect Usage

**Don't use multiple flags:**
```python
# WRONG - Multiple checks
if st.session_state.data_loaded and st.session_state.X_train is not None:
    # This is redundant and error-prone
```

**Don't check dependent keys:**
```python
# WRONG - Checking dependent key
if st.session_state.X_train is not None:
    # This is not the source of truth
```

**Don't create new flags:**
```python
# WRONG - Creating new flag
st.session_state.data_available = True
if st.session_state.data_available:
    # This violates the contract
```

---

## Implementation Guide

### Step 1: Import Contract
```python
from session_state_contract import is_data_loaded, initialize_session_state
```

### Step 2: Initialize at Startup
```python
# In main app file, before any page logic
initialize_session_state()
```

### Step 3: Use Helper in All Pages
```python
# In every page that needs data
if not is_data_loaded():
    st.warning("⚠️ Load data first")
    return
```

### Step 4: Set Dataset When Loading
```python
# When user loads data
st.session_state.dataset = pd.read_csv(uploaded_file)
st.session_state.X_train = X_train
st.session_state.X_test = X_test
# ... other dependent keys
```

---

## Benefits

✅ **Single Source of Truth** - One key to check everywhere  
✅ **No Inconsistency** - Impossible to have stale flags  
✅ **Production Safe** - Clear contract prevents bugs  
✅ **Easy to Maintain** - Change logic in one place  
✅ **Reusable** - Same helper across all pages  
✅ **Testable** - Simple function to unit test  

---

## Example: Complete Page Implementation

```python
from session_state_contract import is_data_loaded, initialize_session_state

# Initialize once at app startup
initialize_session_state()

def page_automl_training():
    st.title("🧠 AutoML Training")
    
    # Use canonical check
    if not is_data_loaded():
        st.warning("⚠️ Load data first in Data Loading tab")
        return
    
    # Data is guaranteed to exist here
    st.markdown("AutoML detects model type and applies optimal strategy.")
    
    # ... rest of page logic
    # Can safely access st.session_state.dataset
    # Can safely access st.session_state.X_train, etc.
```

---

## Migration Guide

### From Old Code
```python
# Old: Multiple flags
if st.session_state.data_loaded and st.session_state.X_train is not None:
    # ...
```

### To New Code
```python
# New: Single check
if is_data_loaded():
    # ...
```

### Changes Required
1. Import `is_data_loaded` from `session_state_contract`
2. Replace all data availability checks with `is_data_loaded()`
3. Remove `data_loaded` flag from session state
4. Remove redundant `X_train is not None` checks

---

## Contract Enforcement

### What the Contract Guarantees

If `is_data_loaded()` returns `True`:
- ✅ `st.session_state.dataset` is not None
- ✅ `st.session_state.X_train` is populated
- ✅ `st.session_state.X_test` is populated
- ✅ `st.session_state.y_train` is populated
- ✅ `st.session_state.y_test` is populated

### What the Contract Requires

- ✅ Always use `is_data_loaded()` to check data availability
- ✅ Always set `st.session_state.dataset` when loading data
- ✅ Always set dependent keys together with dataset
- ✅ Never create alternative data availability flags

---

## Testing

### Unit Test Example
```python
def test_is_data_loaded():
    # Test when no data
    st.session_state.dataset = None
    assert not is_data_loaded()
    
    # Test when data exists
    st.session_state.dataset = pd.DataFrame({'a': [1, 2, 3]})
    assert is_data_loaded()
```

### Integration Test Example
```python
def test_data_loading_page():
    # Load data
    st.session_state.dataset = load_sample_data()
    
    # Check all pages see data
    assert is_data_loaded()
    assert st.session_state.X_train is not None
    assert st.session_state.X_test is not None
```

---

## Summary

| Aspect | Rule |
|--------|------|
| **Source of Truth** | `st.session_state["dataset"]` |
| **Check Function** | `is_data_loaded()` |
| **Usage** | All pages must use `is_data_loaded()` |
| **Flags** | No other data availability flags allowed |
| **Dependent Keys** | Set together with dataset |
| **Guarantee** | If `is_data_loaded()` is True, all data exists |

---

## Files

- **`session_state_contract.py`** - Contract definition and helpers
- **`main_canonical.py`** - Example implementation using contract
- **`SESSION_STATE_CONTRACT.md`** - This documentation

---

**This contract ensures consistency across all pages and prevents multi-page bugs.**
