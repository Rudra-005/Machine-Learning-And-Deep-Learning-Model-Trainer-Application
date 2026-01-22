# Session State Contract - Quick Reference

## The Rule
```
st.session_state["dataset"] is the ONLY indicator that data is loaded.
Use is_data_loaded() helper in ALL pages.
```

---

## Quick Start

### 1. Import
```python
from session_state_contract import is_data_loaded, initialize_session_state
```

### 2. Initialize (once at startup)
```python
initialize_session_state()
```

### 3. Check in Every Page
```python
if not is_data_loaded():
    st.warning("Load data first")
    return
```

### 4. Set When Loading
```python
st.session_state.dataset = df
st.session_state.X_train = X_train
st.session_state.X_test = X_test
st.session_state.y_train = y_train
st.session_state.y_test = y_test
```

---

## Canonical Check

### ✅ Correct
```python
if is_data_loaded():
    # Data exists
```

### ❌ Wrong
```python
if st.session_state.data_loaded:  # Don't use flags
if st.session_state.X_train is not None:  # Don't check dependent keys
if st.session_state.get("dataset"):  # Use is_data_loaded() instead
```

---

## Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `dataset` | pd.DataFrame or None | **CANONICAL** - Only indicator |
| `X_train` | np.ndarray | Training features (dependent) |
| `X_test` | np.ndarray | Test features (dependent) |
| `y_train` | np.ndarray | Training labels (dependent) |
| `y_test` | np.ndarray | Test labels (dependent) |
| `model_trained` | bool | Whether model is trained |

---

## Page Implementation Template

```python
def page_automl_training():
    st.title("🧠 AutoML Training")
    
    # ALWAYS check first
    if not is_data_loaded():
        st.warning("⚠️ Load data first")
        return
    
    # Data is guaranteed to exist here
    st.markdown("AutoML mode active")
    
    # Safe to use all dependent keys
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    
    # ... rest of page
```

---

## Sidebar Status

```python
st.sidebar.markdown("**Status**")
if is_data_loaded():
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.warning("⚠️ No Data")
```

---

## Common Patterns

### Pattern 1: Guard Clause
```python
if not is_data_loaded():
    st.warning("Load data first")
    return
# Safe to proceed
```

### Pattern 2: Conditional Display
```python
if is_data_loaded():
    st.subheader("Data Overview")
    st.metric("Rows", st.session_state.dataset.shape[0])
```

### Pattern 3: Sidebar Status
```python
if is_data_loaded():
    st.sidebar.success("✅ Data Ready")
else:
    st.sidebar.info("ℹ️ Load data")
```

---

## Files

| File | Purpose |
|------|---------|
| `session_state_contract.py` | Contract definition |
| `main_canonical.py` | Example implementation |
| `SESSION_STATE_CONTRACT.md` | Full documentation |
| `SESSION_STATE_QUICK_REF.md` | This file |

---

## Remember

✅ One source of truth: `st.session_state["dataset"]`  
✅ One check function: `is_data_loaded()`  
✅ Use in all pages: No exceptions  
✅ Set together: All dependent keys at once  

**That's it!**
