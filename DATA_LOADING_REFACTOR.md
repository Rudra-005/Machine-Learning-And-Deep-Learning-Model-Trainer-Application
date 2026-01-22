# Data Loading Page Refactor - Integration Guide

## Overview

The refactored Data Loading page stores the dataframe in `st.session_state["dataset"]` as the single source of truth, with optional metadata in `dataset_shape` and `dataset_columns`.

---

## Key Changes

### Before (Old)
```python
# Multiple flags and checks
st.session_state.data_preprocessed = False
st.session_state.dataset = None
st.session_state.X_train = None

# Checks scattered everywhere
if st.session_state.data_preprocessed:
    # ...
if st.session_state.dataset is not None:
    # ...
```

### After (New)
```python
# Single source of truth
st.session_state.dataset = df
st.session_state.dataset_shape = df.shape
st.session_state.dataset_columns = list(df.columns)

# Single canonical check
if is_data_loaded():
    # ...
```

---

## Session State Contract

### Canonical Keys
| Key | Type | Purpose |
|-----|------|---------|
| `dataset` | pd.DataFrame or None | **CANONICAL** - Only indicator |
| `dataset_shape` | tuple or None | Metadata: (rows, cols) |
| `dataset_columns` | list or None | Metadata: column names |

### Dependent Keys
| Key | Type | Purpose |
|-----|------|---------|
| `X_train` | np.ndarray | Training features |
| `X_test` | np.ndarray | Test features |
| `y_train` | np.ndarray | Training labels |
| `y_test` | np.ndarray | Test labels |

---

## Helper Functions

### 1. Check if Data Loaded
```python
def is_data_loaded():
    """Canonical check: data is loaded if dataset exists."""
    return st.session_state.get("dataset") is not None
```

**Usage:**
```python
if is_data_loaded():
    # Data is guaranteed to exist
    df = st.session_state.dataset
    rows, cols = st.session_state.dataset_shape
```

### 2. Set Dataset with Metadata
```python
def set_dataset(df):
    """Set dataset and metadata. Single point for dataset storage."""
    st.session_state.dataset = df
    st.session_state.dataset_shape = df.shape
    st.session_state.dataset_columns = list(df.columns)
```

**Usage:**
```python
# When loading CSV
df = pd.read_csv(uploaded_file)
set_dataset(df)

# When loading sample
df = load_sample_dataset()
set_dataset(df)
```

### 3. Initialize Session State
```python
def initialize_session_state():
    """Initialize session state with dataset as source of truth."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_shape" not in st.session_state:
        st.session_state.dataset_shape = None
    if "dataset_columns" not in st.session_state:
        st.session_state.dataset_columns = None
    # ... other keys
```

**Usage:**
```python
# At app startup
initialize_session_state()
```

---

## Integration Steps

### Step 1: Import Helpers
```python
from data_loading_refactored import is_data_loaded, set_dataset, initialize_session_state
```

### Step 2: Initialize at Startup
```python
# In main app
initialize_session_state()
```

### Step 3: Use in Data Loading Page
```python
def page_data_loading():
    # ... UI code ...
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        set_dataset(df)  # Store with metadata
        st.success("✓ Loaded")
    
    if is_data_loaded():
        # Display dataset info
        st.metric("Rows", st.session_state.dataset_shape[0])
        st.metric("Cols", st.session_state.dataset_shape[1])
```

### Step 4: Use in Other Pages
```python
def page_automl_training():
    if not is_data_loaded():
        st.warning("Load data first")
        return
    
    # Data is guaranteed to exist
    df = st.session_state.dataset
    rows, cols = st.session_state.dataset_shape
    columns = st.session_state.dataset_columns
```

---

## Data Persistence

### Across Page Navigation
```
User loads CSV on Data Loading page
    ↓
st.session_state.dataset = df (persists)
st.session_state.dataset_shape = (rows, cols) (persists)
st.session_state.dataset_columns = [...] (persists)
    ↓
User navigates to AutoML Training page
    ↓
is_data_loaded() returns True
    ↓
Data is available in AutoML page
```

### Across Reruns
```
User interacts with widget
    ↓
Page reruns
    ↓
st.session_state.dataset still exists
    ↓
Data persists across reruns
```

---

## Benefits

✅ **Single Source of Truth** - Only `dataset` indicates data availability  
✅ **No Boolean Flags** - No `data_loaded`, `data_preprocessed`, etc.  
✅ **Metadata Included** - Shape and columns available without recomputing  
✅ **Persists Across Navigation** - Data survives page changes  
✅ **Persists Across Reruns** - Data survives widget interactions  
✅ **Reusable Helpers** - Same functions across all pages  
✅ **Type Safe** - Metadata types are predictable  

---

## Example: Complete Integration

### Data Loading Page
```python
def page_data_loading():
    st.title("📊 Data Loading")
    
    uploaded_file = st.file_uploader("CSV", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        set_dataset(df)  # Store with metadata
        st.success("✓ Loaded")
    
    if is_data_loaded():
        st.metric("Rows", st.session_state.dataset_shape[0])
        st.dataframe(st.session_state.dataset.head())
```

### AutoML Training Page
```python
def page_automl_training():
    st.title("🧠 AutoML Training")
    
    if not is_data_loaded():
        st.warning("Load data first")
        return
    
    # Data is guaranteed to exist
    df = st.session_state.dataset
    rows, cols = st.session_state.dataset_shape
    
    st.write(f"Training on {rows} rows × {cols} columns")
```

### Sidebar Status
```python
st.sidebar.markdown("**Status**")
if is_data_loaded():
    rows, cols = st.session_state.dataset_shape
    st.sidebar.success(f"✅ Data: {rows}×{cols}")
else:
    st.sidebar.warning("⚠️ No Data")
```

---

## Migration Checklist

- [ ] Import helpers from `data_loading_refactored.py`
- [ ] Call `initialize_session_state()` at app startup
- [ ] Replace all `st.session_state.dataset = df` with `set_dataset(df)`
- [ ] Replace all data availability checks with `is_data_loaded()`
- [ ] Remove `data_preprocessed` flag
- [ ] Remove `data_loaded` flag
- [ ] Test data persistence across page navigation
- [ ] Test data persistence across reruns

---

## Files

| File | Purpose |
|------|---------|
| `data_loading_refactored.py` | Refactored page with helpers |
| `DATA_LOADING_REFACTOR.md` | This guide |

---

## Summary

The refactored Data Loading page:
- ✅ Stores dataframe in `st.session_state["dataset"]`
- ✅ Stores metadata in `dataset_shape` and `dataset_columns`
- ✅ Uses `is_data_loaded()` for all checks
- ✅ Uses `set_dataset()` for all storage
- ✅ Persists across page navigation
- ✅ Persists across reruns
- ✅ No boolean flags

**Result: Single source of truth, consistent across all pages.**
