# Data Loading Refactor - Quick Reference

## The Change

### Before
```python
st.session_state.data_preprocessed = False  # Boolean flag
st.session_state.dataset = None
```

### After
```python
st.session_state.dataset = df  # Actual data
st.session_state.dataset_shape = df.shape  # Metadata
st.session_state.dataset_columns = list(df.columns)  # Metadata
```

---

## Three Helper Functions

### 1. Check if Data Loaded
```python
def is_data_loaded():
    return st.session_state.get("dataset") is not None
```

### 2. Set Dataset with Metadata
```python
def set_dataset(df):
    st.session_state.dataset = df
    st.session_state.dataset_shape = df.shape
    st.session_state.dataset_columns = list(df.columns)
```

### 3. Initialize Session State
```python
def initialize_session_state():
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "dataset_shape" not in st.session_state:
        st.session_state.dataset_shape = None
    if "dataset_columns" not in st.session_state:
        st.session_state.dataset_columns = None
```

---

## Usage

### Load CSV
```python
df = pd.read_csv(uploaded_file)
set_dataset(df)  # Store with metadata
```

### Load Sample
```python
df = load_sample_dataset()
set_dataset(df)  # Store with metadata
```

### Check if Loaded
```python
if is_data_loaded():
    rows, cols = st.session_state.dataset_shape
    columns = st.session_state.dataset_columns
    df = st.session_state.dataset
```

### Display Info
```python
if is_data_loaded():
    st.metric("Rows", st.session_state.dataset_shape[0])
    st.metric("Cols", st.session_state.dataset_shape[1])
```

---

## Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `dataset` | pd.DataFrame or None | **CANONICAL** |
| `dataset_shape` | tuple or None | Metadata: (rows, cols) |
| `dataset_columns` | list or None | Metadata: column names |

---

## Persistence

✅ Persists across page navigation  
✅ Persists across reruns  
✅ No boolean flags  
✅ Single source of truth  

---

## Integration

1. Import helpers
2. Call `initialize_session_state()` at startup
3. Use `set_dataset(df)` when loading
4. Use `is_data_loaded()` to check
5. Access via `st.session_state.dataset`

---

## Files

- `data_loading_refactored.py` - Refactored page
- `DATA_LOADING_REFACTOR.md` - Full guide
- `DATA_LOADING_QUICK_REF.md` - This file
