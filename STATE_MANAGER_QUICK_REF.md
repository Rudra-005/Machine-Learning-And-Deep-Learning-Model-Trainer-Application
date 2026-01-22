# State Manager - Quick Reference

## Import

```python
from app.utils.state_manager import (
    is_data_loaded, get_dataset, set_dataset, clear_dataset,
    is_model_trained, get_trained_model, set_trained_model,
    get_metrics, set_metrics, clear_training_state, initialize_defaults
)
```

## Data Operations

```python
# Check if data exists
if is_data_loaded():
    data = get_dataset()

# Set data
set_dataset(df)

# Clear data and all preprocessing
clear_dataset()
```

## Model Operations

```python
# Check if model is trained
if is_model_trained():
    model = get_trained_model()

# Set trained model
set_trained_model(model)

# Clear all training state
clear_training_state()
```

## Metrics Operations

```python
# Get metrics
metrics = get_metrics()

# Set metrics
set_metrics(metrics)
```

## Initialization

```python
# Initialize all defaults (call once at app start)
initialize_defaults()
```

---

## Common Patterns

### Pattern 1: Check and Use Data

```python
if is_data_loaded():
    data = get_dataset()
    # Use data
else:
    st.warning("Please upload data first")
```

### Pattern 2: After Training

```python
set_trained_model(model)
set_metrics(metrics)
st.success("Training complete!")
```

### Pattern 3: Show Results

```python
if is_model_trained():
    model = get_trained_model()
    metrics = get_metrics()
    # Display results
else:
    st.info("Train a model first")
```

### Pattern 4: Clear Everything

```python
if st.button("Clear All"):
    clear_dataset()
    clear_training_state()
    st.success("Cleared!")
```

---

## Why Use State Manager?

| Direct Access | State Manager | Benefit |
|---|---|---|
| `'data' in st.session_state` | `is_data_loaded()` | Clearer intent |
| `st.session_state.data` | `get_dataset()` | Single source of truth |
| `st.session_state.data = df` | `set_dataset(df)` | Atomic updates |
| Manual cleanup | `clear_dataset()` | No orphaned state |

---

## State Manager Functions

### is_data_loaded()
- **Returns**: `bool`
- **Purpose**: Check if dataset is loaded
- **Use**: Guard clauses

### get_dataset()
- **Returns**: `DataFrame` or `None`
- **Purpose**: Get loaded dataset
- **Use**: Access data in pages

### set_dataset(data)
- **Args**: `data` (DataFrame)
- **Purpose**: Set dataset in state
- **Use**: After uploading CSV

### clear_dataset()
- **Purpose**: Clear dataset + preprocessing
- **Clears**: data, X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
- **Use**: Reset button, new upload

### is_model_trained()
- **Returns**: `bool`
- **Purpose**: Check if model is trained
- **Use**: Guard clauses

### get_trained_model()
- **Returns**: Model or `None`
- **Purpose**: Get trained model
- **Use**: Access model in pages

### set_trained_model(model)
- **Args**: `model` (sklearn/keras model)
- **Purpose**: Set trained model + flag
- **Sets**: trained_model, model_trained=True
- **Use**: After training

### get_metrics()
- **Returns**: `dict` or `None`
- **Purpose**: Get evaluation metrics
- **Use**: Display results

### set_metrics(metrics)
- **Args**: `metrics` (dict)
- **Purpose**: Set metrics in state
- **Use**: After evaluation

### clear_training_state()
- **Purpose**: Clear all training state
- **Clears**: trained_model, model_trained, training_history, metrics, last_task_type, last_model_name
- **Use**: Reset training

### initialize_defaults()
- **Purpose**: Initialize all state defaults
- **Use**: Call once at app start

---

## Before vs After

### Before (Direct Access)

```python
# Check data
if 'data' in st.session_state:
    data = st.session_state.data
else:
    st.warning("No data")

# Set data
st.session_state.data = df

# Check model
if st.session_state.get('model_trained', False):
    model = st.session_state.trained_model
else:
    st.info("No model")

# Set model
st.session_state.trained_model = model
st.session_state.model_trained = True  # ❌ Easy to forget!

# Clear data
st.session_state.data = None
# ❌ Forgot to clear preprocessing!
```

### After (State Manager)

```python
# Check data
if is_data_loaded():
    data = get_dataset()
else:
    st.warning("No data")

# Set data
set_dataset(df)

# Check model
if is_model_trained():
    model = get_trained_model()
else:
    st.info("No model")

# Set model
set_trained_model(model)  # ✅ Automatically sets flag

# Clear data
clear_dataset()  # ✅ Clears everything
```

---

## Benefits

✅ **Clearer code** - Intent is obvious  
✅ **Fewer bugs** - Consistent state management  
✅ **Easier maintenance** - Change once, update everywhere  
✅ **Type safety** - Validates state types  
✅ **Atomic updates** - Related state always consistent  
✅ **Self-documenting** - Function names explain purpose  

---

**Status**: ✅ READY TO USE
