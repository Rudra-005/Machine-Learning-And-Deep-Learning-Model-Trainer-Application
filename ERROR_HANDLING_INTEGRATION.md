# Error Handling Integration Guide

## Quick Start

The error handling module is already created and ready to use.

### File Location
```
app/utils/error_handler.py
```

### Import in Your Code

```python
from app.utils.error_handler import ErrorHandler, MemoryMonitor, safe_execute
```

---

## Usage Examples

### 1. Validate Data on Upload

```python
# In Data Upload section
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Validate data
    if not ErrorHandler.handle_data_validation(data):
        return  # Error shown to user, exit
    
    st.success("✅ Data validated!")
    st.session_state.data = data
```

### 2. Validate Target Selection

```python
# In Training section
target_col = st.selectbox("Target Column", data.columns)
task_type = st.selectbox("Task Type", ["Classification", "Regression"])

# Validate target
if not ErrorHandler.handle_target_selection(data, target_col, task_type):
    return  # Error shown to user, exit
```

### 3. Wrap Model Training

```python
@ErrorHandler.handle_model_training
def train_model_safe():
    # Your training code
    model = ModelFactory.create_model(task_type, model_name)
    model.fit(X_train, y_train)
    return model

# Use it
model = train_model_safe()
if model is None:
    st.error("Training failed")
```

### 4. Check Memory Before Training

```python
if not MemoryMonitor.check_memory():
    st.warning("⚠️ Low memory available. Training may fail.")
    return

# Proceed with training
```

### 5. Get Memory Info

```python
info = MemoryMonitor.get_memory_info()
st.write(f"Memory Usage: {info['percent']:.1f}%")
st.write(f"Available: {info['available']:.2f} GB")
```

---

## Error Types Handled

### DataValidationError
- Empty dataset
- Too few columns
- Too few rows

### TargetSelectionError
- Column not found
- All missing values
- Too many missing values
- Invalid unique values
- Non-numeric for regression

### ModelTrainingError
- Training failures
- Memory errors
- Value errors
- Unexpected exceptions

### PreprocessingError
- Preprocessing failures

---

## Logging

### View Logs

```bash
# Real-time
tail -f logs/app.log

# Last 50 lines
tail -50 logs/app.log

# Search errors
grep ERROR logs/app.log
```

### Log Levels

```
DEBUG   - Detailed information
INFO    - General information
WARNING - Warning messages
ERROR   - Error messages
```

### Set Log Level

```bash
# Windows
set LOG_LEVEL=DEBUG

# Linux/Mac
export LOG_LEVEL=DEBUG
```

---

## Production Checklist

- [x] Error handler module created
- [x] Custom exceptions defined
- [x] Memory monitoring implemented
- [x] Logging configured
- [x] User-friendly error messages
- [x] Decorators for easy integration
- [x] Safe execution wrapper
- [x] Documentation provided

---

## Next Steps

1. **Import error handler** in main.py
2. **Add validation** to data upload
3. **Add validation** to target selection
4. **Wrap training** with decorator
5. **Monitor logs** in production
6. **Set log level** for environment

---

## Support

For issues:
1. Check `logs/app.log`
2. Review error messages
3. Refer to `PRODUCTION_ERROR_HANDLING.md`
4. Check memory usage

---

**Status**: ✅ Ready for production deployment!

