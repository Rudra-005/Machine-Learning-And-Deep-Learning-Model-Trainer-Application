# Production Error Handling & Logging Guide

## Overview

The ML/DL Trainer now includes robust error handling and logging for production use.

---

## Error Handling Components

### 1. Custom Exceptions (`app/utils/error_handler.py`)

```python
# Base exception
class MLTrainerException(Exception)

# Specific exceptions
class DataValidationError(MLTrainerException)
class TargetSelectionError(MLTrainerException)
class ModelTrainingError(MLTrainerException)
class MemoryError(MLTrainerException)
class PreprocessingError(MLTrainerException)
```

### 2. ErrorHandler Class

**Data Validation:**
```python
ErrorHandler.handle_data_validation(data)
# Checks:
# - Dataset not empty
# - At least 2 columns
# - At least 10 rows
```

**Target Selection:**
```python
ErrorHandler.handle_target_selection(data, target_col, task_type)
# Checks:
# - Column exists
# - Not completely empty
# - Missing values < 50%
# - Classification: 2+ unique values
# - Regression: numeric values
```

**Model Training:**
```python
@ErrorHandler.handle_model_training
def train_model():
    # Handles:
    # - Memory errors
    # - Training failures
    # - Value errors
    # - Unexpected exceptions
```

### 3. Memory Monitoring

```python
MemoryMonitor.check_memory()  # Returns True if < 90% usage
MemoryMonitor.get_memory_info()  # Returns memory stats
```

---

## Logging Configuration

### Setup (`app/utils/logger.py`)

```python
logger = setup_logger("ml_trainer")
# Logs to:
# - File: logs/app.log
# - Console: stdout
# - Level: INFO (configurable)
```

### Log Format

```
2026-01-19 12:34:56 - ml_trainer - INFO - Message
```

---

## Error Handling Patterns

### Pattern 1: Data Validation

```python
if not ErrorHandler.handle_data_validation(data):
    return  # Error already displayed to user
```

### Pattern 2: Target Selection

```python
if not ErrorHandler.handle_target_selection(data, target_col, task_type):
    return  # Error already displayed to user
```

### Pattern 3: Model Training (Decorator)

```python
@ErrorHandler.handle_model_training
def train_model():
    # Your training code
    pass

result = train_model()
if result is None:
    # Training failed, error already shown
    pass
```

### Pattern 4: Safe Execution

```python
result = safe_execute(some_function, arg1, arg2)
if result is None:
    # Function failed, error logged
    pass
```

---

## Error Messages for Users

### Data Validation Errors

```
❌ Data Validation Error

Dataset is empty
Dataset must have at least 2 columns
Dataset must have at least 10 rows
```

### Target Selection Errors

```
❌ Target Selection Error

Target column 'col_name' not found in dataset
Target column is completely empty
Target column has 75.5% missing values
Classification requires at least 2 unique values. Found: 1
Too many classes (150). Consider regression instead.
Cannot convert target to numeric: [error details]
```

### Training Errors

```
❌ Memory Error

Insufficient memory for training.

Solutions:
1. Use a smaller dataset
2. Reduce batch size
3. Close other applications
```

```
❌ Class Distribution Error

Issue with class distribution in data.

Solutions:
1. Check for missing values in target
2. Ensure at least 2 classes exist
3. Try a different train-test split
```

```
❌ Unexpected Training Error

An unexpected error occurred during training.

Troubleshooting:
1. Check data quality in EDA tab
2. Verify target column selection
3. Try with sample dataset
4. Check application logs
```

---

## Logging Examples

### Info Level

```
2026-01-19 12:34:56 - ml_trainer - INFO - Data validation passed: 1000 rows, 10 columns
2026-01-19 12:34:57 - ml_trainer - INFO - Target validation passed: target (Classification)
2026-01-19 12:34:58 - ml_trainer - INFO - Starting model training: train_model
2026-01-19 12:35:10 - ml_trainer - INFO - Model training completed successfully
```

### Warning Level

```
2026-01-19 12:34:56 - ml_trainer - WARNING - High memory usage: 85.5%
2026-01-19 12:34:57 - ml_trainer - WARNING - EDA operation warning: [details]
```

### Error Level

```
2026-01-19 12:34:56 - ml_trainer - ERROR - Data validation error: Dataset is empty
2026-01-19 12:34:57 - ml_trainer - ERROR - Target selection error: Target column not found
2026-01-19 12:34:58 - ml_trainer - ERROR - Model training error: [details]
2026-01-19 12:34:59 - ml_trainer - ERROR - Unexpected error during training: [details]
[Full traceback]
```

---

## Configuration

### Log Level (`app/config.py`)

```python
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"
```

### Set Environment Variable

```bash
# Windows
set LOG_LEVEL=DEBUG

# Linux/Mac
export LOG_LEVEL=DEBUG
```

---

## Monitoring & Debugging

### View Logs

```bash
# Real-time logs
tail -f logs/app.log

# Last 100 lines
tail -100 logs/app.log

# Search for errors
grep ERROR logs/app.log

# Search for specific error
grep "Memory error" logs/app.log
```

### Memory Monitoring

```python
from app.utils.error_handler import MemoryMonitor

# Check current memory
info = MemoryMonitor.get_memory_info()
print(f"Memory usage: {info['percent']}%")
print(f"Available: {info['available']:.2f} GB")
```

---

## Best Practices

### 1. Always Validate Input

```python
if not ErrorHandler.handle_data_validation(data):
    return
```

### 2. Use Decorators for Functions

```python
@ErrorHandler.handle_model_training
def train_model():
    pass
```

### 3. Log Important Events

```python
logger.info(f"Training started with {len(X_train)} samples")
logger.error(f"Training failed: {error_message}")
```

### 4. Check Memory Before Heavy Operations

```python
if not MemoryMonitor.check_memory():
    st.warning("Low memory available")
```

### 5. Provide Actionable Error Messages

```python
# Good
st.error("❌ Classification requires at least 2 unique values")

# Better
st.error("""
❌ Classification requires at least 2 unique values

Solutions:
1. Switch to Regression
2. Select different target column
""")
```

---

## Error Handling Checklist

- [x] Data validation on upload
- [x] Target column validation
- [x] Memory monitoring
- [x] Model training error handling
- [x] Preprocessing error handling
- [x] EDA operation error handling
- [x] User-friendly error messages
- [x] Comprehensive logging
- [x] Exception traceback logging
- [x] Memory limit warnings

---

## Production Deployment

### 1. Set Log Level

```bash
export LOG_LEVEL=INFO
```

### 2. Monitor Logs

```bash
tail -f logs/app.log
```

### 3. Set Memory Threshold

```python
# In MemoryMonitor class
MEMORY_THRESHOLD = 0.9  # 90%
```

### 4. Enable Error Notifications

```python
# Add to error handler for production
# - Send to monitoring service
# - Alert on critical errors
# - Track error frequency
```

---

## Testing Error Handling

### Test Data Validation

```python
# Empty dataset
data = pd.DataFrame()
ErrorHandler.handle_data_validation(data)  # Should fail

# Single column
data = pd.DataFrame({'col': [1, 2, 3]})
ErrorHandler.handle_data_validation(data)  # Should fail

# Valid dataset
data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
ErrorHandler.handle_data_validation(data)  # Should pass
```

### Test Target Selection

```python
# Missing target column
ErrorHandler.handle_target_selection(data, 'missing_col', 'Classification')

# Empty target
data['target'] = None
ErrorHandler.handle_target_selection(data, 'target', 'Classification')

# Single class
data['target'] = 0
ErrorHandler.handle_target_selection(data, 'target', 'Classification')
```

### Test Memory Monitoring

```python
info = MemoryMonitor.get_memory_info()
print(f"Total: {info['total']:.2f} GB")
print(f"Available: {info['available']:.2f} GB")
print(f"Usage: {info['percent']}%")
```

---

## Summary

The application now includes:

✅ **Custom Exceptions** - Specific error types for different scenarios
✅ **Error Handler** - Centralized error handling with user-friendly messages
✅ **Memory Monitoring** - Track and prevent out-of-memory errors
✅ **Comprehensive Logging** - All events logged to file and console
✅ **Decorators** - Easy error handling for functions
✅ **Safe Execution** - Wrapper for safe function execution
✅ **Production Ready** - Suitable for production deployment

---

## Files Modified

- `app/utils/error_handler.py` - NEW: Error handling module
- `app/utils/logger.py` - EXISTING: Logging configuration
- `app/main.py` - UPDATED: Import error handler
- `app/config.py` - EXISTING: Log configuration

---

**Status**: ✅ Production-ready error handling and logging implemented!

