# Error Handling Quick Reference

## Import
```python
from app.utils.error_handler import ErrorHandler, MemoryMonitor, safe_execute
from app.utils.logger import logger
```

---

## Data Validation
```python
if not ErrorHandler.handle_data_validation(data):
    return
```

**Checks:**
- Not empty
- 2+ columns
- 10+ rows

---

## Target Selection
```python
if not ErrorHandler.handle_target_selection(data, target_col, task_type):
    return
```

**Checks:**
- Column exists
- < 50% missing
- Classification: 2+ unique values
- Regression: numeric values

---

## Model Training (Decorator)
```python
@ErrorHandler.handle_model_training
def train_model():
    model = ModelFactory.create_model(task_type, model_name)
    model.fit(X_train, y_train)
    return model

result = train_model()
```

**Handles:**
- Memory errors
- Training failures
- Value errors
- Unexpected exceptions

---

## Memory Check
```python
if not MemoryMonitor.check_memory():
    st.warning("Low memory available")
    return
```

---

## Memory Info
```python
info = MemoryMonitor.get_memory_info()
# Returns: {'total': GB, 'available': GB, 'percent': %, 'used': GB}
```

---

## Safe Execution
```python
result = safe_execute(some_function, arg1, arg2)
if result is None:
    # Function failed
    pass
```

---

## Logging
```python
logger.info("Training started")
logger.warning("High memory usage")
logger.error("Training failed: details")
```

---

## View Logs
```bash
tail -f logs/app.log
grep ERROR logs/app.log
```

---

## Set Log Level
```bash
export LOG_LEVEL=DEBUG  # or INFO, WARNING, ERROR
```

---

## Error Messages (Auto-Shown)

### Data Validation
```
❌ Data Validation Error
Dataset is empty / Too few columns / Too few rows
```

### Target Selection
```
❌ Target Selection Error
Column not found / Missing values / Invalid values
```

### Training
```
❌ Memory Error / Training Error / Unexpected Error
[Actionable solutions provided]
```

---

## Production Checklist

- [ ] Import error handler
- [ ] Add data validation
- [ ] Add target validation
- [ ] Wrap training with decorator
- [ ] Check memory before training
- [ ] Monitor logs
- [ ] Set log level
- [ ] Test error scenarios

---

## Common Patterns

### Pattern 1: Validate & Return
```python
if not ErrorHandler.handle_data_validation(data):
    return
# Continue with valid data
```

### Pattern 2: Decorator
```python
@ErrorHandler.handle_model_training
def train():
    pass

result = train()
if result is None:
    return
```

### Pattern 3: Memory Check
```python
if not MemoryMonitor.check_memory():
    return
# Proceed with training
```

### Pattern 4: Safe Execute
```python
result = safe_execute(func, *args)
if result is None:
    return
```

---

## Error Types

| Error | Cause | Solution |
|-------|-------|----------|
| DataValidationError | Invalid dataset | Check data quality |
| TargetSelectionError | Invalid target | Select valid column |
| ModelTrainingError | Training failed | Check logs |
| MemoryError | Out of memory | Reduce dataset size |
| PreprocessingError | Preprocessing failed | Check data format |

---

## Status: ✅ Production Ready

All error handling implemented and documented!

