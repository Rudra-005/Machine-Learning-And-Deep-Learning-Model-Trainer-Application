# Parameter Validation Layer

Production-ready validation preventing parameter mismatches with user-friendly warnings.

## Overview

The validation layer prevents:
- ✅ **Epochs for ML models** - Shows warning, ignores parameter
- ✅ **max_iter for tree-based models** - Shows warning, ignores parameter
- ✅ **CV for DL models** - Shows warning, uses train/val/test instead
- ✅ **Batch size for ML models** - Shows warning, ignores parameter
- ✅ **Learning rate for tree-based models** - Shows warning, ignores parameter

## Architecture

```
app/utils/
├── parameter_validator.py       # Core validation logic
└── __init__.py

examples/
└── parameter_validation_example.py  # Usage examples
```

## Core Components

### ParameterValidator Class

```python
from app.utils.parameter_validator import ParameterValidator

# Validate single parameter
is_valid, warning = ParameterValidator.validate_epochs_usage('random_forest', 50)

# Validate all parameters
params = {'epochs': 50, 'cv_folds': 5}
is_valid, warnings = ParameterValidator.validate_all_parameters('random_forest', params)

# Display warnings
ParameterValidator.display_warnings(warnings)

# Filter to valid parameters only
filtered = ParameterValidator.get_valid_parameters('random_forest', params)
```

## Validation Rules

### Epochs
```
✅ Valid for: Sequential, CNN, RNN
❌ Invalid for: Random Forest, Logistic Regression, SVM, etc.

Warning: "Epochs not applicable for {model_name}. 
Use K-Fold CV for ML models."
```

### Max Iterations
```
✅ Valid for: Logistic Regression, SGD, Perceptron
❌ Invalid for: Random Forest, Gradient Boosting, Sequential, CNN, RNN

Warning: "Max iterations not applicable for {model_name}.
Use epochs for DL or CV for tree-based models."
```

### Cross-Validation Folds
```
✅ Valid for: Random Forest, Gradient Boosting, Logistic Regression, SVM
❌ Invalid for: Sequential, CNN, RNN

Warning: "K-fold CV not applicable for {model_name}.
Use train/val/test split for DL models."
```

### Batch Size
```
✅ Valid for: Sequential, CNN, RNN
❌ Invalid for: Random Forest, Logistic Regression, SVM, etc.

Warning: "Batch size not applicable for {model_name}.
ML models train on full dataset at once."
```

### Learning Rate
```
✅ Valid for: Sequential, CNN, RNN, Logistic Regression, SGD
❌ Invalid for: Random Forest, Gradient Boosting, Perceptron

Warning: "Learning rate not applicable for {model_name}.
Tree-based models don't use learning rates."
```

## Usage Examples

### Example 1: Validate Single Parameter

```python
from app.utils.parameter_validator import ParameterValidator

# Check if epochs is valid for Random Forest
is_valid, warning = ParameterValidator.validate_epochs_usage('random_forest', 50)

if not is_valid:
    print(warning)
    # Output: "Epochs not applicable for random_forest..."
```

### Example 2: Validate All Parameters

```python
params = {
    'epochs': 50,
    'cv_folds': 5,
    'batch_size': 32
}

is_valid, warnings = ParameterValidator.validate_all_parameters('random_forest', params)

if not is_valid:
    for warning in warnings:
        st.warning(warning)
```

### Example 3: Filter Parameters

```python
params = {
    'epochs': 50,
    'max_iter': 1000,
    'cv_folds': 5,
    'batch_size': 32,
    'n_estimators': 100
}

# Get only valid parameters for Random Forest
filtered = ParameterValidator.get_valid_parameters('random_forest', params)
# Result: {'cv_folds': 5, 'n_estimators': 100}
```

### Example 4: Streamlit Integration

```python
import streamlit as st
from app.utils.parameter_validator import ParameterValidator

# Collect parameters from UI
params = {
    'epochs': st.slider("Epochs", 1, 100, 50),
    'cv_folds': st.slider("CV Folds", 3, 10, 5),
    'batch_size': st.selectbox("Batch Size", [16, 32, 64])
}

# Validate
is_valid, warnings = ParameterValidator.validate_all_parameters(model_name, params)

# Display warnings
if not is_valid:
    ParameterValidator.display_warnings(warnings)

# Use only valid parameters
valid_params = ParameterValidator.get_valid_parameters(model_name, params)
train_model(model_name, valid_params)
```

## Parameter Compatibility Matrix

| Parameter | Tree-Based | Iterative | DL |
|-----------|-----------|-----------|-----|
| epochs | ❌ | ❌ | ✅ |
| max_iter | ❌ | ✅ | ❌ |
| cv_folds | ✅ | ✅ | ❌ |
| batch_size | ❌ | ❌ | ✅ |
| learning_rate | ❌ | ✅ | ✅ |
| n_estimators | ✅ | ❌ | ❌ |
| max_depth | ✅ | ❌ | ❌ |

## Warning Messages

### Epochs Warning
```
⚠️ Epochs not applicable for {model_name}

Epochs are for deep learning models (Sequential, CNN, RNN).

For ML models:
- Use K-Fold Cross-Validation for robust evaluation
- Use Max Iterations for iterative models (LogisticRegression, SGD)
- Tree-based models don't need iterations

Epochs parameter will be ignored.
```

### Max Iterations Warning (Tree-Based)
```
⚠️ Max Iterations not applicable for {model_name}

Max iterations are for iterative models (LogisticRegression, SGD, Perceptron).

Tree-based models ({model_name}) don't use iterations.
They use:
- n_estimators: Number of trees
- max_depth: Tree depth
- K-Fold CV: For evaluation

Max iterations parameter will be ignored.
```

### CV Warning
```
⚠️ K-Fold CV not applicable for {model_name}

K-fold cross-validation is for ML models.

Deep learning models use:
- Train/Validation/Test split: Separate datasets
- Epochs: Multiple passes through training data
- Early Stopping: Monitor validation loss

CV parameter will be ignored. Using train/val/test split instead.
```

## Validation Flow

```
User selects model and parameters
    ↓
ParameterValidator.validate_all_parameters()
    ↓
Check each parameter against model type
    ↓
Collect warnings for invalid parameters
    ↓
Display warnings to user
    ↓
Filter parameters to valid ones only
    ↓
Train model with valid parameters
```

## Testing

Run examples:
```bash
streamlit run examples/parameter_validation_example.py
```

Test scenarios:
1. Epochs validation (DL vs ML)
2. Max iterations validation (Iterative vs Tree-based vs DL)
3. CV validation (ML vs DL)
4. Batch size validation (DL vs ML)
5. Parameter filtering
6. Comprehensive validation

## Files

- `app/utils/parameter_validator.py` - Core logic (200 lines)
- `examples/parameter_validation_example.py` - Usage examples (200 lines)
- `PARAMETER_VALIDATION_README.md` - This documentation

## Benefits

✅ **Prevents Errors** - Invalid parameters caught early
✅ **User-Friendly** - Warnings instead of crashes
✅ **Clear Guidance** - Explains what's valid and why
✅ **Automatic Filtering** - Removes invalid parameters
✅ **Comprehensive** - Validates all parameter combinations
✅ **Minimal Code** - ~400 lines total
✅ **Production Ready** - Robust error handling

## Integration Checklist

- [x] Epochs validation for ML models
- [x] max_iter validation for tree-based models
- [x] CV validation for DL models
- [x] Batch size validation
- [x] Learning rate validation
- [x] User-friendly warnings
- [x] Parameter filtering
- [x] Streamlit integration
- [x] Examples provided
- [x] Comprehensive documentation

## Best Practices

1. **Always validate before training**
   ```python
   is_valid, warnings = ParameterValidator.validate_all_parameters(model_name, params)
   ```

2. **Display warnings to user**
   ```python
   ParameterValidator.display_warnings(warnings)
   ```

3. **Filter parameters before use**
   ```python
   valid_params = ParameterValidator.get_valid_parameters(model_name, params)
   ```

4. **Log validation results**
   ```python
   logger.info(f"Validation: {is_valid}, Warnings: {len(warnings)}")
   ```

## Future Enhancements

- [ ] Custom validation rules per model
- [ ] Parameter range validation
- [ ] Dependency validation (e.g., batch_size requires epochs)
- [ ] Performance impact warnings
- [ ] Recommendation engine for parameters
