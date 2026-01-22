# Dynamic UI Parameter Logic

Conditional parameter rendering based on model type, tuning settings, and validation rules.

## Overview

The dynamic UI system ensures:
- ✅ **Correct parameters shown** - Only valid parameters for each model
- ✅ **Invalid parameters hidden** - Automatically removed from UI
- ✅ **User validation** - Prevents conceptually incorrect values
- ✅ **Consistent behavior** - Same rules across all models

## Rules

### Rule 1: Show CV Folds for ALL ML Models
```python
if is_tree_based(model_name) or is_iterative(model_name):
    # Show "Cross-Validation Folds (k)" slider
```

**When**: Always for ML models
**Models**: Random Forest, Gradient Boosting, Logistic Regression, SVM, Linear Regression
**Range**: 3-10 folds (configurable per model)

### Rule 2: Show HP Search Iterations ONLY When Tuning Enabled
```python
enable_tuning = st.checkbox("Enable Hyperparameter Tuning")
if enable_tuning:
    # Show "Hyperparameter Search Iterations" slider
```

**When**: User enables tuning checkbox
**Range**: 5-100 iterations
**Applies to**: All models

### Rule 3: Show Max Iterations ONLY for Iterative ML Models
```python
if is_iterative(model_name):
    # Show "Max Iterations" slider
```

**When**: Model is iterative (Logistic Regression, SVM, Linear Regression)
**Range**: 100-10000 iterations
**Hidden for**: Tree-based and deep learning models

### Rule 4: Show Epochs ONLY for Deep Learning Models
```python
if is_deep_learning(model_name):
    # Show "Epochs" slider
```

**When**: Model is deep learning (Sequential NN, CNN, RNN)
**Range**: 1-500 epochs
**Hidden for**: All ML models

## Architecture

```
app/utils/
├── dynamic_ui.py
│   ├── ParameterValidator
│   │   ├── validate_cv_folds()
│   │   ├── validate_max_iter()
│   │   ├── validate_epochs()
│   │   └── validate_hp_iterations()
│   ├── render_training_parameters()
│   ├── validate_training_params()
│   └── display_parameter_summary()
└── __init__.py
```

## Core Functions

### `render_training_parameters(model_name, task_type)`

Renders parameters conditionally based on model type.

**Returns**: `(params: dict, enable_tuning: bool)`

```python
from app.utils.dynamic_ui import render_training_parameters

params, enable_tuning = render_training_parameters('random_forest', 'classification')
# params = {'cv_folds': 5, 'n_estimators': 100, 'max_depth': 10}
# enable_tuning = False
```

### `validate_training_params(params, model_name, task_type)`

Validates all parameters against rules.

**Returns**: `(is_valid: bool, errors: list)`

```python
from app.utils.dynamic_ui import validate_training_params

is_valid, errors = validate_training_params(params, 'random_forest', 'classification')
if not is_valid:
    for error in errors:
        st.error(error)
```

### `display_parameter_summary(params, model_name)`

Shows summary of selected parameters.

```python
from app.utils.dynamic_ui import display_parameter_summary

display_parameter_summary(params, 'random_forest')
```

## Validation Rules

### CV Folds Validation
- **Min**: 3 (configurable per model)
- **Max**: 10 (configurable per model)
- **Error**: "K-fold must be between X and Y"

### Max Iterations Validation
- **Min**: 100
- **Max**: 10000
- **Error**: "Max iterations must be between 100 and 10000"

### Epochs Validation
- **Min**: 1
- **Max**: 500
- **Error**: "Epochs must be between 1 and 500"

### HP Search Iterations Validation
- **Min**: 5
- **Max**: 100
- **Error**: "HP search iterations must be between 5 and 100"

## Usage Example

### Basic Integration

```python
import streamlit as st
from app.utils.dynamic_ui import (
    render_training_parameters,
    validate_training_params,
    display_parameter_summary
)

# Select model
model_name = st.selectbox("Model", ["random_forest", "logistic_regression", "cnn"])
task_type = st.selectbox("Task", ["Classification", "Regression"])

# Render parameters dynamically
params, enable_tuning = render_training_parameters(model_name, task_type)

# Validate
is_valid, errors = validate_training_params(params, model_name, task_type)

if not is_valid:
    for error in errors:
        st.error(error)

# Show summary
display_parameter_summary(params, model_name)

# Train
if st.button("Train", disabled=not is_valid):
    train_model(model_name, params)
```

## Parameter Flow

```
User selects model
    ↓
render_training_parameters() called
    ↓
Check model category:
    - Tree-based? → Show CV folds
    - Iterative? → Show CV folds + Max iterations
    - Deep learning? → Show Epochs
    ↓
Check tuning enabled?
    - Yes → Show HP search iterations
    - No → Hide HP search iterations
    ↓
Render model-specific parameters
    ↓
validate_training_params() called
    ↓
Check all values against rules
    ↓
Return (is_valid, errors)
    ↓
Display summary
    ↓
Enable/disable train button
```

## Parameter Visibility Matrix

| Parameter | Tree-Based | Iterative | Deep Learning |
|-----------|-----------|-----------|---------------|
| CV Folds | ✅ | ✅ | ❌ |
| Max Iterations | ❌ | ✅ | ❌ |
| Epochs | ❌ | ❌ | ✅ |
| HP Search Iterations | ✅ (if tuning) | ✅ (if tuning) | ✅ (if tuning) |
| Model-specific | ✅ | ✅ | ✅ |

## Error Prevention

### Invalid Parameter Combinations
```python
# ❌ PREVENTED: Showing epochs for Random Forest
if is_tree_based(model_name):
    # Epochs NOT shown

# ❌ PREVENTED: Showing max_iter for CNN
if is_deep_learning(model_name):
    # Max iterations NOT shown

# ❌ PREVENTED: Invalid CV fold value
if cv_value < 3 or cv_value > 10:
    st.error("K-fold must be between 3 and 10")
```

### Value Range Validation
```python
# ❌ PREVENTED: Epochs = 0
if epochs < 1:
    st.error("Epochs must be between 1 and 500")

# ❌ PREVENTED: Max iterations = 50
if max_iter < 100:
    st.error("Max iterations must be between 100 and 10000")
```

## Testing

Run the example:
```bash
streamlit run examples/dynamic_ui_example.py
```

Test scenarios:
1. Select Random Forest → CV folds shown, max_iter hidden, epochs hidden
2. Select Logistic Regression → CV folds shown, max_iter shown, epochs hidden
3. Select CNN → CV folds hidden, max_iter hidden, epochs shown
4. Enable tuning → HP search iterations shown
5. Disable tuning → HP search iterations hidden
6. Enter invalid values → Error messages displayed

## Files

- `app/utils/dynamic_ui.py` - Core logic (200 lines)
- `examples/dynamic_ui_example.py` - Integration example (80 lines)
- `DYNAMIC_UI_README.md` - This documentation

## Benefits

✅ **User-Friendly** - Only relevant parameters shown
✅ **Error Prevention** - Invalid values caught early
✅ **Consistent** - Same rules everywhere
✅ **Maintainable** - Centralized logic
✅ **Extensible** - Easy to add new models/parameters
✅ **Minimal Code** - ~200 lines total
