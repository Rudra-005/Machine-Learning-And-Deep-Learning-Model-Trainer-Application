# Model Configuration System

Centralized, reusable model categorization and parameter management for ML/DL Trainer.

## Overview

The model configuration system provides:
- **Model Categorization**: Tree-based, Iterative, Deep Learning
- **Parameter Definitions**: Type, range, defaults for each model
- **Training Strategies**: K-fold CV for ML, Epochs for DL
- **UI Integration**: Dynamic parameter rendering
- **Backend Integration**: Consistent model handling

## Architecture

```
models/
├── model_config.py          # Core configuration & queries
├── training_utils.py        # Backend training logic
└── __init__.py

app/utils/
├── model_ui.py              # Streamlit UI rendering
└── __init__.py

examples/
└── model_config_usage.py    # Usage examples
```

## Model Categories

### 1. Tree-Based ML
- **Strategy**: K-fold cross-validation
- **Models**: Random Forest, Gradient Boosting
- **Parameters**: n_estimators, max_depth, min_samples_split

### 2. Iterative ML
- **Strategy**: K-fold cross-validation
- **Models**: Logistic Regression, Linear Regression, SVM
- **Parameters**: C, max_iter, kernel

### 3. Deep Learning
- **Strategy**: Epochs
- **Models**: Sequential NN, CNN, RNN
- **Parameters**: epochs, batch_size, learning_rate

## Core Functions

### Query Functions

```python
from models.model_config import *

# Get model category
category = get_model_category('random_forest')  # 'tree_based'

# Get full model config
config = get_model_config('random_forest')

# Get models for a task
models = get_models_by_task('classification')

# Get training strategy
strategy = get_category_strategy('random_forest')  # 'k-fold_cv'

# Get CV configuration
cv_config = get_cv_folds_config('random_forest')  # {'min': 3, 'max': 10, 'default': 5}

# Type checking
is_tree_based('random_forest')      # True
is_iterative('logistic_regression') # True
is_deep_learning('cnn')             # True

# Get model parameters
params = get_model_params('random_forest')
```

### UI Functions

```python
from app.utils.model_ui import *

# Render hyperparameters dynamically
hyperparams = render_hyperparameters('random_forest')
# Returns: {'cv_folds': 5, 'n_estimators': 100, 'max_depth': 10, ...}

# Get strategy info
info = get_strategy_info('random_forest')
# Returns: {'type': 'cross_validation', 'description': '...', 'uses_epochs': False}
```

### Backend Functions

```python
from models.training_utils import *

# Train with appropriate strategy
model, cv_scores = train_model_with_strategy(
    model, X_train, y_train, 'random_forest', 'classification'
)

# Get training info
info = get_training_info('random_forest')
# Returns: {'model_name': '...', 'strategy': '...', 'is_tree_based': True, ...}

# Apply hyperparameters
model = apply_hyperparams(model, {'n_estimators': 200, 'max_depth': 15})
```

## Usage Examples

### Example 1: Get Classification Models

```python
from models.model_config import get_models_by_task

models = get_models_by_task('classification')
for model_name, config in models.items():
    print(f"{config['name']} ({model_name})")
```

### Example 2: Dynamic UI Rendering

```python
import streamlit as st
from app.utils.model_ui import render_hyperparameters

model_name = st.selectbox("Select Model", ['random_forest', 'logistic_regression', 'cnn'])
hyperparams = render_hyperparameters(model_name)
```

### Example 3: Backend Training

```python
from sklearn.ensemble import RandomForestClassifier
from models.training_utils import train_model_with_strategy

model = RandomForestClassifier()
trained_model, cv_scores = train_model_with_strategy(
    model, X_train, y_train, 'random_forest', 'classification'
)

if cv_scores is not None:
    print(f"CV Score: {cv_scores.mean():.4f}")
```

### Example 4: Type Checking

```python
from models.model_config import is_tree_based, is_iterative, is_deep_learning

if is_tree_based(model_name):
    print("Using tree-based model with k-fold CV")
elif is_iterative(model_name):
    print("Using iterative model with k-fold CV")
elif is_deep_learning(model_name):
    print("Using deep learning model with epochs")
```

## Configuration Structure

```python
MODEL_CONFIG = {
    "tree_based": {
        "category": "Tree-Based ML",
        "strategy": "k-fold_cv",
        "models": {
            "random_forest": {
                "name": "Random Forest",
                "task_types": ["classification", "regression"],
                "params": {
                    "n_estimators": {
                        "type": "slider",
                        "min": 10,
                        "max": 500,
                        "default": 100,
                        "label": "Trees"
                    },
                    ...
                },
                "cv_folds": {"min": 3, "max": 10, "default": 5}
            },
            ...
        }
    },
    ...
}
```

## Adding New Models

1. Add to `MODEL_CONFIG` in `models/model_config.py`:

```python
"new_model": {
    "name": "New Model Name",
    "task_types": ["classification"],
    "params": {
        "param1": {
            "type": "slider",
            "min": 0,
            "max": 100,
            "default": 50,
            "label": "Parameter 1"
        }
    },
    "cv_folds": {"min": 3, "max": 10, "default": 5}
}
```

2. Update `ModelFactory` to create the model
3. UI and backend automatically support it

## Benefits

✅ **Single Source of Truth**: All model configs in one place
✅ **Reusable**: Works across UI and backend
✅ **Extensible**: Easy to add new models
✅ **Type-Safe**: Clear parameter definitions
✅ **Consistent**: Same behavior everywhere
✅ **Maintainable**: Changes propagate automatically

## Testing

Run examples:
```bash
python examples/model_config_usage.py
```

## Files

- `models/model_config.py` - Core configuration (150 lines)
- `models/training_utils.py` - Backend utilities (50 lines)
- `app/utils/model_ui.py` - UI utilities (80 lines)
- `examples/model_config_usage.py` - Usage examples (100 lines)
