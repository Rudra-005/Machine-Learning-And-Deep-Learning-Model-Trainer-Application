# Hyperparameter Optimization

Production-ready hyperparameter optimization using RandomizedSearchCV.

## Overview

The HP optimization system provides:
- ✅ **RandomizedSearchCV** - Efficient parameter search
- ✅ **User-defined n_iter** - Configurable search iterations (5-100)
- ✅ **K-fold CV reuse** - Leverages existing CV infrastructure
- ✅ **Best estimator** - Automatically selects best model
- ✅ **Results display** - Best params, best score, top combinations
- ✅ **Optional feature** - Advanced settings, not forced
- ✅ **No epochs** - Never labeled as epochs

## Architecture

```
evaluation/
├── hp_optimizer.py              # Core HP optimization
└── __init__.py

app/utils/
├── hp_streamlit.py              # Streamlit integration
└── __init__.py

examples/
└── hp_optimization_example.py   # Usage examples
```

## Core Components

### 1. HyperparameterOptimizer Class

```python
from evaluation.hp_optimizer import HyperparameterOptimizer

# Get parameter distribution
param_dist = HyperparameterOptimizer.get_param_distribution('random_forest')

# Optimize hyperparameters
best_model, search_results = HyperparameterOptimizer.optimize(
    model, X_train, y_train, 'random_forest', 'classification',
    n_iter=20, cv=5
)

# Display results
HyperparameterOptimizer.display_optimization_results(search_results)
```

### 2. Streamlit Integration

```python
from app.utils.hp_streamlit import render_hp_optimization_config, train_with_optional_tuning

# Render UI
enable_tuning, n_iter = render_hp_optimization_config(model_name)

# Train with optional tuning
model, search_results, predictions = train_with_optional_tuning(
    model, X_train, y_train, X_test, y_test,
    model_name, task_type, cv_folds,
    enable_tuning, n_iter
)
```

## Parameter Distributions

### Random Forest
```python
{
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```

### Gradient Boosting
```python
{
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}
```

### Logistic Regression
```python
{
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'max_iter': [100, 500, 1000],
    'solver': ['lbfgs', 'liblinear']
}
```

### SVM
```python
{
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
```

## Usage Examples

### Example 1: Basic HP Optimization

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from evaluation.hp_optimizer import HyperparameterOptimizer

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)

best_model, search_results = HyperparameterOptimizer.optimize(
    model, X_train, y_train, 'random_forest', 'classification',
    n_iter=20, cv=5
)

print(f"Best Score: {search_results['best_score']:.4f}")
print(f"Best Params: {search_results['best_params']}")
```

### Example 2: Streamlit Integration

```python
import streamlit as st
from app.utils.hp_streamlit import render_hp_optimization_config, train_with_optional_tuning

# Render UI
enable_tuning, n_iter = render_hp_optimization_config('random_forest')

if st.button("Train"):
    model, search_results, predictions = train_with_optional_tuning(
        model, X_train, y_train, X_test, y_test,
        'random_forest', 'classification', cv_folds=5,
        enable_tuning=enable_tuning, n_iter=n_iter
    )
    
    if search_results:
        st.success("✅ HP optimization complete!")
```

### Example 3: Optional Feature

```python
# User can enable/disable HP optimization
enable_tuning = st.checkbox("Enable Hyperparameter Optimization", value=False)

if enable_tuning:
    n_iter = st.slider("Search Iterations", 5, 100, 20)
    # Run optimization
else:
    # Standard training
```

## Output Format

### Search Results Dictionary

```python
search_results = {
    'best_params': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt'
    },
    'best_score': 0.9667,
    'n_iter': 20,
    'cv_folds': 5,
    'scoring': 'accuracy',
    'cv_results': {...}
}
```

### Display Output

```
Hyperparameter Optimization Results
├── Best Score: 0.9667
├── Iterations: 20
├── CV Folds: 5
├── Best Parameters:
│   ├── n_estimators: 200
│   ├── max_depth: 15
│   ├── min_samples_split: 5
│   ├── min_samples_leaf: 2
│   └── max_features: sqrt
└── Top 5 Parameter Combinations:
    ├── 1. Score: 0.9667, Params: {...}
    ├── 2. Score: 0.9633, Params: {...}
    └── ...
```

## Configuration

### Search Iterations (n_iter)
- **Min**: 5
- **Max**: 100
- **Default**: 20
- **Recommendation**: 20-50 for good balance

### Cross-Validation Folds
- **Min**: 3
- **Max**: 10
- **Default**: 5
- **Reused from CV config**

### Scoring Metrics
- **Classification**: accuracy
- **Regression**: r2

## Model Support

### Supported Models
```
✅ Random Forest
✅ Gradient Boosting
✅ Logistic Regression
✅ SVM
✅ Linear Regression
```

### Not Supported
```
❌ Deep Learning (Sequential, CNN, RNN)
```

## Key Features

### 1. Efficient Search
- RandomizedSearchCV tests random combinations
- Faster than GridSearchCV
- Parallelized with n_jobs=-1

### 2. K-Fold CV Reuse
- Uses same CV splitter as standard training
- Stratified for classification
- Consistent random state (42)

### 3. Best Estimator Selection
- Automatically selects best model
- Trained on full training set
- Ready for predictions

### 4. Results Display
- Best parameters shown
- Best score displayed
- Top 5 combinations listed
- Confidence metrics included

### 5. Optional Feature
- Checkbox to enable/disable
- In advanced settings
- No forced usage
- Never labeled as epochs

## Workflow

```
User selects model
    ↓
Standard training UI shown
    ↓
Advanced settings expanded
    ↓
"Enable Hyperparameter Optimization" checkbox
    ↓
If enabled:
    - Show n_iter slider (5-100)
    - Run RandomizedSearchCV
    - Display best params and score
    - Use best estimator for predictions
    ↓
If disabled:
    - Standard training
    - No HP optimization
```

## Performance Considerations

### Time Complexity
- **n_iter=5**: ~30 seconds
- **n_iter=20**: ~2 minutes
- **n_iter=50**: ~5 minutes
- **n_iter=100**: ~10 minutes

(Approximate, depends on dataset size and model)

### Memory Usage
- Minimal overhead
- Parallel jobs use more memory
- Manageable for typical datasets

## Testing

Run examples:
```bash
streamlit run examples/hp_optimization_example.py
```

Test scenarios:
1. Basic HP optimization
2. Parameter distributions
3. Full training pipeline
4. Default vs optimized comparison
5. Impact of n_iter

## Files

- `evaluation/hp_optimizer.py` - Core logic (120 lines)
- `app/utils/hp_streamlit.py` - Streamlit integration (60 lines)
- `examples/hp_optimization_example.py` - Usage examples (200 lines)
- `HP_OPTIMIZATION_README.md` - This documentation

## Benefits

✅ **Better Models** - Finds optimal hyperparameters
✅ **Efficient** - RandomizedSearchCV is fast
✅ **Reusable** - Leverages k-fold CV
✅ **Optional** - Advanced feature, not forced
✅ **Clear Results** - Best params and scores displayed
✅ **No Epochs** - Never confused with DL training
✅ **Minimal Code** - ~380 lines total
✅ **Production Ready** - Error handling included

## Integration Checklist

- [x] RandomizedSearchCV used
- [x] User-defined n_iter (5-100, default 20)
- [x] K-fold CV reused
- [x] Best estimator selected
- [x] Best params displayed
- [x] Best score displayed
- [x] Optional feature (advanced settings)
- [x] Not labeled as epochs
- [x] Skipped for deep learning
- [x] Streamlit integration complete
- [x] Examples provided
