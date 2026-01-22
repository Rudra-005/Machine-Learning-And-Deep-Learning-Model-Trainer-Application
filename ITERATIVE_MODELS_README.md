# Iterative ML Models

Production-ready handling of iterative ML models with max_iter parameter.

## Overview

The iterative models system provides:
- ✅ **LogisticRegression** - Linear classifier
- ✅ **SGDClassifier** - Stochastic Gradient Descent
- ✅ **Perceptron** - Linear perceptron algorithm
- ✅ **Max Iterations** - Exposed as "Max Iterations" parameter
- ✅ **Internal mapping** - Mapped to max_iter internally
- ✅ **CV integration** - Works with k-fold cross-validation
- ✅ **Clear explanation** - Comments explain why NOT epochs

## Why Not Epochs?

### Max Iterations (Iterative ML)
```
Single pass through training data
↓
Optimization algorithm (SGD, Newton, etc.)
↓
Iterates until convergence or max_iter reached
↓
Example: max_iter=1000 means up to 1000 optimization iterations
```

### Epochs (Deep Learning)
```
Multiple passes through training data
↓
Each epoch processes all data in batches
↓
Typically 10-100+ epochs
↓
Example: epochs=50 means 50 complete passes through data
```

**Key Difference**: Iterative ML trains ONCE with convergence iterations. Deep Learning trains MULTIPLE times (epochs) on batches.

## Architecture

```
models/
├── iterative_models.py          # Core iterative model logic
└── __init__.py

app/utils/
├── iterative_streamlit.py       # Streamlit integration
└── __init__.py

examples/
└── iterative_models_example.py  # Usage examples
```

## Core Components

### 1. IterativeModelHandler Class

```python
from models.iterative_models import IterativeModelHandler

# Check if model is iterative
is_iterative = IterativeModelHandler.is_iterative_model('logistic_regression')

# Get list of iterative models
models = IterativeModelHandler.get_iterative_models()
# Returns: ['logistic_regression', 'sgd_classifier', 'perceptron']

# Create model with max_iter
model = IterativeModelHandler.create_iterative_model(
    'logistic_regression', max_iter=1000
)

# Train with CV
model, cv_scores, predictions = IterativeModelHandler.train_iterative_with_cv(
    model, X_train, y_train, X_test, y_test, 'logistic_regression', cv_folds=5
)

# Get model info
info = IterativeModelHandler.get_model_info('logistic_regression')
```

### 2. Streamlit Integration

```python
from app.utils.iterative_streamlit import render_iterative_model_config

# Render UI
max_iter, additional_params = render_iterative_model_config('logistic_regression')

# Train model
trained_model, cv_scores, predictions = train_iterative_model(
    'logistic_regression', X_train, y_train, X_test, y_test,
    max_iter, cv_folds=5
)

# Display results
display_iterative_training_results(cv_scores, metrics)
```

## Supported Models

### LogisticRegression
```python
{
    'name': 'Logistic Regression',
    'description': 'Linear model for binary/multiclass classification',
    'max_iter_range': (100, 10000),
    'default_max_iter': 100,
    'parameters': ['solver', 'C', 'max_iter']
}
```

### SGDClassifier
```python
{
    'name': 'SGD Classifier',
    'description': 'Stochastic Gradient Descent for classification',
    'max_iter_range': (100, 10000),
    'default_max_iter': 1000,
    'parameters': ['loss', 'max_iter', 'learning_rate']
}
```

### Perceptron
```python
{
    'name': 'Perceptron',
    'description': 'Linear classifier using perceptron algorithm',
    'max_iter_range': (100, 10000),
    'default_max_iter': 1000,
    'parameters': ['max_iter', 'learning_rate']
}
```

## Usage Examples

### Example 1: Basic Training

```python
from models.iterative_models import IterativeModelHandler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create model with max_iter
model = IterativeModelHandler.create_iterative_model(
    'logistic_regression', max_iter=1000
)

# Train
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Test Score: {score:.4f}")
```

### Example 2: With Cross-Validation

```python
# Train with CV
model, cv_scores, predictions = IterativeModelHandler.train_iterative_with_cv(
    model, X_train, y_train, X_test, y_test,
    'logistic_regression', cv_folds=5
)

print(f"CV Mean: {cv_scores.mean():.4f}")
print(f"CV Std: {cv_scores.std():.4f}")
```

### Example 3: Streamlit Integration

```python
import streamlit as st
from app.utils.iterative_streamlit import render_iterative_model_config

# Render UI
max_iter, additional_params = render_iterative_model_config('logistic_regression')

if st.button("Train"):
    trained_model, cv_scores, predictions = train_iterative_model(
        'logistic_regression', X_train, y_train, X_test, y_test,
        max_iter, cv_folds=5, **additional_params
    )
```

## Parameter Mapping

### User-Facing
```
"Max Iterations" slider (100-10000)
```

### Internal Mapping
```python
max_iter = st.slider("Max Iterations", 100, 10000, 1000)
model = IterativeModelHandler.create_iterative_model(
    'logistic_regression', max_iter=max_iter  # Mapped internally
)
```

## Cross-Validation Integration

### How It Works
```
1. Create iterative model with max_iter
2. Use StratifiedKFold for CV splits
3. Compute cross_val_score across folds
4. Train on full training set
5. Evaluate on test set
```

### Code
```python
model, cv_scores, predictions = IterativeModelHandler.train_iterative_with_cv(
    model, X_train, y_train, X_test, y_test,
    'logistic_regression', cv_folds=5
)
```

## Model-Specific Parameters

### LogisticRegression
- **solver**: Algorithm for optimization (lbfgs, liblinear, newton-cg, sag, saga)
- **C**: Inverse regularization strength
- **max_iter**: Convergence iterations

### SGDClassifier
- **loss**: Loss function (hinge, log, modified_huber, squared_hinge)
- **learning_rate**: Learning rate schedule
- **max_iter**: Convergence iterations

### Perceptron
- **learning_rate**: Learning rate
- **max_iter**: Convergence iterations

## Configuration

### Max Iterations Range
- **Min**: 100
- **Max**: 10000
- **Default**: 100 (LogisticRegression), 1000 (SGD, Perceptron)

### Cross-Validation
- **Min folds**: 3
- **Max folds**: 10
- **Default**: 5
- **Type**: StratifiedKFold (for classification)

## Important Notes

### Feature Scaling
Iterative models are sensitive to feature scaling. Always scale features:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Convergence
- Higher max_iter allows more convergence iterations
- May not always improve performance
- Check convergence warnings in logs

### Random State
- Set to 42 for reproducibility
- Affects initialization and shuffling

## Testing

Run examples:
```bash
streamlit run examples/iterative_models_example.py
```

Test scenarios:
1. Available iterative models
2. Max iterations vs epochs explanation
3. Basic training
4. CV integration
5. Impact of max_iter
6. Model comparison

## Files

- `models/iterative_models.py` - Core logic (100 lines)
- `app/utils/iterative_streamlit.py` - Streamlit integration (80 lines)
- `examples/iterative_models_example.py` - Usage examples (200 lines)
- `ITERATIVE_MODELS_README.md` - This documentation

## Benefits

✅ **Clear Parameter** - "Max Iterations" not "Epochs"
✅ **Proper Mapping** - Internally mapped to max_iter
✅ **CV Integration** - Works with k-fold cross-validation
✅ **Well Documented** - Comments explain design choices
✅ **Multiple Models** - LogisticRegression, SGD, Perceptron
✅ **Model-Specific** - Additional parameters per model
✅ **Minimal Code** - ~380 lines total
✅ **Production Ready** - Error handling included

## Integration Checklist

- [x] LogisticRegression supported
- [x] SGDClassifier supported
- [x] Perceptron supported
- [x] "Max Iterations" parameter exposed
- [x] Mapped to max_iter internally
- [x] Comments explain why NOT epochs
- [x] K-fold CV integration
- [x] Model-specific parameters
- [x] Feature scaling recommended
- [x] Streamlit integration complete
- [x] Examples provided
