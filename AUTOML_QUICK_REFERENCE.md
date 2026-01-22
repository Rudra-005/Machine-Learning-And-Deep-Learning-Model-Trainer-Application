# AutoML Mode: Quick Reference Guide

## For Users

### What is AutoML Mode?

AutoML Mode automatically detects your model type and applies the best training strategy. You don't need to understand CV, epochs, or convergence—the system handles it.

### How to Use

1. **Select Task Type**: Classification or Regression
2. **Select Model**: Choose from available models
3. **AutoML Detects**: Model category and optimal strategy
4. **Configure Parameters**: Only relevant controls shown
5. **Train**: Click "Start AutoML Training"
6. **View Results**: Metrics displayed appropriately

### Parameter Guide

| Parameter | When Shown | What It Does |
|-----------|-----------|--------------|
| **CV Folds** | All ML models | Number of cross-validation folds (3-10) |
| **Max Iter** | Iterative models only | Convergence iterations (100-10000) |
| **Epochs** | Deep learning only | Number of training passes (10-200) |
| **Batch Size** | Deep learning only | Samples per gradient update (16-128) |
| **Learning Rate** | Iterative & DL | Step size for optimization (0.0001-0.1) |
| **HP Tuning** | All ML models | Enable hyperparameter search (optional) |
| **Early Stopping** | Deep learning only | Stop if validation loss doesn't improve |

### Model Categories

```
Tree-Based (Random Forest, Gradient Boosting)
    ↓
    Strategy: K-Fold CV
    Shows: CV Folds, HP Tuning
    Hides: Epochs, Max Iter

Iterative (Logistic Regression, SGD)
    ↓
    Strategy: K-Fold CV + Max Iter
    Shows: CV Folds, Max Iter, HP Tuning
    Hides: Epochs

SVM (SVC, SVR)
    ↓
    Strategy: K-Fold CV
    Shows: CV Folds, HP Tuning
    Hides: Epochs, Max Iter

Deep Learning (Sequential, CNN, LSTM)
    ↓
    Strategy: Epochs + Early Stopping
    Shows: Epochs, Batch Size, Learning Rate, Early Stopping
    Hides: CV Folds, Max Iter
```

---

## For Developers

### Core Files

```
models/automl.py                    # Model detection & configuration
models/automl_trainer.py            # Training orchestration
app/utils/automl_ui.py              # Streamlit UI components
app/pages/automl_training.py        # Training page
examples/automl_examples.py         # Usage examples
```

### Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from models.automl_trainer import train_with_automl

# Create model
model = RandomForestClassifier()

# Train with AutoML
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 5, 'enable_hp_tuning': True}
)

# Access results
print(f"CV Score: {results['cv_mean']:.4f}")
print(f"Test Score: {results['test_score']:.4f}")
```

### Adding a New Model

1. **Add to MODEL_REGISTRY** in `models/automl.py`:

```python
MODEL_REGISTRY[ModelCategory.TREE_BASED].append('MyNewModel')
```

2. **Add hyperparameter distributions** in `models/automl_trainer.py`:

```python
'MyNewModel': {
    'param1': [value1, value2, value3],
    'param2': [value1, value2, value3]
}
```

3. **Test with AutoMLConfig**:

```python
from models.automl import AutoMLConfig
model = MyNewModel()
automl = AutoMLConfig(model)
print(automl.config)  # Verify configuration
```

### Key Functions

#### Model Detection

```python
from models.automl import detect_model_category, ModelCategory

category = detect_model_category(model)
# Returns: ModelCategory.TREE_BASED, ITERATIVE, SVM, or DEEP_LEARNING
```

#### Configuration

```python
from models.automl import AutoMLConfig

automl = AutoMLConfig(model)
config = automl.get_training_config()
visible = automl.get_visible_parameters()
ui_config = automl.get_ui_config()
```

#### Training

```python
from models.automl_trainer import train_with_automl

results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 5, 'enable_hp_tuning': True, 'hp_iterations': 30}
)
```

#### UI Rendering

```python
from app.utils.automl_ui import render_automl_mode, render_automl_summary

# Render parameter controls
params = render_automl_mode(model)

# Display summary
render_automl_summary(model, params)
```

### Architecture Diagram

```
User Interface (Streamlit)
    ↓
render_automl_mode()
    ↓
AutoMLConfig.get_ui_config()
    ↓
detect_model_category()
    ↓
STRATEGY_CONFIG lookup
    ↓
get_visible_parameters()
    ↓
Display only relevant controls
    ↓
User clicks "Train"
    ↓
train_with_automl()
    ↓
AutoMLTrainer.train()
    ↓
should_use_cv() or should_use_epochs()?
    ├─ CV: _train_with_cv()
    │   ├─ Optional: _tune_hyperparameters()
    │   └─ Return: cv_mean, cv_std, test_score
    └─ Epochs: _train_with_epochs()
        └─ Return: train_loss, val_loss, test_accuracy
    ↓
display_automl_results()
    ↓
Show results appropriately
```

### Testing

```python
# Test model detection
from models.automl import detect_model_category, ModelCategory
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
assert detect_model_category(model) == ModelCategory.TREE_BASED

# Test configuration
from models.automl import AutoMLConfig

automl = AutoMLConfig(model)
assert automl.config['use_epochs'] is False
assert automl.visible_params['cv_folds'] is True

# Test training
from models.automl_trainer import train_with_automl
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)

results = train_with_automl(model, X_train, y_train, X_test, y_test)
assert 'cv_mean' in results
assert 'test_score' in results
```

---

## Common Patterns

### Pattern 1: Train Multiple Models

```python
from models.automl_trainer import train_with_automl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

models = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression()
]

results = {}
for model in models:
    results[model.__class__.__name__] = train_with_automl(
        model, X_train, y_train, X_test, y_test
    )

# Compare results
for name, result in results.items():
    print(f"{name}: {result['test_score']:.4f}")
```

### Pattern 2: Hyperparameter Tuning

```python
# Without tuning
results_no_tune = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'enable_hp_tuning': False}
)

# With tuning
results_tune = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'enable_hp_tuning': True, 'hp_iterations': 50}
)

# Compare
print(f"Without tuning: {results_no_tune['test_score']:.4f}")
print(f"With tuning: {results_tune['test_score']:.4f}")
print(f"Best params: {results_tune['best_params']}")
```

### Pattern 3: Custom CV Folds

```python
# Default (5 folds)
results_default = train_with_automl(
    model, X_train, y_train, X_test, y_test
)

# Custom (10 folds for more robust evaluation)
results_custom = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 10}
)

print(f"Default CV: {results_default['cv_mean']:.4f}")
print(f"Custom CV: {results_custom['cv_mean']:.4f}")
```

---

## Troubleshooting

### Debug Model Detection

```python
from models.automl import AutoMLConfig, detect_model_category

model = MyModel()
try:
    category = detect_model_category(model)
    print(f"Category: {category}")
except ValueError as e:
    print(f"Error: {e}")
    print(f"Model name: {model.__class__.__name__}")
    print(f"Module: {model.__module__}")
```

### Debug Configuration

```python
from models.automl import AutoMLConfig

automl = AutoMLConfig(model)
print("Full config:")
print(automl.config)
print("\nVisible parameters:")
print(automl.visible_params)
print("\nUI config:")
print(automl.get_ui_config())
```

### Debug Training

```python
from models.automl_trainer import AutoMLTrainer

trainer = AutoMLTrainer(model)
print(f"Strategy: {trainer.config['strategy']}")
print(f"Use CV: {trainer.config['use_epochs'] is False}")
print(f"Use Epochs: {trainer.config['use_epochs']}")

# Train with verbose output
results = trainer.train(X_train, y_train, X_test, y_test, params)
print(f"Results: {results}")
```

---

## Performance Tips

### 1. Reduce HP Tuning Iterations

```python
# Slow (100 iterations)
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'enable_hp_tuning': True, 'hp_iterations': 100}
)

# Fast (20 iterations)
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'enable_hp_tuning': True, 'hp_iterations': 20}
)
```

### 2. Reduce CV Folds

```python
# Slow (10 folds)
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 10}
)

# Fast (3 folds)
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 3}
)
```

### 3. Disable HP Tuning for Quick Testing

```python
# Quick test
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'enable_hp_tuning': False}
)
```

---

## API Reference

### AutoMLConfig

```python
class AutoMLConfig:
    def __init__(self, model: Any)
    def get_training_params(self, user_params: Dict = None) -> Dict
    def get_ui_config(self) -> Dict
```

### AutoMLTrainer

```python
class AutoMLTrainer:
    def __init__(self, model: Any)
    def train(self, X_train, y_train, X_test, y_test, params) -> Dict
```

### Functions

```python
def detect_model_category(model) -> ModelCategory
def get_training_config(model) -> Dict
def get_visible_parameters(model) -> Dict
def should_use_cv(model) -> bool
def should_use_epochs(model) -> bool
def train_with_automl(model, X_train, y_train, X_test, y_test, params) -> Dict
```

### Streamlit Functions

```python
def render_automl_mode(model) -> Dict
def render_automl_summary(model, params) -> None
def render_automl_comparison(model) -> None
def display_automl_results(model, results) -> None
```

---

## Summary

**AutoML Mode** provides automatic model detection and intelligent strategy selection:

- ✅ Detects model category automatically
- ✅ Selects optimal training strategy
- ✅ Shows only relevant parameters
- ✅ Applies K-Fold CV, epochs, or tuning intelligently
- ✅ Provides clean, intuitive UI
- ✅ Follows ML best practices

**Result**: Users can train models effectively without understanding underlying strategies.
