# AutoML Mode: Automatic Model Detection & Strategy Selection

## Overview

AutoML Mode automatically detects model types and applies the optimal training strategy without user configuration. Users select a model, and the system intelligently applies K-Fold CV, epochs, or hyperparameter tuning as appropriate.

---

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: Model Detection (automl.py)                    │
│ - Detect model category (tree, iterative, SVM, DL)      │
│ - Map to training strategy                              │
│ - Determine visible parameters                          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│ Layer 2: Training Orchestration (automl_trainer.py)     │
│ - Execute K-Fold CV for ML models                       │
│ - Execute epochs for DL models                          │
│ - Apply hyperparameter tuning if enabled                │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│ Layer 3: UI Rendering (automl_ui.py)                    │
│ - Show only relevant parameters                         │
│ - Display strategy explanation                          │
│ - Render results appropriately                          │
└─────────────────────────────────────────────────────────┘
```

---

## Model Categories & Strategies

### 1. Tree-Based Models

**Models**: Random Forest, Gradient Boosting, Decision Trees, Extra Trees

**Strategy**: K-Fold Cross-Validation (single pass)

**Why**:
- Converge in a single pass through data
- Epochs would redundantly retrain the same model
- K-Fold CV provides robust overfitting detection
- All data used for training (no wasted validation set)

**Visible Parameters**:
- ✓ CV Folds (3-10, default 5)
- ✓ HP Tuning (optional)
- ✗ Epochs
- ✗ Max Iter
- ✗ Batch Size

**Example**:
```python
model = RandomForestClassifier()
automl = AutoMLConfig(model)
# Returns: K-Fold CV with 5 folds, optional HP tuning
```

### 2. Iterative Models

**Models**: Logistic Regression, SGD, Perceptron, Ridge, Lasso

**Strategy**: K-Fold CV with Convergence Control

**Why**:
- Need convergence control (max_iter parameter)
- K-Fold CV for robust evaluation
- max_iter prevents infinite training loops
- More efficient than epochs for this model class

**Visible Parameters**:
- ✓ CV Folds (3-10, default 5)
- ✓ Max Iter (100-10000, default 1000)
- ✓ Learning Rate (optional)
- ✓ HP Tuning (optional)
- ✗ Epochs
- ✗ Batch Size

**Example**:
```python
model = LogisticRegression()
automl = AutoMLConfig(model)
# Returns: K-Fold CV with max_iter convergence control
```

### 3. SVM Models

**Models**: SVC, SVR, LinearSVC, LinearSVR

**Strategy**: K-Fold CV with Kernel Tuning

**Why**:
- Kernel selection critical for performance
- K-Fold CV validates kernel choices
- Hyperparameter tuning essential
- Single-pass convergence

**Visible Parameters**:
- ✓ CV Folds (3-10, default 5)
- ✓ HP Tuning (optional, recommended)
- ✗ Epochs
- ✗ Max Iter
- ✗ Batch Size

**Example**:
```python
model = SVC()
automl = AutoMLConfig(model)
# Returns: K-Fold CV with kernel tuning
```

### 4. Deep Learning Models

**Models**: Sequential, CNN, LSTM, RNN, Functional

**Strategy**: Epochs with Early Stopping

**Why**:
- Require multiple passes through data in batches
- Epochs track training progress
- Early stopping prevents overfitting automatically
- K-Fold CV too expensive for deep learning

**Visible Parameters**:
- ✓ Epochs (10-200, default 50)
- ✓ Batch Size (16-128, default 32)
- ✓ Learning Rate (0.0001-0.1, default 0.001)
- ✓ Early Stopping (default True)
- ✗ CV Folds
- ✗ Max Iter

**Example**:
```python
model = Sequential([Dense(64), Dense(32), Dense(1)])
automl = AutoMLConfig(model)
# Returns: Epochs with early stopping
```

---

## Core Components

### 1. Model Detection (`models/automl.py`)

```python
def detect_model_category(model) -> ModelCategory:
    """Auto-detect model category from instance or class name."""
    # Checks MODEL_REGISTRY for model name
    # Falls back to module inspection (keras, sklearn)
    # Raises ValueError if unknown
```

**Supported Detection**:
- By class name (e.g., "RandomForestClassifier")
- By module (e.g., "tensorflow.keras")
- By registry lookup

### 2. Configuration Management

```python
class AutoMLConfig:
    """Manages AutoML configuration for a model."""
    
    def __init__(self, model):
        self.config = get_training_config(model)
        self.visible_params = get_visible_parameters(model)
    
    def get_training_params(self, user_params=None):
        """Merge defaults with user input."""
    
    def get_ui_config(self):
        """Get configuration for UI rendering."""
```

### 3. Training Orchestration (`models/automl_trainer.py`)

```python
class AutoMLTrainer:
    """Executes optimal training strategy."""
    
    def train(self, X_train, y_train, X_test, y_test, params):
        """Route to appropriate training method."""
        if should_use_cv(model):
            return self._train_with_cv(...)
        elif should_use_epochs(model):
            return self._train_with_epochs(...)
```

**Training Methods**:
- `_train_with_cv()`: K-Fold cross-validation
- `_train_with_epochs()`: Epochs with early stopping
- `_tune_hyperparameters()`: RandomizedSearchCV

### 4. UI Rendering (`app/utils/automl_ui.py`)

```python
def render_automl_mode(model) -> Dict[str, Any]:
    """Render only relevant parameters."""
    # Shows CV folds for ML models
    # Shows epochs for DL models
    # Shows max_iter for iterative models
    # Hides irrelevant parameters
```

---

## Usage Examples

### Example 1: Tree-Based Classification

```python
from sklearn.ensemble import RandomForestClassifier
from models.automl_trainer import train_with_automl

# Create model
model = RandomForestClassifier(random_state=42)

# Train with AutoML
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 5, 'enable_hp_tuning': True, 'hp_iterations': 30}
)

# Results
print(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
print(f"Test Score: {results['test_score']:.4f}")
print(f"Best Params: {results.get('best_params', {})}")
```

**Output**:
```
CV Score: 0.9533 ± 0.0245
Test Score: 0.9667
Best Params: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 2}
```

### Example 2: Iterative Classification

```python
from sklearn.linear_model import LogisticRegression
from models.automl_trainer import train_with_automl

# Create model
model = LogisticRegression(random_state=42)

# Train with AutoML
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 5, 'max_iter': 1000, 'enable_hp_tuning': False}
)

# Results
print(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
print(f"Test Score: {results['test_score']:.4f}")
```

**Output**:
```
CV Score: 0.9200 ± 0.0356
Test Score: 0.9333
```

### Example 3: Deep Learning

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from models.automl_trainer import train_with_automl

# Create model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with AutoML
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'epochs': 50, 'batch_size': 32, 'early_stopping': True}
)

# Results
print(f"Train Loss: {results['train_loss']:.4f}")
print(f"Val Loss: {results['val_loss']:.4f}")
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

**Output**:
```
Train Loss: 0.2145
Val Loss: 0.2389
Test Accuracy: 0.9200
```

---

## Streamlit Integration

### AutoML Training Page

```python
# app/pages/automl_training.py

def page_automl_training():
    """Complete AutoML workflow."""
    
    # Step 1: Task type
    task_type = st.radio("Classification or Regression?", ['Classification', 'Regression'])
    
    # Step 2: Model selection
    model_name = st.selectbox("Choose model", list(ML_MODELS[task_type].keys()))
    model = ML_MODELS[task_type][model_name]
    
    # Step 3: AutoML configuration (auto-detected)
    render_automl_summary(model, {})
    params = render_automl_mode(model)
    
    # Step 4: Training
    if st.button("Start AutoML Training"):
        results = train_with_automl(model, X_train, y_train, X_test, y_test, params)
        display_automl_results(model, results)
```

### UI Components

**render_automl_mode(model)**
- Shows only relevant parameters
- Hides irrelevant controls
- Provides helpful tooltips

**render_automl_summary(model, params)**
- Displays model type
- Shows selected strategy
- Explains why this strategy

**render_automl_comparison(model)**
- Compares strategies
- Explains trade-offs
- Shows alternatives

---

## Parameter Visibility Logic

### Decision Tree

```
Model Selected
    ↓
Detect Category
    ├─ Tree-Based? → Show: CV, HP Tuning | Hide: Epochs, Max Iter
    ├─ Iterative? → Show: CV, Max Iter, HP Tuning | Hide: Epochs
    ├─ SVM? → Show: CV, HP Tuning | Hide: Epochs, Max Iter
    └─ Deep Learning? → Show: Epochs, Batch Size, LR | Hide: CV, Max Iter
```

### Validation

```python
def validate_parameters(model, params):
    """Ensure parameters match model strategy."""
    config = get_training_config(model)
    
    # Check epochs
    if params.get('epochs') and not config['use_epochs']:
        return False, "Epochs not applicable for this model"
    
    # Check max_iter
    if params.get('max_iter') and not config['use_max_iter']:
        return False, "max_iter not applicable for this model"
    
    # Check CV
    if params.get('cv_folds') and not should_use_cv(model):
        return False, "CV not applicable for this model"
    
    return True, ""
```

---

## Hyperparameter Tuning

### Automatic Tuning Configuration

Each model has predefined hyperparameter distributions:

```python
PARAM_DISTRIBUTIONS = {
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    },
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
}
```

### Tuning Process

```
1. User enables HP tuning
2. User specifies iterations (5-100)
3. RandomizedSearchCV samples random combinations
4. Each combination evaluated with K-Fold CV
5. Best combination selected
6. Model retrained with best parameters
```

---

## Results Format

### ML Models (K-Fold CV)

```python
{
    'cv_mean': 0.9533,           # Mean CV score
    'cv_std': 0.0245,            # Standard deviation
    'cv_scores': [0.95, 0.96, ...],  # Individual fold scores
    'test_score': 0.9667,        # Test set score
    'strategy': 'k_fold_cv',
    'hp_tuning_enabled': False,
    'best_params': {...}         # If tuning enabled
}
```

### Deep Learning (Epochs)

```python
{
    'train_loss': 0.2145,        # Final training loss
    'val_loss': 0.2389,          # Final validation loss
    'test_accuracy': 0.9200,     # Test accuracy
    'history': {...},            # Training history
    'strategy': 'epochs_with_early_stopping',
    'hp_tuning_enabled': False
}
```

---

## Design Decisions

### 1. Why Model Registry?

**Decision**: Use dictionary-based model registry instead of class inspection

**Rationale**:
- Explicit and maintainable
- Easy to add new models
- Clear categorization
- No magic or reflection

### 2. Why AutoMLConfig Class?

**Decision**: Encapsulate configuration in class instead of functions

**Rationale**:
- Stateful configuration management
- Easy to extend with new methods
- Clear separation of concerns
- Testable and mockable

### 3. Why Separate UI Layer?

**Decision**: Separate UI rendering from training logic

**Rationale**:
- Reusable training logic
- Easy to add new UI frameworks
- Testable without Streamlit
- Clear responsibility boundaries

### 4. Why RandomizedSearchCV?

**Decision**: Use RandomizedSearchCV instead of GridSearchCV

**Rationale**:
- More efficient for large parameter spaces
- Faster convergence
- Better for production use
- Scales better with more parameters

---

## Best Practices

### 1. Always Use AutoML for New Models

```python
# ✓ Good: Let AutoML decide
model = RandomForestClassifier()
results = train_with_automl(model, X_train, y_train, X_test, y_test)

# ✗ Avoid: Manual strategy selection
model = RandomForestClassifier()
results = train_with_epochs(model, X_train, y_train, epochs=50)  # Wrong!
```

### 2. Enable HP Tuning for Important Models

```python
# ✓ Good: Tune hyperparameters
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'enable_hp_tuning': True, 'hp_iterations': 50}
)

# ✗ Avoid: Skip tuning
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'enable_hp_tuning': False}
)
```

### 3. Respect CV Folds for Robust Evaluation

```python
# ✓ Good: Use sufficient folds
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 5}  # At least 5
)

# ✗ Avoid: Too few folds
results = train_with_automl(
    model, X_train, y_train, X_test, y_test,
    params={'cv_folds': 2}  # Too few
)
```

---

## Troubleshooting

### Issue: "Unknown model type"

**Cause**: Model not in registry

**Solution**: Add to MODEL_REGISTRY in automl.py

```python
MODEL_REGISTRY[ModelCategory.TREE_BASED].append('MyCustomModel')
```

### Issue: Wrong parameters shown

**Cause**: Model category misdetected

**Solution**: Check detect_model_category() logic

```python
# Debug
automl = AutoMLConfig(model)
print(automl.config['category'])  # Check detected category
```

### Issue: Training too slow

**Cause**: Too many HP tuning iterations

**Solution**: Reduce iterations

```python
params = {'enable_hp_tuning': True, 'hp_iterations': 10}  # Reduce from 50
```

---

## Future Enhancements

- [ ] Ensemble AutoML (combine multiple models)
- [ ] Neural Architecture Search (NAS)
- [ ] Feature engineering automation
- [ ] Model explainability (SHAP, LIME)
- [ ] Automated feature selection
- [ ] Imbalanced data handling
- [ ] Multi-objective optimization

---

## Summary

AutoML Mode provides:

✅ **Automatic model detection** - No manual categorization  
✅ **Intelligent strategy selection** - Right approach for each model  
✅ **Clean UI** - Only relevant parameters shown  
✅ **Robust evaluation** - K-Fold CV for ML, epochs for DL  
✅ **Optional tuning** - Hyperparameter optimization available  
✅ **Production ready** - Follows ML best practices  

**Result**: Users can train models effectively without understanding the underlying strategies.
