# K-Fold Cross-Validation Implementation

Production-ready k-fold cross-validation for traditional ML models using sklearn.

## Overview

The k-fold CV system provides:
- ✅ **sklearn cross_val_score** - Robust CV implementation
- ✅ **User-defined k** - Default k=5, configurable 3-10
- ✅ **Mean & Std** - Automatic computation
- ✅ **Clear display** - Fold scores, CI, metrics
- ✅ **DL skip** - No CV for deep learning models
- ✅ **Stratified** - For classification tasks

## Architecture

```
evaluation/
├── kfold_validator.py       # Core CV logic
└── __init__.py

app/utils/
├── cv_streamlit.py          # Streamlit integration
└── __init__.py

examples/
└── kfold_cv_example.py      # Usage examples
```

## Core Components

### 1. KFoldCrossValidator Class

```python
from evaluation.kfold_validator import KFoldCrossValidator

# Get CV splitter
cv_splitter = KFoldCrossValidator.get_cv_splitter('classification', k=5)

# Compute CV scores
cv_scores, mean, std = KFoldCrossValidator.compute_cv_scores(
    model, X, y, k=5, task_type='classification'
)

# Display results
KFoldCrossValidator.display_cv_results(cv_scores, mean, std, 'classification')
```

### 2. Training Pipeline

```python
from evaluation.kfold_validator import train_ml_with_cv

model, cv_results, predictions = train_ml_with_cv(
    model, X_train, y_train, X_test, y_test,
    k=5, task_type='classification', model_name='random_forest'
)
```

### 3. Streamlit Integration

```python
from app.utils.cv_streamlit import render_cv_config, train_with_cv_pipeline

# Render UI
k, enable_cv = render_cv_config(model_name)

# Train with CV
model, cv_results, predictions, metrics = train_with_cv_pipeline(
    model, X_train, y_train, X_test, y_test, k, task_type, model_name
)

# Display results
display_training_results(cv_results, metrics, task_type)
```

## Usage Examples

### Example 1: Basic K-Fold CV

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from evaluation.kfold_validator import KFoldCrossValidator

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100)

# Compute CV scores
cv_scores, mean_score, std_score = KFoldCrossValidator.compute_cv_scores(
    model, X_train, y_train, k=5, task_type='classification'
)

print(f"Mean: {mean_score:.4f}, Std: {std_score:.4f}")
# Output: Mean: 0.9667, Std: 0.0211
```

### Example 2: Full Training Pipeline

```python
from evaluation.kfold_validator import train_ml_with_cv

model, cv_results, predictions = train_ml_with_cv(
    model, X_train, y_train, X_test, y_test,
    k=5, task_type='classification', model_name='random_forest'
)

print(f"CV Mean: {cv_results['mean_score']:.4f}")
print(f"CV Std: {cv_results['std_score']:.4f}")
```

### Example 3: Streamlit Integration

```python
import streamlit as st
from app.utils.cv_streamlit import render_cv_config, train_with_cv_pipeline

# Render CV config UI
k, enable_cv = render_cv_config('random_forest')

if st.button("Train"):
    model, cv_results, predictions, metrics = train_with_cv_pipeline(
        model, X_train, y_train, X_test, y_test, k, 'classification', 'random_forest'
    )
```

## Key Features

### 1. Automatic CV Splitter Selection

```python
# Classification → StratifiedKFold
cv_splitter = KFoldCrossValidator.get_cv_splitter('classification', k=5)

# Regression → KFold
cv_splitter = KFoldCrossValidator.get_cv_splitter('regression', k=5)
```

### 2. Scoring Metric Selection

```python
# Classification → 'accuracy'
# Regression → 'r2'
```

### 3. Deep Learning Skip

```python
if is_deep_learning(model_name):
    # Skip CV, use epochs instead
    model.fit(X_train, y_train)
else:
    # Use k-fold CV
    cv_scores, mean, std = compute_cv_scores(...)
```

### 4. Confidence Interval Calculation

```python
ci_lower = mean_score - 1.96 * std_score
ci_upper = mean_score + 1.96 * std_score
# 95% confidence interval
```

## Output Format

### CV Results Dictionary

```python
cv_results = {
    'cv_scores': array([0.96, 0.97, 0.95, 0.98, 0.96]),
    'mean_score': 0.9640,
    'std_score': 0.0089,
    'k_folds': 5,
    'task_type': 'classification'
}
```

### Display Output

```
Cross-Validation Results
├── Mean CV Score: 0.9640
├── Std Dev: 0.0089
├── Folds: 5
├── Fold Scores:
│   ├── Fold 1: 0.9600
│   ├── Fold 2: 0.9700
│   ├── Fold 3: 0.9500
│   ├── Fold 4: 0.9800
│   └── Fold 5: 0.9600
└── 95% CI: [0.9466, 0.9814]
```

## Configuration

### Default Settings

```python
{
    'n_splits': 5,           # Default k
    'shuffle': True,         # Shuffle before split
    'random_state': 42,      # Reproducibility
    'stratified': True       # For classification
}
```

### User-Configurable

```python
k = st.slider("K-Fold Splits", min_value=3, max_value=10, value=5)
```

## Model Support

### ML Models (with CV)

```
✅ Random Forest
✅ Gradient Boosting
✅ Logistic Regression
✅ Linear Regression
✅ SVM
```

### Deep Learning Models (no CV)

```
❌ Sequential NN
❌ CNN
❌ RNN
```

## Validation Rules

### K-Fold Range
- **Min**: 3
- **Max**: 10
- **Default**: 5

### Scoring Metrics
- **Classification**: accuracy
- **Regression**: r2

## Testing

Run examples:
```bash
streamlit run examples/kfold_cv_example.py
```

Test scenarios:
1. Basic CV with k=5
2. Different k values (3, 5, 10)
3. Classification vs Regression
4. ML vs Deep Learning
5. Confidence interval calculation

## Files

- `evaluation/kfold_validator.py` - Core logic (100 lines)
- `app/utils/cv_streamlit.py` - Streamlit integration (80 lines)
- `examples/kfold_cv_example.py` - Usage examples (150 lines)
- `KFOLD_CV_README.md` - This documentation

## Benefits

✅ **Robust Evaluation** - Multiple fold assessment
✅ **Reproducible** - Fixed random state
✅ **Stratified** - Balanced class distribution
✅ **Clear Results** - Mean, std, CI displayed
✅ **DL Compatible** - Skipped for deep learning
✅ **Minimal Code** - ~280 lines total
✅ **Production Ready** - Error handling included

## Integration Checklist

- [x] sklearn cross_val_score used
- [x] User-defined k (3-10, default 5)
- [x] Mean and std computed
- [x] Results displayed clearly
- [x] No epochs in ML pipeline
- [x] DL models skip CV
- [x] Stratified for classification
- [x] Confidence interval calculated
- [x] Streamlit integration complete
- [x] Examples provided
