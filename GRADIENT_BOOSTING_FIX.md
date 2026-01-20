# Gradient Boosting Model Fix

**Issue**: `gradient_boosting` model was referenced in UI but not registered in ModelFactory  
**Error**: `Invalid model_name: gradient_boosting. Available models for classification: ['logistic_regression', 'random_forest', 'svm', 'neural_network']`

---

## Changes Made

### 1. **models/model_factory.py**

#### Added imports:
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
```

#### Added default hyperparameters:
```python
'gradient_boosting': {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}
```

#### Added builder functions:
```python
def build_gradient_boosting_classifier(**hyperparams) -> GradientBoostingClassifier:
    """Build Gradient Boosting classifier."""
    logger.info(f"Building Gradient Boosting Classifier with params: {hyperparams}")
    return GradientBoostingClassifier(**hyperparams)

def build_gradient_boosting_regressor(**hyperparams) -> GradientBoostingRegressor:
    """Build Gradient Boosting regressor."""
    logger.info(f"Building Gradient Boosting Regressor with params: {hyperparams}")
    return GradientBoostingRegressor(**hyperparams)
```

#### Registered in ModelFactory:
```python
_BUILDERS = {
    'classification': {
        ...
        'gradient_boosting': build_gradient_boosting_classifier,
        ...
    },
    'regression': {
        ...
        'gradient_boosting': build_gradient_boosting_regressor,
        ...
    }
}
```

---

### 2. **app/main.py**

#### Updated model selection dropdown:
```python
# Classification
model_name = st.selectbox(
    "Algorithm",
    ["logistic_regression", "random_forest", "svm", "gradient_boosting"]
)

# Regression
model_name = st.selectbox(
    "Algorithm",
    ["linear_regression", "random_forest", "svm", "gradient_boosting"]
)
```

#### Added hyperparameter UI for gradient_boosting:
```python
elif model_name == "gradient_boosting":
    n_estimators = st.slider("Number of Estimators", 10, 500, 100)
    learning_rate = st.slider("Learning Rate", 0.001, 0.5, 0.1, step=0.01)
    max_depth = st.slider("Max Depth", 2, 20, 5)
```

#### Removed KNN (not implemented):
- Removed from classification dropdown
- Removed from regression dropdown
- Removed hyperparameter UI

---

## Available Models Now

### Classification
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ SVM
- ✅ Gradient Boosting
- ✅ Neural Network

### Regression
- ✅ Linear Regression
- ✅ Random Forest
- ✅ SVM
- ✅ Gradient Boosting
- ✅ Neural Network

---

## Testing

Try training with gradient_boosting:
1. Upload data (e.g., Iris dataset)
2. Select "Classification" task
3. Select "Gradient Boosting" algorithm
4. Adjust hyperparameters if desired
5. Click "Start Training"

Should now work without errors!

