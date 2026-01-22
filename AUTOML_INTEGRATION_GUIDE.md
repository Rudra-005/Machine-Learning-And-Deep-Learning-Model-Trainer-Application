# AutoML Mode Integration Guide

## How AutoML Fits Into ML/DL Trainer

### Application Architecture

```
ML/DL Trainer Application
│
├── Data Loading & Preprocessing
│   └── Preprocessed data (X_train, y_train, X_test, y_test)
│
├── Model Selection
│   ├── Manual Mode (existing)
│   │   └── User selects model, configures parameters manually
│   │
│   └── AutoML Mode (NEW)
│       ├── User selects model
│       ├── AutoML detects category
│       ├── AutoML selects strategy
│       ├── AutoML shows relevant parameters
│       └── AutoML trains with optimal approach
│
├── Training
│   ├── Manual Mode: User-configured strategy
│   └── AutoML Mode: Auto-selected strategy
│
└── Results & Evaluation
    ├── Manual Mode: Generic results display
    └── AutoML Mode: Strategy-specific results display
```

---

## Integration Points

### 1. Data Preprocessing (Existing)

```python
# app.py or data_preprocessing.py
X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_dataset(...)

# Store in session state
st.session_state.X_train = X_train
st.session_state.y_train = y_train
st.session_state.X_test = X_test
st.session_state.y_test = y_test
```

### 2. Model Selection (New AutoML Page)

```python
# app/pages/automl_training.py
def page_automl_training():
    # Step 1: Task type
    task_type = st.radio("Classification or Regression?", ['Classification', 'Regression'])
    
    # Step 2: Model selection
    model_name = st.selectbox("Choose model", list(ML_MODELS[task_type].keys()))
    model = ML_MODELS[task_type][model_name]
    
    # Step 3: AutoML configuration (auto-detected)
    automl = AutoMLConfig(model)
    params = render_automl_mode(model)
    
    # Step 4: Training
    if st.button("Start AutoML Training"):
        results = train_with_automl(
            model,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.X_test,
            st.session_state.y_test,
            params
        )
        display_automl_results(model, results)
```

### 3. Training Orchestration (New)

```python
# models/automl_trainer.py
class AutoMLTrainer:
    def train(self, X_train, y_train, X_test, y_test, params):
        if should_use_cv(self.model):
            return self._train_with_cv(...)
        elif should_use_epochs(self.model):
            return self._train_with_epochs(...)
```

### 4. Results Display (New)

```python
# app/utils/automl_ui.py
def display_automl_results(model, results):
    if results['strategy'] == 'k_fold_cv':
        # Show CV results
        st.metric("CV Score", f"{results['cv_mean']:.4f}")
        st.metric("Std Dev", f"{results['cv_std']:.4f}")
    elif results['strategy'] == 'epochs_with_early_stopping':
        # Show DL results
        st.metric("Train Loss", f"{results['train_loss']:.4f}")
        st.metric("Val Loss", f"{results['val_loss']:.4f}")
```

---

## Sidebar Navigation

### Updated Sidebar Structure

```python
# app.py or main page
st.sidebar.title("ML/DL Trainer")

page = st.sidebar.radio(
    "Select Mode",
    options=[
        "📊 Data Loading",
        "🧠 Manual Training",
        "🤖 AutoML Training",
        "📈 Results",
        "ℹ️ About"
    ]
)

if page == "📊 Data Loading":
    page_data_loading()
elif page == "🧠 Manual Training":
    page_manual_training()  # Existing
elif page == "🤖 AutoML Training":
    page_automl_training()  # New
elif page == "📈 Results":
    page_results()
elif page == "ℹ️ About":
    page_about()
```

---

## Session State Management

### Extended Session State

```python
def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        # Data
        'dataset': None,
        'data_preprocessed': False,
        'X_train': None,
        'X_val': None,
        'X_test': None,
        'y_train': None,
        'y_val': None,
        'y_test': None,
        'preprocessor': None,
        
        # Manual training
        'model': None,
        'trained_model': None,
        'training_history': None,
        'metrics': None,
        'model_trained': False,
        
        # AutoML training (NEW)
        'automl_model': None,
        'automl_trained_model': None,
        'automl_results': None,
        'automl_trained': False,
        'automl_config': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

---

## Model Registry

### Available Models in AutoML

```python
ML_MODELS = {
    'Classification': {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42)
    },
    'Regression': {
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor(),
        'Linear Regression': LinearRegression()
    }
}
```

---

## Workflow Comparison

### Manual Mode (Existing)

```
1. Upload Data
   ↓
2. Preprocess Data
   ↓
3. Select Model
   ↓
4. Configure Parameters (Manual)
   - CV Folds: 5
   - Epochs: 50
   - Max Iter: 1000
   - Batch Size: 32
   ↓
5. Train (User-selected strategy)
   ↓
6. View Results
```

### AutoML Mode (New)

```
1. Upload Data
   ↓
2. Preprocess Data
   ↓
3. Select Model
   ↓
4. AutoML Detects Category
   ↓
5. AutoML Selects Strategy
   ↓
6. AutoML Shows Relevant Parameters
   - Only CV Folds (for ML)
   - Only Epochs (for DL)
   ↓
7. Train (Auto-selected strategy)
   ↓
8. View Results (Strategy-specific)
```

---

## Code Integration Examples

### Example 1: Add AutoML Page to Main App

```python
# app.py
import streamlit as st
from app.pages.automl_training import page_automl_training

# ... existing code ...

# Sidebar navigation
page = st.sidebar.radio(
    "Select Page",
    options=[
        "Data Loading",
        "Manual Training",
        "AutoML Training",  # NEW
        "Results",
        "About"
    ]
)

if page == "Data Loading":
    page_data_loading()
elif page == "Manual Training":
    page_model_training()
elif page == "AutoML Training":
    page_automl_training()  # NEW
elif page == "Results":
    page_results()
elif page == "About":
    page_about()
```

### Example 2: Reuse Preprocessed Data

```python
# app/pages/automl_training.py
def page_automl_training():
    # Check if data is preprocessed
    if not st.session_state.get('data_preprocessed'):
        st.warning("Please preprocess data first")
        return
    
    # Use preprocessed data
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # ... rest of training code ...
```

### Example 3: Store AutoML Results

```python
# app/pages/automl_training.py
if st.button("Start AutoML Training"):
    results = train_with_automl(model, X_train, y_train, X_test, y_test, params)
    
    # Store in session state
    st.session_state.automl_model = model
    st.session_state.automl_trained_model = results.get('best_estimator', model)
    st.session_state.automl_results = results
    st.session_state.automl_trained = True
    st.session_state.automl_config = automl.get_ui_config()
    
    # Display results
    display_automl_results(model, results)
```

---

## File Structure

### Updated Project Structure

```
ML_DL_Trainer/
├── app/
│   ├── main.py                          # Entry point
│   ├── config.py                        # Configuration
│   ├── pages/
│   │   ├── eda_page.py                 # EDA visualization
│   │   ├── manual_training.py          # Manual training (existing)
│   │   └── automl_training.py          # AutoML training (NEW)
│   └── utils/
│       ├── error_handler.py
│       ├── file_handler.py
│       ├── logger.py
│       ├── validators.py
│       ├── automl_ui.py                # AutoML UI (NEW)
│       └── dynamic_ui.py
│
├── core/
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── target_analyzer.py
│   └── validator.py
│
├── models/
│   ├── model_factory.py
│   ├── automl.py                       # AutoML detection (NEW)
│   ├── automl_trainer.py               # AutoML training (NEW)
│   ├── ml/
│   │   ├── classifier.py
│   │   └── regressor.py
│   └── dl/
│       ├── cnn_models.py
│       └── rnn_models.py
│
├── evaluation/
│   ├── metrics.py
│   ├── visualizer.py
│   ├── cross_validator.py
│   └── reporter.py
│
├── storage/
│   ├── model_repository.py
│   ├── result_repository.py
│   └── cache_manager.py
│
├── data/
│   ├── uploads/
│   ├── preprocessed/
│   ├── models/
│   └── results/
│
├── examples/
│   └── automl_examples.py               # AutoML examples (NEW)
│
├── tests/
│   ├── test_automl.py                  # AutoML tests (NEW)
│   └── ...
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── TRAINING_STRATEGY.md                # Training strategy docs
├── AUTOML_DOCUMENTATION.md             # AutoML docs (NEW)
├── AUTOML_QUICK_REFERENCE.md           # Quick reference (NEW)
└── AUTOML_IMPLEMENTATION_SUMMARY.md    # Implementation summary (NEW)
```

---

## Dependencies

### New Dependencies (if needed)

```python
# requirements.txt additions
scikit-learn>=1.0.0  # For RandomizedSearchCV
tensorflow>=2.10.0   # For deep learning (optional)
streamlit>=1.28.0    # Already required
```

### Existing Dependencies Used

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
import streamlit as st
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_automl.py
import pytest
from models.automl import detect_model_category, AutoMLConfig, ModelCategory
from models.automl_trainer import train_with_automl
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def test_detect_tree_based():
    model = RandomForestClassifier()
    assert detect_model_category(model) == ModelCategory.TREE_BASED

def test_automl_config():
    model = RandomForestClassifier()
    automl = AutoMLConfig(model)
    assert automl.config['use_epochs'] is False
    assert automl.visible_params['cv_folds'] is True

def test_train_with_automl():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2
    )
    
    model = RandomForestClassifier()
    results = train_with_automl(model, X_train, y_train, X_test, y_test)
    
    assert 'cv_mean' in results
    assert 'test_score' in results
    assert results['strategy'] == 'k_fold_cv'
```

### Integration Tests

```python
# tests/test_automl_integration.py
def test_automl_streamlit_integration():
    """Test AutoML integration with Streamlit."""
    from app.utils.automl_ui import render_automl_mode
    
    model = RandomForestClassifier()
    params = render_automl_mode(model)
    
    assert 'cv_folds' in params
    assert 'enable_hp_tuning' in params
```

---

## Deployment Considerations

### Docker Integration

```dockerfile
# Dockerfile (updated)
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/main.py"]
```

### Environment Variables

```bash
# .env
AUTOML_ENABLED=true
AUTOML_DEFAULT_CV_FOLDS=5
AUTOML_DEFAULT_HP_ITERATIONS=30
AUTOML_DEFAULT_EPOCHS=50
```

---

## Performance Optimization

### Caching

```python
# app/pages/automl_training.py
@st.cache_resource
def get_model_registry():
    """Cache model registry."""
    return ML_MODELS

@st.cache_data
def get_automl_config(model_name):
    """Cache AutoML configuration."""
    model = get_model_registry()[model_name]
    return AutoMLConfig(model).get_ui_config()
```

### Parallel Processing

```python
# models/automl_trainer.py
searcher = RandomizedSearchCV(
    self.model,
    param_dist,
    n_iter=hp_iterations,
    cv=cv,
    n_jobs=-1  # Use all available cores
)
```

---

## Monitoring & Logging

### Training Logs

```python
# app/pages/automl_training.py
import logging

logger = logging.getLogger(__name__)

def page_automl_training():
    logger.info(f"Starting AutoML training with {model.__class__.__name__}")
    
    try:
        results = train_with_automl(...)
        logger.info(f"Training completed. CV Score: {results['cv_mean']:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        st.error(f"Training failed: {str(e)}")
```

---

## Summary

### Integration Checklist

- ✅ Create `models/automl.py` - Model detection & configuration
- ✅ Create `models/automl_trainer.py` - Training orchestration
- ✅ Create `app/utils/automl_ui.py` - Streamlit UI components
- ✅ Create `app/pages/automl_training.py` - Training page
- ✅ Create `examples/automl_examples.py` - Usage examples
- ✅ Create documentation files
- ✅ Update session state management
- ✅ Add to sidebar navigation
- ✅ Create unit tests
- ✅ Update requirements.txt
- ✅ Update Docker configuration
- ✅ Add logging & monitoring

### Key Benefits

✅ **Automatic model detection** - No manual categorization  
✅ **Intelligent strategy selection** - Right approach for each model  
✅ **Clean UI** - Only relevant parameters shown  
✅ **Seamless integration** - Works with existing data pipeline  
✅ **Production ready** - Comprehensive testing & documentation  

### Result

AutoML Mode is fully integrated into ML/DL Trainer, providing users with an intelligent, automatic training experience while maintaining compatibility with the existing manual training mode.
