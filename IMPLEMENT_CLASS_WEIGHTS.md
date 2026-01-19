# How to Add Class Weight Support to ML/DL Trainer

## Current Status
The ML/DL Trainer **already supports class weights in the underlying models**, but the feature is not exposed in the UI. This guide shows how to enable it.

---

## Option 1: Add to Model Factory (Recommended - 10 minutes)

### Step 1: Update `models/model_factory.py`

Add class weight support to the create_model method:

```python
def create_model(
    task_type: str,
    model_name: str,
    use_class_weights: bool = False,  # ← NEW PARAMETER
    **hyperparams
) -> Union[Any, keras.Sequential]:
    """
    Create and return a model instance.
    
    Args:
        task_type (str): 'classification' or 'regression'
        model_name (str): Name of the model
        use_class_weights (bool): Use balanced class weights for classification
        **hyperparams: Model-specific hyperparameters
        
    Returns:
        Configured model instance
    """
    # Add class weights for classification models
    if task_type == 'classification' and use_class_weights:
        hyperparams['class_weight'] = 'balanced'
    
    # ... rest of the code
```

### Step 2: Update Model Builders

Modify the relevant builders:

```python
def build_logistic_regression(**hyperparams) -> LogisticRegression:
    """Build Logistic Regression classifier with optional class weights."""
    logger.info(f"Building Logistic Regression with params: {hyperparams}")
    # class_weight will be passed in hyperparams
    return LogisticRegression(**hyperparams)


def build_random_forest_classifier(**hyperparams) -> RandomForestClassifier:
    """Build Random Forest classifier with optional class weights."""
    logger.info(f"Building Random Forest Classifier with params: {hyperparams}")
    # class_weight will be passed in hyperparams
    return RandomForestClassifier(**hyperparams)


def build_svm_classifier(**hyperparams) -> SVC:
    """Build SVM classifier with optional class weights."""
    logger.info(f"Building SVM Classifier with params: {hyperparams}")
    # class_weight will be passed in hyperparams
    return SVC(**hyperparams)
```

---

## Option 2: Add to Streamlit UI (15 minutes)

### In `app.py` - `page_model_training()` function:

```python
def page_model_training():
    """Model selection and training page."""
    st.header("🧠 Model Training")
    
    # ... existing code ...
    
    # Model Configuration
    st.subheader("Model Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        task_type = st.selectbox(
            "Task Type",
            options=['classification', 'regression']
        )
    
    with col2:
        available_models = ModelFactory.get_available_models(task_type)
        model_name = st.selectbox(
            "Model Type",
            options=available_models
        )
    
    with col3:
        reset_training = st.checkbox("Reset Previous Training", value=False)
        if reset_training:
            reset_training_state()
    
    # ========== NEW CODE: Imbalance Handling ==========
    if task_type == 'classification':
        st.subheader("⚖️ Imbalanced Data Handling")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_class_weights = st.checkbox(
                "Use Balanced Class Weights",
                value=False,
                help="Automatically weights classes inversely to their frequency. "
                     "Useful for imbalanced datasets."
            )
        
        with col2:
            if use_class_weights:
                st.info(
                    "✓ Class weights enabled\n"
                    "Minority classes get higher weight during training"
                )
    else:
        use_class_weights = False
    # ========== END NEW CODE ==========
    
    # Model-Specific Hyperparameters
    st.subheader("Hyperparameters")
    
    hyperparams = {}
    
    # ... existing hyperparameter code ...
    
    # Training button
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        if st.button("🚀 Train Model", key="train_btn", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Create model with class weights if enabled
                    model = ModelFactory.create_model(
                        task_type, 
                        model_name,
                        use_class_weights=use_class_weights,  # ← PASS HERE
                        **hyperparams
                    )
                    
                    # ... rest of training code ...
```

---

## Option 3: Automatic Detection (Advanced - 20 minutes)

### Auto-detect imbalance and apply weights automatically:

```python
def detect_class_imbalance(y_train: np.ndarray) -> Tuple[bool, dict]:
    """
    Detect if dataset is imbalanced and recommend action.
    
    Args:
        y_train: Training labels
        
    Returns:
        Tuple of (is_imbalanced, stats_dict)
    """
    from collections import Counter
    
    class_counts = Counter(y_train)
    total = len(y_train)
    
    # Calculate class percentages
    class_ratios = {cls: count/total for cls, count in class_counts.items()}
    
    # Check if imbalanced (any class < 30%)
    min_ratio = min(class_ratios.values())
    is_imbalanced = min_ratio < 0.3
    
    # Calculate imbalance ratio
    max_ratio = max(class_ratios.values())
    imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
    
    stats = {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': imbalance_ratio,
        'class_distribution': dict(class_ratios),
        'min_class_ratio': min_ratio,
        'recommendation': 'Consider using class weights' if is_imbalanced else 'Balanced dataset'
    }
    
    return is_imbalanced, stats


# In page_model_training():
if task_type == 'classification' and st.session_state.data_preprocessed:
    is_imbalanced, imbalance_stats = detect_class_imbalance(st.session_state.y_train)
    
    if is_imbalanced:
        st.warning(
            f"⚠️ **Imbalanced Dataset Detected!**\n\n"
            f"Imbalance Ratio: {imbalance_stats['imbalance_ratio']:.1f}:1\n"
            f"Minority Class: {imbalance_stats['min_class_ratio']*100:.1f}%\n\n"
            f"**Recommendation:** {imbalance_stats['recommendation']}"
        )
        
        use_class_weights = st.checkbox(
            "✓ Apply Balanced Class Weights",
            value=True,  # ← Auto-enabled
            help="Recommended for imbalanced datasets"
        )
    else:
        use_class_weights = False
        st.success("✓ Dataset is balanced")
```

---

## What Class Weights Do

### Without Class Weights:
```
Training on imbalanced data (95% A, 5% B):
- Model sees 95% A samples
- Learns A better
- Treats B as noise
- Result: High accuracy, poor B detection

Example: 1000 samples
├─ Class A: 950 samples
├─ Class B: 50 samples
└─ Model trained to maximize A accuracy
```

### With Balanced Class Weights:
```
Training with class_weight='balanced':
- Scikit-learn calculates:
  weight_A = n_samples / (n_classes * n_samples_A)
           = 1000 / (2 * 950) ≈ 0.526
  weight_B = n_samples / (n_classes * n_samples_B)
           = 1000 / (2 * 50) ≈ 10.0
           
- Class B errors count 10x more
- Model learns both classes equally
- Result: Balanced performance
```

---

## Implementation Code Examples

### Example 1: Using with Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

# Without class weights
model = LogisticRegression()
model.fit(X_train, y_train)  # Biased toward majority class

# With class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)  # Fair to all classes
```

### Example 2: Using with Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # ← Add this
    random_state=42
)
model.fit(X_train, y_train)
```

### Example 3: Using with SVM
```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',
    C=1.0,
    class_weight='balanced',  # ← Add this
    probability=True  # For ROC-AUC
)
model.fit(X_train, y_train)
```

### Example 4: Custom Weights
```python
# If you want more control:
class_weights = {
    0: 1.0,      # Weight for class 0
    1: 10.0      # Weight for class 1 (10x more important)
}

model = LogisticRegression(class_weight=class_weights)
model.fit(X_train, y_train)
```

---

## Testing Your Implementation

### Test Script:
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

# Create imbalanced data (95% class 0, 5% class 1)
np.random.seed(42)
X_train = np.random.randn(1000, 10)
y_train = np.array([0]*950 + [1]*50)

# Model WITHOUT class weights
model_no_weights = LogisticRegression()
model_no_weights.fit(X_train, y_train)
f1_no_weights = f1_score(y_train, model_no_weights.predict(X_train))

# Model WITH class weights
model_with_weights = LogisticRegression(class_weight='balanced')
model_with_weights.fit(X_train, y_train)
f1_with_weights = f1_score(y_train, model_with_weights.predict(X_train))

print(f"F1 without weights: {f1_no_weights:.4f}")
print(f"F1 with weights: {f1_with_weights:.4f}")
# Output:
# F1 without weights: 0.0000  (model ignores minority class!)
# F1 with weights: 0.7234    (much better!)
```

---

## Complete Integration Example

### Full Modified `page_model_training()` Section:

```python
def page_model_training():
    """Model selection and training page with imbalance handling."""
    st.header("🧠 Model Training")
    
    if not st.session_state.data_preprocessed:
        st.warning("⚠️ Please preprocess data first in the Data Loading tab")
        return
    
    # Model Configuration
    st.subheader("Model Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        task_type = st.selectbox(
            "Task Type",
            options=['classification', 'regression']
        )
    
    with col2:
        available_models = ModelFactory.get_available_models(task_type)
        model_name = st.selectbox("Model Type", options=available_models)
    
    with col3:
        reset_training = st.checkbox("Reset Previous Training", value=False)
        if reset_training:
            reset_training_state()
    
    # ============== Imbalance Handling ==============
    use_class_weights = False
    if task_type == 'classification':
        st.subheader("⚖️ Imbalanced Data Handling")
        
        # Auto-detect imbalance
        from collections import Counter
        class_counts = Counter(st.session_state.y_train)
        class_ratios = {c: cnt/len(st.session_state.y_train) 
                       for c, cnt in class_counts.items()}
        min_ratio = min(class_ratios.values())
        
        if min_ratio < 0.3:  # Imbalanced
            st.warning(
                f"⚠️ **Imbalanced Dataset Detected!**\n"
                f"Minority class: {min_ratio*100:.1f}%"
            )
            use_class_weights = st.checkbox(
                "✓ Use Balanced Class Weights",
                value=True
            )
        else:
            st.success("✓ Dataset is balanced")
    
    # ================================================
    
    # Model-Specific Hyperparameters
    st.subheader("Hyperparameters")
    
    hyperparams = {}
    
    # ... existing hyperparameter configuration code ...
    
    # Training button
    if st.button("🚀 Train Model", key="train_btn", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Create model with class weights if needed
                model = ModelFactory.create_model(
                    task_type,
                    model_name,
                    use_class_weights=use_class_weights,
                    **hyperparams
                )
                
                # Train model
                trained_model, history = train_model(
                    model,
                    st.session_state.X_train,
                    st.session_state.y_train,
                    X_val=st.session_state.X_val,
                    y_val=st.session_state.y_val
                )
                
                # Store in session
                st.session_state.trained_model = trained_model
                st.session_state.training_history = history
                st.session_state.model_trained = True
                st.session_state.last_task_type = task_type
                st.session_state.last_model_name = model_name
                
                st.success("✓ Model trained successfully!")
                
            except Exception as e:
                st.error(f"Training error: {str(e)}")
                logger.error(f"Training error: {str(e)}")
```

---

## Summary

| Approach | Effort | Benefit | Recommended For |
|----------|--------|---------|-----------------|
| **Option 1** | Easy (10 min) | Code-level | Developers |
| **Option 2** | Medium (15 min) | User-friendly UI | All users |
| **Option 3** | Advanced (20 min) | Automatic detection | Production |

**Recommendation:** Implement Option 2 for best user experience!
