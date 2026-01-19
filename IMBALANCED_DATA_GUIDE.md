# Imbalanced Dataset Handling in ML/DL Trainer

## Summary
Yes, the ML/DL Trainer application **does handle imbalanced datasets** through several built-in mechanisms.

---

## How Imbalanced Datasets Are Handled

### 1. **Stratified Data Splitting** ✓
**Implementation:** `data_preprocessing.py` (lines 305-360)

The most critical feature for imbalanced data is **stratified train-test-validation splitting**.

```python
# Automatic stratification in preprocessing
stratify_y = y if stratify and len(y.unique()) < 20 else None

# Ensures all splits maintain class distribution
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=test_size + val_size,
    random_state=42,
    stratify=stratify_y  # ← Maintains class balance across splits
)
```

**Benefits:**
- Maintains class distribution in train/val/test splits
- Prevents biased evaluation metrics
- Ensures each fold has proportional class representation
- Critical for imbalanced datasets where random splits would create imbalanced subsets

**Example:**
```
Original dataset: 95% class 0, 5% class 1
With stratified split:
  - Train: 95% class 0, 5% class 1
  - Val: 95% class 0, 5% class 1
  - Test: 95% class 0, 5% class 1

Without stratification (random split):
  - Train: 98% class 0, 2% class 1  ← Biased!
  - Val: 92% class 0, 8% class 1
  - Test: 94% class 0, 6% class 1
```

### 2. **Appropriate Evaluation Metrics** ✓
**Implementation:** `evaluate.py` (lines 50-150)

The application computes metrics that are sensitive to class imbalance:

#### Metrics Computed for Classification:
- **Accuracy**: Shows overall performance (can be misleading for imbalanced data)
- **Precision**: How many positive predictions were correct
- **Recall (Sensitivity)**: How many actual positives were detected
- **F1-Score**: Harmonic mean of precision and recall (balances both)
- **ROC-AUC**: Threshold-independent metric (excellent for imbalanced data)
- **Confusion Matrix**: Shows true positives, false positives, etc.

#### Weighted Averaging:
```python
metrics['precision'] = float(precision_score(
    y_true, y_pred, 
    average='weighted',  # ← Weights by class support
    zero_division=0
))
```

**Why This Matters:**
- **Accuracy alone misleads:** A model predicting all samples as majority class can have 95% accuracy
- **ROC-AUC is better:** Evaluates model across all thresholds regardless of class distribution
- **F1-Score helps:** Balances precision and recall for both classes
- **Weighted averaging:** Accounts for class imbalance in metrics

### 3. **Class-Aware Model Training** (Ready to Implement)

While not currently automatic, the infrastructure supports class weights:

```python
# Available in scikit-learn models:
LogisticRegression(class_weight='balanced')  # ← Supported
RandomForestClassifier(class_weight='balanced')  # ← Supported
SVC(class_weight='balanced')  # ← Supported

# Formula for balanced weights:
# weight_class_i = n_samples / (n_classes * n_samples_of_class_i)
```

### 4. **Visual Inspection Tools** ✓

The app provides visualizations to detect imbalance:
- **Confusion Matrix:** Shows per-class performance
- **Per-class metrics in logs:** Breakdown of precision/recall per class
- **ROC curves:** Visual representation of trade-offs

---

## Current Capabilities vs Limitations

| Feature | Status | Details |
|---------|--------|---------|
| Stratified Splitting | ✅ Implemented | Automatic for classification tasks |
| Class Imbalance Metrics | ✅ Implemented | ROC-AUC, F1, weighted precision/recall |
| Class Weights (sklearn) | 🟡 Ready | Not exposed in UI, but supported by models |
| SMOTE/Oversampling | ❌ Not Implemented | Would require `imbalanced-learn` library |
| Cost-Sensitive Learning | 🟡 Ready | Models support `class_weight` parameter |
| Focal Loss (Deep Learning) | ❌ Not Implemented | Would require custom Keras loss |
| Threshold Optimization | ❌ Not Implemented | Could be added for custom thresholds |

---

## How to Handle Imbalanced Data - Best Practices

### Step 1: Upload Your Imbalanced Dataset
Simply upload your CSV file in the **Data Loading** tab. The app automatically handles it.

### Step 2: Use Stratified Evaluation
The app automatically uses stratified splitting. Your train/val/test sets will maintain class distribution.

### Step 3: Focus on Right Metrics
When evaluating results in the **Evaluation** tab, prioritize:
1. **ROC-AUC** - Best metric for imbalanced data
2. **F1-Score** - Balanced metric
3. **Per-class Precision & Recall** - See which class is predicted better
4. **Confusion Matrix** - Visualize true/false positives/negatives

### Step 4: Model Selection
Choose models appropriately:
- **Random Forest**: Less affected by imbalance than linear models
- **SVM with class_weight**: Can handle imbalance well
- **Logistic Regression**: Can handle imbalance with proper weighting

---

## Example Workflow for Imbalanced Data

### Scenario: Fraud Detection Dataset (99% legitimate, 1% fraud)

**Step 1:** Upload fraud dataset
```
Total samples: 10,000
- Legitimate: 9,900 (99%)
- Fraud: 100 (1%)
```

**Step 2:** Preprocess (Data Loading tab)
```
✓ Auto-stratified split applied:
- Train: 6,930 legitimate, 70 fraud
- Val: 990 legitimate, 10 fraud
- Test: 1,980 legitimate, 20 fraud
```

**Step 3:** Train Model (Model Training tab)
```
Select: Random Forest Classifier
- Training set maintains class distribution
- Model learns both classes proportionally
```

**Step 4:** Evaluate (Evaluation tab)
```
Results shown:
- Accuracy: 99.4%  ← Misleading! (predicting all as legitimate would be 99%)
- ROC-AUC: 0.92    ← Great! Shows real performance
- F1-Score: 0.68   ← Good balance of precision/recall
- Recall (Fraud): 0.85 ← Catches 85% of frauds
```

---

## Recommended Enhancements (Future)

To further improve imbalanced dataset handling, consider these additions:

### 1. **Add SMOTE Oversampling**
```python
# Would require: pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

### 2. **Class Weight Options in UI**
```python
# Add to app.py Model Training tab:
use_class_weight = st.checkbox("Use Balanced Class Weights")
if use_class_weight:
    hyperparams['class_weight'] = 'balanced'
```

### 3. **Custom Decision Threshold**
```python
# Optimize threshold based on use case
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_pred_proba > 0.3).astype(int)  # Lower threshold for fraud
```

### 4. **Cost-Sensitive Learning**
```python
# Assign higher cost to minority class errors
weights = {0: 1, 1: 99}  # Fraud is 99x more expensive to miss
```

---

## Code References

### Stratified Splitting Location
**File:** `data_preprocessing.py`
**Lines:** 305-360
**Function:** `split_data()`

### Metrics Computation
**File:** `evaluate.py`
**Lines:** 50-150
**Function:** `compute_classification_metrics()`

### Model Factory (Class Weight Ready)
**File:** `models/model_factory.py`
**Lines:** 90-150
**Functions:** `build_logistic_regression()`, `build_random_forest_classifier()`, `build_svm_classifier()`

---

## Summary

| Aspect | Current | How It Works |
|--------|---------|-------------|
| **Imbalanced Data Support** | ✅ YES | Stratified splitting + imbalance-aware metrics |
| **Automatic Handling** | ✅ YES | Applied automatically for classification |
| **User Configuration** | 🟡 Partial | Happens automatically, could be exposed |
| **Metrics for Imbalance** | ✅ YES | ROC-AUC, F1, weighted precision/recall |
| **Advanced Methods** | ❌ NO | SMOTE, focal loss not implemented |
| **Production Ready** | ✅ YES | Works well for typical imbalance scenarios |

**Bottom Line:** The application **handles imbalanced datasets well** through stratified splitting and appropriate metrics. For datasets with extreme imbalance (>99% single class), consider implementing SMOTE oversampling or class weighting as enhancements.
