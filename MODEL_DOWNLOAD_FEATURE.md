# Model Download Feature - PKL Format

**Feature**: Download trained models in pickle (.pkl) format from Results page

---

## Changes Made

### 1. **app/main.py** - Results Page

#### Updated Download Section:
```python
with col2:
    st.write("#### Download Options")
    
    if 'trained_model' in st.session_state:
        import pickle
        model_bytes = pickle.dumps(st.session_state.trained_model)
        model_name = st.session_state.get('last_model_name', 'model')
        
        st.download_button(
            label="📥 Download Model (PKL)",
            data=model_bytes,
            file_name=f"{model_name}_trained.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )
    
    import json
    metrics_json = json.dumps(metrics, indent=2, default=str)
    st.download_button(
        label="📊 Download Metrics (JSON)",
        data=metrics_json,
        file_name="metrics.json",
        mime="application/json",
        use_container_width=True
    )
```

#### Store Model Name in Session:
```python
st.session_state.last_model_name = model_name
```

---

## How It Works

### 1. **Model Serialization**
- Trained model is serialized using `pickle.dumps()`
- Converts model object to bytes for download

### 2. **Download Button**
- Uses Streamlit's `st.download_button()` component
- Filename includes model type: `{model_name}_trained.pkl`
- MIME type: `application/octet-stream` (binary)

### 3. **Metrics Export**
- Metrics dictionary converted to JSON
- Filename: `metrics.json`
- MIME type: `application/json`

---

## Usage

### Download Trained Model:
1. Train a model in **Training** tab
2. Go to **Results** tab
3. Click **"📥 Download Model (PKL)"** button
4. File saves as `{model_name}_trained.pkl`

### Download Metrics:
1. After training, go to **Results** tab
2. Click **"📊 Download Metrics (JSON)"** button
3. File saves as `metrics.json`

---

## Loading Downloaded Model

### Python Code:
```python
import pickle

# Load the downloaded model
with open('logistic_regression_trained.pkl', 'rb') as f:
    model = pickle.load(f)

# Use the model for predictions
predictions = model.predict(X_new)
```

---

## Supported Models

All scikit-learn models can be downloaded:
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ SVM
- ✅ Gradient Boosting
- ✅ Linear Regression

---

## File Format

### PKL File Structure:
```
{model_name}_trained.pkl
├── Model object (serialized)
├── Hyperparameters
├── Fitted coefficients/weights
└── Preprocessing info
```

### JSON Metrics Structure:
```json
{
  "accuracy": 0.95,
  "precision": 0.93,
  "recall": 0.94,
  "f1_score": 0.935,
  "roc_auc": 0.98
}
```

---

## Testing

1. Upload Iris dataset
2. Train Logistic Regression
3. Go to Results tab
4. Click "Download Model (PKL)"
5. Verify file downloads as `logistic_regression_trained.pkl`
6. Load and test the model

