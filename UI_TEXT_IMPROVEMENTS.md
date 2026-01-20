# UI Text & Wording Improvements

Specific text suggestions for production-ready UX.

---

## 1. Training Page - Opening Section

### Current
```
st.title("⚙️ Model Training")

if 'data' not in st.session_state:
    st.warning("⚠️ Please upload data first in 'Data Upload' tab")
```

### Improved
```
st.title("⚙️ Model Training")

if 'data' not in st.session_state:
    st.warning(
        "⚠️ **No data loaded**\n\n"
        "Please upload data first in the **Data Upload** tab, "
        "or use one of the sample datasets."
    )
    st.info(
        "💡 **Tip**: After uploading, explore your data in the "
        "**EDA / Data Understanding** tab to understand it better."
    )
    return
```

---

## 2. Configuration Section - Add Context

### Current
```
st.write("### Configuration")
task_type = st.selectbox("Task Type", ["Classification", "Regression"])
```

### Improved
```
st.write("### Configuration")
st.caption("Choose your task type and target variable")

col1, col2 = st.columns([1, 2])
with col1:
    task_type = st.selectbox(
        "Task Type",
        ["Classification", "Regression"],
        help="Classification: Predict categories | Regression: Predict numbers"
    )
with col2:
    st.write("")  # Spacing
    if task_type == "Classification":
        st.caption("📊 Predicting categories (e.g., Yes/No, A/B/C)")
    else:
        st.caption("📈 Predicting continuous values (e.g., price, temperature)")
```

---

## 3. Target Column Selection - Enhanced Validation

### Current
```
target_col = st.selectbox("Target Column", data.columns)
target_data = data[target_col]
unique_count = target_data.nunique()

if task_type == "Classification" and unique_count < 2:
    st.error("❌ Target column must have at least 2 unique values for classification!")
```

### Improved
```
target_col = st.selectbox(
    "Target Column",
    data.columns,
    help="The column you want to predict"
)
target_data = data[target_col]
unique_count = target_data.nunique()
missing_count = target_data.isna().sum()

# Show target info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Unique Values", unique_count)
with col2:
    st.metric("Missing Values", missing_count)
with col3:
    st.metric("Data Type", str(target_data.dtype))

# Validation with actionable errors
if missing_count > 0:
    st.error(
        "❌ **Target column has missing values**\n\n"
        "**Why this matters:** Model can't learn from incomplete data\n\n"
        "**How to fix:**\n"
        "1. Go to **Data Upload** tab\n"
        "2. Remove rows with missing target values\n"
        "3. Or select a different target column"
    )
elif task_type == "Classification" and unique_count < 2:
    st.error(
        "❌ **Classification needs at least 2 categories**\n\n"
        "**Why this matters:** Can't classify if there's only one option\n\n"
        "**How to fix:**\n"
        "1. Switch to **Regression** if target is continuous\n"
        "2. Or select a different target column with multiple categories"
    )
elif task_type == "Classification" and unique_count > 50:
    st.warning(
        f"⚠️ **Many categories detected** ({unique_count} unique values)\n\n"
        "**Suggestion:** This might work better as a regression problem. "
        "Consider switching task type."
    )
else:
    st.success("✅ Target column looks good!")
```

---

## 4. Model Selection - Add Decision Helper

### Current
```
if model_type == "Machine Learning":
    if task_type == "Classification":
        model_name = st.selectbox(
            "Algorithm",
            ["logistic_regression", "random_forest", "svm", "knn", "gradient_boosting"]
        )
```

### Improved
```
if model_type == "Machine Learning":
    st.write("### Model Selection")
    
    # Add decision helper
    with st.expander("🤔 Not sure which model to choose?", expanded=False):
        st.markdown("""
        **Quick Decision Guide:**
        
        | Situation | Best Choice | Why |
        |-----------|-------------|-----|
        | **First time** | Logistic Regression | Simple, fast, interpretable |
        | **Want best accuracy** | Random Forest | Handles most data well |
        | **Imbalanced data** | Gradient Boosting | Handles class imbalance |
        | **Need speed** | Logistic Regression | Trains in seconds |
        | **Complex patterns** | Random Forest | Captures non-linear relationships |
        | **High-dimensional data** | SVM | Works well with many features |
        | **Unsure?** | Random Forest | Safe default choice |
        
        **Pro Tip:** You can train multiple models and compare results!
        """)
    
    if task_type == "Classification":
        model_name = st.selectbox(
            "Algorithm",
            ["logistic_regression", "random_forest", "svm", "knn", "gradient_boosting"],
            help="Choose based on your data characteristics and speed requirements"
        )
```

---

## 5. Hyperparameters - Clear Defaults

### Current
```
st.write("### Hyperparameters")
if model_type == "Machine Learning":
    if model_name == "random_forest":
        n_estimators = st.slider("Number of Trees", 10, 500, 100)
        max_depth = st.slider("Max Depth", 2, 20, 10)
```

### Improved
```
st.write("### Hyperparameters")
st.caption("💡 Default values are optimized for most datasets")

if model_type == "Machine Learning":
    if model_name == "random_forest":
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider(
                "Number of Trees",
                10, 500, 100,
                help=(
                    "**Default: 100**\n\n"
                    "More trees = better accuracy but slower training\n\n"
                    "Typical range: 50-500"
                )
            )
        
        with col2:
            max_depth = st.slider(
                "Max Depth",
                2, 20, 10,
                help=(
                    "**Default: 10**\n\n"
                    "Deeper trees = more complex patterns\n"
                    "Too deep = overfitting\n\n"
                    "Typical range: 5-20"
                )
            )
        
        st.info(
            "ℹ️ **Note:** These parameters won't change automatically. "
            "Your choices are final for this training run."
        )
```

---

## 6. Advanced Options - Opt-In Design

### Current
```
# All options shown equally
```

### Improved
```
st.write("### Hyperparameters")

# Basic options (always visible)
st.write("**Basic Settings** (Recommended for most users)")
col1, col2 = st.columns(2)
with col1:
    n_estimators = st.slider("Number of Trees", 10, 500, 100)
with col2:
    max_depth = st.slider("Max Depth", 2, 20, 10)

# Advanced options (opt-in)
with st.expander("⚙️ Advanced Options (Optional)", expanded=False):
    st.caption("💡 Only adjust if you understand what these do")
    
    col1, col2 = st.columns(2)
    with col1:
        min_samples_split = st.slider(
            "Min Samples to Split",
            2, 20, 5,
            help="Minimum samples required to split a node. Higher = simpler trees"
        )
    with col2:
        min_samples_leaf = st.slider(
            "Min Samples per Leaf",
            1, 10, 2,
            help="Minimum samples at leaf node. Higher = prevents overfitting"
        )
    
    st.warning(
        "⚠️ **Advanced settings can significantly impact model performance**\n\n"
        "Only modify if you have experience with these parameters."
    )
```

---

## 7. Pre-Training Checklist

### Current
```
# No checklist
```

### Improved
```
st.divider()

# Pre-training checklist
if not train_disabled:
    with st.expander("✅ Pre-Training Checklist", expanded=False):
        st.markdown("""
        Before training, make sure:
        
        - ✓ Data is uploaded and loaded
        - ✓ Target column is selected
        - ✓ Target column has no missing values
        - ✓ Task type matches your data (Classification vs Regression)
        - ✓ You've reviewed data in EDA tab
        
        **Ready?** Click the button below to start training!
        """)
```

---

## 8. Training Button - Clear Call-to-Action

### Current
```
if st.button("🚀 Start Training", key="train_btn", use_container_width=True, disabled=train_disabled):
```

### Improved
```
st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    if st.button(
        "🚀 Start Training",
        key="train_btn",
        use_container_width=True,
        disabled=train_disabled,
        type="primary"
    ):
        # Training code...
with col2:
    if train_disabled:
        st.caption("⚠️ Fix errors above to train")
    else:
        st.caption("✅ Ready to train")
```

---

## 9. Success Messages - Celebratory

### Current
```
st.success("✅ Training completed!")
```

### Improved
```
st.success(
    "🎉 **Training completed successfully!**\n\n"
    "Your model is ready to use. "
    "Check the **Results** tab to see performance metrics and visualizations."
)
st.balloons()

# Add next steps
st.info(
    "**What's next?**\n\n"
    "1. Go to **Results** tab to see metrics\n"
    "2. Download your trained model\n"
    "3. Try different hyperparameters to improve accuracy"
)
```

---

## 10. Error Messages - Actionable

### Current
```
except Exception as e:
    st.error(f"❌ Error during training: {str(e)}")
```

### Improved
```
except ValueError as e:
    if "The least populated class" in str(e):
        st.error(
            "❌ **Class imbalance too severe**\n\n"
            "**What happened:** One class has too few samples for stratified split\n\n"
            "**Solutions:**\n"
            "1. Collect more data for the minority class\n"
            "2. Use class weights (check Advanced Options)\n"
            "3. Try a different model (Gradient Boosting handles imbalance better)\n"
            "4. Use a different train-test split strategy"
        )
    else:
        st.error(
            f"❌ **Data validation error**\n\n"
            f"**Issue:** {str(e)}\n\n"
            "**How to fix:**\n"
            "1. Check that all features are numeric or properly encoded\n"
            "2. Ensure target column has valid values\n"
            "3. Remove ID columns from training"
        )

except Exception as e:
    st.error(
        "❌ **Unexpected error during training**\n\n"
        "**Please check:**\n"
        "1. All features are numeric or properly encoded\n"
        "2. Target column has valid values\n"
        "3. No ID columns are included\n"
        "4. Dataset has enough samples (at least 10 rows)\n\n"
        "**Still having issues?** Check the logs for more details."
    )
    logger.error(f"Training error: {str(e)}")
```

---

## 11. EDA Page - Add Training Recommendations

### Current
```
# No recommendations
```

### Improved
```
st.divider()
st.subheader("📋 Recommendations for Training")

recommendations = []
warnings = []

# Check for missing values
if data.isnull().sum().sum() > 0:
    warnings.append(
        "❌ **Missing values detected**\n"
        "Handle these before training for best results"
    )

# Check for class imbalance
if task_type == 'classification':
    class_analysis = analyze_classification(target_data)
    if class_analysis['imbalance_ratio'] > 1.5:
        recommendations.append(
            "⚠️ **Class imbalance detected**\n"
            f"Ratio: {class_analysis['imbalance_ratio']:.1f}:1\n"
            "Use class weights in Training tab"
        )

# Check for outliers
if task_type == 'regression':
    recommendations.append(
        "💡 **Tip:** Consider feature scaling for better results"
    )

if warnings or recommendations:
    if warnings:
        st.warning("\n\n".join(warnings))
    if recommendations:
        st.info("\n\n".join(recommendations))
    
    st.success(
        "**Next Step:** Go to the **Training** tab and apply these recommendations "
        "when configuring your model."
    )
else:
    st.success("✅ Data looks great! Ready for training.")
```

---

## 12. Results Page - Clear Presentation

### Current
```
st.write("### Model Performance Metrics")
```

### Improved
```
st.write("### 📊 Model Performance Metrics")

if 'metrics' not in st.session_state:
    st.info(
        "ℹ️ **No results yet**\n\n"
        "Train a model first to see results here.\n\n"
        "**Steps:**\n"
        "1. Go to **Data Upload** and load data\n"
        "2. Go to **Training** and click **Start Training**\n"
        "3. Results will appear here automatically"
    )
else:
    st.success("✅ Model trained successfully!")
    
    # Display metrics
    metrics = st.session_state.metrics
    
    col1, col2, col3, col4 = st.columns(4)
    # ... display metrics ...
    
    st.divider()
    
    st.write("### 💾 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Model", use_container_width=True):
            st.success("✅ Model ready for download")
    with col2:
        if st.button("📊 Export Metrics", use_container_width=True):
            st.success("✅ Metrics exported")
```

---

## Summary of Wording Improvements

| Section | Current | Improved |
|---------|---------|----------|
| Errors | Generic | Actionable with solutions |
| Defaults | Hidden | Visible with explanations |
| Options | All equal | Basic vs Advanced |
| Success | Plain | Celebratory with next steps |
| Guidance | None | Decision helpers included |
| Tooltips | Minimal | Comprehensive |
| Warnings | Vague | Specific with impact |

---

## Key Principles Applied

1. **Clarity**: Every message explains what, why, and how
2. **Actionability**: Errors include solutions
3. **Hierarchy**: Basic options first, advanced opt-in
4. **Feedback**: Success messages are celebratory
5. **Guidance**: Decision helpers for uncertain users
6. **Safety**: No auto-modifications, explicit statements
7. **Polish**: Emoji, formatting, professional tone

