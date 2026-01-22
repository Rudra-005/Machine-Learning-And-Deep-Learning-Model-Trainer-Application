# Debug Validation Block: Quick Reference

## 🚀 Quick Start

### Enable Debug Mode
1. Open app in Streamlit
2. Look for **🐛 Debug Mode** in sidebar
3. Click checkbox to enable
4. Go to **3️⃣ Training** page
5. Select target column
6. Click **🐛 Debug Info** expander

### What You'll See
```json
{
  "target_column": "target",
  "missing_values": 0,
  "data_type": "int64",
  "unique_values": 3
}
```

---

## 📝 Code Implementation

### Debug Function
```python
def log_target_validation_debug(target_col, data, target_data):
    """Log target validation details for debugging."""
    debug_info = {
        'target_column': target_col,
        'missing_values': int(data[target_col].isna().sum()),
        'data_type': str(data[target_col].dtype),
        'unique_values': int(target_data.nunique())
    }
    logger.debug(f"Target Validation: {debug_info}")
    return debug_info
```

### Sidebar Toggle
```python
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show validation debug information")
```

### Display in Training Page
```python
if debug_mode:
    debug_info = log_target_validation_debug(target_col, data, target_data)
    with st.expander("🐛 Debug Info", expanded=False):
        st.json(debug_info)
```

---

## 📊 Debug Information

| Field | Type | Example | Purpose |
|-------|------|---------|---------|
| target_column | str | "target" | Column name |
| missing_values | int | 0 | Count of NaN values |
| data_type | str | "int64" | NumPy/Pandas dtype |
| unique_values | int | 3 | Distinct value count |

---

## 🔍 Examples

### Example 1: Clean Numeric Target
```json
{
  "target_column": "target",
  "missing_values": 0,
  "data_type": "int64",
  "unique_values": 3
}
```
✅ Good: No missing values, numeric type, 3 classes

### Example 2: Categorical Target
```json
{
  "target_column": "category",
  "missing_values": 0,
  "data_type": "object",
  "unique_values": 4
}
```
✅ Good: Categorical type, 4 categories

### Example 3: Target with Missing Values
```json
{
  "target_column": "target",
  "missing_values": 5,
  "data_type": "float64",
  "unique_values": 10
}
```
⚠️ Warning: 5 missing values detected

### Example 4: Single Value Target
```json
{
  "target_column": "target",
  "missing_values": 0,
  "data_type": "int64",
  "unique_values": 1
}
```
❌ Error: Only 1 unique value (need >=2)

---

## 🎯 Use Cases

### Troubleshooting Validation Errors
```
User sees: "Not Enough Categories" error
Debug info shows: unique_values: 1
Solution: Select different column with 2+ values
```

### Checking Data Quality
```
User wonders: Does my target have missing values?
Debug info shows: missing_values: 0
Result: Target is clean, ready for training
```

### Verifying Data Type
```
User asks: Is my target numeric or categorical?
Debug info shows: data_type: "object"
Result: Target is categorical (text)
```

### Analyzing Class Distribution
```
User needs: How many classes in target?
Debug info shows: unique_values: 5
Result: 5-class classification problem
```

---

## 🔧 Customization

### Add More Metrics
```python
def log_target_validation_debug(target_col, data, target_data):
    debug_info = {
        'target_column': target_col,
        'missing_values': int(data[target_col].isna().sum()),
        'data_type': str(data[target_col].dtype),
        'unique_values': int(target_data.nunique()),
        # Add new metrics:
        'memory_usage': str(data[target_col].memory_usage(deep=True)),
        'min_value': str(target_data.min()) if is_numeric_target(target_data) else 'N/A',
        'max_value': str(target_data.max()) if is_numeric_target(target_data) else 'N/A',
    }
    logger.debug(f"Target Validation: {debug_info}")
    return debug_info
```

### Change Display Format
```python
# Instead of JSON, use table format:
if debug_mode:
    debug_info = log_target_validation_debug(target_col, data, target_data)
    with st.expander("🐛 Debug Info", expanded=False):
        st.dataframe(pd.DataFrame([debug_info]))
```

### Add to Different Page
```python
# Add debug info to EDA page:
if debug_mode and 'data' in st.session_state:
    target_col = st.selectbox("Select column", st.session_state.data.columns)
    target_data = st.session_state.data[target_col].dropna()
    debug_info = log_target_validation_debug(target_col, st.session_state.data, target_data)
    st.json(debug_info)
```

---

## 📍 Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Debug Function | app/main.py | 73-85 |
| Sidebar Toggle | app/main.py | 103-104 |
| Display Logic | app/main.py | 237-241 |

---

## ✅ Features

- ✅ **Toggle-based**: Easy on/off control
- ✅ **Production-safe**: OFF by default
- ✅ **Non-intrusive**: Expandable section
- ✅ **Logged**: Integrated with logger
- ✅ **Minimal**: Only ~15 lines of code
- ✅ **Extensible**: Easy to add metrics
- ✅ **JSON format**: Clear and readable

---

## 🚫 What NOT to Do

❌ Enable debug mode by default  
❌ Display actual data values  
❌ Add expensive computations  
❌ Log sensitive information  
❌ Clutter the main UI  

---

## 🎓 Learning Path

1. **Understand**: Read DEBUG_VALIDATION_GUIDE.md
2. **Locate**: Find code in app/main.py (lines 73-85, 103-104, 237-241)
3. **Enable**: Toggle debug mode in sidebar
4. **Observe**: View debug info in Training page
5. **Extend**: Add more metrics as needed

---

## 🔗 Related Files

- **DEBUG_VALIDATION_GUIDE.md** - Comprehensive guide
- **app/main.py** - Implementation code
- **app/utils/logger.py** - Logger configuration

---

## 💡 Tips

### Tip 1: Check Logs
When debug mode is ON, check application logs:
```
DEBUG:__main__:Target Validation: {'target_column': 'target', ...}
```

### Tip 2: Expand Section
Click **🐛 Debug Info** to expand/collapse debug details

### Tip 3: Copy JSON
Right-click on JSON to copy for analysis

### Tip 4: Multiple Targets
Select different target columns to compare debug info

### Tip 5: Troubleshoot
Use debug info to understand validation errors

---

## 📞 Support

### Questions?
- Check DEBUG_VALIDATION_GUIDE.md for details
- Review code examples above
- Check application logs

### Issues?
- Verify debug mode is enabled
- Check target column selection
- Reload page and try again

---

## Summary

**What**: Debug validation block for target column metadata  
**Where**: Sidebar toggle + Training page  
**When**: Enable when troubleshooting  
**Why**: Understand validation issues  
**How**: Toggle checkbox, view expandable section  

**Metrics Logged**:
- Target column name
- Missing value count
- Data type
- Unique value count

**Status**: ✅ Production-ready, safe by default

---

## Code Snippet Reference

### Copy-Paste Ready

**Enable Debug Mode**:
```python
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show validation debug information")
```

**Log Debug Info**:
```python
def log_target_validation_debug(target_col, data, target_data):
    debug_info = {
        'target_column': target_col,
        'missing_values': int(data[target_col].isna().sum()),
        'data_type': str(data[target_col].dtype),
        'unique_values': int(target_data.nunique())
    }
    logger.debug(f"Target Validation: {debug_info}")
    return debug_info
```

**Display Debug Info**:
```python
if debug_mode:
    debug_info = log_target_validation_debug(target_col, data, target_data)
    with st.expander("🐛 Debug Info", expanded=False):
        st.json(debug_info)
```

---

**Made for developers, by developers** 🚀
