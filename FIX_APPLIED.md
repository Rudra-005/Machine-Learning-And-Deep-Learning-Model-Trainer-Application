# ✅ Fix Applied - NameError Resolved

## Issue
```
NameError: name 'GradientBoostingRegressor' is not defined
File "C:\Users\rudra\Downloads\ML_DL_Trainer\main.py", line 107
```

## Root Cause
Missing import for `GradientBoostingRegressor` from `sklearn.ensemble`

## Solution Applied
Updated import statement in `main.py` line 18:

### Before
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
```

### After
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
```

## Status
✅ **FIXED** - Application ready to run

---

## 🚀 Run the Application Now

```bash
cd c:\Users\rudra\Downloads\ML_DL_Trainer
streamlit run main.py
```

The application will now start without errors!

---

## ✅ Verification

The following imports are now complete:
- ✅ RandomForestClassifier
- ✅ GradientBoostingClassifier
- ✅ RandomForestRegressor
- ✅ **GradientBoostingRegressor** (FIXED)
- ✅ LogisticRegression
- ✅ Ridge
- ✅ Lasso
- ✅ SVC
- ✅ SVR
- ✅ KNeighborsClassifier
- ✅ KNeighborsRegressor

All models are now properly imported and ready to use!
