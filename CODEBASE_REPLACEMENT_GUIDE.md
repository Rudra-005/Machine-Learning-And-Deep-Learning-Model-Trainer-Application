"""
Codebase-Wide Replacement Guide: Unified Data Availability Check

Replace ALL data availability checks with single canonical condition:
"dataset" in st.session_state

This ensures all pages rely on the same logic.
"""

# ============================================================================
# CANONICAL CHECK (Use everywhere)
# ============================================================================

# CORRECT - Use this everywhere
if "dataset" in st.session_state and st.session_state.dataset is not None:
    # Data is available
    pass

# SHORTER VERSION (Preferred)
if st.session_state.get("dataset") is not None:
    # Data is available
    pass


# ============================================================================
# FIND & REPLACE PATTERNS
# ============================================================================

# Pattern 1: data_loaded flag
# FIND:    if st.session_state.data_loaded:
# REPLACE: if st.session_state.get("dataset") is not None:

# Pattern 2: is_data_loaded() function
# FIND:    if is_data_loaded():
# REPLACE: if st.session_state.get("dataset") is not None:

# Pattern 3: X_train existence check
# FIND:    if st.session_state.X_train is not None:
# REPLACE: if st.session_state.get("dataset") is not None:

# Pattern 4: Multiple checks
# FIND:    if st.session_state.data_loaded and st.session_state.X_train is not None:
# REPLACE: if st.session_state.get("dataset") is not None:

# Pattern 5: data_preprocessed flag
# FIND:    if not st.session_state.data_preprocessed:
# REPLACE: if st.session_state.get("dataset") is None:

# Pattern 6: Guard clause with return
# FIND:    if not st.session_state.data_loaded:
#              st.warning("Load data first")
#              return
# REPLACE: if st.session_state.get("dataset") is None:
#              st.warning("Load data first")
#              st.stop()


# ============================================================================
# FILES TO UPDATE
# ============================================================================

FILES_TO_UPDATE = [
    "app.py",
    "main.py",
    "main_canonical.py",
    "app_demo.py",
    "app/main.py",
    "app/pages/automl_training.py",
    "app/pages/eda_page.py",
    "app/utils/automl_ui.py",
    "app/utils/cv_streamlit.py",
    "app/utils/dl_streamlit.py",
    "app/utils/dynamic_ui.py",
    "app/utils/hp_streamlit.py",
    "app/utils/iterative_streamlit.py",
    "app/utils/logger_streamlit.py",
    "app/utils/model_ui.py",
    "app/utils/parameter_validator.py",
    "app/utils/training_logger.py",
    "automl_training_refactored.py",
    "data_loading_refactored.py",
    "sidebar_integration_example.py",
    "sidebar_refactored.py",
    "session_state_contract.py",
]


# ============================================================================
# GUARD CLAUSE PATTERN (Use in all pages)
# ============================================================================

# CORRECT PATTERN
def page_automl_training():
    st.title("🧠 AutoML Training")
    
    # Guard: Check ONLY if dataset exists
    if st.session_state.get("dataset") is None:
        st.warning("Please load data first in the Data Loading tab")
        st.stop()
    
    # Safe to proceed - dataset exists
    # ... rest of page logic


# ============================================================================
# SIDEBAR STATUS PATTERN
# ============================================================================

# CORRECT PATTERN
st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")

if st.session_state.get("dataset") is not None:
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.warning("⚠️ No Data Loaded")

if st.session_state.get("model_trained"):
    st.sidebar.success("✅ Model Trained")
else:
    st.sidebar.info("ℹ️ No Model Trained")


# ============================================================================
# CONDITIONAL DISPLAY PATTERN
# ============================================================================

# CORRECT PATTERN
if st.session_state.get("dataset") is not None:
    st.subheader("📋 Data Overview")
    st.metric("Rows", st.session_state.dataset.shape[0])
    st.metric("Cols", st.session_state.dataset.shape[1])


# ============================================================================
# INITIALIZATION PATTERN
# ============================================================================

# CORRECT PATTERN
def initialize_session_state():
    """Initialize session state with dataset as source of truth."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    # ... other keys


# ============================================================================
# DEPRECATED PATTERNS (Remove these)
# ============================================================================

# ❌ DON'T USE: data_loaded flag
# st.session_state.data_loaded = True
# if st.session_state.data_loaded:

# ❌ DON'T USE: is_data_loaded() function
# def is_data_loaded():
#     return st.session_state.get("dataset") is not None
# if is_data_loaded():

# ❌ DON'T USE: X_train existence check
# if st.session_state.X_train is not None:

# ❌ DON'T USE: data_preprocessed flag
# st.session_state.data_preprocessed = True
# if st.session_state.data_preprocessed:

# ❌ DON'T USE: Multiple checks
# if st.session_state.data_loaded and st.session_state.X_train is not None:


# ============================================================================
# SUMMARY
# ============================================================================

"""
CANONICAL RULE:
- Use ONLY: st.session_state.get("dataset") is not None
- For guards: Use st.stop() not return
- For initialization: Set "dataset" to None
- For display: Check if dataset is not None
- For sidebar: Show status based on dataset existence

BENEFITS:
✅ Single source of truth
✅ No conflicting flags
✅ Consistent across all pages
✅ Easy to maintain
✅ No deprecated patterns
"""
