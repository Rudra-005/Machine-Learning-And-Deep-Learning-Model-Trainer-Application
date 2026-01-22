"""
AutoML Training Page Guard Logic - Minimal Reference

Canonical guard pattern for checking dataset existence.
"""

# ============================================================================
# BEFORE (Old - Multiple checks)
# ============================================================================

# OLD CODE - DON'T USE
"""
def page_automl_training():
    st.title("🧠 AutoML Training")
    
    # Multiple checks - WRONG
    if not st.session_state.data_preprocessed:
        st.warning("Please preprocess data first")
        return
    
    if st.session_state.X_train is None:
        st.warning("No training data")
        return
"""


# ============================================================================
# AFTER (New - Canonical check only)
# ============================================================================

# NEW CODE - USE THIS
"""
def page_automl_training():
    st.title("🧠 AutoML Training")
    
    # GUARD: Check ONLY if dataset exists
    if st.session_state.get("dataset") is None:
        st.warning("Please load data first in the Data Loading tab")
        st.stop()
    
    # Dataset is guaranteed to exist here
    # Proceed with page logic
"""


# ============================================================================
# PATTERN
# ============================================================================

# Canonical Guard Pattern:
# 1. Check ONLY: st.session_state.get("dataset") is None
# 2. Show warning with clear message
# 3. Use st.stop() to halt execution
# 4. Proceed normally if dataset exists


# ============================================================================
# USAGE
# ============================================================================

import streamlit as st

def page_automl_training():
    st.title("🧠 AutoML Training")
    
    # Guard: Check ONLY if dataset exists
    if st.session_state.get("dataset") is None:
        st.warning("Please load data first in the Data Loading tab")
        st.stop()
    
    # Safe to proceed - dataset exists
    st.write("Dataset loaded successfully")
    st.write(f"Shape: {st.session_state.dataset.shape}")
