"""
Sidebar Refactor - Minimal Reference

Before and after code for sidebar status section.
"""

# ============================================================================
# BEFORE (Old - with deprecated flags)
# ============================================================================

# OLD CODE - DON'T USE
"""
st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")
if st.session_state.data_loaded:  # DEPRECATED FLAG
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.warning("⚠️ No Data Loaded")

if st.session_state.model_trained:
    st.sidebar.success("✅ Model Trained")
else:
    st.sidebar.info("ℹ️ No Model Trained")
"""


# ============================================================================
# AFTER (New - canonical check only)
# ============================================================================

# NEW CODE - USE THIS
"""
st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")

# Check ONLY if dataset exists
if st.session_state.get("dataset") is not None:
    st.sidebar.success("✅ Data Loaded")
else:
    st.sidebar.warning("⚠️ No Data Loaded")

# Check if model is trained
if st.session_state.get("model_trained"):
    st.sidebar.success("✅ Model Trained")
else:
    st.sidebar.info("ℹ️ No Model Trained")
"""


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def render_sidebar_status():
    """Render sidebar status section."""
    import streamlit as st
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status**")
    
    # Data status: check ONLY if dataset exists
    if st.session_state.get("dataset") is not None:
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.warning("⚠️ No Data Loaded")
    
    # Model status: check if model is trained
    if st.session_state.get("model_trained"):
        st.sidebar.success("✅ Model Trained")
    else:
        st.sidebar.info("ℹ️ No Model Trained")


# ============================================================================
# USAGE
# ============================================================================

# In main app:
# render_sidebar_status()
