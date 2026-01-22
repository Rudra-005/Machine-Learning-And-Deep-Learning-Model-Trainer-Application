"""
Debug Panel Integration Guide

Add debug panel to main app with single import and function call.
"""

# ============================================================================
# INTEGRATION STEPS
# ============================================================================

# Step 1: Import at top of main app
# from debug_panel import render_debug_panel, DEBUG_MODE

# Step 2: Call in main app (after sidebar setup)
# render_debug_panel()

# Step 3: Toggle debug mode
# In debug_panel.py, change:
# DEBUG_MODE = True   # Enable debug panel
# DEBUG_MODE = False  # Disable debug panel


# ============================================================================
# EXAMPLE INTEGRATION
# ============================================================================

"""
# In app.py or main.py

import streamlit as st
from debug_panel import render_debug_panel

st.set_page_config(page_title="ML/DL Trainer", layout="wide")

# ... rest of app setup ...

# Sidebar
st.sidebar.title("🤖 ML/DL Trainer")
st.sidebar.markdown("---")

# ... sidebar navigation ...

# Add debug panel at end of sidebar
render_debug_panel()

# ... rest of app ...
"""


# ============================================================================
# DEBUG PANEL FEATURES
# ============================================================================

"""
The debug panel shows:

1. Dataset Status
   - ✅ Dataset exists (or ❌ No dataset)
   - Shape: rows × columns

2. Session State Keys
   - Total keys count
   - Populated keys count
   - Empty keys count
   - List of populated keys with types
   - List of empty keys

3. Quick Validation
   - ✅/❌ Dataset loaded
   - ✅/❌ X_train exists
   - ✅/❌ y_train exists
   - ✅/❌ Model trained

4. Expandable/Collapsible
   - Located in sidebar under "🔧 Debug Panel"
   - Can be expanded/collapsed as needed
"""


# ============================================================================
# DISABLING DEBUG PANEL
# ============================================================================

"""
To disable debug panel:

1. Open debug_panel.py
2. Change: DEBUG_MODE = True
3. To:     DEBUG_MODE = False
4. Save file

The debug panel will no longer appear in the sidebar.
"""


# ============================================================================
# MINIMAL CODE EXAMPLE
# ============================================================================

import streamlit as st

DEBUG_MODE = True

def render_debug_panel():
    """Render debug panel in sidebar."""
    if not DEBUG_MODE:
        return
    
    with st.sidebar:
        st.markdown("---")
        with st.expander("🔧 Debug Panel"):
            # Dataset status
            dataset_exists = st.session_state.get("dataset") is not None
            st.write(f"**Dataset**: {'✅ Exists' if dataset_exists else '❌ Missing'}")
            
            if dataset_exists:
                shape = st.session_state.dataset.shape
                st.write(f"**Shape**: {shape[0]} rows × {shape[1]} cols")
            
            # Session state summary
            keys = list(st.session_state.keys())
            populated = [k for k in keys if st.session_state[k] is not None]
            
            st.write(f"**Keys**: {len(keys)} total, {len(populated)} populated")
            
            # Quick checks
            st.write("**Checks**:")
            st.write(f"  • Dataset: {'✅' if dataset_exists else '❌'}")
            st.write(f"  • X_train: {'✅' if st.session_state.get('X_train') is not None else '❌'}")
            st.write(f"  • Model: {'✅' if st.session_state.get('model_trained') else '❌'}")
