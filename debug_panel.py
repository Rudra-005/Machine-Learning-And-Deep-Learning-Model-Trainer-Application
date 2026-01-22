"""
Debug Panel: Session State Inspector

Temporary debug panel for validating session state.
Can be disabled by setting DEBUG_MODE = False
"""

import streamlit as st
import pandas as pd

# Toggle debug mode here
DEBUG_MODE = True


def render_debug_panel():
    """Render debug panel showing session state info."""
    if not DEBUG_MODE:
        return
    
    with st.sidebar:
        st.markdown("---")
        with st.expander("🔧 Debug Panel"):
            st.subheader("Session State Inspector")
            
            # Dataset status
            st.markdown("**Dataset Status**")
            dataset_exists = st.session_state.get("dataset") is not None
            
            if dataset_exists:
                st.success("✅ Dataset exists")
                dataset_shape = st.session_state.dataset.shape
                st.write(f"Shape: {dataset_shape[0]} rows × {dataset_shape[1]} columns")
            else:
                st.error("❌ No dataset")
            
            # Session state keys
            st.markdown("**Session State Keys**")
            keys = list(st.session_state.keys())
            
            # Group keys by status
            data_keys = [k for k in keys if st.session_state[k] is not None]
            empty_keys = [k for k in keys if st.session_state[k] is None]
            
            st.write(f"**Total Keys**: {len(keys)}")
            st.write(f"**Populated**: {len(data_keys)}")
            st.write(f"**Empty**: {len(empty_keys)}")
            
            # Show populated keys
            if data_keys:
                st.write("**Populated Keys:**")
                for key in sorted(data_keys):
                    value = st.session_state[key]
                    if isinstance(value, pd.DataFrame):
                        st.write(f"  • {key}: DataFrame {value.shape}")
                    elif isinstance(value, bool):
                        st.write(f"  • {key}: {value}")
                    else:
                        st.write(f"  • {key}: {type(value).__name__}")
            
            # Show empty keys
            if empty_keys:
                st.write("**Empty Keys:**")
                for key in sorted(empty_keys):
                    st.write(f"  • {key}: None")
            
            # Quick validation
            st.markdown("**Validation**")
            checks = {
                "Dataset loaded": dataset_exists,
                "X_train exists": st.session_state.get("X_train") is not None,
                "y_train exists": st.session_state.get("y_train") is not None,
                "Model trained": st.session_state.get("model_trained", False),
            }
            
            for check_name, check_result in checks.items():
                status = "✅" if check_result else "❌"
                st.write(f"{status} {check_name}")


if __name__ == "__main__":
    # Test the debug panel
    st.set_page_config(page_title="Debug Panel Test", layout="wide")
    
    # Initialize session state
    if "dataset" not in st.session_state:
        st.session_state.dataset = None
    if "X_train" not in st.session_state:
        st.session_state.X_train = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    
    st.title("Debug Panel Test")
    
    # Test buttons
    if st.button("Load Sample Data"):
        import numpy as np
        np.random.seed(42)
        X = np.random.randn(100, 5)
        st.session_state.dataset = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        st.success("Sample data loaded")
    
    if st.button("Clear Data"):
        st.session_state.dataset = None
        st.session_state.X_train = None
        st.session_state.y_train = None
        st.success("Data cleared")
    
    # Render debug panel
    render_debug_panel()
    
    st.write("Debug panel is in the sidebar →")
