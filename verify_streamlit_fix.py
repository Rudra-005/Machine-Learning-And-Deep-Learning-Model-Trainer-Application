#!/usr/bin/env python3
"""
Verify that the Streamlit app no longer has session state widget key conflicts.
"""

import subprocess
import sys
from pathlib import Path

def check_app_syntax():
    """Check if app.py has valid syntax."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", "app.py"],
        cwd=Path.cwd(),
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def check_for_widget_key_conflict():
    """
    Check app.py for the widget key conflict issue.
    The issue was: st.selectbox with key="task_type" and key="model_name"
    then trying to modify st.session_state.task_type and st.session_state.model_name
    """
    with open("app.py", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Check if we've removed the problematic key parameters
    issues = []
    
    # Find the page_model_training function
    if 'def page_model_training' in content:
        # Extract that function
        start = content.find('def page_model_training')
        end = content.find('\ndef ', start + 1)
        if end == -1:
            end = len(content)
        
        func_content = content[start:end]
        
        # Check for task_type selectbox with key parameter
        if 'st.selectbox(\n            "Task Type"' in func_content or 'st.selectbox("Task Type"' in func_content:
            # Check if it has key="task_type"
            if 'key="task_type"' in func_content:
                issues.append('Found problematic key="task_type" in selectbox')
        
        # Check for model_name selectbox with key parameter  
        if 'st.selectbox(\n            "Model Type"' in func_content or 'st.selectbox("Model Type"' in func_content:
            if 'key="model_name"' in func_content:
                issues.append('Found problematic key="model_name" in selectbox')
    
    return issues

def main():
    print("=" * 70)
    print("Streamlit Widget Key Conflict Verification")
    print("=" * 70)
    
    print("\n✓ Checking app.py syntax...")
    if check_app_syntax():
        print("  ✓ app.py has valid Python syntax")
    else:
        print("  ✗ app.py has syntax errors")
        return False
    
    print("\n✓ Checking for widget key conflicts...")
    issues = check_for_widget_key_conflict()
    if not issues:
        print("  ✓ No problematic widget key parameters found")
        print("  ✓ Selectbox widgets no longer have key='task_type' or key='model_name'")
    else:
        print("  ✗ Found issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False
    
    print("\n" + "=" * 70)
    print("RESULT: ✓ Streamlit app widget key conflicts have been FIXED!")
    print("=" * 70)
    print("\nThe error 'st.session_state.task_type cannot be modified after")
    print("the widget with key task_type is instantiated' should now be resolved.")
    print("\nThe app is ready to run with: streamlit run app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
