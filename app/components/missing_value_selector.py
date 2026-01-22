"""
Streamlit UI Component for Missing Value Strategy Selection

Interactive table for users to review and select missing value handling strategies.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Any


STRATEGY_OPTIONS = {
    'numeric': ['median', 'mean', 'drop_column'],
    'categorical': ['most_frequent', 'constant', 'drop_column']
}


def render_missing_value_selector(
    recommendations: List[Dict[str, Any]]
) -> Dict[str, str]:
    """
    Render interactive UI for missing value strategy selection.
    
    Args:
        recommendations: List from recommend_missing_value_strategy()
                        Each item has: column, data_type, missing_percentage, 
                                      strategy, explanation
        
    Returns:
        Dictionary mapping column names to user-selected strategies
        
    Example:
        >>> recommendations = [
        ...     {
        ...         'column': 'age',
        ...         'data_type': 'numeric',
        ...         'missing_percentage': 25.0,
        ...         'strategy': 'median',
        ...         'explanation': 'Impute numeric column with median...'
        ...     }
        ... ]
        >>> user_selections = render_missing_value_selector(recommendations)
        >>> user_selections['age']
        'median'
    """
    st.subheader("Missing Value Handling Strategy")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    total_missing = sum(r['missing_percentage'] for r in recommendations)
    cols_with_missing = len([r for r in recommendations if r['missing_percentage'] > 0])
    
    col1.metric("Columns with Missing", cols_with_missing)
    col2.metric("Total Missing %", f"{total_missing:.1f}%")
    col3.metric("Recommendations", len(recommendations))
    
    st.divider()
    
    # Filter to only columns with missing values
    missing_recs = [r for r in recommendations if r['missing_percentage'] > 0]
    
    if not missing_recs:
        st.success("✅ No missing values detected!")
        return {}
    
    st.write(f"**Configure handling for {len(missing_recs)} columns with missing values**")
    
    # Create interactive selection table
    user_selections = {}
    
    for rec in missing_recs:
        col = rec['column']
        data_type = rec['data_type']
        missing_pct = rec['missing_percentage']
        recommended = rec['strategy']
        explanation = rec['explanation']
        
        # Create columns for layout
        col1, col2, col3, col4 = st.columns([2, 1.5, 2, 3])
        
        with col1:
            st.write(f"**{col}**")
            st.caption(f"{data_type} • {missing_pct}% missing")
        
        with col2:
            st.write("📋 Recommended")
            st.caption(f"`{recommended}`")
        
        with col3:
            # Get valid strategies for this data type
            valid_strategies = STRATEGY_OPTIONS.get(data_type, ['median', 'most_frequent'])
            
            # Selectbox with recommended as default
            selected = st.selectbox(
                label="Strategy",
                options=valid_strategies,
                index=valid_strategies.index(recommended) if recommended in valid_strategies else 0,
                key=f"strategy_{col}",
                label_visibility="collapsed"
            )
            user_selections[col] = selected
        
        with col4:
            st.write("💡 Why?")
            st.caption(explanation)
        
        st.divider()
    
    return user_selections


def render_strategy_summary(
    recommendations: List[Dict[str, Any]],
    user_selections: Dict[str, str]
) -> None:
    """
    Display summary of selected strategies.
    
    Args:
        recommendations: Original recommendations
        user_selections: User-selected strategies
    """
    st.subheader("Strategy Summary")
    
    # Build summary table
    summary_data = []
    for rec in recommendations:
        col = rec['column']
        if col in user_selections:
            summary_data.append({
                'Column': col,
                'Type': rec['data_type'],
                'Missing %': f"{rec['missing_percentage']:.1f}%",
                'Recommended': rec['strategy'],
                'Selected': user_selections[col],
                'Match': '✅' if user_selections[col] == rec['strategy'] else '⚠️'
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Show warnings for non-recommended selections
        non_matching = [s for s in summary_data if s['Match'] == '⚠️']
        if non_matching:
            st.info(
                f"ℹ️ {len(non_matching)} column(s) using non-recommended strategy. "
                "This is fine if you have domain knowledge!"
            )


def render_strategy_details(
    recommendations: List[Dict[str, Any]]
) -> None:
    """
    Display detailed information about each strategy.
    
    Args:
        recommendations: List of recommendations
    """
    with st.expander("📚 Strategy Guide"):
        st.markdown("""
        **Numeric Columns:**
        - `median`: Robust to outliers, preserves distribution
        - `mean`: Simple average, affected by outliers
        - `drop_column`: Remove if >40% missing
        
        **Categorical Columns:**
        - `most_frequent`: Fill with most common value
        - `constant`: Fill with fixed value (e.g., "Unknown")
        - `drop_column`: Remove if >40% missing
        """)


def get_preprocessing_config(
    recommendations: List[Dict[str, Any]],
    user_selections: Dict[str, str]
) -> Dict[str, Dict[str, Any]]:
    """
    Convert user selections to preprocessing configuration.
    
    Args:
        recommendations: Original recommendations
        user_selections: User-selected strategies
        
    Returns:
        Configuration dictionary for preprocessing pipeline
    """
    config = {}
    
    for rec in recommendations:
        col = rec['column']
        strategy = user_selections.get(col, rec['strategy'])
        
        config[col] = {'strategy': strategy}
        
        # Add value for constant strategy
        if strategy == 'constant':
            config[col]['value'] = 'Unknown'
    
    return config


def page_missing_value_configuration(df: pd.DataFrame) -> Dict[str, str]:
    """
    Complete page for missing value strategy configuration.
    
    Args:
        df: Input DataFrame
        
    Returns:
        User-selected strategies
    """
    from core.missing_value_analyzer import recommend_missing_value_strategy
    
    st.title("⚙️ Missing Value Configuration")
    st.subheader("Review and approve missing value handling strategies")
    st.divider()
    
    # Get recommendations
    recommendations = recommend_missing_value_strategy(df)
    
    # Render selector
    user_selections = render_missing_value_selector(recommendations)
    
    st.divider()
    
    # Show summary
    if user_selections:
        render_strategy_summary(recommendations, user_selections)
        st.divider()
    
    # Show guide
    render_strategy_details(recommendations)
    
    return user_selections
