"""
Redesigned EDA / Data Understanding page - Production UI
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.missing_value_analyzer import (
    compute_missing_stats,
    get_columns_above_threshold,
    get_missing_patterns,
    create_missing_bar_chart,
    create_missing_heatmap
)
from core.target_analyzer import (
    detect_task_type,
    analyze_classification,
    analyze_regression,
    create_class_distribution_plot,
    create_regression_histogram,
    create_regression_boxplot
)
from core.feature_analyzer import (
    detect_feature_types,
    get_feature_stats,
    plot_numerical_histogram,
    plot_numerical_boxplot,
    plot_categorical_bar
)
from core.relationship_analyzer import (
    compute_correlation_matrix,
    get_top_correlated_features,
    analyze_categorical_regression,
    analyze_categorical_classification,
    plot_correlation_heatmap,
    plot_categorical_regression,
    plot_categorical_classification
)
from app.utils.logger import logger
from app.utils.eda_optimizer import (
    DataQualityChecker,
    CachedEDAOperations,
    display_data_quality_warnings,
    should_sample_data,
    get_sampled_data,
    create_selective_plot_selector
)


def render_eda_page():
    """Render optimized EDA page with clean UI."""
    st.title("2️⃣ Exploratory Data Analysis")
    st.subheader("Understand your data before training")
    st.divider()
    
    if 'data' not in st.session_state or st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    data = st.session_state.data
    
    # Initialize session state
    if 'eda_target_col' not in st.session_state:
        st.session_state.eda_target_col = data.columns[0]
    if 'eda_selected_features' not in st.session_state:
        st.session_state.eda_selected_features = []
    
    # Data sampling info
    should_sample, sample_size = should_sample_data(data)
    if should_sample:
        st.info(f"📊 Large dataset ({len(data):,} rows). Using {sample_size:,} samples for visualizations.")
        viz_data = get_sampled_data(data, sample_size)
    else:
        viz_data = data
    
    # Data quality warnings
    with st.expander("⚠️ Data Quality Report", expanded=False):
        display_data_quality_warnings(data, st.session_state.eda_target_col)
    
    st.divider()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview",
        "❌ Missing",
        "🎯 Target",
        "🔍 Features",
        "📈 Correlation"
    ])
    
    # ============ TAB 1: OVERVIEW ============
    with tab1:
        st.markdown("### Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(data):,}")
        col2.metric("Columns", len(data.columns))
        col3.metric("Missing", int(data.isnull().sum().sum()))
        col4.metric("Duplicates", len(data) - len(data.drop_duplicates()))
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Preview**")
            st.dataframe(data.head(10), use_container_width=True)
        
        with col2:
            st.markdown("**Column Information**")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes,
                'Non-Null': data.notna().sum(),
                'Null': data.isnull().sum(),
                'Unique': data.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        st.divider()
        
        st.markdown("**Descriptive Statistics**")
        st.dataframe(data.describe(), use_container_width=True)
    
    # ============ TAB 2: MISSING VALUES ============
    with tab2:
        st.markdown("### Missing Values Analysis")
        
        try:
            missing_stats = compute_missing_stats(data)
            total_missing = missing_stats['missing_count'].sum()
            
            if total_missing == 0:
                st.success("✅ No missing values detected!")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Missing", int(total_missing))
                col2.metric("Missing %", f"{(total_missing / (len(data) * len(data.columns)) * 100):.2f}%")
                col3.metric("Affected Columns", (missing_stats['missing_count'] > 0).sum())
                
                st.divider()
                
                st.markdown("**Missing by Column**")
                st.dataframe(missing_stats[missing_stats['missing_count'] > 0], use_container_width=True)
                
                st.divider()
                
                threshold = st.slider("Threshold (%)", 0, 100, 50)
                above_threshold = get_columns_above_threshold(data, threshold / 100)
                
                if above_threshold['count'] > 0:
                    st.warning(f"⚠️ Above {threshold}%: {above_threshold['columns']}")
                else:
                    st.success(f"✅ No columns above {threshold}%")
                
                st.divider()
                
                patterns = get_missing_patterns(data)
                st.markdown("**Missing Patterns**")
                st.write(f"Rows with missing: {patterns['rows_with_missing']} ({patterns['rows_with_missing_pct']:.1f}%)")
                
                st.divider()
                
                st.markdown("**Visualization**")
                viz_type = st.selectbox("Type", ["Bar Chart", "Heatmap"])
                
                if st.button("📊 Generate", key="missing_plot"):
                    try:
                        if viz_type == "Bar Chart":
                            fig = create_missing_bar_chart(viz_data, backend='plotly')
                        else:
                            fig = create_missing_heatmap(viz_data, backend='plotly')
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Missing plot error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Missing analysis error: {str(e)}")
    
    # ============ TAB 3: TARGET ANALYSIS ============
    with tab3:
        st.markdown("### Target Variable Analysis")
        
        try:
            target_col = st.selectbox("Select Target", data.columns, key="target_select")
            st.session_state.eda_target_col = target_col
            
            target_data = data[target_col].dropna()
            
            if len(target_data) == 0:
                st.error("❌ Target has no valid values")
                return
            
            task_info = detect_task_type(target_data)
            task_type = task_info.task_type
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Task", task_type.upper())
            col2.metric("Unique", target_data.nunique())
            col3.metric("Type", str(target_data.dtype))
            
            st.divider()
            
            if task_type == 'classification':
                st.markdown("**Classification Analysis**")
                
                class_analysis = analyze_classification(target_data)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Classes", class_analysis['n_classes'])
                col2.metric("Imbalance", f"{class_analysis['imbalance_ratio']:.2f}:1")
                col3.metric("Minority %", f"{(1/class_analysis['imbalance_ratio']*100):.2f}%")
                
                st.divider()
                
                st.markdown("**Class Distribution**")
                dist_df = pd.DataFrame({
                    'Class': list(class_analysis['class_distribution'].keys()),
                    'Percentage': list(class_analysis['class_distribution'].values())
                })
                st.dataframe(dist_df, use_container_width=True)
                
                if class_analysis['imbalance_ratio'] > 1.5:
                    st.warning(
                        f"⚠️ **Imbalanced Dataset**\n\n"
                        f"Ratio: {class_analysis['imbalance_ratio']:.2f}:1\n"
                        f"Tip: Use class weights during training"
                    )
            
            else:
                st.markdown("**Regression Analysis**")
                
                reg_analysis = analyze_regression(target_data)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{reg_analysis['mean']:.4f}")
                col2.metric("Std", f"{reg_analysis['std']:.4f}")
                col3.metric("Skew", f"{reg_analysis['skewness']:.4f}")
                col4.metric("Kurt", f"{reg_analysis['kurtosis']:.4f}")
                
                st.divider()
                
                st.markdown("**Distribution Stats**")
                stats_df = pd.DataFrame({
                    'Metric': ['Min', 'Q1', 'Median', 'Q3', 'Max'],
                    'Value': [
                        reg_analysis['min'],
                        reg_analysis['q25'],
                        reg_analysis['median'],
                        reg_analysis['q75'],
                        reg_analysis['max']
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)
            
            st.divider()
            
            st.markdown("**Distribution Plot**")
            if st.button("📊 Generate", key="target_plot"):
                try:
                    if task_type == 'classification':
                        fig = create_class_distribution_plot(target_data, backend='plotly')
                    else:
                        fig = create_regression_histogram(target_data, backend='plotly')
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Target analysis error: {str(e)}")
    
    # ============ TAB 4: FEATURE ANALYSIS ============
    with tab4:
        st.markdown("### Feature Analysis")
        
        try:
            target_col = st.session_state.eda_target_col
            feature_cols = [col for col in data.columns if col != target_col]
            
            if not feature_cols:
                st.warning("⚠️ No features available")
                return
            
            feature_types = detect_feature_types(data[feature_cols])
            
            col1, col2 = st.columns(2)
            col1.metric("Numerical", len(feature_types['numerical']))
            col2.metric("Categorical", len(feature_types['categorical']))
            
            st.divider()
            
            st.markdown("**Select Features**")
            selected_features = create_selective_plot_selector(feature_cols, max_default=3)
            st.session_state.eda_selected_features = selected_features
            
            if not selected_features:
                st.info("ℹ️ Select features to analyze")
                return
            
            st.divider()
            
            st.markdown("**Feature Statistics**")
            
            for feature in selected_features:
                with st.expander(f"📊 {feature}"):
                    try:
                        stats = get_feature_stats(data, feature)
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Type", stats['type'])
                        col2.metric("Missing", stats['missing'])
                        col3.metric("Unique", stats.get('unique', 'N/A'))
                        
                        if stats['type'] == 'numerical':
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean", f"{stats['mean']:.4f}")
                            col2.metric("Std", f"{stats['std']:.4f}")
                            col3.metric("Min", f"{stats['min']:.4f}")
                            col4.metric("Max", f"{stats['max']:.4f}")
                        
                        if st.button(f"📊 Plot", key=f"plot_{feature}"):
                            try:
                                if stats['type'] == 'numerical':
                                    fig = plot_numerical_histogram(viz_data, feature, backend='plotly')
                                else:
                                    fig = plot_categorical_bar(viz_data, feature, backend='plotly')
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Feature analysis error: {str(e)}")
    
    # ============ TAB 5: CORRELATION ============
    with tab5:
        st.markdown("### Feature-Target Relationships")
        
        try:
            target_col = st.session_state.eda_target_col
            target_data = data[target_col].dropna()
            feature_cols = [col for col in data.columns if col != target_col]
            
            if not feature_cols:
                st.warning("⚠️ No features available")
                return
            
            task_info = detect_task_type(target_data)
            task_type = task_info.task_type
            
            st.markdown("**Correlation Analysis**")
            
            corr_method = st.selectbox("Method", ["pearson", "spearman"])
            
            if st.button("📊 Compute", key="corr_btn"):
                try:
                    temp_df = data[feature_cols + [target_col]].copy()
                    top_result = get_top_correlated_features(temp_df, target_col, method=corr_method, top_n=10)
                    
                    st.markdown(f"**Top {top_result['count']} Correlated**")
                    top_df = pd.DataFrame({
                        'Feature': top_result['features'],
                        'Correlation': top_result['correlations']
                    })
                    st.dataframe(top_df, use_container_width=True)
                    
                    st.divider()
                    
                    # Categorical analysis
                    feature_types = detect_feature_types(data[feature_cols])
                    cat_features = feature_types['categorical']
                    
                    if cat_features:
                        st.markdown("**Categorical Features**")
                        selected_cat = st.selectbox("Select", cat_features)
                        
                        if st.button("📊 Plot", key="cat_plot"):
                            try:
                                if task_type == 'classification':
                                    fig = plot_categorical_classification(viz_data, selected_cat, target_col, backend='plotly')
                                else:
                                    fig = plot_categorical_regression(viz_data, selected_cat, target_col, backend='plotly')
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    
                    st.divider()
                    
                    # Heatmap
                    st.markdown("**Correlation Heatmap**")
                    num_features = feature_types['numerical']
                    
                    if num_features:
                        heatmap_features = st.multiselect(
                            "Features",
                            num_features,
                            default=num_features[:min(5, len(num_features))],
                            key="heatmap_select"
                        )
                        
                        if heatmap_features and st.button("📊 Generate", key="heatmap_btn"):
                            try:
                                temp_hm = viz_data[heatmap_features].copy()
                                fig = plot_correlation_heatmap(temp_hm, heatmap_features[0] if heatmap_features else None, backend='plotly')
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                    else:
                        st.info("ℹ️ No numerical features for heatmap")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Correlation error: {str(e)}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Correlation analysis error: {str(e)}")


if __name__ == "__main__":
    render_eda_page()
