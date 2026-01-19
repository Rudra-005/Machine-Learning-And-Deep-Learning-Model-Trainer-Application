"""
EDA optimization utilities: caching, performance, and data quality checks
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from functools import lru_cache
import hashlib


class DataQualityChecker:
    """Check data quality and generate warnings."""
    
    @staticmethod
    def check_data_quality(data: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality assessment.
        
        Returns dict with quality metrics and warnings.
        """
        warnings = []
        quality_score = 100
        
        # Check missing values
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_pct > 50:
            warnings.append(("🔴 CRITICAL", f"Over 50% missing values ({missing_pct:.1f}%)"))
            quality_score -= 30
        elif missing_pct > 20:
            warnings.append(("🟠 WARNING", f"High missing values ({missing_pct:.1f}%)"))
            quality_score -= 15
        
        # Check duplicates
        dup_pct = (len(data) - len(data.drop_duplicates())) / len(data) * 100
        if dup_pct > 10:
            warnings.append(("🟠 WARNING", f"High duplicates ({dup_pct:.1f}%)"))
            quality_score -= 10
        
        # Check for constant columns
        for col in data.columns:
            if data[col].nunique() == 1:
                warnings.append(("🟡 INFO", f"Constant column: {col}"))
                quality_score -= 5
        
        # Check for low variance
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() < 1e-6:
                warnings.append(("🟡 INFO", f"Near-zero variance: {col}"))
                quality_score -= 3
        
        # Check dataset size
        if len(data) < 50:
            warnings.append(("🟠 WARNING", f"Very small dataset ({len(data)} rows)"))
            quality_score -= 15
        
        return {
            'quality_score': max(0, quality_score),
            'warnings': warnings,
            'missing_pct': missing_pct,
            'duplicate_pct': dup_pct
        }
    
    @staticmethod
    def check_target_quality(target: pd.Series, task_type: str) -> Dict:
        """Check target variable quality."""
        warnings = []
        
        if task_type == 'classification':
            # Check class imbalance
            class_counts = target.value_counts()
            min_class_pct = (class_counts.min() / len(target)) * 100
            max_class_pct = (class_counts.max() / len(target)) * 100
            imbalance_ratio = max_class_pct / min_class_pct if min_class_pct > 0 else float('inf')
            
            if imbalance_ratio > 10:
                warnings.append(("🔴 CRITICAL", f"Severe imbalance ({imbalance_ratio:.1f}:1)"))
            elif imbalance_ratio > 3:
                warnings.append(("🟠 WARNING", f"Moderate imbalance ({imbalance_ratio:.1f}:1)"))
            
            if len(class_counts) < 2:
                warnings.append(("🔴 CRITICAL", "Only 1 class in target"))
            
            return {
                'imbalance_ratio': imbalance_ratio,
                'min_class_pct': min_class_pct,
                'warnings': warnings
            }
        
        else:  # regression
            # Check target distribution
            skewness = target.skew()
            if abs(skewness) > 2:
                warnings.append(("🟠 WARNING", f"Highly skewed target (skewness: {skewness:.2f})"))
            
            # Check for outliers
            Q1 = target.quantile(0.25)
            Q3 = target.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((target < Q1 - 1.5*IQR) | (target > Q3 + 1.5*IQR)).sum()
            outlier_pct = (outliers / len(target)) * 100
            
            if outlier_pct > 10:
                warnings.append(("🟠 WARNING", f"Many outliers ({outlier_pct:.1f}%)"))
            
            return {
                'skewness': skewness,
                'outlier_pct': outlier_pct,
                'warnings': warnings
            }


class CachedEDAOperations:
    """Cached EDA operations to prevent recomputation."""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def cached_missing_stats(data_hash: str, data: pd.DataFrame) -> Dict:
        """Cache missing value statistics."""
        from core.missing_value_analyzer import compute_missing_statistics
        return compute_missing_statistics(data)
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def cached_feature_types(data_hash: str, data: pd.DataFrame) -> Dict:
        """Cache feature type detection."""
        from core.feature_analyzer import detect_feature_types
        return detect_feature_types(data)
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def cached_correlation(data_hash: str, features: pd.DataFrame, 
                          target: pd.Series, method: str) -> pd.DataFrame:
        """Cache correlation computation."""
        from core.relationship_analyzer import compute_correlation_matrix
        return compute_correlation_matrix(features, target, method=method)
    
    @staticmethod
    def get_data_hash(data: pd.DataFrame) -> str:
        """Generate hash of dataframe for cache key."""
        return hashlib.md5(
            pd.util.hash_pandas_object(data, index=True).values
        ).hexdigest()[:8]


def display_data_quality_warnings(data: pd.DataFrame, target_col: str = None):
    """Display data quality warnings in UI."""
    checker = DataQualityChecker()
    
    # Overall data quality
    quality = checker.check_data_quality(data)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("**Data Quality Assessment**")
    with col2:
        score = quality['quality_score']
        if score >= 80:
            st.metric("Quality Score", f"{score}/100", "✅")
        elif score >= 60:
            st.metric("Quality Score", f"{score}/100", "⚠️")
        else:
            st.metric("Quality Score", f"{score}/100", "🔴")
    
    # Display warnings
    if quality['warnings']:
        for severity, msg in quality['warnings']:
            if "CRITICAL" in severity:
                st.error(f"{severity} {msg}")
            elif "WARNING" in severity:
                st.warning(f"{severity} {msg}")
            else:
                st.info(f"{severity} {msg}")
    
    # Target quality if specified
    if target_col and target_col in data.columns:
        target = data[target_col].dropna()
        
        # Detect task type
        from core.target_analyzer import detect_task_type
        task_type = detect_task_type(target)
        
        target_quality = checker.check_target_quality(target, task_type)
        
        st.divider()
        st.write("**Target Variable Quality**")
        
        if target_quality['warnings']:
            for severity, msg in target_quality['warnings']:
                if "CRITICAL" in severity:
                    st.error(f"{severity} {msg}")
                elif "WARNING" in severity:
                    st.warning(f"{severity} {msg}")
                else:
                    st.info(f"{severity} {msg}")


def should_sample_data(data: pd.DataFrame, threshold: int = 100000) -> Tuple[bool, int]:
    """
    Determine if data should be sampled for performance.
    
    Returns (should_sample, sample_size)
    """
    if len(data) > threshold:
        sample_size = min(threshold, int(len(data) * 0.1))
        return True, sample_size
    return False, len(data)


def get_sampled_data(data: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """Get sampled data for large datasets."""
    if sample_size is None or len(data) <= sample_size:
        return data
    return data.sample(n=sample_size, random_state=42)


def create_selective_plot_selector(feature_cols: List[str], 
                                   max_default: int = 3) -> List[str]:
    """
    Create UI for selective feature plotting.
    
    Returns selected features.
    """
    st.write("**Select Features to Visualize**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Choose up to {len(feature_cols)} features")
    with col2:
        select_all = st.checkbox("Select All", value=False)
    
    if select_all:
        return feature_cols
    
    selected = st.multiselect(
        "Features",
        feature_cols,
        default=feature_cols[:min(max_default, len(feature_cols))],
        label_visibility="collapsed"
    )
    
    return selected


def add_tooltip(text: str, tooltip: str):
    """Add tooltip to UI element."""
    st.write(f"{text} ℹ️")
    with st.expander("Help", expanded=False):
        st.caption(tooltip)
