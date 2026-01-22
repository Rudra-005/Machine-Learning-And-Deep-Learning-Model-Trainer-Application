"""
Training Logger

Explains training decisions in user-friendly language:
- Why cross-validation was used instead of epochs
- Why certain parameters were hidden
- What strategy was applied based on model type
"""

import streamlit as st
from datetime import datetime
from models.model_config import is_tree_based, is_iterative, is_deep_learning


class TrainingLogger:
    """Logs training decisions in user-friendly language."""
    
    @staticmethod
    def log_model_selection(model_name, task_type):
        """Log model selection decision."""
        return f"""
        📋 **Model Selected**: {model_name.replace('_', ' ').title()}
        📊 **Task Type**: {task_type}
        ⏰ **Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
    
    @staticmethod
    def log_strategy_decision(model_name):
        """Log training strategy decision with explanation."""
        
        if is_deep_learning(model_name):
            return """
            🧠 **Training Strategy**: Deep Learning (Epochs)
            
            **Why this strategy?**
            - Deep learning models learn through multiple passes over data
            - Each pass (epoch) helps the model improve gradually
            - We use epochs to control how many times the model sees the data
            - Validation loss is monitored to prevent overfitting
            
            **What happens:**
            1. Model processes all training data (1 epoch)
            2. Checks performance on validation data
            3. Repeats for specified number of epochs
            4. Stops early if validation loss stops improving
            """
        
        elif is_iterative(model_name):
            return """
            🔄 **Training Strategy**: Iterative ML with Cross-Validation
            
            **Why this strategy?**
            - Iterative models converge through optimization iterations
            - Cross-validation tests the model on different data splits
            - This gives us confidence the model works on unseen data
            - More reliable than training once on a single split
            
            **What happens:**
            1. Data is split into k folds (e.g., 5 parts)
            2. Model trains on k-1 folds, tests on 1 fold
            3. Repeats k times with different test folds
            4. Reports average performance across all folds
            """
        
        elif is_tree_based(model_name):
            return """
            🌳 **Training Strategy**: Tree-Based ML with Cross-Validation
            
            **Why this strategy?**
            - Tree models don't need iterations to converge
            - Cross-validation ensures robust evaluation
            - Tests model on multiple data splits
            - Prevents overfitting to one particular split
            
            **What happens:**
            1. Data is split into k folds (e.g., 5 parts)
            2. Model trains on k-1 folds, tests on 1 fold
            3. Repeats k times with different test folds
            4. Reports average performance across all folds
            """
        
        return "Unknown strategy"
    
    @staticmethod
    def log_parameter_decisions(model_name, params_dict):
        """Log why certain parameters were shown/hidden."""
        
        explanations = []
        
        # Epochs
        if is_deep_learning(model_name):
            explanations.append("""
            ✅ **Epochs**: Shown
            - Deep learning models need epochs to train
            - Controls how many times model sees the data
            """)
        else:
            explanations.append("""
            ❌ **Epochs**: Hidden
            - Not applicable for {model_type} models
            - {model_type} models don't use epochs
            - Use cross-validation instead for evaluation
            """.format(
                model_type="tree-based" if is_tree_based(model_name) else "iterative"
            ))
        
        # Max Iterations
        if is_iterative(model_name):
            explanations.append("""
            ✅ **Max Iterations**: Shown
            - Iterative models need a convergence limit
            - Controls maximum optimization iterations
            """)
        else:
            explanations.append("""
            ❌ **Max Iterations**: Hidden
            - Not applicable for {model_type} models
            - {model_type} models don't use iterations
            """.format(
                model_type="deep learning" if is_deep_learning(model_name) else "tree-based"
            ))
        
        # Cross-Validation Folds
        if not is_deep_learning(model_name):
            explanations.append("""
            ✅ **K-Fold Cross-Validation**: Shown
            - ML models benefit from cross-validation
            - Tests model on multiple data splits
            - Gives more reliable performance estimate
            """)
        else:
            explanations.append("""
            ❌ **K-Fold Cross-Validation**: Hidden
            - Not applicable for deep learning models
            - DL models use train/validation/test split instead
            - Early stopping monitors validation performance
            """)
        
        # Batch Size
        if is_deep_learning(model_name):
            explanations.append("""
            ✅ **Batch Size**: Shown
            - Deep learning processes data in batches
            - Smaller batches = more frequent updates
            - Larger batches = faster training
            """)
        else:
            explanations.append("""
            ❌ **Batch Size**: Hidden
            - Not applicable for ML models
            - ML models process all data at once
            - No batching needed
            """)
        
        # Learning Rate
        if is_deep_learning(model_name) or is_iterative(model_name):
            explanations.append("""
            ✅ **Learning Rate**: Shown
            - Controls how fast the model learns
            - Higher = faster learning (but may be unstable)
            - Lower = slower learning (but more stable)
            """)
        else:
            explanations.append("""
            ❌ **Learning Rate**: Hidden
            - Not applicable for tree-based models
            - Tree models don't use gradient descent
            """)
        
        return "\n".join(explanations)
    
    @staticmethod
    def log_cv_explanation(model_name, cv_folds):
        """Explain why cross-validation was used."""
        
        if is_deep_learning(model_name):
            return """
            ℹ️ **Why No Cross-Validation for Deep Learning?**
            
            Deep learning models use a different approach:
            - **Train Set**: Used to train the model
            - **Validation Set**: Used to monitor performance during training
            - **Test Set**: Used to evaluate final performance
            
            This is better for DL because:
            1. DL models are computationally expensive
            2. K-fold CV would require training k models (too slow)
            3. Train/val/test split is more practical
            4. Early stopping uses validation set to prevent overfitting
            """
        
        else:
            return f"""
            ℹ️ **Why Cross-Validation for ML Models?**
            
            We're using {cv_folds}-fold cross-validation because:
            
            1. **More Reliable**: Tests on multiple data splits
            2. **Prevents Luck**: One good split might be lucky
            3. **Better Estimate**: Average of {cv_folds} tests is more trustworthy
            4. **Detects Overfitting**: If CV score is much lower than train score
            
            **How it works:**
            - Data split into {cv_folds} equal parts
            - Train on {cv_folds-1} parts, test on 1 part
            - Repeat {cv_folds} times with different test parts
            - Report average performance
            
            **Example with 5-fold CV:**
            - Fold 1: Train on parts 2-5, test on part 1
            - Fold 2: Train on parts 1,3-5, test on part 2
            - Fold 3: Train on parts 1-2,4-5, test on part 3
            - Fold 4: Train on parts 1-3,5, test on part 4
            - Fold 5: Train on parts 1-4, test on part 5
            - Final score = average of all 5 test scores
            """
    
    @staticmethod
    def log_training_start(model_name, task_type, params_dict):
        """Log training start with all decisions."""
        
        log_text = f"""
        🚀 **TRAINING STARTED**
        
        {TrainingLogger.log_model_selection(model_name, task_type)}
        
        {TrainingLogger.log_strategy_decision(model_name)}
        
        **Parameters Used:**
        """
        
        for key, value in params_dict.items():
            log_text += f"\n- {key.replace('_', ' ').title()}: {value}"
        
        return log_text
    
    @staticmethod
    def log_training_complete(model_name, metrics):
        """Log training completion with results."""
        
        log_text = """
        ✅ **TRAINING COMPLETED**
        
        **Final Performance:**
        """
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_text += f"\n- {key.replace('_', ' ').title()}: {value:.4f}"
        
        return log_text
    
    @staticmethod
    def display_training_log(log_text):
        """Display training log in Streamlit."""
        with st.expander("📋 Training Log", expanded=False):
            st.markdown(log_text)


def log_training_decision(model_name, task_type, cv_folds=None, epochs=None):
    """
    Log complete training decision in user-friendly format.
    
    Returns:
        formatted_log: String with complete explanation
    """
    
    log_parts = []
    
    # Model selection
    log_parts.append(f"**Model**: {model_name.replace('_', ' ').title()}")
    log_parts.append(f"**Task**: {task_type}")
    
    # Strategy
    if is_deep_learning(model_name):
        log_parts.append("**Strategy**: Deep Learning with Epochs")
        log_parts.append(f"- Epochs: {epochs} (multiple passes through data)")
        log_parts.append("- Validation: Monitored during training")
        log_parts.append("- Early Stopping: Stops if validation loss plateaus")
    
    elif is_iterative(model_name):
        log_parts.append("**Strategy**: Iterative ML with Cross-Validation")
        log_parts.append(f"- Max Iterations: Convergence limit")
        log_parts.append(f"- Cross-Validation: {cv_folds}-fold")
        log_parts.append("- Reason: Tests model on multiple data splits")
    
    else:  # tree-based
        log_parts.append("**Strategy**: Tree-Based ML with Cross-Validation")
        log_parts.append(f"- Cross-Validation: {cv_folds}-fold")
        log_parts.append("- Reason: Ensures robust evaluation")
    
    return "\n".join(log_parts)
