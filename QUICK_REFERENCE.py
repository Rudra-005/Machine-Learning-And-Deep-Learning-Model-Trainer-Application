#!/usr/bin/env python
"""
ML/DL TRAINER - QUICK REFERENCE CARD

How to use the complete ML/DL training system
"""

# ============================================================================
# 1. INSTALLATION
# ============================================================================

# Install all dependencies:
# pip install -r requirements.txt

# ============================================================================
# 2. QUICK START (Streamlit App)
# ============================================================================

# Launch the interactive dashboard:
# streamlit run app.py

# Then:
# 1. Upload CSV or load sample data
# 2. Configure preprocessing options
# 3. Select model type and hyperparameters
# 4. Click "Train Model"
# 5. View evaluation metrics
# 6. Download trained model

# ============================================================================
# 3. PROGRAMMATIC USAGE
# ============================================================================

# Example 1: Complete ML Workflow
# ─────────────────────────────────

from data_preprocessing import preprocess_dataset
from models.model_factory import ModelFactory
from train import train_model
from evaluate import evaluate_model
import pandas as pd

# Load and preprocess data
df = pd.read_csv('data.csv')
X_train, X_val, X_test, y_train, y_val, y_test, prep = preprocess_dataset(
    df,  # Now accepts DataFrame directly!
    target_col='target',
    test_size=0.2,
    val_size=0.1
)

# Create model
model = ModelFactory.create_model(
    'classification',  # or 'regression'
    'random_forest',   # or 'svm', 'logistic_regression', 'neural_network'
    n_estimators=200,
    max_depth=15
)

# Train model
trained_model, history = train_model(
    model, X_train, y_train,
    X_val=X_val, y_val=y_val
)

# Evaluate model
metrics = evaluate_model(
    trained_model, X_test, y_test,
    task_type='classification'
)

# Print results
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")


# Example 2: Neural Network Training
# ────────────────────────────────────

from models.model_factory import ModelFactory
from train import train_model

nn = ModelFactory.create_model(
    'classification',
    'neural_network',
    input_dim=20,           # Number of features
    num_classes=3,          # Number of output classes
    hidden_layers=[256, 128, 64],
    dropout_rate=0.2
)

trained_nn, nn_history = train_model(
    nn, X_train, y_train,
    X_val=X_val, y_val=y_val,
    epochs=100,
    batch_size=32,
    verbose=1
)


# Example 3: Model Evaluation & Visualization
# ─────────────────────────────────────────────

from evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    generate_evaluation_report,
    save_metrics_json
)

metrics = evaluate_model(trained_model, X_test, y_test, 'classification')

# Visualize
plot_confusion_matrix(y_test, metrics['y_pred'], 
                     save_path='confusion_matrix.png')
plot_roc_curve(y_test, trained_model.predict_proba(X_test)[:, 1],
              save_path='roc_curve.png')

# Generate report
generate_evaluation_report(metrics, 'classification', 'RandomForest',
                          save_path='report.txt')

# Save metrics
save_metrics_json(metrics, 'metrics.json')


# ============================================================================
# 4. AVAILABLE MODELS
# ============================================================================

# Classification Models
# ─────────────────────
# - logistic_regression
# - random_forest
# - svm (Support Vector Machine)
# - neural_network

# Regression Models
# ─────────────────
# - linear_regression
# - random_forest
# - svm
# - neural_network


# ============================================================================
# 5. HYPERPARAMETER TUNING
# ============================================================================

# Random Forest
# ─────────────
clf = ModelFactory.create_model(
    'classification', 'random_forest',
    n_estimators=200,      # Number of trees
    max_depth=15,          # Max tree depth
    min_samples_split=5,   # Min samples to split
    min_samples_leaf=2,    # Min samples in leaf
    random_state=42
)

# SVM
# ───
clf = ModelFactory.create_model(
    'classification', 'svm',
    C=1.0,                 # Regularization strength
    kernel='rbf',          # 'rbf', 'linear', 'poly'
    gamma='scale'
)

# Logistic Regression
# ────────────────────
clf = ModelFactory.create_model(
    'classification', 'logistic_regression',
    max_iter=1000,
    solver='lbfgs',        # 'lbfgs', 'liblinear', 'saga'
    random_state=42
)

# Neural Network
# ──────────────
nn = ModelFactory.create_model(
    'classification', 'neural_network',
    input_dim=20,
    num_classes=3,
    hidden_layers=[256, 128, 64],
    dropout_rate=0.2,
    activation='relu',
    output_activation='softmax'
)


# ============================================================================
# 6. METRICS AVAILABLE
# ============================================================================

# Classification Metrics
# ──────────────────────
metrics = evaluate_model(model, X_test, y_test, 'classification')
print(metrics['accuracy'])      # 0-1 score
print(metrics['precision'])     # 0-1 score
print(metrics['recall'])        # 0-1 score
print(metrics['f1'])            # 0-1 score
print(metrics['roc_auc'])       # 0-1 score (if probabilities available)
print(metrics['confusion_matrix'])  # Matrix representation


# Regression Metrics
# ──────────────────
metrics = evaluate_model(model, X_test, y_test, 'regression')
print(metrics['mae'])           # Mean Absolute Error
print(metrics['mse'])           # Mean Squared Error
print(metrics['rmse'])          # Root Mean Squared Error
print(metrics['r2'])            # R² Score (0-1)
print(metrics['mape'])          # Mean Absolute Percentage Error


# ============================================================================
# 7. SAVING & LOADING MODELS
# ============================================================================

import joblib

# Save model
joblib.dump(trained_model, 'my_model.pkl')

# Load model
loaded_model = joblib.load('my_model.pkl')

# For Keras models:
# trained_nn.save('my_neural_network.h5')
# loaded_nn = tf.keras.models.load_model('my_neural_network.h5')


# ============================================================================
# 8. TESTING
# ============================================================================

# Run integration tests to verify everything works:
# python test_integration.py

# This tests:
# ✓ Data preprocessing with DataFrame input
# ✓ Model factory creation (5 models)
# ✓ Model training (sklearn)
# ✓ Model evaluation (metrics)


# ============================================================================
# 9. TROUBLESHOOTING
# ============================================================================

# Issue: Import errors
# Solution: pip install -r requirements.txt

# Issue: StringIO error (FIXED)
# Solution: Now pass DataFrames directly instead of StringIO objects
# Example: preprocess_dataset(df, target_col='target')

# Issue: Memory errors with large datasets
# Solution: Reduce test_size, val_size parameters

# Issue: Model not improving
# Solution: Try different models or adjust hyperparameters

# For more help, see:
# - SETUP_GUIDE.md
# - FIX_SUMMARY.md
# - STATUS_REPORT.md


# ============================================================================
# 10. FILE STRUCTURE
# ============================================================================

# ML_DL_Trainer/
# ├── app.py                      # Main Streamlit app
# ├── data_preprocessing.py       # Data handling
# ├── train.py                    # Training pipeline
# ├── evaluate.py                 # Evaluation & metrics
# ├── models/
# │   └── model_factory.py        # Model creation
# ├── test_integration.py         # Integration tests
# ├── requirements.txt            # Dependencies
# ├── SETUP_GUIDE.md             # Usage guide
# ├── FIX_SUMMARY.md             # Technical fix details
# └── STATUS_REPORT.md           # Project status


# ============================================================================
# 11. COMMON COMMANDS
# ============================================================================

# Run Streamlit app:
# streamlit run app.py

# Run integration tests:
# python test_integration.py

# Check Python version:
# python --version

# Install/upgrade dependencies:
# pip install -r requirements.txt --upgrade


# ============================================================================
# 12. KEY IMPROVEMENTS IN THIS VERSION
# ============================================================================

# ✓ StringIO error FIXED - now accepts DataFrames directly
# ✓ 8+ pre-configured models ready to use
# ✓ Comprehensive evaluation metrics (10+)
# ✓ Beautiful Plotly visualizations
# ✓ Interactive Streamlit dashboard
# ✓ Model export & download functionality
# ✓ Full type hints and documentation
# ✓ Production-ready code quality
# ✓ Integration tests for verification


if __name__ == "__main__":
    print(__doc__)
