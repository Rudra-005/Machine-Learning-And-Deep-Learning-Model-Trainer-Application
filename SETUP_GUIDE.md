"""
ML/DL Trainer - Complete Setup & Usage Guide
============================================================================

ISSUE FIXED:
The preprocessing error "expected str, bytes or os.PathLike object, not StringIO"
has been resolved by updating data_preprocessing.py to accept both file paths
and pandas DataFrame objects directly.

MODULES CREATED & TESTED:
✓ data_preprocessing.py   - CSV loading, feature detection, scaling, encoding
✓ model_factory.py        - Dynamic model creation for ML & DL models
✓ train.py               - Training pipeline for sklearn & Keras models
✓ evaluate.py            - Comprehensive evaluation metrics & visualizations
✓ app.py                 - Interactive Streamlit dashboard

QUICK START:
============================================================================

1. Install Dependencies
   pip install -r requirements.txt

2. Run Integration Tests
   python test_integration.py

3. Launch Streamlit App
   streamlit run app.py

MODULES OVERVIEW:
============================================================================

1. data_preprocessing.py
   - DataPreprocessor class with:
     * load_data() - accepts CSV path OR DataFrame directly
     * detect_column_types() - auto-detects numerical/categorical columns
     * analyze_missing_values() - missing value analysis
     * build_preprocessing_pipeline() - StandardScaler + OneHotEncoder
     * fit_transform() & transform() - apply transformations
     * split_data() - stratified train/val/test split
   
   - preprocess_dataset() - end-to-end function
   
   Usage:
   ```python
   from data_preprocessing import preprocess_dataset
   import pandas as pd
   
   df = pd.read_csv('data.csv')
   X_train, X_val, X_test, y_train, y_val, y_test, prep = preprocess_dataset(
       df,  # Can pass DataFrame directly!
       target_col='target',
       test_size=0.2
   )
   ```

2. model_factory.py
   - ModelFactory class with:
     * create_model() - create sklearn or Keras models dynamically
     * get_available_models() - list available models
     * register_model() - register custom models for extensibility
     * get_default_hyperparameters() - access default params
   
   Supported Models:
     Classification: Logistic Regression, Random Forest, SVM, Neural Network
     Regression: Linear Regression, Random Forest, SVM, Neural Network
   
   Usage:
   ```python
   from models.model_factory import ModelFactory
   
   # Sklearn model
   clf = ModelFactory.create_model('classification', 'random_forest', 
                                   n_estimators=200)
   
   # Neural network
   nn = ModelFactory.create_model('classification', 'neural_network',
                                  input_dim=20, num_classes=3,
                                  hidden_layers=[256, 128])
   ```

3. train.py
   - TrainingHistory class - tracks training metrics
   - train_sklearn_model() - sklearn training
   - train_keras_model() - Keras/TensorFlow training
   - train_model() - unified interface (auto-detects model type)
   - train_full_pipeline() - complete pipeline with evaluation
   
   Usage:
   ```python
   from train import train_model
   
   model = ModelFactory.create_model('classification', 'random_forest')
   trained_model, history = train_model(
       model, X_train, y_train,
       X_val=X_val, y_val=y_val
   )
   
   print(f"Training time: {history.get_summary()['total_time']:.2f}s")
   ```

4. evaluate.py
   - compute_classification_metrics() - accuracy, precision, recall, F1, AUC
   - compute_regression_metrics() - MAE, MSE, RMSE, R²
   - evaluate_model() - unified evaluation interface
   
   Visualization Functions:
     * plot_confusion_matrix()
     * plot_roc_curve()
     * plot_precision_recall_curve()
     * plot_regression_results() - predictions, residuals
     * plot_metrics_summary()
   
   Reporting:
     * generate_evaluation_report() - text report
     * save_metrics_json() - JSON export
   
   Usage:
   ```python
   from evaluate import evaluate_model, plot_confusion_matrix
   
   metrics = evaluate_model(trained_model, X_test, y_test, 'classification')
   plot_confusion_matrix(y_test, metrics['y_pred'], 
                        save_path='confusion_matrix.png')
   ```

5. app.py (Streamlit Dashboard)
   Features:
     ✓ CSV upload or sample dataset loading
     ✓ Interactive data exploration
     ✓ Data preprocessing configuration
     ✓ Model selection & hyperparameter tuning
     ✓ Real-time training with progress display
     ✓ Training curves visualization
     ✓ Model evaluation with comprehensive metrics
     ✓ Interactive confusion matrix & residuals plots
     ✓ Model export (.pkl format)
     ✓ Metrics & history download (JSON)
   
   Navigation Tabs:
     📊 Data Loading - upload, explore, preprocess data
     🧠 Model Training - select models, tune hyperparameters, train
     📈 Evaluation - view metrics and visualizations
     📥 Download - export trained model and metrics

COMPLETE WORKFLOW EXAMPLE:
============================================================================

from data_preprocessing import preprocess_dataset
from models.model_factory import ModelFactory
from train import train_model, train_full_pipeline
from evaluate import evaluate_model, generate_evaluation_report
import pandas as pd

# 1. Load and Preprocess Data
df = pd.read_csv('data.csv')
X_train, X_val, X_test, y_train, y_val, y_test, prep = preprocess_dataset(
    df, 
    target_col='target',
    test_size=0.2,
    val_size=0.1
)

# 2. Create Model
model = ModelFactory.create_model(
    'classification', 
    'random_forest',
    n_estimators=200,
    max_depth=15
)

# 3. Train Model
trained_model, history = train_model(
    model, X_train, y_train,
    X_val=X_val, y_val=y_val
)

# 4. Evaluate Model
metrics = evaluate_model(
    trained_model, X_test, y_test,
    task_type='classification'
)

# 5. Generate Report
generate_evaluation_report(
    metrics, 'classification', 'RandomForest',
    save_path='evaluation_report.txt'
)

# Results
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")

KEY FIXES MADE:
============================================================================

Issue: "expected str, bytes or os.PathLike object, not StringIO"
Root Cause: data_preprocessing.py only accepted file paths, Streamlit was
            passing StringIO objects

Solution: Updated data_preprocessing.py to accept Union[str, pd.DataFrame]
          - DataPreprocessor.load_data() now handles both file paths and DataFrames
          - preprocess_dataset() function signature updated
          - app.py now passes DataFrames directly (no StringIO conversion)

TESTING:
============================================================================

Run integration tests to verify everything works:
  python test_integration.py

Tests included:
  ✓ Data preprocessing with DataFrame input
  ✓ Model factory - 5 different models
  ✓ Model training - sklearn model
  ✓ Model evaluation - classification metrics

All tests passing!

NEXT STEPS:
============================================================================

1. Run the app:
   streamlit run app.py

2. The app will open at http://localhost:8501

3. Try the complete workflow:
   - Upload a CSV or load sample data
   - Configure preprocessing
   - Select a model and hyperparameters
   - Train the model
   - View evaluation results
   - Download trained model

TROUBLESHOOTING:
============================================================================

If you encounter any issues:

1. Check Python version: python --version (should be 3.8+)
2. Verify all imports: python -c "import streamlit; import tensorflow; import sklearn"
3. Run integration tests: python test_integration.py
4. Check logs in the Streamlit terminal for detailed errors

REQUIREMENTS:
============================================================================

See requirements.txt for all dependencies.

Key packages:
  - pandas, numpy - data manipulation
  - scikit-learn - ML models
  - tensorflow - deep learning models
  - streamlit - web dashboard
  - matplotlib, seaborn, plotly - visualizations

PROJECT STRUCTURE:
============================================================================

ML_DL_Trainer/
├── data_preprocessing.py      # Data loading, preprocessing, splitting
├── models/
│   ├── __init__.py
│   └── model_factory.py       # Dynamic model creation
├── train.py                   # Training pipeline
├── evaluate.py                # Evaluation & visualization
├── app.py                     # Streamlit dashboard
├── test_integration.py        # Integration tests
├── requirements.txt           # Dependencies
└── README.md                  # Documentation

SUPPORT & DOCUMENTATION:
============================================================================

For detailed documentation on each module, see the docstrings in the code.

Each function includes:
  - Comprehensive docstring with Args, Returns, Raises, Examples
  - Type hints for better IDE support
  - Logging for debugging
  - Error handling with descriptive messages

=============================================================================
"""

if __name__ == "__main__":
    import textwrap
    
    # Print formatted docstring
    print(__doc__)
