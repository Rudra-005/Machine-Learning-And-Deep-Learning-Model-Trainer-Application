"""
AutoML Mode Examples: Demonstrates automatic model detection and strategy selection.

Shows how different models automatically get the right training approach.
"""

import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR

from models.automl import AutoMLConfig, detect_model_category, get_strategy_explanation
from models.automl_trainer import train_with_automl


def example_1_tree_based_classification():
    """Example 1: Tree-based model (Random Forest) - Auto-detects K-Fold CV."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Tree-Based Classification (Random Forest)")
    print("="*70)
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Create model
    model = RandomForestClassifier(random_state=42)
    
    # AutoML detection
    automl = AutoMLConfig(model)
    print(f"\n✓ Model Detected: {automl.config['model_name']}")
    print(f"✓ Category: {automl.config['category']}")
    print(f"✓ Strategy: {automl.config['description']}")
    print(f"✓ Visible Parameters: {[k for k, v in automl.visible_params.items() if v]}")
    
    # Train with AutoML
    print("\n🚀 Training with AutoML...")
    results = train_with_automl(
        model, X_train, y_train, X_test, y_test,
        params={'cv_folds': 5, 'enable_hp_tuning': False}
    )
    
    print(f"\n📊 Results:")
    print(f"   CV Mean Score: {results['cv_mean']:.4f}")
    print(f"   CV Std Dev: {results['cv_std']:.4f}")
    print(f"   Test Score: {results['test_score']:.4f}")
    print(f"   Strategy Used: {results['strategy']}")


def example_2_iterative_classification():
    """Example 2: Iterative model (Logistic Regression) - Auto-detects CV + max_iter."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Iterative Classification (Logistic Regression)")
    print("="*70)
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Create model
    model = LogisticRegression(random_state=42)
    
    # AutoML detection
    automl = AutoMLConfig(model)
    print(f"\n✓ Model Detected: {automl.config['model_name']}")
    print(f"✓ Category: {automl.config['category']}")
    print(f"✓ Strategy: {automl.config['description']}")
    print(f"✓ Visible Parameters: {[k for k, v in automl.visible_params.items() if v]}")
    
    # Train with AutoML
    print("\n🚀 Training with AutoML...")
    results = train_with_automl(
        model, X_train, y_train, X_test, y_test,
        params={'cv_folds': 5, 'max_iter': 1000, 'enable_hp_tuning': False}
    )
    
    print(f"\n📊 Results:")
    print(f"   CV Mean Score: {results['cv_mean']:.4f}")
    print(f"   CV Std Dev: {results['cv_std']:.4f}")
    print(f"   Test Score: {results['test_score']:.4f}")
    print(f"   Strategy Used: {results['strategy']}")


def example_3_svm_classification():
    """Example 3: SVM model - Auto-detects K-Fold CV with kernel tuning."""
    print("\n" + "="*70)
    print("EXAMPLE 3: SVM Classification")
    print("="*70)
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Create model
    model = SVC(random_state=42)
    
    # AutoML detection
    automl = AutoMLConfig(model)
    print(f"\n✓ Model Detected: {automl.config['model_name']}")
    print(f"✓ Category: {automl.config['category']}")
    print(f"✓ Strategy: {automl.config['description']}")
    print(f"✓ Visible Parameters: {[k for k, v in automl.visible_params.items() if v]}")
    
    # Train with AutoML
    print("\n🚀 Training with AutoML...")
    results = train_with_automl(
        model, X_train, y_train, X_test, y_test,
        params={'cv_folds': 5, 'enable_hp_tuning': True, 'hp_iterations': 20}
    )
    
    print(f"\n📊 Results:")
    print(f"   CV Mean Score: {results['cv_mean']:.4f}")
    print(f"   CV Std Dev: {results['cv_std']:.4f}")
    print(f"   Test Score: {results['test_score']:.4f}")
    print(f"   Best Params: {results.get('best_params', {})}")
    print(f"   Strategy Used: {results['strategy']}")


def example_4_regression():
    """Example 4: Regression model - Auto-detects K-Fold CV for regression."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Regression (Ridge)")
    print("="*70)
    
    # Load data
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(
        diabetes.data, diabetes.target, test_size=0.2, random_state=42
    )
    
    # Create model
    model = Ridge(random_state=42)
    
    # AutoML detection
    automl = AutoMLConfig(model)
    print(f"\n✓ Model Detected: {automl.config['model_name']}")
    print(f"✓ Category: {automl.config['category']}")
    print(f"✓ Strategy: {automl.config['description']}")
    print(f"✓ Visible Parameters: {[k for k, v in automl.visible_params.items() if v]}")
    
    # Train with AutoML
    print("\n🚀 Training with AutoML...")
    results = train_with_automl(
        model, X_train, y_train, X_test, y_test,
        params={'cv_folds': 5, 'enable_hp_tuning': False}
    )
    
    print(f"\n📊 Results:")
    print(f"   CV Mean Score (R²): {results['cv_mean']:.4f}")
    print(f"   CV Std Dev: {results['cv_std']:.4f}")
    print(f"   Test Score (R²): {results['test_score']:.4f}")
    print(f"   Strategy Used: {results['strategy']}")


def example_5_deep_learning():
    """Example 5: Deep Learning model - Auto-detects Epochs with Early Stopping."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Deep Learning (Sequential Neural Network)")
    print("="*70)
    
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
    except ImportError:
        print("⚠️  TensorFlow not installed. Skipping DL example.")
        return
    
    # Load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Normalize
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    
    # Create model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # AutoML detection
    automl = AutoMLConfig(model)
    print(f"\n✓ Model Detected: {automl.config['model_name']}")
    print(f"✓ Category: {automl.config['category']}")
    print(f"✓ Strategy: {automl.config['description']}")
    print(f"✓ Visible Parameters: {[k for k, v in automl.visible_params.items() if v]}")
    
    # Train with AutoML
    print("\n🚀 Training with AutoML...")
    results = train_with_automl(
        model, X_train, y_train, X_test, y_test,
        params={'epochs': 30, 'batch_size': 16, 'early_stopping': True}
    )
    
    print(f"\n📊 Results:")
    print(f"   Train Loss: {results['train_loss']:.4f}")
    print(f"   Val Loss: {results['val_loss']:.4f}")
    print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"   Strategy Used: {results['strategy']}")


def example_6_parameter_visibility():
    """Example 6: Show parameter visibility for different models."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Parameter Visibility Comparison")
    print("="*70)
    
    models = [
        ("Random Forest", RandomForestClassifier()),
        ("Logistic Regression", LogisticRegression()),
        ("SVM", SVC()),
    ]
    
    print("\n{:<25} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "CV Folds", "Max Iter", "Epochs", "HP Tuning"
    ))
    print("-" * 85)
    
    for name, model in models:
        automl = AutoMLConfig(model)
        visible = automl.visible_params
        
        print("{:<25} {:<15} {:<15} {:<15} {:<15}".format(
            name,
            "✓" if visible['cv_folds'] else "✗",
            "✓" if visible['max_iter'] else "✗",
            "✓" if visible['epochs'] else "✗",
            "✓" if visible['hp_tuning'] else "✗"
        ))


def example_7_strategy_explanation():
    """Example 7: Show strategy explanations for each model."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Strategy Explanations")
    print("="*70)
    
    models = [
        RandomForestClassifier(),
        LogisticRegression(),
        SVC(),
    ]
    
    for model in models:
        automl = AutoMLConfig(model)
        print(f"\n📌 {automl.config['model_name']}:")
        print(f"   {get_strategy_explanation(model)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AutoML MODE EXAMPLES")
    print("="*70)
    
    example_1_tree_based_classification()
    example_2_iterative_classification()
    example_3_svm_classification()
    example_4_regression()
    example_5_deep_learning()
    example_6_parameter_visibility()
    example_7_strategy_explanation()
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
