"""
Integration Test - Verify all modules work together
"""

import pandas as pd
import numpy as np
import sys

print("=" * 70)
print("ML/DL Trainer Integration Test")
print("=" * 70)

# Test 1: Data Preprocessing
print("\n[1/4] Testing Data Preprocessing...")
try:
    from data_preprocessing import preprocess_dataset
    
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.randn(100) * 50000 + 50000,
        'score': np.random.randint(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.randint(0, 2, 100)
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test, prep = preprocess_dataset(
        df, 
        target_col='target',
        test_size=0.2,
        val_size=0.1
    )
    
    print("✓ Data preprocessing successful")
    print(f"  - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
except Exception as e:
    print(f"✗ Data preprocessing failed: {str(e)}")
    sys.exit(1)

# Test 2: Model Factory
print("\n[2/4] Testing Model Factory...")
try:
    from models.model_factory import ModelFactory
    
    # Test classification models
    clf_rf = ModelFactory.create_model('classification', 'random_forest', n_estimators=50)
    clf_lr = ModelFactory.create_model('classification', 'logistic_regression')
    clf_svm = ModelFactory.create_model('classification', 'svm')
    
    # Test regression models
    reg_rf = ModelFactory.create_model('regression', 'random_forest', n_estimators=50)
    reg_lr = ModelFactory.create_model('regression', 'linear_regression')
    
    print("✓ Model factory successful")
    print("  - Created 5 models (3 classification, 2 regression)")
    
except Exception as e:
    print(f"✗ Model factory failed: {str(e)}")
    sys.exit(1)

# Test 3: Training
print("\n[3/4] Testing Model Training...")
try:
    from train import train_model
    
    trained_model, history = train_model(
        clf_rf, X_train, y_train,
        X_val=X_val, y_val=y_val
    )
    
    print("✓ Model training successful")
    print(f"  - Training time: {history.get_summary()['total_time']:.2f}s")
    print(f"  - Model type: {history.get_summary()['model_type']}")
    
except Exception as e:
    print(f"✗ Model training failed: {str(e)}")
    sys.exit(1)

# Test 4: Evaluation
print("\n[4/4] Testing Model Evaluation...")
try:
    from evaluate import evaluate_model
    
    metrics = evaluate_model(
        trained_model,
        X_test,
        y_test,
        task_type='classification'
    )
    
    print("✓ Model evaluation successful")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    
except Exception as e:
    print(f"✗ Model evaluation failed: {str(e)}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL INTEGRATION TESTS PASSED!")
print("=" * 70)
print("\nYou can now run the Streamlit app with:")
print("  streamlit run app.py")
