# AutoML Mode: Visual Reference Guide

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)                   │
│                                                                 │
│  1. Select Task Type (Classification/Regression)               │
│  2. Select Model (from dropdown)                               │
│  3. AutoML Configuration (auto-detected)                       │
│  4. Train Model (click button)                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Layer 1: Model Detection          │
        │  (models/automl.py)                │
        │                                    │
        │  detect_model_category()           │
        │  get_training_config()             │
        │  get_visible_parameters()          │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │  MODEL_REGISTRY Lookup             │
        │                                    │
        │  RandomForest → TREE_BASED         │
        │  LogisticReg → ITERATIVE           │
        │  SVC → SVM                         │
        │  Sequential → DEEP_LEARNING        │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────▼───────────────────┐
        │  STRATEGY_CONFIG Lookup            │
        │                                    │
        │  TREE_BASED → K_FOLD_CV            │
        │  ITERATIVE → K_FOLD_CV_CONVERGENCE │
        │  SVM → K_FOLD_CV                   │
        │  DEEP_LEARNING → EPOCHS_EARLY_STOP │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Layer 2: Training Orchestration   │
        │  (models/automl_trainer.py)        │
        │                                    │
        │  AutoMLTrainer.train()             │
        └────────────────┬───────────────────┘
                         │
        ┌────────────────┴────────────────────────────┐
        │                                             │
        ▼                                             ▼
   ┌─────────────────┐                      ┌──────────────────┐
   │ should_use_cv() │                      │ should_use_epochs()
   │                 │                      │                  │
   │ _train_with_cv()│                      │ _train_with_epochs()
   │                 │                      │                  │
   │ Optional:       │                      │ With:            │
   │ _tune_hp()      │                      │ EarlyStopping    │
   └────────┬────────┘                      └────────┬─────────┘
            │                                        │
            ▼                                        ▼
   ┌─────────────────────────────┐      ┌──────────────────────┐
   │ Results (K-Fold CV)         │      │ Results (Epochs)     │
   │                             │      │                      │
   │ cv_mean: 0.9533             │      │ train_loss: 0.2145   │
   │ cv_std: 0.0245              │      │ val_loss: 0.2389     │
   │ test_score: 0.9667          │      │ test_accuracy: 0.92  │
   │ best_params: {...}          │      │ history: {...}       │
   └────────┬────────────────────┘      └────────┬─────────────┘
            │                                    │
            └────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │  Layer 3: UI Rendering             │
        │  (app/utils/automl_ui.py)          │
        │                                    │
        │  display_automl_results()          │
        │  render_automl_summary()           │
        │  render_automl_comparison()        │
        └────────────────┬───────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────┐
        │  Display Results to User           │
        │                                    │
        │  Metrics, Charts, Best Params      │
        └────────────────────────────────────┘
```

---

## Model Category Decision Tree

```
                    Model Selected
                         │
                         ▼
                  Detect Model Name
                         │
        ┌────────────────┼────────────────┬──────────────┐
        │                │                │              │
        ▼                ▼                ▼              ▼
   RandomForest    LogisticReg         SVC          Sequential
   GradientBoosting SGDClassifier      SVR          CNN
   DecisionTree     Perceptron         LinearSVC    LSTM
   ExtraTree        Ridge              LinearSVR    RNN
        │                │                │              │
        ▼                ▼                ▼              ▼
   TREE_BASED      ITERATIVE            SVM        DEEP_LEARNING
        │                │                │              │
        ▼                ▼                ▼              ▼
   K-Fold CV       K-Fold CV +       K-Fold CV    Epochs +
                   max_iter                       Early Stop
        │                │                │              │
        ▼                ▼                ▼              ▼
   Show: CV        Show: CV, max_iter Show: CV    Show: Epochs,
   Hide: Epochs    Hide: Epochs       Hide: Epochs Batch Size
```

---

## Parameter Visibility Matrix

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Parameter        │ Tree     │ Iterative│ SVM      │ DL       │
├──────────────────┼──────────┼──────────┼──────────┼──────────┤
│ CV Folds         │ ✓ Show   │ ✓ Show   │ ✓ Show   │ ✗ Hide   │
│ Max Iter         │ ✗ Hide   │ ✓ Show   │ ✗ Hide   │ ✗ Hide   │
│ Epochs           │ ✗ Hide   │ ✗ Hide   │ ✗ Hide   │ ✓ Show   │
│ Batch Size       │ ✗ Hide   │ ✗ Hide   │ ✗ Hide   │ ✓ Show   │
│ Learning Rate    │ ✗ Hide   │ ✓ Show   │ ✗ Hide   │ ✓ Show   │
│ HP Tuning        │ ✓ Show   │ ✓ Show   │ ✓ Show   │ ✗ Hide   │
│ Early Stopping   │ ✗ Hide   │ ✗ Hide   │ ✗ Hide   │ ✓ Show   │
└──────────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## Training Strategy Flowchart

### K-Fold Cross-Validation (ML Models)

```
Input: X_train, y_train, X_test, y_test
       cv_folds=5, enable_hp_tuning=False
       │
       ▼
   Setup StratifiedKFold(n_splits=5)
       │
       ├─ Fold 1: Train on [2-5], Validate on [1]
       ├─ Fold 2: Train on [1,3-5], Validate on [2]
       ├─ Fold 3: Train on [1-2,4-5], Validate on [3]
       ├─ Fold 4: Train on [1-3,5], Validate on [4]
       └─ Fold 5: Train on [1-4], Validate on [5]
       │
       ▼
   Compute CV Scores: [0.95, 0.96, 0.94, 0.97, 0.95]
       │
       ▼
   Train on Full X_train, y_train
       │
       ▼
   Evaluate on X_test, y_test
       │
       ▼
   Return:
   {
       'cv_mean': 0.9533,
       'cv_std': 0.0245,
       'test_score': 0.9667
   }
```

### Epochs with Early Stopping (DL Models)

```
Input: X_train, y_train, X_test, y_test
       epochs=50, batch_size=32, early_stopping=True
       │
       ▼
   Setup EarlyStopping(monitor='val_loss', patience=5)
       │
       ├─ Epoch 1: Train Loss=0.5234, Val Loss=0.5123
       ├─ Epoch 2: Train Loss=0.4156, Val Loss=0.4234
       ├─ Epoch 3: Train Loss=0.3245, Val Loss=0.3456
       ├─ ...
       ├─ Epoch 25: Train Loss=0.2145, Val Loss=0.2389
       ├─ Epoch 26: Train Loss=0.2134, Val Loss=0.2401 (no improve)
       ├─ Epoch 27: Train Loss=0.2123, Val Loss=0.2415 (no improve)
       ├─ Epoch 28: Train Loss=0.2112, Val Loss=0.2428 (no improve)
       ├─ Epoch 29: Train Loss=0.2101, Val Loss=0.2441 (no improve)
       ├─ Epoch 30: Train Loss=0.2090, Val Loss=0.2454 (no improve)
       │           ↑ STOP (patience=5 reached)
       │
       ▼
   Evaluate on X_test, y_test
       │
       ▼
   Return:
   {
       'train_loss': 0.2145,
       'val_loss': 0.2389,
       'test_accuracy': 0.9200
   }
```

### Hyperparameter Tuning (Optional)

```
Input: Model, X_train, y_train, X_test, y_test
       enable_hp_tuning=True, hp_iterations=30
       │
       ▼
   Get PARAM_DISTRIBUTIONS for model
   {
       'n_estimators': [50, 100, 200],
       'max_depth': [5, 10, 15, None],
       'min_samples_split': [2, 5, 10]
   }
       │
       ▼
   RandomizedSearchCV(n_iter=30, cv=5)
       │
       ├─ Combination 1: n_est=50, depth=5, split=2
       │  ├─ Fold 1: Score=0.92
       │  ├─ Fold 2: Score=0.93
       │  ├─ Fold 3: Score=0.91
       │  ├─ Fold 4: Score=0.94
       │  └─ Fold 5: Score=0.92
       │  Mean: 0.924
       │
       ├─ Combination 2: n_est=200, depth=15, split=5
       │  Mean: 0.953 ← BEST
       │
       ├─ ...
       │
       └─ Combination 30: ...
       │
       ▼
   Select Best Combination
   best_params = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5}
       │
       ▼
   Return:
   {
       'cv_mean': 0.953,
       'test_score': 0.967,
       'best_params': {...},
       'best_estimator': model
   }
```

---

## User Workflow Diagram

### AutoML Mode Workflow

```
START
  │
  ▼
┌─────────────────────────────────────┐
│ 1. Upload Data                      │
│    - CSV file or sample dataset     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 2. Preprocess Data                  │
│    - Handle missing values          │
│    - Scale features                 │
│    - Encode categories              │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 3. Select Task Type                 │
│    - Classification or Regression   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 4. Select Model                     │
│    - Random Forest                  │
│    - Logistic Regression            │
│    - SVM                            │
│    - Neural Network                 │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 5. AutoML Detects Category          │
│    - Tree-based / Iterative / SVM   │
│    - Deep Learning                  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 6. AutoML Selects Strategy          │
│    - K-Fold CV                      │
│    - Epochs + Early Stop            │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 7. Show Relevant Parameters         │
│    - Only CV for ML                 │
│    - Only Epochs for DL             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 8. User Configures (Optional)       │
│    - CV Folds: 5                    │
│    - HP Tuning: Yes/No              │
│    - Epochs: 50                     │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 9. Click "Train"                    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 10. AutoML Trains                   │
│     - Applies optimal strategy      │
│     - Shows progress                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 11. View Results                    │
│     - CV Score ± Std Dev (ML)       │
│     - Train/Val Loss (DL)           │
│     - Best Hyperparameters          │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│ 12. Download Model & Metrics        │
│     - PKL file                      │
│     - JSON metrics                  │
└────────────┬────────────────────────┘
             │
             ▼
           END
```

---

## Code Organization Diagram

```
ML_DL_Trainer/
│
├── models/
│   ├── automl.py
│   │   ├── ModelCategory (enum)
│   │   ├── TrainingStrategy (enum)
│   │   ├── MODEL_REGISTRY (dict)
│   │   ├── STRATEGY_CONFIG (dict)
│   │   ├── detect_model_category()
│   │   ├── get_training_config()
│   │   ├── get_visible_parameters()
│   │   └── AutoMLConfig (class)
│   │
│   └── automl_trainer.py
│       ├── AutoMLTrainer (class)
│       │   ├── train()
│       │   ├── _train_with_cv()
│       │   ├── _train_with_epochs()
│       │   ├── _tune_hyperparameters()
│       │   └── _get_param_distributions()
│       └── train_with_automl()
│
├── app/
│   ├── utils/
│   │   └── automl_ui.py
│   │       ├── render_automl_mode()
│   │       ├── render_automl_summary()
│   │       ├── render_automl_comparison()
│   │       ├── display_automl_training_progress()
│   │       └── display_automl_results()
│   │
│   └── pages/
│       └── automl_training.py
│           ├── page_automl_training()
│           ├── page_automl_comparison()
│           └── page_automl_guide()
│
├── examples/
│   └── automl_examples.py
│       ├── example_1_tree_based_classification()
│       ├── example_2_iterative_classification()
│       ├── example_3_svm_classification()
│       ├── example_4_regression()
│       ├── example_5_deep_learning()
│       ├── example_6_parameter_visibility()
│       └── example_7_strategy_explanation()
│
└── Documentation/
    ├── AUTOML_DOCUMENTATION.md
    ├── AUTOML_QUICK_REFERENCE.md
    ├── AUTOML_INTEGRATION_GUIDE.md
    ├── AUTOML_IMPLEMENTATION_SUMMARY.md
    ├── AUTOML_COMPLETE_SUMMARY.md
    └── AUTOML_VISUAL_REFERENCE.md (this file)
```

---

## Class Hierarchy Diagram

```
┌─────────────────────────────────────┐
│         AutoMLConfig                │
├─────────────────────────────────────┤
│ - model: Any                        │
│ - config: Dict                      │
│ - visible_params: Dict              │
├─────────────────────────────────────┤
│ + __init__(model)                   │
│ + get_training_params()             │
│ + get_ui_config()                   │
└─────────────────────────────────────┘
           △
           │ uses
           │
┌─────────────────────────────────────┐
│      AutoMLTrainer                  │
├─────────────────────────────────────┤
│ - model: Any                        │
│ - automl: AutoMLConfig              │
│ - config: Dict                      │
├─────────────────────────────────────┤
│ + train()                           │
│ + _train_with_cv()                  │
│ + _train_with_epochs()              │
│ + _tune_hyperparameters()           │
│ + _get_param_distributions()        │
└─────────────────────────────────────┘
           △
           │ uses
           │
┌─────────────────────────────────────┐
│    Streamlit UI Functions           │
├─────────────────────────────────────┤
│ + render_automl_mode()              │
│ + render_automl_summary()           │
│ + render_automl_comparison()        │
│ + display_automl_results()          │
└─────────────────────────────────────┘
```

---

## Data Flow Diagram

```
User Input
    │
    ├─ Model Selection
    │  └─ model_name: str
    │
    ├─ Task Type
    │  └─ task_type: str (Classification/Regression)
    │
    └─ Parameters (Optional)
       ├─ cv_folds: int
       ├─ epochs: int
       ├─ batch_size: int
       ├─ enable_hp_tuning: bool
       └─ hp_iterations: int
           │
           ▼
    ┌──────────────────────────────┐
    │ AutoMLConfig                 │
    │ - Detect category            │
    │ - Get strategy               │
    │ - Get visible params         │
    └──────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────┐
    │ AutoMLTrainer                │
    │ - Route to strategy          │
    │ - Execute training           │
    │ - Return results             │
    └──────────────────────────────┘
           │
           ├─ K-Fold CV Results
           │  ├─ cv_mean
           │  ├─ cv_std
           │  ├─ test_score
           │  └─ best_params (if tuning)
           │
           └─ Epochs Results
              ├─ train_loss
              ├─ val_loss
              ├─ test_accuracy
              └─ history
                  │
                  ▼
           ┌──────────────────────────────┐
           │ Streamlit UI                 │
           │ - Display results            │
           │ - Show metrics               │
           │ - Plot charts                │
           └──────────────────────────────┘
                  │
                  ▼
           User Views Results
```

---

## Strategy Selection Logic

```
Model Instance
    │
    ▼
detect_model_category()
    │
    ├─ Check MODEL_REGISTRY
    │  ├─ Found in TREE_BASED? → ModelCategory.TREE_BASED
    │  ├─ Found in ITERATIVE? → ModelCategory.ITERATIVE
    │  ├─ Found in SVM? → ModelCategory.SVM
    │  └─ Found in DEEP_LEARNING? → ModelCategory.DEEP_LEARNING
    │
    └─ Not found? Check module
       ├─ 'keras' or 'tensorflow'? → DEEP_LEARNING
       ├─ 'sklearn'? → TREE_BASED
       └─ Unknown? → Raise ValueError
           │
           ▼
get_training_config(category)
    │
    ├─ TREE_BASED → {strategy: K_FOLD_CV, cv_folds: 5, use_epochs: False, ...}
    ├─ ITERATIVE → {strategy: K_FOLD_CV_WITH_CONVERGENCE, cv_folds: 5, use_max_iter: True, ...}
    ├─ SVM → {strategy: K_FOLD_CV, cv_folds: 5, use_epochs: False, ...}
    └─ DEEP_LEARNING → {strategy: EPOCHS_WITH_EARLY_STOPPING, epochs: 50, use_epochs: True, ...}
           │
           ▼
get_visible_parameters(config)
    │
    ├─ use_epochs? → Show: epochs, batch_size, learning_rate, early_stopping
    ├─ use_max_iter? → Show: max_iter
    ├─ use_cv? → Show: cv_folds
    └─ use_hp_tuning? → Show: hp_tuning, hp_iterations
           │
           ▼
render_automl_mode()
    │
    └─ Display only visible parameters
```

---

## Performance Comparison

```
Model Type          Strategy              Time (approx)
─────────────────────────────────────────────────────────
Random Forest       K-Fold CV (5 folds)   5-10 seconds
Logistic Reg        K-Fold CV (5 folds)   1-2 seconds
SVM                 K-Fold CV (5 folds)   10-30 seconds
Neural Network      Epochs (50)           30-60 seconds

With HP Tuning:
Random Forest       K-Fold CV + Tuning    2-5 minutes
Logistic Reg        K-Fold CV + Tuning    1-3 minutes
SVM                 K-Fold CV + Tuning    5-15 minutes
```

---

## Summary

This visual reference guide shows:

✅ **System Architecture** - Three-layer design  
✅ **Model Categories** - Decision tree for categorization  
✅ **Parameter Visibility** - Matrix of visible parameters  
✅ **Training Strategies** - Flowcharts for each strategy  
✅ **User Workflow** - Complete workflow diagram  
✅ **Code Organization** - File and class structure  
✅ **Data Flow** - How data flows through system  
✅ **Strategy Selection** - Logic for selecting strategy  
✅ **Performance** - Approximate timing for each model  

Use these diagrams to understand and explain the AutoML system.
