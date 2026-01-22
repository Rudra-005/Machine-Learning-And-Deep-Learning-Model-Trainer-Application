# Deep Learning Training Pipeline

Production-ready deep learning training completely isolated from ML pipeline.

## Overview

The DL training system provides:
- ✅ **Epochs ONLY** - Never shown for ML models
- ✅ **Batch Size** - Configurable (16, 32, 64, 128, 256)
- ✅ **Learning Rate** - User-defined (0.0001-0.1)
- ✅ **Early Stopping** - Prevents overfitting
- ✅ **Training & Validation Loss** - Displayed with curves
- ✅ **Isolated Pipeline** - Completely separate from ML
- ✅ **Multiple Architectures** - Sequential, CNN, RNN

## Architecture

```
models/
├── dl_trainer.py                # Core DL training logic
└── __init__.py

app/utils/
├── dl_streamlit.py              # Streamlit integration
└── __init__.py

examples/
└── dl_training_example.py       # Usage examples
```

## Core Components

### 1. DLTrainer Class

```python
from models.dl_trainer import DLTrainer

# Build Sequential model
model = DLTrainer.build_sequential_model(input_dim=10, output_dim=3, task_type='classification')

# Build CNN model
model = DLTrainer.build_cnn_model(input_shape=(28, 28, 1), output_dim=10, task_type='classification')

# Build RNN model
model = DLTrainer.build_rnn_model(input_shape=(100, 1), output_dim=10, task_type='classification')

# Train with epochs
history, trained_model = DLTrainer.train_dl_model(
    model, X_train, y_train, X_val, y_val,
    epochs=50, batch_size=32, learning_rate=0.001, early_stopping=True
)
```

### 2. Streamlit Integration

```python
from app.utils.dl_streamlit import render_dl_config, train_dl_model, display_dl_training_results

# Render UI
epochs, batch_size, learning_rate, early_stopping = render_dl_config('sequential')

# Train
trained_model, history, predictions = train_dl_model(
    'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
    'classification', epochs, batch_size, learning_rate, early_stopping
)

# Display results
display_dl_training_results(history, predictions, y_test, 'classification')
```

## Supported Architectures

### Sequential Neural Network
```python
model = DLTrainer.build_sequential_model(input_dim=10, output_dim=3, task_type='classification')
# Layers: Dense(128) → Dropout → Dense(64) → Dropout → Dense(32) → Dense(output)
```

### Convolutional Neural Network (CNN)
```python
model = DLTrainer.build_cnn_model(input_shape=(28, 28, 1), output_dim=10, task_type='classification')
# Layers: Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → Flatten → Dense
```

### Recurrent Neural Network (RNN/LSTM)
```python
model = DLTrainer.build_rnn_model(input_shape=(100, 1), output_dim=10, task_type='classification')
# Layers: LSTM → Dropout → LSTM → Dropout → Dense
```

## Parameters

### Epochs
- **Definition**: Number of complete passes through training data
- **Range**: 1-500
- **Default**: 50
- **Impact**: More epochs = longer training, potential overfitting

### Batch Size
- **Definition**: Number of samples per gradient update
- **Options**: 16, 32, 64, 128, 256
- **Default**: 32
- **Impact**: Larger batches = faster training, less memory

### Learning Rate
- **Definition**: Optimizer step size
- **Range**: 0.0001-0.1
- **Default**: 0.001
- **Impact**: Higher LR = faster convergence, potential instability

### Early Stopping
- **Definition**: Stop training if validation loss plateaus
- **Patience**: 5 epochs
- **Benefit**: Prevents overfitting, saves training time

## Usage Examples

### Example 1: Basic Sequential Training

```python
from models.dl_trainer import DLTrainer, prepare_dl_data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Prepare data
X_train, y_train, X_val, y_val, X_test, y_test = prepare_dl_data(
    X_train, y_train, X_val, y_val, X_test, y_test, 'classification'
)

# Build model
model = DLTrainer.build_sequential_model(X_train.shape[1], 3, 'classification')

# Train with epochs
history, trained_model = DLTrainer.train_dl_model(
    model, X_train, y_train, X_val, y_val,
    epochs=50, batch_size=32, learning_rate=0.001, early_stopping=True
)

print(f"Final Loss: {history.history['loss'][-1]:.4f}")
```

### Example 2: Streamlit Integration

```python
import streamlit as st
from app.utils.dl_streamlit import render_dl_config, train_dl_model

# Render UI
epochs, batch_size, learning_rate, early_stopping = render_dl_config('sequential')

if st.button("Train"):
    trained_model, history, predictions = train_dl_model(
        'sequential', X_train, y_train, X_val, y_val, X_test, y_test,
        'classification', epochs, batch_size, learning_rate, early_stopping
    )
```

## Output Format

### Training History
```python
history.history = {
    'loss': [0.8, 0.6, 0.4, ...],           # Training loss per epoch
    'val_loss': [0.85, 0.65, 0.45, ...],    # Validation loss per epoch
    'accuracy': [0.6, 0.7, 0.8, ...],       # Training accuracy
    'val_accuracy': [0.58, 0.68, 0.78, ...]  # Validation accuracy
}
```

### Display Output
```
Training Results
├── Training & Validation Loss (curve)
├── Performance Metrics
│   ├── Accuracy: 0.9200
│   ├── Precision: 0.9100
│   └── Recall: 0.9300
└── Training Summary
    ├── Final Training Loss: 0.1234
    ├── Final Validation Loss: 0.1456
    ├── Total Epochs: 45
    └── Best Epoch: 40
```

## Isolation from ML Pipeline

### ML Pipeline
```
K-Fold Cross-Validation
↓
max_iter (convergence iterations)
↓
Single pass through data
↓
No epochs
```

### DL Pipeline
```
Train/Val/Test Split
↓
Epochs (multiple passes)
↓
Batch-based updates
↓
Early stopping
```

**Key Difference**: ML trains ONCE with convergence iterations. DL trains MULTIPLE times (epochs) on batches.

## Loss Curves Interpretation

### Healthy Training
```
Loss decreases smoothly
Training loss < Validation loss (slight gap)
No sudden spikes
```

### Overfitting
```
Training loss continues decreasing
Validation loss plateaus or increases
Large gap between training and validation
```

### Underfitting
```
Both losses remain high
No significant improvement
Model too simple
```

## Early Stopping Details

- **Monitor**: Validation loss
- **Patience**: 5 epochs without improvement
- **Action**: Restore best weights and stop
- **Benefit**: Prevents overfitting, saves time

## Testing

Run examples:
```bash
streamlit run examples/dl_training_example.py
```

Test scenarios:
1. Epochs vs max_iter explanation
2. Sequential training
3. Loss curves visualization
4. Batch size impact
5. Learning rate impact
6. Early stopping effect

## Files

- `models/dl_trainer.py` - Core logic (150 lines)
- `app/utils/dl_streamlit.py` - Streamlit integration (100 lines)
- `examples/dl_training_example.py` - Usage examples (250 lines)
- `DL_TRAINING_README.md` - This documentation

## Benefits

✅ **Epochs ONLY for DL** - Never shown for ML models
✅ **Complete Isolation** - Separate from ML pipeline
✅ **Loss Visualization** - Training & validation curves
✅ **Early Stopping** - Prevents overfitting
✅ **Multiple Architectures** - Sequential, CNN, RNN
✅ **Configurable** - Epochs, batch size, learning rate
✅ **Minimal Code** - ~500 lines total
✅ **Production Ready** - Error handling included

## Integration Checklist

- [x] Epochs shown ONLY for DL models
- [x] Batch size configurable (16-256)
- [x] Learning rate user-defined (0.0001-0.1)
- [x] Early stopping implemented
- [x] Training & validation loss displayed
- [x] Loss curves visualized
- [x] DL pipeline isolated from ML
- [x] Multiple architectures supported
- [x] Streamlit integration complete
- [x] Examples provided
