# ML/DL Training Platform

A scalable, production-ready web-based Machine Learning and Deep Learning training platform.

## Features

- **Data Upload & Exploration**: Upload CSV datasets with automatic data quality checks
- **Flexible Model Selection**: Choose from ML (Scikit-learn) or DL (TensorFlow/Keras) models
- **Hyperparameter Configuration**: Tune learning rate, epochs, batch size, and more
- **Automatic Preprocessing**: Missing value imputation, scaling, categorical encoding
- **Model Training**: Single and cross-validation training modes
- **Comprehensive Evaluation**: Classification and regression metrics
- **Visualization**: Confusion matrices, feature importance, residual plots
- **Model Persistence**: Save and load trained models
- **Session Management**: Track multiple training sessions

## Architecture

```
┌─────────────────────────────────────┐
│     Streamlit Frontend UI           │
│  (Data Upload, Config, Training)    │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│   FastAPI Backend (Optional)        │
│  (API Routes, Session Management)   │
└────────────┬────────────────────────┘
             │
     ┌───────┴──────────┬──────────────┬──────────┐
     │                  │              │          │
┌────▼───────┐   ┌────▼────┐   ┌────▼──┐   ┌──▼────┐
│ Preprocessing│   │Model    │   │Eval.  │   │Storage │
│ & Features   │   │Training │   │Metrics│   │Repos   │
└──────────────┘   └─────────┘   └───────┘   └────────┘
```

### Folder Structure

```
ML_DL_Trainer/
├── app/                          # Frontend application
│   ├── main.py                   # Streamlit entry point
│   ├── config.py                 # Configuration
│   └── utils/                    # Utilities
│       ├── file_handler.py
│       ├── logger.py
│       └── validators.py
├── backend/                      # Backend services
│   ├── session_manager.py        # Session management
│   └── task_queue.py             # Async task handling
├── core/                         # Core ML operations
│   ├── preprocessor.py           # Data preprocessing
│   ├── feature_engineer.py       # Feature engineering
│   └── validator.py              # Data validation
├── models/                       # ML/DL models
│   ├── model_factory.py          # Model creation
│   ├── ml/                       # SKL models
│   │   ├── classifier.py
│   │   └── regressor.py
│   └── dl/                       # TensorFlow models
│       ├── cnn_models.py
│       └── rnn_models.py
├── evaluation/                   # Evaluation utilities
│   ├── metrics.py                # Metrics calculation
│   ├── visualizer.py             # Plotting utilities
│   ├── reporter.py               # Report generation
│   └── cross_validator.py        # CV utilities
├── storage/                      # Data persistence
│   ├── model_repository.py       # Model storage
│   ├── result_repository.py      # Results storage
│   └── cache_manager.py          # Caching
├── data/                         # Data directories
│   ├── uploads/                  # User uploaded files
│   ├── preprocessed/             # Processed data
│   ├── models/                   # Trained models
│   └── results/                  # Experiment results
├── tests/                        # Unit tests
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ML_DL_Trainer.git
cd ML_DL_Trainer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set environment variables:
```bash
# Create .env file
echo "DEBUG=False" > .env
echo "LOG_LEVEL=INFO" >> .env
```

## Usage

### Run Streamlit Application

```bash
streamlit run app/main.py
```

The application will open at `http://localhost:8501`

### Workflow

1. **Home Page**: Overview and getting started guide
2. **Data Upload**: Upload CSV file and explore data
3. **Training**: Select model, configure hyperparameters, and train
4. **Results**: View metrics, visualizations, and download model
5. **About**: Platform information

## Supported Models

### Machine Learning (Scikit-learn)

**Classification:**
- **Logistic Regression** - Fast, interpretable, baseline model
- **Random Forest** - Robust ensemble, handles non-linear relationships
- **Support Vector Machine (SVM)** - Excellent for high-dimensional data
- **Gradient Boosting** - Sequential ensemble learning with proven accuracy
- **XGBoost** *(Optional)* - Optimized gradient boosting, industry standard
- **LightGBM** *(Optional)* - Fast boosting with lower memory footprint

**Regression:**
- **Linear Regression** - Simple baseline
- **Random Forest** - Robust ensemble approach
- **Support Vector Regression (SVR)** - For non-linear problems
- **Gradient Boosting** - Sequential boosting for complex patterns
- **XGBoost** *(Optional)* - High-performance boosting
- **LightGBM** *(Optional)* - Memory-efficient boosting

### Deep Learning (TensorFlow/Keras)

- Sequential Neural Networks
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN/LSTM)

### Optional Libraries

| Library | Status | Installation |
|---------|--------|---------------|
| XGBoost | Optional | `pip install xgboost` |
| LightGBM | Optional | `pip install lightgbm` |
| SMOTE | Optional | `pip install imbalanced-learn` |

✅ **Core functionality works without optional libraries** - graceful fallback if not installed

## Model Selection Guide

### When to Use Each Model

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **Logistic Regression** | Baseline, interpretability | Fast, explainable | Limited for complex patterns |
| **Random Forest** | General-purpose | Robust, feature importance | Can overfit with defaults |
| **SVM** | High-dimensional data | Powerful, versatile | Slow on large datasets |
| **Gradient Boosting** | Kaggle competitions, production | High accuracy, handles imbalance | Slower training |
| **XGBoost** | Production ML, tabular data | Industry-standard, optimized | Requires tuning |
| **LightGBM** | Large datasets, fast iteration | Memory-efficient, rapid training | Fewer hyperparameters |
| **Neural Networks** | Complex patterns, images, sequences | Flexible, scalable | Needs more data, tuning |

### Quick Decision Tree

```
┌─ Small dataset (<10K rows)?          → Random Forest
├─ Structured/tabular data?           → XGBoost or LightGBM
├─ Need interpretability?             → Logistic Regression
├─ Imbalanced classification?         → Gradient Boosting (with class weights)
├─ High-dimensional (>1000 features)? → SVM or Neural Network
├─ Images/sequences?                  → Neural Networks (CNN/RNN)
└─ Unsure?                            → Start with Random Forest
```

### Why These Models Were Added

**Gradient Boosting (Built-in)**
- ✅ Scikit-learn native - no extra dependencies
- ✅ Excellent imbalanced dataset support
- ✅ Industry-proven accuracy
- ✅ Reasonable training time for most datasets

**XGBoost (Optional)**
- ✅ 10-20% accuracy improvement over standard GB
- ✅ Industry standard in Kaggle competitions
- ✅ Advanced regularization features
- ✅ Handles missing values automatically
- ⚠️ Separate installation (faster iteration for users without it)

**LightGBM (Optional)**
- ✅ 2-5x faster than XGBoost on large datasets
- ✅ Lower memory requirements
- ✅ Better with millions of rows
- ⚠️ Different hyperparameter meanings (separate installation)

### Factory Pattern for Easy Extension

The `ModelFactory` class enables effortless model addition:

```python
# Adding a new model is just 3 lines:
def build_my_model(**params):
    return MyModel(**params)

ModelFactory.register_model(
    'classification', 'my_model', build_my_model,
    defaults={'param1': value}
)
```

**Benefits:**
- ✅ No UI changes needed - automatically appears in dropdown
- ✅ No train.py or evaluate.py modifications
- ✅ Hyperparameters configurable per-model
- ✅ Graceful fallback if optional libraries missing

## Key Design Patterns

1. **Factory Pattern**: ModelFactory for flexible model creation
2. **Repository Pattern**: Model and result storage
3. **Pipeline Pattern**: Data preprocessing pipeline
4. **Observer Pattern**: Real-time training callbacks
5. **Session Pattern**: User session management

## API Endpoints (FastAPI)

Future endpoints for backend integration:

```
POST   /api/upload               - Upload dataset
POST   /api/train                - Start training
GET    /api/train/{session_id}   - Get training status
GET    /api/results/{session_id} - Get results
GET    /api/models               - List models
GET    /api/models/{model_id}    - Download model
```

## Configuration

Edit `app/config.py` to customize:

```python
MAX_FILE_SIZE = 500 * 1024 * 1024  # Max upload size
DEFAULT_TEST_SIZE = 0.2             # Train-test split
DEFAULT_CV_FOLDS = 5                # Cross-validation folds
DEFAULT_EPOCHS = 50                 # DL epochs
DEFAULT_BATCH_SIZE = 32             # DL batch size
```

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

## Production Deployment

### Docker

```bash
docker-compose up -d
```

### Cloud Deployment

**AWS:**
- Use EC2 for app hosting
- S3 for model/data storage
- RDS for metadata database

**Google Cloud:**
- Cloud Run for serverless deployment
- Cloud Storage for models
- Cloud SQL for database

**Azure:**
- App Service for hosting
- Blob Storage for models
- SQL Database for metadata

## Scalability Path

| Component | Dev | Prod |
|-----------|-----|------|
| Frontend | Streamlit | Streamlit + Load Balancer |
| Backend | Single thread | Celery + Redis |
| Database | SQLite | PostgreSQL |
| Storage | Local FS | S3/GCS |
| Caching | In-memory | Redis |
| Monitoring | Logs | ELK Stack |

## Security Considerations

- ✅ Input validation for file uploads
- ✅ CSRF protection
- ✅ Secure model serialization
- ✅ Environment-based configuration
- ✅ Logging and audit trails
- ⚠️ TODO: User authentication
- ⚠️ TODO: Role-based access control

## Performance Tips

1. Use stratified split for imbalanced datasets
2. Enable cross-validation for robust evaluation
3. Use feature scaling for distance-based algorithms
4. Cache preprocessed data for large datasets
5. Use GPU acceleration for DL models

## Troubleshooting

**Issue**: Memory error with large datasets
- **Solution**: Increase system RAM or use data streaming

**Issue**: Slow training
- **Solution**: Use smaller batch size or fewer features

**Issue**: Import errors
- **Solution**: Verify all dependencies: `pip install -r requirements.txt`

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Create Pull Request

## License

MIT License - see LICENSE file

## Support

- 📧 Email: support@example.com
- 💬 Issues: GitHub Issues
- 📖 Documentation: [Wiki](https://github.com/yourusername/ML_DL_Trainer/wiki)

## Roadmap

- [ ] User authentication
- [ ] Model versioning
- [ ] Hyperparameter optimization
- [ ] AutoML integration
- [ ] Model explainability (SHAP, LIME)
- [ ] Real-time collaboration
- [ ] Mobile app
- [ ] Advanced visualizations

---

**Made with ❤️ for the ML/DL community**
