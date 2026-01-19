# ML/DL Training Platform - Architecture Design

## 1. HIGH-LEVEL ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│                    (Streamlit Frontend)                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Dashboard │ Data Upload │ Config │ Training │ Results    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────▼────────────────────────────────────────────┐
│                   APPLICATION LAYER                             │
│                  (Fastapi/Flask Backend)                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ API Routes │ Session Manager │ Task Queue Handler       │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────┬──────────────┬──────────────────┬────────────────────┬─┘
         │              │                  │                    │
    ┌────▼───────┐ ┌────▼───────┐ ┌───────▼────┐ ┌────────────▼─┐
    │   CORE     │ │   MODEL    │ │ EVALUATION │ │  UTILITIES   │
    │  SERVICES  │ │   LAYER    │ │   LAYER    │ │    LAYER     │
    └────────────┘ └────────────┘ └────────────┘ └──────────────┘
         │              │                  │                    │
         │         ┌─────┴──────┐         │                    │
         │         │            │         │                    │
         │      ┌──▼──┐    ┌────▼──┐     │                    │
         │      │ SKL │    │ TF/   │     │                    │
         │      │     │    │Keras  │     │                    │
         │      └─────┘    └───────┘     │                    │
         │                               │                    │
┌────────▼───────────────────────────────▼────────────────────▼──┐
│                   DATA PERSISTENCE LAYER                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ File Storage │ Models Repo │ Results DB │ Logs Cache    │  │
│  │ (CSV/Models)  │ (Trained)   │ (Metrics)  │ (Training)    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. FOLDER STRUCTURE

```
ML_DL_Trainer/
│
├── app/
│   ├── __init__.py
│   ├── main.py                         # Streamlit entry point
│   ├── config.py                       # Global configuration
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                   # Logging utility
│       ├── file_handler.py             # File upload/storage
│       └── validators.py               # Input validation
│
├── backend/
│   ├── __init__.py
│   ├── api.py                          # FastAPI/Flask routes
│   ├── session_manager.py              # Session & state management
│   └── task_queue.py                   # Async task handling (Celery/Threading)
│
├── core/
│   ├── __init__.py
│   ├── preprocessor.py                 # Data preprocessing pipeline
│   ├── feature_engineer.py             # Feature engineering
│   └── validator.py                    # Data quality checks
│
├── models/
│   ├── __init__.py
│   ├── model_factory.py                # Factory pattern for model creation
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── classifier.py               # SKL Classification models
│   │   ├── regressor.py                # SKL Regression models
│   │   └── ensemble.py                 # Ensemble methods
│   └── dl/
│       ├── __init__.py
│       ├── cnn_models.py               # CNN architectures
│       ├── rnn_models.py               # RNN architectures
│       └── transformer_models.py       # Transformer architectures
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                      # Classification & Regression metrics
│   ├── visualizer.py                   # Plot generation
│   ├── reporter.py                     # Report generation
│   └── cross_validator.py              # K-fold, stratified splits
│
├── storage/
│   ├── __init__.py
│   ├── model_repository.py             # Save/load trained models
│   ├── result_repository.py            # Save/load results
│   └── cache_manager.py                # In-memory caching
│
├── data/
│   ├── uploads/                        # User uploaded datasets
│   ├── preprocessed/                   # Processed datasets
│   ├── models/                         # Trained models (.pkl, .h5)
│   └── results/                        # Experiment results (JSON/CSV)
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_integration.py
│
├── requirements.txt                    # Python dependencies
├── .env                                # Environment variables
├── docker-compose.yml                  # Docker setup (optional)
├── README.md                           # Documentation
└── main.py                             # Application launcher
```

---

## 3. DATA FLOW EXPLANATION

### Step-by-Step Flow:

```
1. USER UPLOAD PHASE
   └─> Streamlit UI: User uploads CSV file
       └─> app/utils/file_handler.py: Validate & store file
           └─> data/uploads/ (persistent storage)

2. CONFIGURATION PHASE
   └─> Streamlit UI: User selects
       • Task: Classification / Regression
       • Model: RandomForest, SVM, Neural Network, etc.
       • Hyperparameters: epochs, batch_size, learning_rate
       └─> backend/session_manager.py: Store session config

3. PREPROCESSING PHASE
   └─> core/preprocessor.py: 
       • Handle missing values (mean, median, drop)
       • Encode categorical variables (One-Hot, Label encoding)
       • Normalize/Standardize numerical features
       └─> data/preprocessed/ (cache for reuse)

4. FEATURE ENGINEERING PHASE
   └─> core/feature_engineer.py:
       • Create polynomial features
       • Interaction features
       • Domain-specific features
       └─> storage/cache_manager.py

5. DATA SPLITTING PHASE
   └─> evaluation/cross_validator.py:
       • Train-Test split (80-20)
       • Stratified split for imbalanced data
       └─> Return X_train, X_test, y_train, y_test

6. MODEL TRAINING PHASE
   └─> models/model_factory.py:
       • Instantiate selected model
       • For ML: models/ml/*.py (SKL models)
       • For DL: models/dl/*.py (Keras models)
       └─> backend/task_queue.py (async execution)
           └─> Real-time training logs → storage/cache_manager.py

7. EVALUATION PHASE
   └─> evaluation/metrics.py:
       • Calculate: Accuracy, Precision, Recall, F1 (Classification)
       • Calculate: MSE, RMSE, MAE, R² (Regression)
       • Confusion Matrix, AUC-ROC, Learning curves
       └─> evaluation/visualizer.py: Generate plots

8. RESULTS STORAGE PHASE
   └─> storage/result_repository.py:
       • Save trained model → data/models/
       • Save metrics → data/results/ (JSON)
       • Save visualizations → data/results/ (PNG)

9. RESULTS PRESENTATION PHASE
   └─> Streamlit UI:
       • Display metrics dashboard
       • Show visualizations
       • Allow model download
       • Export report (PDF/HTML)
```

---

## 4. TECHNOLOGY JUSTIFICATION

### Frontend: Streamlit
✅ **Why Streamlit?**
- Rapid prototyping with minimal code
- Built-in widgets for file upload, sliders, dropdowns
- Real-time updates without manual page refresh
- Excellent for ML/Data Science workflows
- Lower learning curve vs React/Vue
- Native support for Plotly, Matplotlib visualizations

### Backend: FastAPI (or Flask)
✅ **Why FastAPI?**
- High performance (3x faster than Flask)
- Automatic API documentation (OpenAPI/Swagger)
- Built-in data validation with Pydantic
- Async support for long-running tasks
- Type hints for better code quality
- WebSocket support for real-time training updates

### ML Framework: Scikit-learn
✅ **Why Scikit-learn?**
- Industry standard for traditional ML
- Comprehensive algorithms: SVM, RandomForest, XGBoost, KNN
- Excellent preprocessing utilities
- Low memory footprint
- Mature & well-documented
- Easy integration with pipelines

### DL Framework: TensorFlow/Keras
✅ **Why TensorFlow?**
- Leading deep learning framework
- High-level Keras API (simple and intuitive)
- Excellent for CNNs, RNNs, Transformers
- Production-ready deployment options
- GPU acceleration support
- Rich ecosystem (TensorBoard, TFLite, TFServing)

### Storage: File-based + SQLite/PostgreSQL
✅ **Why File-based?**
- Models: .pkl (SKL), .h5 (Keras) for reproducibility
- Results: JSON for lightweight, queryable data
- Scalable to cloud storage (S3, GCS) later

✅ **Why SQLite (for dev) / PostgreSQL (for prod)?**
- Track experiment metadata
- Query historical results
- Easy migration path to production

### Async Task Queue: Celery (or Python Threading)
✅ **Why Async Handling?**
- Long training jobs don't block UI
- User can track progress in real-time
- Multiple concurrent trainings possible
- Celery: scales to distributed workers (Redis/RabbitMQ backend)

---

## 5. KEY DESIGN PATTERNS

### Factory Pattern
```python
# models/model_factory.py
ModelFactory.create(task_type='classification', algo='random_forest')
→ returns SKL RandomForestClassifier instance
```

### Pipeline Pattern
```python
# Preprocessing → Feature Engineering → Scaling → Splitting
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('preprocessor', Preprocessor()),
                     ('scaler', StandardScaler())])
```

### Repository Pattern
```python
# storage/model_repository.py
model_repo.save(model, metadata)
model_repo.load(model_id)
```

### Observer Pattern
```python
# Real-time training updates via callbacks
model.fit(X_train, y_train, callbacks=[ProgressCallback()])
```

---

## 6. SCALABILITY CONSIDERATIONS

| Aspect | Current | Future |
|--------|---------|--------|
| **Storage** | Local filesystem | S3/GCS bucket |
| **Database** | SQLite | PostgreSQL |
| **Task Queue** | Threading | Celery + Redis |
| **Deployment** | Single container | Kubernetes cluster |
| **Caching** | In-memory | Redis |
| **Logging** | File-based | ELK Stack |
| **Monitoring** | None | Prometheus + Grafana |

---

## 7. SECURITY CONSIDERATIONS

- Input validation for file uploads (size, format)
- Sandboxed execution environment
- User authentication & authorization
- Model versioning & reproducibility
- HTTPS for API communication
- Rate limiting on API endpoints
- Secure credential management (.env files)

---

## 8. DEPLOYMENT OPTIONS

1. **Development**: Docker + docker-compose
2. **Staging**: Single cloud VM (AWS EC2, GCP Compute)
3. **Production**: Kubernetes cluster with:
   - Streamlit app pod
   - FastAPI pod
   - Celery worker pods
   - PostgreSQL pod
   - Redis pod

