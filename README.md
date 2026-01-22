# ML/DL Trainer

**A production-ready web platform for training, evaluating, and deploying machine learning and deep learning models.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

## Overview

ML/DL Trainer is an end-to-end machine learning platform that simplifies the model development lifecycle. Upload data, configure hyperparameters, train models, and download results—all through an intuitive web interface. Supports 9 ML algorithms and 3 DL architectures with automatic preprocessing, cross-validation, and comprehensive evaluation metrics.

## ✨ Key Features

| Feature | Details |
|---------|---------|
| **📤 Data Upload** | CSV file upload with automatic validation and quality checks |
| **🔍 EDA** | Exploratory data analysis with missing value detection, feature relationships, and target analysis |
| **🎯 Model Selection** | 9 ML algorithms (Scikit-learn) + 3 DL architectures (TensorFlow/Keras) |
| **⚙️ Hyperparameter Tuning** | Per-model configuration for learning rate, epochs, batch size, tree depth, etc. |
| **🔄 Preprocessing** | Automatic missing value imputation, feature scaling, categorical encoding |
| **📊 Evaluation** | Classification & regression metrics, confusion matrices, feature importance plots |
| **💾 Model Persistence** | Download trained models (PKL) and metrics (JSON) |
| **🚀 Production Ready** | Error handling, logging, memory monitoring, Docker support |

## 🤖 Supported Models

### Machine Learning (Scikit-learn)
- **Classification**: Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting
- **Regression**: Linear Regression, Random Forest, SVR, Gradient Boosting

### Deep Learning (TensorFlow/Keras)
- Sequential Neural Networks
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (LSTM)

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│     Streamlit Frontend (Port 8501)  │
│  (Upload, Config, Training, Results)│
└────────────┬────────────────────────┘
             │
     ┌───────┴──────────┬──────────────┬──────────┐
     │                  │              │          │
┌────▼───────┐   ┌────▼────┐   ┌────▼──┐   ┌──▼────┐
│Preprocessing│   │Model    │   │Eval.  │   │Storage │
│ & Features  │   │Training │   │Metrics│   │(PKL)   │
└─────────────┘   └─────────┘   └───────┘   └────────┘
```

## 📋 Project Structure

```
ML_DL_Trainer/
├── app/                          # Frontend (Streamlit)
│   ├── main.py                   # Entry point
│   ├── config.py                 # Configuration
│   ├── pages/
│   │   └── eda_page.py          # EDA visualization
│   └── utils/
│       ├── error_handler.py      # Error handling & logging
│       ├── file_handler.py       # File operations
│       ├── logger.py             # Logging setup
│       └── validators.py         # Data validation
├── core/                         # ML operations
│   ├── preprocessor.py           # Data preprocessing
│   ├── feature_engineer.py       # Feature engineering
│   ├── target_analyzer.py        # Target analysis
│   └── validator.py              # Data validation
├── models/                       # Model implementations
│   ├── model_factory.py          # Factory pattern
│   ├── ml/
│   │   ├── classifier.py         # ML classifiers
│   │   └── regressor.py          # ML regressors
│   └── dl/
│       ├── cnn_models.py         # CNN architectures
│       └── rnn_models.py         # RNN architectures
├── evaluation/                   # Evaluation utilities
│   ├── metrics.py                # Metrics calculation
│   ├── visualizer.py             # Plotting
│   ├── cross_validator.py        # Cross-validation
│   └── reporter.py               # Report generation
├── storage/                      # Data persistence
│   ├── model_repository.py       # Model storage
│   ├── result_repository.py      # Results storage
│   └── cache_manager.py          # Caching
├── data/                         # Data directories
│   ├── uploads/                  # User uploads
│   ├── preprocessed/             # Processed data
│   ├── models/                   # Trained models
│   └── results/                  # Experiment results
├── tests/                        # Unit tests
├── Dockerfile                    # Container image
├── docker-compose.yml            # Multi-container setup
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- 2GB RAM minimum

### Local Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/ML_DL_Trainer.git
   cd ML_DL_Trainer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run application**
   ```bash
   streamlit run app/main.py
   ```

   Application opens at `http://localhost:8501`

### Docker Deployment

1. **Build image**
   ```bash
   docker build -t ml-dl-trainer:latest .
   ```

2. **Run container**
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/data:/app/data \
     ml-dl-trainer:latest
   ```

3. **Using Docker Compose**
   ```bash
   docker-compose up -d
   ```

## 📖 Usage Workflow

### Step 1: Upload Data
- Navigate to **Data Upload** page
- Upload CSV file or load sample dataset (Iris, Wine)
- Review data preview, statistics, and column info

### Step 2: Explore Data (Optional)
- Go to **EDA / Data Understanding** page
- Analyze missing values, feature distributions, relationships
- Identify target variable characteristics

### Step 3: Configure & Train
- Select **Training** page
- Choose task type: Classification or Regression
- Select algorithm and set hyperparameters
- Click **Start Training**

### Step 4: Review Results
- View performance metrics on **Results** page
- Download trained model (PKL format)
- Export metrics (JSON format)

## ⚙️ Configuration

Edit `app/config.py` to customize:

```python
MAX_FILE_SIZE = 500 * 1024 * 1024  # Max upload: 500MB
DEFAULT_TEST_SIZE = 0.2             # Train-test split
DEFAULT_CV_FOLDS = 5                # Cross-validation folds
DEFAULT_EPOCHS = 50                 # DL epochs
DEFAULT_BATCH_SIZE = 32             # DL batch size
LOG_LEVEL = "INFO"                  # Logging level
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_core.py -v

# Run with coverage
pytest tests/ --cov=core --cov=models
```

## 📊 Screenshots

### Home Page
- Platform overview with feature highlights
- Quick start guide with 3-step workflow
- Supported models showcase
- Call-to-action buttons

### Data Upload
- Drag-and-drop CSV upload
- Sample dataset loading (Iris, Wine)
- Data preview with statistics
- Column information display

### EDA / Data Understanding
- Missing value analysis
- Feature distribution plots
- Correlation heatmaps
- Target variable analysis
- Relationship visualization

### Training
- Task type selection (Classification/Regression)
- Algorithm selection with model-specific hyperparameters
- Real-time training progress
- Target validation with warnings

### Results
- Performance metrics display
- Model download (PKL)
- Metrics export (JSON)
- Detailed evaluation results

### About
- Platform information
- Supported algorithms list
- Architecture overview
- Quick links and acknowledgments

## 🔒 Security & Production Features

- ✅ Input validation for file uploads
- ✅ Error handling with custom exceptions
- ✅ Memory monitoring (90% threshold)
- ✅ Comprehensive logging with file rotation
- ✅ Non-root Docker user
- ✅ Health checks in container
- ✅ Environment-based configuration
- ⚠️ TODO: User authentication
- ⚠️ TODO: Role-based access control

## 📈 Performance Optimization

| Optimization | Implementation |
|--------------|-----------------|
| **Caching** | @st.cache_data with TTL for expensive computations |
| **Sampling** | 10% sampling for datasets >100K rows in visualizations |
| **Preprocessing** | Vectorized operations with NumPy/Pandas |
| **Memory** | MemoryMonitor tracks usage, prevents OOM |
| **Logging** | Async logging to avoid blocking UI |

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory error with large datasets | Increase system RAM or use data streaming |
| Slow training | Reduce batch size or feature count |
| Import errors | Run `pip install -r requirements.txt` |
| Port 8501 already in use | `streamlit run app/main.py --server.port 8502` |
| Docker build fails | Ensure Docker daemon is running |

## 📦 Deployment Options

### AWS
```bash
# EC2 + S3 + RDS
- EC2 for app hosting
- S3 for model/data storage
- RDS for metadata database
```

### Google Cloud
```bash
# Cloud Run + Cloud Storage
- Cloud Run for serverless deployment
- Cloud Storage for models
- Cloud SQL for database
```

### Azure
```bash
# App Service + Blob Storage
- App Service for hosting
- Blob Storage for models
- SQL Database for metadata
```

### Heroku
```bash
git push heroku main
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/amazing`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing`
5. Create Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/ML_DL_Trainer/issues)
- 📖 Docs: [Wiki](https://github.com/yourusername/ML_DL_Trainer/wiki)

## 🗺️ Roadmap

- [ ] User authentication & RBAC
- [ ] Model versioning & registry
- [ ] Hyperparameter optimization (Optuna)
- [ ] AutoML integration
- [ ] Model explainability (SHAP, LIME)
- [ ] Real-time collaboration
- [ ] Mobile app
- [ ] Advanced visualizations (Plotly)
- [ ] API endpoint documentation
- [ ] Performance benchmarking

## 📊 Resume-Ready Project Explanation

**ML/DL Trainer** is a full-stack machine learning platform demonstrating end-to-end software engineering practices:

### Technical Stack
- **Frontend**: Streamlit (Python web framework)
- **Backend**: FastAPI, Python
- **ML/DL**: Scikit-learn (9 algorithms), TensorFlow/Keras (3 architectures)
- **Data**: Pandas, NumPy
- **DevOps**: Docker, Docker Compose
- **Testing**: Pytest

### Key Accomplishments
1. **Architecture**: Implemented factory pattern for extensible model creation, repository pattern for data persistence
2. **Data Pipeline**: Built preprocessing pipeline with automatic missing value handling, feature scaling, categorical encoding
3. **Error Handling**: Developed comprehensive error handling module with custom exceptions, memory monitoring, production logging
4. **UI/UX**: Created intuitive Streamlit interface with 6 pages, real-time feedback, sample datasets
5. **Production Ready**: Added Docker support, health checks, non-root user, environment configuration
6. **Testing**: Wrote unit tests for core components (feature analysis, target detection, preprocessing)

### Design Patterns Used
- **Factory Pattern**: ModelFactory for flexible model creation
- **Repository Pattern**: Model and result storage abstraction
- **Pipeline Pattern**: Data preprocessing pipeline
- **Observer Pattern**: Real-time training callbacks
- **Decorator Pattern**: Error handling decorators

### Scalability Features
- Caching with TTL for expensive computations
- Data sampling for large datasets
- Memory monitoring to prevent OOM
- Async logging
- Containerization for cloud deployment

