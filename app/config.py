"""
Global configuration for the ML/DL Training Platform
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
MODELS_DIR = DATA_DIR / "models"
RESULTS_DIR = DATA_DIR / "results"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Create directories if not exist
for directory in [UPLOADS_DIR, MODELS_DIR, RESULTS_DIR, PREPROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Application settings
APP_NAME = "ML/DL Training Platform"
APP_VERSION = "1.0.0"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# File upload settings
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB
ALLOWED_EXTENSIONS = {"csv"}

# Model training settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# Neural Network defaults
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "app.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Database (for metadata, results)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ml_trainer.db")

# Redis (for caching, task queue)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
