"""
Model repository for saving and loading trained models
"""
import pickle
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from app.config import MODELS_DIR
from app.utils.logger import logger

class ModelRepository:
    """Repository for model persistence"""
    
    @staticmethod
    def save_sklearn_model(model, model_name: str, metadata: dict = None) -> str:
        """
        Save scikit-learn model to disk
        
        Args:
            model: Trained sklearn model
            model_name: Name of the model
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.pkl"
            filepath = MODELS_DIR / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            if metadata:
                meta_path = MODELS_DIR / f"{model_name}_{timestamp}_meta.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            logger.info(f"Model saved: {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @staticmethod
    def save_keras_model(model, model_name: str, metadata: dict = None) -> str:
        """
        Save Keras/TensorFlow model to disk
        
        Args:
            model: Trained Keras model
            model_name: Name of the model
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = MODELS_DIR / f"{model_name}_{timestamp}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model.save(model_dir / "model.h5")
            
            # Save metadata
            if metadata:
                meta_path = model_dir / "metadata.json"
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            logger.info(f"Keras model saved: {model_dir}")
            return str(model_dir)
        
        except Exception as e:
            logger.error(f"Error saving Keras model: {str(e)}")
            raise
    
    @staticmethod
    def load_sklearn_model(filepath: str):
        """
        Load scikit-learn model from disk
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Model loaded: {filepath}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @staticmethod
    def load_keras_model(model_dir: str):
        """
        Load Keras model from disk
        
        Args:
            model_dir: Directory containing model
            
        Returns:
            Loaded model
        """
        try:
            model = tf.keras.models.load_model(Path(model_dir) / "model.h5")
            logger.info(f"Keras model loaded: {model_dir}")
            return model
        
        except Exception as e:
            logger.error(f"Error loading Keras model: {str(e)}")
            raise
