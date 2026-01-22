"""
Deep Learning Training Pipeline

Completely isolated from ML pipeline.
Uses epochs, batch size, learning rate, early stopping.
Displays training & validation loss.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import streamlit as st


class DLTrainer:
    """Deep Learning trainer with epochs and early stopping."""
    
    @staticmethod
    def build_sequential_model(input_dim, output_dim, task_type):
        """
        Build Sequential Neural Network.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output classes/units
            task_type: 'classification' or 'regression'
        
        Returns:
            model: Compiled Keras model
        """
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_dim, activation='softmax' if task_type == 'classification' else 'linear')
        ])
        
        loss = 'categorical_crossentropy' if task_type == 'classification' else 'mse'
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy' if task_type == 'classification' else 'mae']
        )
        
        return model
    
    @staticmethod
    def build_cnn_model(input_shape, output_dim, task_type):
        """
        Build Convolutional Neural Network.
        
        Args:
            input_shape: Input shape (height, width, channels)
            output_dim: Number of output classes
            task_type: 'classification' or 'regression'
        
        Returns:
            model: Compiled Keras model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(output_dim, activation='softmax' if task_type == 'classification' else 'linear')
        ])
        
        loss = 'categorical_crossentropy' if task_type == 'classification' else 'mse'
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy' if task_type == 'classification' else 'mae']
        )
        
        return model
    
    @staticmethod
    def build_rnn_model(input_shape, output_dim, task_type):
        """
        Build Recurrent Neural Network (LSTM).
        
        Args:
            input_shape: Input shape (sequence_length, features)
            output_dim: Number of output classes
            task_type: 'classification' or 'regression'
        
        Returns:
            model: Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(output_dim, activation='softmax' if task_type == 'classification' else 'linear')
        ])
        
        loss = 'categorical_crossentropy' if task_type == 'classification' else 'mse'
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy' if task_type == 'classification' else 'mae']
        )
        
        return model
    
    @staticmethod
    def train_dl_model(model, X_train, y_train, X_val, y_val,
                       epochs, batch_size, learning_rate, early_stopping=True):
        """
        Train DL model with epochs and early stopping.
        
        Note: Epochs = multiple passes through entire training data.
        This is DIFFERENT from max_iter (iterative ML convergence iterations).
        
        Args:
            model: Keras model
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            early_stopping: Whether to use early stopping
        
        Returns:
            history: Training history with loss/accuracy
            trained_model: Trained model
        """
        
        # Update learning rate
        model.optimizer.learning_rate = learning_rate
        
        # Early stopping callback
        callbacks = []
        if early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return history, model


def prepare_dl_data(X_train, y_train, X_val, y_val, X_test, y_test, task_type):
    """
    Prepare data for deep learning.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        task_type: 'classification' or 'regression'
    
    Returns:
        Prepared data ready for DL training
    """
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Encode targets for classification
    if task_type == 'classification':
        from tensorflow.keras.utils import to_categorical
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_val = to_categorical(y_val, num_classes)
        y_test = to_categorical(y_test, num_classes)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_dl_predictions(model, X_test, task_type):
    """Get predictions from DL model."""
    predictions = model.predict(X_test, verbose=0)
    
    if task_type == 'classification':
        return np.argmax(predictions, axis=1)
    else:
        return predictions.flatten()
