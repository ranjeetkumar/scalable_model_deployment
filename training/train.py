# train_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
from sklearn.model_selection import train_test_split
import os
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build a simple CNN model"""
        model = models.Sequential([
            Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            #layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            #layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        # Add data augmentation for training
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
            .shuffle(10000)\
            .batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))\
            .batch(batch_size)
            
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/model_{epoch:02d}_{val_accuracy:.3f}.keras',
                save_weights_only=False,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, version="v1", local_path="model.h5", bucket_name=None):
        """Save the model locally and to GCS if bucket is provided"""
        if self.model is None:
            raise ValueError("No model to save")
            
        # Save locally
        self.model.save(local_path)
        logger.info(f"Model saved locally to {local_path}")
        
        # Upload to GCS if bucket is provided
        if bucket_name:
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(f"models/model_{version}.h5")
                blob.upload_from_filename(local_path)
                logger.info(f"Model uploaded to gs://{bucket_name}/models/model_{version}.h5")
            except Exception as e:
                logger.error(f"Error uploading model to GCS: {str(e)}")
                raise

def generate_sample_data(num_samples=1000):
    """Generate sample data for demonstration"""
    # Generate random data (replace this with your actual data)
    X = np.random.randn(num_samples, 28, 28, 1)
    y = np.random.randint(0, 10, num_samples)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Generate sample data
    X_train, X_val, y_train, y_val = generate_sample_data()
    
    # Initialize and train model
    trainer = ModelTrainer()
    trainer.build_model()
    
    # Train the model
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Save the model
    trainer.save_model(
        version="v1",
        local_path="model.h5",
        bucket_name=os.getenv("GCP_BUCKET_NAME")
    )

if __name__ == "__main__":
    main()