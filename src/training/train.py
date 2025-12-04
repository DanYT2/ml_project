"""
Training pipeline for ASL gesture recognition models.

This module provides a comprehensive training pipeline with:
- Data loading and augmentation
- Model training with callbacks
- Hyperparameter configuration
- Training visualization
- Model checkpointing
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import matplotlib.pyplot as plt

from ..data.dataset import ASLDataset, load_kaggle_dataset, CONFIG as DATA_CONFIG
from ..data.augmentation import create_augmented_dataset, AugmentationConfig
from ..models.landmark_model import create_landmark_model, create_conv_landmark_model
from ..models.hybrid_model import create_hybrid_model, create_static_model
from .callbacks import get_callbacks


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Data
    data_dir: str = "data"
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Model
    model_type: str = "static"  # "static", "hybrid", "conv", "ensemble"
    num_classes: int = 27
    sequence_length: int = 30  # For hybrid model
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    min_learning_rate: float = 1e-6
    warmup_epochs: int = 5
    
    # Regularization
    dropout_rate: float = 0.4
    l2_reg: float = 0.001
    label_smoothing: float = 0.1
    
    # Augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0
    scale_range: Tuple[float, float] = (0.9, 1.1)
    noise_stddev: float = 0.02
    
    # Callbacks
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    
    # Checkpointing
    checkpoint_dir: str = "models/checkpoints"
    save_best_only: bool = True
    
    # Logging
    log_dir: str = "logs"
    experiment_name: str = ""
    
    def __post_init__(self):
        if not self.experiment_name:
            self.experiment_name = f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class Trainer:
    """
    Training pipeline for ASL gesture recognition.
    
    Handles the complete training workflow including data loading,
    model creation, training, and evaluation.
    """
    
    def __init__(self, config: TrainingConfig = None):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model = None
        self.history = None
        
        # Create directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Save config
        self._save_config()
        
    def _save_config(self):
        """Save training configuration to file."""
        config_path = os.path.join(
            self.config.log_dir,
            f"{self.config.experiment_name}_config.json"
        )
        with open(config_path, 'w') as f:
            # Convert tuple to list for JSON serialization
            config_dict = asdict(self.config)
            config_dict['scale_range'] = list(config_dict['scale_range'])
            json.dump(config_dict, f, indent=2)
            
    def load_data(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Load and prepare training data.
        
        Returns:
            Dictionary with train/val/test splits
        """
        print("Loading dataset...")
        
        data = load_kaggle_dataset(
            data_dir=self.config.data_dir,
            test_size=self.config.test_size,
            val_size=self.config.val_size
        )
        
        print(f"Training samples: {len(data['train'][0])}")
        print(f"Validation samples: {len(data['val'][0])}")
        print(f"Test samples: {len(data['test'][0])}")
        
        return data
    
    def create_model(self) -> keras.Model:
        """
        Create model based on configuration.
        
        Returns:
            Keras model
        """
        print(f"Creating {self.config.model_type} model...")
        
        if self.config.model_type == "static":
            model = create_static_model(
                num_classes=self.config.num_classes,
                dropout_rate=self.config.dropout_rate
            )
        elif self.config.model_type == "conv":
            model = create_conv_landmark_model(
                num_classes=self.config.num_classes,
                dropout_rate=self.config.dropout_rate
            )
        elif self.config.model_type == "hybrid":
            model = create_hybrid_model(
                sequence_length=self.config.sequence_length,
                num_classes=self.config.num_classes,
                dropout_rate=self.config.dropout_rate
            )
        elif self.config.model_type == "landmark":
            model = create_landmark_model(
                num_classes=self.config.num_classes,
                dropout_rate=self.config.dropout_rate
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
            
        return model
    
    def compile_model(self, model: keras.Model) -> keras.Model:
        """
        Compile model with optimizer and loss.
        
        Args:
            model: Keras model
            
        Returns:
            Compiled model
        """
        # Learning rate schedule with warmup
        total_steps = self.config.epochs * 1000  # Approximate
        warmup_steps = self.config.warmup_epochs * 1000
        
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=total_steps - warmup_steps,
            alpha=self.config.min_learning_rate / self.config.learning_rate
        )
        
        # Warmup wrapper
        if self.config.warmup_epochs > 0:
            lr_schedule = WarmupSchedule(
                lr_schedule,
                warmup_steps=warmup_steps,
                target_lr=self.config.learning_rate
            )
            
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        # Loss with label smoothing
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
            ]
        )
        
        return model
    
    def create_datasets(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow datasets with augmentation.
        
        Args:
            data: Data dictionary from load_data
            
        Returns:
            Train, validation, and test datasets
        """
        X_train, y_train = data['train']
        X_val, y_val = data['val']
        X_test, y_test = data['test']
        
        # Augmentation config
        aug_config = AugmentationConfig(
            rotation_range=self.config.rotation_range,
            scale_range=self.config.scale_range,
            noise_stddev=self.config.noise_stddev
        ) if self.config.use_augmentation else None
        
        # For hybrid model, need to create sequences
        if self.config.model_type == "hybrid":
            X_train = self._create_sequences(X_train)
            X_val = self._create_sequences(X_val)
            X_test = self._create_sequences(X_test)
            
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_ds = train_ds.shuffle(len(X_train)).batch(self.config.batch_size)
        else:
            train_ds = create_augmented_dataset(
                X_train, y_train,
                batch_size=self.config.batch_size,
                augment=self.config.use_augmentation,
                config=aug_config
            )
            
        # Validation and test datasets (no augmentation)
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(self.config.batch_size)
        
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(self.config.batch_size)
        
        # Prefetch
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds
    
    def _create_sequences(self, landmarks: np.ndarray) -> np.ndarray:
        """Create pseudo-sequences for hybrid model from static landmarks."""
        seq_len = self.config.sequence_length
        sequences = np.repeat(landmarks[:, np.newaxis, :], seq_len, axis=1)
        
        # Add temporal noise
        noise = np.random.normal(0, 0.01, sequences.shape).astype(np.float32)
        sequences = sequences + noise
        
        return sequences.astype(np.float32)
    
    def train(
        self,
        data: Dict[str, Tuple[np.ndarray, np.ndarray]] = None
    ) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            data: Optional pre-loaded data (will load if not provided)
            
        Returns:
            Training history
        """
        # Load data if not provided
        if data is None:
            data = self.load_data()
            
        # Create datasets
        train_ds, val_ds, test_ds = self.create_datasets(data)
        
        # Create and compile model
        self.model = self.create_model()
        self.model = self.compile_model(self.model)
        
        # Print model summary
        self.model.summary()
        
        # Get callbacks
        callbacks = get_callbacks(
            checkpoint_dir=self.config.checkpoint_dir,
            log_dir=self.config.log_dir,
            experiment_name=self.config.experiment_name,
            patience=self.config.early_stopping_patience,
            reduce_lr_patience=self.config.reduce_lr_patience,
            reduce_lr_factor=self.config.reduce_lr_factor,
            min_lr=self.config.min_learning_rate
        )
        
        # Train
        print(f"\nStarting training for {self.config.epochs} epochs...")
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = self.model.evaluate(test_ds, verbose=1)
        
        print(f"\nTest Results:")
        for name, value in zip(self.model.metrics_names, test_results):
            print(f"  {name}: {value:.4f}")
            
        # Save training history
        self._save_history()
        
        return self.history
    
    def _save_history(self):
        """Save training history to file."""
        if self.history is None:
            return
            
        history_path = os.path.join(
            self.config.log_dir,
            f"{self.config.experiment_name}_history.json"
        )
        
        history_dict = {k: [float(v) for v in vals] 
                        for k, vals in self.history.history.items()}
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
            
    def plot_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Optional path to save plot
        """
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 Accuracy
        if 'top3_acc' in self.history.history:
            axes[1, 0].plot(self.history.history['top3_acc'], label='Train')
            axes[1, 0].plot(self.history.history['val_top3_acc'], label='Validation')
            axes[1, 0].set_title('Top-3 Accuracy')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved training plot to {save_path}")
        else:
            plt.show()
            
    def save_model(self, path: str = None):
        """
        Save the trained model.
        
        Args:
            path: Path to save model (default: models/{experiment_name})
        """
        if self.model is None:
            print("No model to save")
            return
            
        if path is None:
            path = os.path.join("models", self.config.experiment_name)
            
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Save in SavedModel format
        self.model.save(path)
        print(f"Saved model to {path}")
        
        # Also save weights separately
        weights_path = f"{path}_weights.h5"
        self.model.save_weights(weights_path)
        print(f"Saved weights to {weights_path}")
        
    def load_model(self, path: str):
        """
        Load a saved model.
        
        Args:
            path: Path to saved model
        """
        self.model = keras.models.load_model(path)
        print(f"Loaded model from {path}")


class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup."""
    
    def __init__(
        self,
        after_warmup_schedule: keras.optimizers.schedules.LearningRateSchedule,
        warmup_steps: int,
        target_lr: float
    ):
        """
        Initialize warmup schedule.
        
        Args:
            after_warmup_schedule: Schedule to use after warmup
            warmup_steps: Number of warmup steps
            target_lr: Target learning rate at end of warmup
        """
        super().__init__()
        self.after_warmup_schedule = after_warmup_schedule
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        
    def __call__(self, step):
        """Get learning rate for step."""
        warmup_lr = self.target_lr * step / self.warmup_steps
        
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.after_warmup_schedule(step - self.warmup_steps)
        )
        
    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'target_lr': self.target_lr
        }


def train_model(
    data_dir: str = "data",
    model_type: str = "static",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Trainer:
    """
    Convenience function to train a model.
    
    Args:
        data_dir: Path to data directory
        model_type: Type of model to train
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Trained Trainer object
    """
    config = TrainingConfig(
        data_dir=data_dir,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    trainer = Trainer(config)
    trainer.train()
    trainer.save_model()
    trainer.plot_history(
        save_path=os.path.join(config.log_dir, f"{config.experiment_name}_history.png")
    )
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ASL recognition model")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model-type", default="static", 
                        choices=["static", "conv", "hybrid", "landmark"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

