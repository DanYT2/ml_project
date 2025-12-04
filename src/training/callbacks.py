"""
Custom callbacks for training ASL gesture recognition models.

This module provides callbacks for:
- Model checkpointing
- Early stopping
- Learning rate scheduling
- Training visualization
- Custom logging
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Optional, Dict, Any
from datetime import datetime


class LearningRateLogger(keras.callbacks.Callback):
    """Log learning rate at each epoch."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Log current learning rate."""
        logs = logs or {}
        
        lr = self.model.optimizer.learning_rate
        
        # Handle learning rate schedules
        if hasattr(lr, 'numpy'):
            logs['lr'] = float(lr.numpy())
        elif callable(lr):
            logs['lr'] = float(lr(self.model.optimizer.iterations))
        else:
            logs['lr'] = float(lr)


class ConfusionMatrixCallback(keras.callbacks.Callback):
    """Compute and log confusion matrix during training."""
    
    def __init__(
        self,
        validation_data: tf.data.Dataset,
        class_names: List[str],
        log_dir: str,
        log_frequency: int = 5
    ):
        """
        Initialize confusion matrix callback.
        
        Args:
            validation_data: Validation dataset
            class_names: List of class names
            log_dir: Directory to save confusion matrices
            log_frequency: How often to compute (every N epochs)
        """
        super().__init__()
        self.validation_data = validation_data
        self.class_names = class_names
        self.log_dir = log_dir
        self.log_frequency = log_frequency
        
        os.makedirs(log_dir, exist_ok=True)
        
    def on_epoch_end(self, epoch, logs=None):
        """Compute confusion matrix at epoch end."""
        if (epoch + 1) % self.log_frequency != 0:
            return
            
        # Get predictions
        y_true = []
        y_pred = []
        
        for x_batch, y_batch in self.validation_data:
            predictions = self.model.predict(x_batch, verbose=0)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(y_batch.numpy())
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute confusion matrix
        cm = tf.math.confusion_matrix(y_true, y_pred)
        
        # Save to file
        cm_path = os.path.join(self.log_dir, f'confusion_matrix_epoch_{epoch+1}.npy')
        np.save(cm_path, cm.numpy())


class GradientMonitorCallback(keras.callbacks.Callback):
    """Monitor gradient norms during training."""
    
    def __init__(self, log_frequency: int = 100):
        """
        Initialize gradient monitor.
        
        Args:
            log_frequency: How often to log (every N batches)
        """
        super().__init__()
        self.log_frequency = log_frequency
        self.gradient_norms = []
        
    def on_train_batch_end(self, batch, logs=None):
        """Monitor gradients at batch end."""
        if batch % self.log_frequency != 0:
            return
            
        # Get gradient norms from optimizer
        if hasattr(self.model.optimizer, '_gradients'):
            gradients = self.model.optimizer._gradients
            if gradients:
                grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
                if grad_norms:
                    mean_norm = np.mean(grad_norms)
                    self.gradient_norms.append(mean_norm)


class EarlyStoppingWithWarmup(keras.callbacks.EarlyStopping):
    """Early stopping that ignores first N warmup epochs."""
    
    def __init__(self, warmup_epochs: int = 10, **kwargs):
        """
        Initialize early stopping with warmup.
        
        Args:
            warmup_epochs: Number of epochs to ignore before early stopping kicks in
            **kwargs: Arguments passed to EarlyStopping
        """
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        """Check for early stopping after warmup."""
        if epoch < self.warmup_epochs:
            return
        super().on_epoch_end(epoch, logs)


class TrainingProgressCallback(keras.callbacks.Callback):
    """Display training progress with rich formatting."""
    
    def __init__(self, total_epochs: int):
        """
        Initialize progress callback.
        
        Args:
            total_epochs: Total number of training epochs
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = None
        
    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time."""
        self.epoch_start_time = datetime.now()
        
    def on_epoch_end(self, epoch, logs=None):
        """Display progress summary."""
        logs = logs or {}
        
        duration = (datetime.now() - self.epoch_start_time).total_seconds()
        
        # Format metrics
        train_acc = logs.get('accuracy', 0) * 100
        val_acc = logs.get('val_accuracy', 0) * 100
        train_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        # Progress bar
        progress = (epoch + 1) / self.total_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n[{bar}] {epoch+1}/{self.total_epochs} ({progress*100:.1f}%)")
        print(f"  Train: acc={train_acc:.2f}%, loss={train_loss:.4f}")
        print(f"  Val:   acc={val_acc:.2f}%, loss={val_loss:.4f}")
        print(f"  Time:  {duration:.1f}s")


def get_callbacks(
    checkpoint_dir: str = "models/checkpoints",
    log_dir: str = "logs",
    experiment_name: str = "",
    patience: int = 15,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    monitor: str = "val_accuracy",
    mode: str = "max"
) -> List[keras.callbacks.Callback]:
    """
    Get standard training callbacks.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
        experiment_name: Name for this experiment
        patience: Early stopping patience
        reduce_lr_patience: ReduceLROnPlateau patience
        reduce_lr_factor: LR reduction factor
        min_lr: Minimum learning rate
        monitor: Metric to monitor
        mode: 'min' or 'max' for monitored metric
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Model checkpoint
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{experiment_name}_best.keras"
    )
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=monitor,
            mode=mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    )
    
    # Also save weights periodically
    weights_path = os.path.join(
        checkpoint_dir,
        f"{experiment_name}_epoch_{{epoch:03d}}.weights.h5"
    )
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            weights_path,
            save_weights_only=True,
            save_freq='epoch',
            period=10  # Save every 10 epochs
        )
    )
    
    # Early stopping
    callbacks.append(
        EarlyStoppingWithWarmup(
            warmup_epochs=10,
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    )
    
    # Reduce learning rate on plateau
    callbacks.append(
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        )
    )
    
    # TensorBoard
    tb_log_dir = os.path.join(log_dir, "tensorboard", experiment_name)
    callbacks.append(
        keras.callbacks.TensorBoard(
            log_dir=tb_log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )
    )
    
    # CSV logger
    csv_path = os.path.join(log_dir, f"{experiment_name}_training.csv")
    callbacks.append(
        keras.callbacks.CSVLogger(csv_path, append=True)
    )
    
    # Learning rate logger
    callbacks.append(LearningRateLogger())
    
    # Terminate on NaN
    callbacks.append(keras.callbacks.TerminateOnNaN())
    
    return callbacks


def create_lr_scheduler(
    schedule_type: str = "cosine",
    initial_lr: float = 0.001,
    min_lr: float = 1e-6,
    epochs: int = 100,
    warmup_epochs: int = 5,
    decay_epochs: int = None
) -> keras.callbacks.LearningRateScheduler:
    """
    Create a learning rate scheduler callback.
    
    Args:
        schedule_type: Type of schedule ('cosine', 'exponential', 'step')
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        decay_epochs: Epochs for step decay
        
    Returns:
        LearningRateScheduler callback
    """
    decay_epochs = decay_epochs or epochs // 3
    
    def cosine_schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    def exponential_schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        decay_rate = (min_lr / initial_lr) ** (1 / (epochs - warmup_epochs))
        return initial_lr * (decay_rate ** (epoch - warmup_epochs))
    
    def step_schedule(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (epoch + 1) / warmup_epochs
        num_decays = epoch // decay_epochs
        return max(initial_lr * (0.1 ** num_decays), min_lr)
    
    schedules = {
        'cosine': cosine_schedule,
        'exponential': exponential_schedule,
        'step': step_schedule
    }
    
    return keras.callbacks.LearningRateScheduler(
        schedules.get(schedule_type, cosine_schedule),
        verbose=0
    )


if __name__ == "__main__":
    # Test callbacks
    print("Testing callbacks...")
    
    callbacks = get_callbacks(
        experiment_name="test",
        patience=10
    )
    
    print(f"Created {len(callbacks)} callbacks:")
    for cb in callbacks:
        print(f"  - {cb.__class__.__name__}")

