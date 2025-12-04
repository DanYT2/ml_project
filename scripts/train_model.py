#!/usr/bin/env python3
"""
Train ASL gesture recognition model.

This script provides a CLI for training models with various configurations.

Usage:
    python scripts/train_model.py --model-type static --epochs 100
    python scripts/train_model.py --model-type hybrid --epochs 50 --batch-size 64
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.train import Trainer, TrainingConfig
from src.utils.helpers import set_seed, setup_gpu, get_device_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ASL gesture recognition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        choices=["static", "conv", "hybrid", "landmark"],
        default="static",
        help="Model architecture type"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=27,
        help="Number of output classes (26 letters + delete)"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Sequence length for hybrid model"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs"
    )
    
    # Regularization
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout rate"
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=0.001,
        help="L2 regularization coefficient"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor"
    )
    
    # Augmentation
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable data augmentation"
    )
    parser.add_argument(
        "--rotation-range",
        type=float,
        default=15.0,
        help="Random rotation range (degrees)"
    )
    parser.add_argument(
        "--noise-stddev",
        type=float,
        default=0.02,
        help="Noise standard deviation for augmentation"
    )
    
    # Callbacks
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience"
    )
    
    # Output
    parser.add_argument(
        "--checkpoint-dir",
        default="models/checkpoints",
        help="Directory for model checkpoints"
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--experiment-name",
        default="",
        help="Name for this experiment"
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--gpu-memory-limit",
        type=int,
        default=None,
        help="GPU memory limit in MB"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup GPU
    setup_gpu(memory_limit=args.gpu_memory_limit)
    
    # Print device info
    device_info = get_device_info()
    print(f"\nTensorFlow version: {device_info['tensorflow_version']}")
    print(f"GPUs available: {device_info['num_gpus']}")
    
    # Create training config
    config = TrainingConfig(
        # Data
        data_dir=args.data_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        
        # Model
        model_type=args.model_type,
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        
        # Training
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_learning_rate=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        
        # Regularization
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
        label_smoothing=args.label_smoothing,
        
        # Augmentation
        use_augmentation=not args.no_augmentation,
        rotation_range=args.rotation_range,
        noise_stddev=args.noise_stddev,
        
        # Callbacks
        early_stopping_patience=args.patience,
        reduce_lr_patience=args.reduce_lr_patience,
        
        # Output
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Model type: {config.model_type}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Experiment: {config.experiment_name}")
    
    # Create trainer and train
    trainer = Trainer(config)
    history = trainer.train()
    
    # Save model
    model_path = os.path.join("models", config.experiment_name)
    trainer.save_model(model_path)
    
    # Plot training history
    plot_path = os.path.join(config.log_dir, f"{config.experiment_name}_history.png")
    trainer.plot_history(save_path=plot_path)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Training plot saved to: {plot_path}")
    
    # Final accuracy
    final_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")


if __name__ == "__main__":
    main()

