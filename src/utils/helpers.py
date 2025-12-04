"""
Utility functions for ASL gesture recognition.

General purpose helpers for training, evaluation, and deployment.
"""

import os
import random
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # For additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'tensorflow_version': tf.__version__,
        'gpus': [],
        'cpu_count': os.cpu_count()
    }
    
    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            info['gpus'].append({
                'name': gpu.name,
                'device_type': gpu.device_type,
                'details': details
            })
        except Exception:
            info['gpus'].append({
                'name': gpu.name,
                'device_type': gpu.device_type
            })
            
    info['num_gpus'] = len(gpus)
    info['gpu_available'] = len(gpus) > 0
    
    return info


def setup_gpu(memory_limit: Optional[int] = None, allow_growth: bool = True):
    """
    Configure GPU settings.
    
    Args:
        memory_limit: Memory limit in MB (None for no limit)
        allow_growth: Allow dynamic memory allocation
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPUs available")
        return
        
    try:
        for gpu in gpus:
            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            elif allow_growth:
                tf.config.experimental.set_memory_growth(gpu, True)
                
        print(f"Configured {len(gpus)} GPU(s)")
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")


def create_experiment_dir(
    base_dir: str = "experiments",
    experiment_name: str = None
) -> str:
    """
    Create directory for experiment outputs.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of experiment (auto-generated if None)
        
    Returns:
        Path to experiment directory
    """
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'visualizations', 'exports']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
    print(f"Created experiment directory: {experiment_dir}")
    
    return experiment_dir


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(
        np.prod(v.shape) for v in model.trainable_variables
    )
    non_trainable = sum(
        np.prod(v.shape) for v in model.non_trainable_variables
    )
    
    return {
        'trainable': int(trainable),
        'non_trainable': int(non_trainable),
        'total': int(trainable + non_trainable)
    }


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def get_model_size(model: tf.keras.Model) -> Dict[str, str]:
    """
    Get model size in memory.
    
    Args:
        model: Keras model
        
    Returns:
        Dictionary with size information
    """
    params = count_parameters(model)
    
    # Assuming float32 weights (4 bytes per parameter)
    size_bytes = params['total'] * 4
    
    return {
        'parameters': params['total'],
        'size_bytes': size_bytes,
        'size_formatted': format_size(size_bytes)
    }


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self.start_time
        if self.name:
            print(f"{self.name}: {self.elapsed:.4f}s")


def load_class_names(path: str = None) -> list:
    """
    Load class names for ASL alphabet.
    
    Args:
        path: Path to class names file (uses default if None)
        
    Returns:
        List of class names
    """
    default_classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DELETE"]
    
    if path is None:
        return default_classes
        
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]
            
    return default_classes


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    set_seed(42)
    print("Set random seed")
    
    device_info = get_device_info()
    print(f"TensorFlow version: {device_info['tensorflow_version']}")
    print(f"GPUs available: {device_info['num_gpus']}")
    
    exp_dir = create_experiment_dir(experiment_name="test_experiment")
    print(f"Experiment dir: {exp_dir}")

