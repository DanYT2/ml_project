#!/usr/bin/env python3
"""
Convert Keras model to TensorFlow.js format.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def get_weight_name(layer, weight_index, weight_shape):
    """Generate proper TensorFlow.js compatible weight name."""
    layer_name = layer.name
    layer_class = layer.__class__.__name__
    
    if layer_class == 'Dense':
        return f"{layer_name}/{'kernel' if weight_index == 0 else 'bias'}"
    elif layer_class == 'Conv1D' or layer_class == 'Conv2D':
        return f"{layer_name}/{'kernel' if weight_index == 0 else 'bias'}"
    elif layer_class == 'BatchNormalization':
        # BatchNorm has 4 weights: gamma, beta, moving_mean, moving_variance
        weight_names = ['gamma', 'beta', 'moving_mean', 'moving_variance']
        return f"{layer_name}/{weight_names[weight_index]}"
    else:
        return f"{layer_name}/w{weight_index}"


def create_tfjs_model(model, output_path: str):
    """
    Create TF.js model files with proper weight naming.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Collect all weights with proper names
    weight_data = []
    weight_specs = []
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        for i, w in enumerate(layer_weights):
            weight_name = get_weight_name(layer, i, w.shape)
            weight_specs.append({
                "name": weight_name,
                "shape": list(w.shape),
                "dtype": "float32"
            })
            weight_data.append(w.astype(np.float32).flatten())
    
    # Concatenate all weights into single buffer
    if weight_data:
        all_weights = np.concatenate(weight_data)
    else:
        all_weights = np.array([], dtype=np.float32)
    
    # Save weights as binary file
    weights_path = os.path.join(output_path, "group1-shard1of1.bin")
    all_weights.tofile(weights_path)
    print(f"  Saved weights: {len(all_weights) * 4 / 1024:.2f} KB")
    
    # Get model config for topology
    config = model.get_config()
    
    # Create model.json with proper structure
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.keras.__version__}",
        "convertedBy": "TFjs-Keras-Converter v1.0",
        "modelTopology": {
            "class_name": model.__class__.__name__,
            "config": config,
            "keras_version": tf.keras.__version__,
            "backend": "tensorflow"
        },
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": weight_specs
        }]
    }
    
    model_json_path = os.path.join(output_path, "model.json")
    with open(model_json_path, "w") as f:
        json.dump(model_json, f, indent=2)
    print(f"  Saved model.json")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Keras model to TensorFlow.js")
    parser.add_argument("--model", required=True, help="Path to .keras model")
    parser.add_argument("--output", default="web/public/model", help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ASL Model to TensorFlow.js Converter")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)
    
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"\nModel architecture:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Convert to TF.js format
    print(f"\nConverting to TensorFlow.js format...")
    create_tfjs_model(model, args.output)
    
    # Create metadata
    metadata = {
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "num_classes": model.output_shape[-1],
        "quantized": False,
        "class_names": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DELETE"],
        "description": "ASL Alphabet Recognition Model",
        "version": "1.0.0",
        "total_params": total_params,
        "notes": "Trained on Kaggle ASL Alphabet dataset with MediaPipe landmarks"
    }
    
    metadata_path = os.path.join(args.output, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata.json")
    
    # List output files
    print(f"\nOutput files in {args.output}:")
    total_size = 0
    for file in sorted(Path(args.output).glob("*")):
        size = file.stat().st_size
        total_size += size
        print(f"  {file.name}: {size / 1024:.2f} KB")
    
    print(f"\nTotal size: {total_size / 1024 / 1024:.2f} MB")
    print("\n✅ Conversion complete!")
    print("\nNext steps:")
    print("  1. cd web")
    print("  2. npm install")
    print("  3. npm run dev")
    print("  4. Open http://localhost:3000")


if __name__ == "__main__":
    main()












