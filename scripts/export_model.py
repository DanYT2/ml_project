#!/usr/bin/env python3
"""
Export trained model to TensorFlow.js format for web deployment.

This script converts a trained Keras model to TensorFlow.js format
which can be loaded in the browser for real-time inference.

Usage:
    python scripts/export_model.py --model models/static_model --output web/public/model
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_to_tfjs(
    model_path: str,
    output_path: str,
    quantize: bool = True,
    optimize: bool = True
):
    """
    Export Keras model to TensorFlow.js format.
    
    Args:
        model_path: Path to saved Keras model
        output_path: Output directory for TF.js model
        quantize: Whether to quantize weights (reduces size)
        optimize: Whether to apply graph optimizations
    """
    import tensorflow as tf
    import tensorflowjs as tfjs
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    print(f"Model loaded successfully")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Conversion options
    quantization_dtype = 'uint16' if quantize else None
    
    print(f"Exporting to TensorFlow.js format...")
    print(f"Output directory: {output_path}")
    print(f"Quantization: {quantize}")
    
    # Convert to TF.js
    if quantize:
        tfjs.converters.save_keras_model(
            model,
            output_path,
            quantization_dtype_map={'uint16': '*'}
        )
    else:
        tfjs.converters.save_keras_model(model, output_path)
        
    print(f"Export complete!")
    
    # Calculate model size
    total_size = 0
    for file in Path(output_path).glob("*"):
        total_size += file.stat().st_size
        
    print(f"Total model size: {total_size / 1024 / 1024:.2f} MB")
    
    # Save model metadata
    metadata = {
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'num_classes': model.output_shape[-1],
        'quantized': quantize,
        'class_names': list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DELETE"]
    }
    
    metadata_path = os.path.join(output_path, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved metadata to: {metadata_path}")


def create_dummy_model(output_path: str):
    """
    Create a dummy model for testing web app.
    
    Args:
        output_path: Output directory for model
    """
    import tensorflow as tf
    from src.models.landmark_model import create_landmark_model
    from src.models.hybrid_model import create_static_model
    
    print("Creating dummy model for testing...")
    
    # Create model
    model = create_static_model(num_classes=27)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Initialize weights with a forward pass
    dummy_input = tf.random.normal((1, 63))
    _ = model(dummy_input)
    
    print("Model created and initialized")
    
    # Export
    export_to_tfjs(
        model_path=None,  # We'll save directly
        output_path=output_path,
        quantize=True
    )
    
    return model


def export_to_tfjs_direct(model, output_path: str, quantize: bool = True):
    """
    Export model directly without saving to disk first.
    
    Args:
        model: Keras model instance
        output_path: Output directory
        quantize: Whether to quantize
    """
    import tensorflowjs as tfjs
    
    os.makedirs(output_path, exist_ok=True)
    
    if quantize:
        tfjs.converters.save_keras_model(
            model,
            output_path,
            quantization_dtype_map={'uint16': '*'}
        )
    else:
        tfjs.converters.save_keras_model(model, output_path)
        
    # Save metadata
    metadata = {
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'num_classes': model.output_shape[-1] if model.output_shape[-1] else 27,
        'quantized': quantize,
        'class_names': list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DELETE"]
    }
    
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


def optimize_for_web(model_path: str, output_path: str):
    """
    Apply additional optimizations for web deployment.
    
    Args:
        model_path: Path to model
        output_path: Output path
    """
    import tensorflow as tf
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to float16 for faster inference
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # This creates a TFLite model (alternative to TF.js)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(output_path, 'model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"TFLite model saved to: {tflite_path}")
    print(f"TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export model to TensorFlow.js format"
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Path to saved Keras model"
    )
    parser.add_argument(
        "--output", "-o",
        default="web/public/model",
        help="Output directory for TF.js model"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable weight quantization"
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create a dummy model for testing"
    )
    parser.add_argument(
        "--tflite",
        action="store_true",
        help="Also export TFLite model"
    )
    
    args = parser.parse_args()
    
    if args.create_dummy:
        # Create and export dummy model
        import tensorflow as tf
        from src.models.hybrid_model import create_static_model
        
        print("Creating dummy model for testing...")
        model = create_static_model(num_classes=27)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Initialize
        dummy_input = tf.random.normal((1, 63))
        _ = model(dummy_input)
        
        export_to_tfjs_direct(
            model,
            args.output,
            quantize=not args.no_quantize
        )
        
        # Verify export
        print("\nVerifying exported model...")
        files = list(Path(args.output).glob("*"))
        for f in files:
            print(f"  {f.name}: {f.stat().st_size / 1024:.2f} KB")
            
    elif args.model:
        export_to_tfjs(
            args.model,
            args.output,
            quantize=not args.no_quantize
        )
        
        if args.tflite:
            optimize_for_web(args.model, args.output)
    else:
        print("Error: Please provide --model path or use --create-dummy")
        sys.exit(1)


if __name__ == "__main__":
    main()

