#!/usr/bin/env python3
"""
Download ASL Alphabet dataset from Kaggle.

This script downloads and extracts the ASL Alphabet dataset
which contains images of hand gestures for letters A-Z.

Prerequisites:
- Kaggle API credentials (~/.kaggle/kaggle.json)
- kaggle package installed (pip install kaggle)

Usage:
    python scripts/download_dataset.py --output data/
"""

import os
import sys
import argparse
import zipfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_kaggle_dataset(output_dir: str = "data"):
    """
    Download ASL Alphabet dataset from Kaggle.
    
    Args:
        output_dir: Directory to save dataset
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Error: kaggle package not installed.")
        print("Install with: pip install kaggle")
        sys.exit(1)
        
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_name = "grassknoted/asl-alphabet"
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    # Download
    api.dataset_download_files(
        dataset_name,
        path=output_dir,
        unzip=True
    )
    
    print("Download complete!")
    
    # Verify structure
    verify_dataset_structure(output_dir)


def verify_dataset_structure(data_dir: str):
    """
    Verify the downloaded dataset has expected structure.
    
    Args:
        data_dir: Path to data directory
    """
    print("\nVerifying dataset structure...")
    
    expected_train = Path(data_dir) / "asl_alphabet_train" / "asl_alphabet_train"
    expected_test = Path(data_dir) / "asl_alphabet_test" / "asl_alphabet_test"
    
    # Alternative paths (some versions have different structure)
    alt_train = Path(data_dir) / "asl_alphabet_train"
    
    train_path = None
    if expected_train.exists():
        train_path = expected_train
    elif alt_train.exists():
        train_path = alt_train
        
    if train_path:
        classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
        print(f"Found training directory: {train_path}")
        print(f"Classes found: {len(classes)}")
        
        # Count images per class
        total_images = 0
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                total_images += len(images)
                
        print(f"Total training images: {total_images}")
    else:
        print("Warning: Training directory not found in expected locations")
        print("Please verify the dataset structure manually")
        
    if expected_test.exists():
        test_images = list(expected_test.glob("*.jpg")) + list(expected_test.glob("*.png"))
        print(f"Test images: {len(test_images)}")
        

def create_sample_dataset(output_dir: str = "data/sample"):
    """
    Create a small sample dataset for testing.
    
    Uses random noise as placeholder data.
    
    Args:
        output_dir: Output directory for sample data
    """
    import numpy as np
    import cv2
    
    print(f"Creating sample dataset in {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    samples_per_class = 10
    
    for class_name in classes:
        class_dir = os.path.join(output_dir, "asl_alphabet_train", "asl_alphabet_train", class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            # Create random image (placeholder)
            img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
            img_path = os.path.join(class_dir, f"{class_name}_{i}.jpg")
            cv2.imwrite(img_path, img)
            
    print(f"Created sample dataset with {len(classes) * samples_per_class} images")


def main():
    parser = argparse.ArgumentParser(
        description="Download ASL Alphabet dataset from Kaggle"
    )
    parser.add_argument(
        "--output", "-o",
        default="data",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create sample dataset instead of downloading"
    )
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_dataset(args.output)
    else:
        download_kaggle_dataset(args.output)


if __name__ == "__main__":
    main()

