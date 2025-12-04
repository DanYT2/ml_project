"""
Model evaluation metrics and utilities.

This module provides comprehensive model evaluation including:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Per-class performance
- Error analysis
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


class ModelEvaluator:
    """
    Comprehensive model evaluation.
    
    Provides detailed evaluation metrics and visualizations
    for ASL gesture recognition models.
    """
    
    # Default class names
    DEFAULT_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DELETE"]
    
    def __init__(
        self,
        model: keras.Model,
        class_names: List[str] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.num_classes = len(self.class_names)
        
        # Store evaluation results
        self.results = {}
        
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            X: Input features
            y: True labels
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary of evaluation results
        """
        print("Running model evaluation...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X, batch_size=batch_size, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Basic metrics
        results = {
            'accuracy': accuracy_score(y, y_pred),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # Top-k accuracy
        for k in [3, 5]:
            if self.num_classes >= k:
                results[f'top{k}_accuracy'] = top_k_accuracy_score(y, y_pred_proba, k=k)
                
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y, y_pred)
        
        # Per-class metrics
        results['per_class'] = self._per_class_metrics(y, y_pred)
        
        # Error analysis
        results['errors'] = self._error_analysis(X, y, y_pred, y_pred_proba)
        
        self.results = results
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class precision, recall, F1."""
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        per_class = {}
        for class_name in self.class_names:
            if class_name in report:
                per_class[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
                
        return per_class
    
    def _error_analysis(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = {
            'indices': np.where(y_true != y_pred)[0],
            'confusion_pairs': [],
            'low_confidence_correct': [],
            'high_confidence_wrong': []
        }
        
        # Find most confused class pairs
        cm = confusion_matrix(y_true, y_pred)
        np.fill_diagonal(cm, 0)  # Remove diagonal
        
        # Top confused pairs
        n_pairs = 10
        flat_indices = np.argsort(cm.flatten())[-n_pairs:]
        for idx in flat_indices[::-1]:
            true_class = idx // self.num_classes
            pred_class = idx % self.num_classes
            count = cm[true_class, pred_class]
            if count > 0:
                errors['confusion_pairs'].append({
                    'true': self.class_names[true_class],
                    'predicted': self.class_names[pred_class],
                    'count': int(count)
                })
                
        # Confidence analysis
        confidences = np.max(y_proba, axis=1)
        
        # Low confidence correct predictions
        correct_mask = y_true == y_pred
        low_conf_correct = np.where(correct_mask & (confidences < 0.7))[0]
        errors['low_confidence_correct'] = {
            'count': len(low_conf_correct),
            'mean_confidence': float(np.mean(confidences[low_conf_correct])) if len(low_conf_correct) > 0 else 0
        }
        
        # High confidence wrong predictions
        wrong_mask = y_true != y_pred
        high_conf_wrong = np.where(wrong_mask & (confidences > 0.9))[0]
        errors['high_confidence_wrong'] = {
            'count': len(high_conf_wrong),
            'mean_confidence': float(np.mean(confidences[high_conf_wrong])) if len(high_conf_wrong) > 0 else 0
        }
        
        return errors
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Precision (macro): {results['precision_macro']:.4f}")
        print(f"  Recall (macro):    {results['recall_macro']:.4f}")
        print(f"  F1 (macro):        {results['f1_macro']:.4f}")
        
        if 'top3_accuracy' in results:
            print(f"  Top-3 Accuracy:   {results['top3_accuracy']:.4f}")
        if 'top5_accuracy' in results:
            print(f"  Top-5 Accuracy:   {results['top5_accuracy']:.4f}")
            
        # Worst performing classes
        per_class = results['per_class']
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['f1'])
        
        print(f"\nWorst Performing Classes:")
        for class_name, metrics in sorted_classes[:5]:
            print(f"  {class_name}: F1={metrics['f1']:.3f}, "
                  f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
            
        # Most confused pairs
        if results['errors']['confusion_pairs']:
            print(f"\nMost Confused Class Pairs:")
            for pair in results['errors']['confusion_pairs'][:5]:
                print(f"  {pair['true']} → {pair['predicted']}: {pair['count']} errors")
                
        print("=" * 60)
        
    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        figsize: Tuple[int, int] = (15, 12),
        save_path: str = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            normalize: Whether to normalize the matrix
            figsize: Figure size
            save_path: Path to save figure
        """
        if 'confusion_matrix' not in self.results:
            print("No evaluation results. Run evaluate() first.")
            return
            
        cm = self.results['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)
            
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True if len(self.class_names) <= 27 else False,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            square=True
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved confusion matrix to {save_path}")
        else:
            plt.show()
            
    def plot_per_class_metrics(
        self,
        metric: str = 'f1',
        figsize: Tuple[int, int] = (14, 6),
        save_path: str = None
    ):
        """
        Plot per-class metrics.
        
        Args:
            metric: Metric to plot ('precision', 'recall', 'f1')
            figsize: Figure size
            save_path: Path to save figure
        """
        if 'per_class' not in self.results:
            print("No evaluation results. Run evaluate() first.")
            return
            
        per_class = self.results['per_class']
        
        classes = list(per_class.keys())
        values = [per_class[c][metric] for c in classes]
        
        # Sort by value
        sorted_indices = np.argsort(values)
        classes = [classes[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        plt.figure(figsize=figsize)
        colors = plt.cm.RdYlGn(np.array(values))
        plt.barh(classes, values, color=colors)
        plt.xlabel(metric.capitalize())
        plt.title(f'Per-Class {metric.capitalize()} Score')
        plt.xlim(0, 1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved per-class metrics to {save_path}")
        else:
            plt.show()
            
    def save_results(self, path: str):
        """
        Save evaluation results to JSON.
        
        Args:
            path: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                results_json[key] = value.tolist()
            elif isinstance(value, dict):
                results_json[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                results_json[key] = value
                
        with open(path, 'w') as f:
            json.dump(results_json, f, indent=2)
            
        print(f"Saved evaluation results to {path}")


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str] = None,
    output_dir: str = "evaluation_results"
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Class names
        output_dir: Directory to save results
        
    Returns:
        Evaluation results dictionary
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = ModelEvaluator(model, class_names)
    results = evaluator.evaluate(X_test, y_test)
    
    # Save visualizations
    evaluator.plot_confusion_matrix(
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    evaluator.plot_per_class_metrics(
        save_path=os.path.join(output_dir, "per_class_f1.png")
    )
    
    # Save results
    evaluator.save_results(os.path.join(output_dir, "results.json"))
    
    return results


def compute_latency(
    model: keras.Model,
    input_shape: Tuple[int, ...] = (63,),
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Measure model inference latency.
    
    Args:
        model: Model to measure
        input_shape: Input shape for test data
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Dictionary with latency statistics
    """
    import time
    
    # Create dummy input
    dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model.predict(dummy_input, verbose=0)
        
    # Measure
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(dummy_input, verbose=0)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
        
    latencies = np.array(latencies)
    
    return {
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99))
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing model evaluation...")
    
    # Create dummy model and data
    from ..models.landmark_model import create_landmark_model
    
    model = create_landmark_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Dummy data
    X_test = np.random.randn(100, 63).astype(np.float32)
    y_test = np.random.randint(0, 27, 100)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    # Measure latency
    latency = compute_latency(model)
    print(f"\nLatency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")

