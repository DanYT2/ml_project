"""
Model explainability for ASL gesture recognition.

This module provides tools to understand model predictions:
- Grad-CAM for landmark importance visualization
- Feature attribution
- Prediction explanations
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for landmarks.
    
    Adapts Grad-CAM for 1D landmark data to show which landmarks
    are most important for classification decisions.
    """
    
    # Hand landmark names
    LANDMARK_NAMES = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]
    
    # Hand connections for visualization
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17)  # Palm
    ]
    
    def __init__(self, model: keras.Model, target_layer: str = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            target_layer: Name of target layer for gradients (auto-detected if None)
        """
        self.model = model
        
        # Find target layer (last conv or dense before output)
        if target_layer is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower() or 'dense' in layer.name.lower():
                    if layer != model.layers[-1]:  # Not the output layer
                        target_layer = layer.name
                        break
                        
        if target_layer is None:
            raise ValueError("Could not find suitable target layer")
            
        self.target_layer = target_layer
        
        # Create gradient model
        self.grad_model = self._create_grad_model()
        
    def _create_grad_model(self) -> keras.Model:
        """Create model for computing gradients."""
        target_layer = self.model.get_layer(self.target_layer)
        
        return keras.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output]
        )
        
    def compute_heatmap(
        self,
        landmarks: np.ndarray,
        class_idx: int = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for landmarks.
        
        Args:
            landmarks: Input landmarks of shape (63,) or (1, 63)
            class_idx: Target class index (uses predicted class if None)
            
        Returns:
            Heatmap array of shape (21,) showing landmark importance
        """
        if len(landmarks.shape) == 1:
            landmarks = landmarks[np.newaxis, :]
            
        landmarks = tf.cast(landmarks, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(landmarks)
            conv_output, predictions = self.grad_model(landmarks)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
                
            class_output = predictions[:, class_idx]
            
        # Compute gradients
        grads = tape.gradient(class_output, conv_output)
        
        # Global average pooling of gradients
        if len(grads.shape) == 3:
            pooled_grads = tf.reduce_mean(grads, axis=1)
        else:
            pooled_grads = grads
            
        # Weight feature maps by pooled gradients
        conv_output = conv_output[0]
        if len(conv_output.shape) == 2:
            heatmap = tf.reduce_sum(conv_output * pooled_grads[0], axis=-1)
        else:
            heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
            
        # ReLU and normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
    
    def compute_landmark_importance(
        self,
        landmarks: np.ndarray,
        class_idx: int = None
    ) -> np.ndarray:
        """
        Compute importance score for each of the 21 landmarks.
        
        Uses gradient-based attribution to determine which landmarks
        contribute most to the prediction.
        
        Args:
            landmarks: Input landmarks of shape (63,)
            class_idx: Target class index
            
        Returns:
            Importance scores of shape (21,)
        """
        if len(landmarks.shape) == 1:
            landmarks = landmarks[np.newaxis, :]
            
        landmarks = tf.cast(landmarks, tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(landmarks)
            predictions = self.model(landmarks)
            
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
                
            class_output = predictions[:, class_idx]
            
        # Compute gradients w.r.t. input
        grads = tape.gradient(class_output, landmarks)
        grads = grads.numpy()[0]
        
        # Reshape to (21, 3) and compute norm per landmark
        grads = grads.reshape(21, 3)
        importance = np.linalg.norm(grads, axis=1)
        
        # Normalize
        importance = importance / (importance.max() + 1e-10)
        
        return importance
    
    def visualize_landmark_importance(
        self,
        landmarks: np.ndarray,
        class_idx: int = None,
        class_name: str = None,
        figsize: Tuple[int, int] = (10, 10),
        save_path: str = None
    ):
        """
        Visualize landmark importance on a hand diagram.
        
        Args:
            landmarks: Input landmarks of shape (63,)
            class_idx: Target class index
            class_name: Name of the class for title
            figsize: Figure size
            save_path: Path to save figure
        """
        importance = self.compute_landmark_importance(landmarks, class_idx)
        
        # Reshape landmarks to (21, 3)
        points = landmarks.reshape(21, 3)[:, :2]  # Use only x, y
        
        # Flip y for display
        points[:, 1] = 1 - points[:, 1]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw connections
        for start, end in self.CONNECTIONS:
            ax.plot(
                [points[start, 0], points[end, 0]],
                [points[start, 1], points[end, 1]],
                'gray', linewidth=2, alpha=0.5
            )
            
        # Draw landmarks with importance coloring
        scatter = ax.scatter(
            points[:, 0], points[:, 1],
            c=importance, cmap='hot',
            s=300, edgecolors='black', linewidths=2,
            vmin=0, vmax=1
        )
        
        # Add landmark labels for high importance
        for i, (x, y) in enumerate(points):
            if importance[i] > 0.5:
                ax.annotate(
                    self.LANDMARK_NAMES[i],
                    (x, y), textcoords="offset points",
                    xytext=(10, 10), fontsize=8
                )
                
        plt.colorbar(scatter, ax=ax, label='Importance')
        
        title = f"Landmark Importance"
        if class_name:
            title += f" for '{class_name}'"
        ax.set_title(title)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
        return importance


class LandmarkImportance:
    """
    Integrated Gradients for landmark importance.
    
    More accurate attribution method compared to simple gradients.
    """
    
    def __init__(self, model: keras.Model, num_steps: int = 50):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: Keras model
            num_steps: Number of interpolation steps
        """
        self.model = model
        self.num_steps = num_steps
        
    def compute_integrated_gradients(
        self,
        landmarks: np.ndarray,
        class_idx: int = None,
        baseline: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute integrated gradients.
        
        Args:
            landmarks: Input landmarks of shape (63,)
            class_idx: Target class index
            baseline: Baseline input (zeros if None)
            
        Returns:
            Attribution scores of shape (63,)
        """
        if len(landmarks.shape) == 1:
            landmarks = landmarks[np.newaxis, :]
            
        if baseline is None:
            baseline = np.zeros_like(landmarks)
            
        landmarks = tf.cast(landmarks, tf.float32)
        baseline = tf.cast(baseline, tf.float32)
        
        # Generate interpolated inputs
        alphas = tf.linspace(0.0, 1.0, self.num_steps + 1)
        interpolated = baseline + alphas[:, tf.newaxis, tf.newaxis] * (landmarks - baseline)
        interpolated = tf.reshape(interpolated, (-1, landmarks.shape[-1]))
        
        # Compute gradients for all interpolated inputs
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            predictions = self.model(interpolated)
            
            if class_idx is None:
                # Use the prediction of the actual input
                class_idx = tf.argmax(self.model(landmarks)[0])
                
            class_outputs = predictions[:, class_idx]
            
        grads = tape.gradient(class_outputs, interpolated)
        
        # Average gradients
        avg_grads = tf.reduce_mean(grads, axis=0)
        
        # Integrated gradients = (input - baseline) * avg_grads
        integrated_grads = (landmarks[0] - baseline[0]) * avg_grads
        
        return integrated_grads.numpy()
    
    def get_landmark_importance(
        self,
        landmarks: np.ndarray,
        class_idx: int = None
    ) -> np.ndarray:
        """
        Get per-landmark importance scores.
        
        Args:
            landmarks: Input landmarks of shape (63,)
            class_idx: Target class index
            
        Returns:
            Importance scores of shape (21,)
        """
        attributions = self.compute_integrated_gradients(landmarks, class_idx)
        
        # Reshape to (21, 3) and compute norm
        attributions = attributions.reshape(21, 3)
        importance = np.abs(attributions).sum(axis=1)
        
        # Normalize
        importance = importance / (importance.max() + 1e-10)
        
        return importance


class PredictionExplainer:
    """
    Comprehensive prediction explanation.
    
    Combines multiple explanation methods to provide
    interpretable insights into model predictions.
    """
    
    DEFAULT_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DELETE"]
    
    def __init__(
        self,
        model: keras.Model,
        class_names: List[str] = None
    ):
        """
        Initialize explainer.
        
        Args:
            model: Keras model
            class_names: List of class names
        """
        self.model = model
        self.class_names = class_names or self.DEFAULT_CLASSES
        
        # Initialize explanation methods
        self.gradcam = GradCAM(model)
        self.integrated_grads = LandmarkImportance(model)
        
    def explain(
        self,
        landmarks: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            landmarks: Input landmarks of shape (63,)
            top_k: Number of top predictions to explain
            
        Returns:
            Dictionary with explanation details
        """
        if len(landmarks.shape) == 1:
            landmarks_batch = landmarks[np.newaxis, :]
        else:
            landmarks_batch = landmarks
            
        # Get prediction
        predictions = self.model.predict(landmarks_batch, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        explanation = {
            'predictions': [],
            'landmark_importance': {},
            'key_landmarks': []
        }
        
        # Top predictions
        for idx in top_indices:
            explanation['predictions'].append({
                'class': self.class_names[idx],
                'class_idx': int(idx),
                'confidence': float(predictions[idx])
            })
            
        # Landmark importance for top prediction
        predicted_class = top_indices[0]
        gradcam_importance = self.gradcam.compute_landmark_importance(
            landmarks, predicted_class
        )
        ig_importance = self.integrated_grads.get_landmark_importance(
            landmarks, predicted_class
        )
        
        # Average both methods
        combined_importance = (gradcam_importance + ig_importance) / 2
        
        explanation['landmark_importance'] = {
            'gradcam': gradcam_importance.tolist(),
            'integrated_gradients': ig_importance.tolist(),
            'combined': combined_importance.tolist()
        }
        
        # Key landmarks (top 5 most important)
        top_landmarks = np.argsort(combined_importance)[-5:][::-1]
        for idx in top_landmarks:
            explanation['key_landmarks'].append({
                'index': int(idx),
                'name': GradCAM.LANDMARK_NAMES[idx],
                'importance': float(combined_importance[idx])
            })
            
        return explanation
    
    def visualize_explanation(
        self,
        landmarks: np.ndarray,
        save_path: str = None,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Create visual explanation of prediction.
        
        Args:
            landmarks: Input landmarks
            save_path: Path to save visualization
            figsize: Figure size
        """
        explanation = self.explain(landmarks)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Prediction confidence
        classes = [p['class'] for p in explanation['predictions']]
        confidences = [p['confidence'] for p in explanation['predictions']]
        
        axes[0].barh(classes, confidences)
        axes[0].set_xlabel('Confidence')
        axes[0].set_title('Top Predictions')
        axes[0].set_xlim(0, 1)
        
        # Plot 2: Landmark importance (bar chart)
        importance = explanation['landmark_importance']['combined']
        top_indices = np.argsort(importance)[-10:]
        
        axes[1].barh(
            [GradCAM.LANDMARK_NAMES[i] for i in top_indices],
            [importance[i] for i in top_indices]
        )
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Key Landmarks')
        axes[1].set_xlim(0, 1)
        
        # Plot 3: Hand diagram
        points = landmarks.reshape(21, 3)[:, :2]
        points[:, 1] = 1 - points[:, 1]
        
        for start, end in GradCAM.CONNECTIONS:
            axes[2].plot(
                [points[start, 0], points[end, 0]],
                [points[start, 1], points[end, 1]],
                'gray', linewidth=2, alpha=0.5
            )
            
        scatter = axes[2].scatter(
            points[:, 0], points[:, 1],
            c=importance, cmap='hot',
            s=200, edgecolors='black', linewidths=1,
            vmin=0, vmax=1
        )
        plt.colorbar(scatter, ax=axes[2], label='Importance')
        axes[2].set_title(f"Predicted: {explanation['predictions'][0]['class']}")
        axes[2].set_xlim(-0.1, 1.1)
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_aspect('equal')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved explanation to {save_path}")
        else:
            plt.show()
            
        plt.close()


def explain_prediction(
    model: keras.Model,
    landmarks: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None
) -> Dict[str, Any]:
    """
    Convenience function to explain a prediction.
    
    Args:
        model: Keras model
        landmarks: Input landmarks
        class_names: Class names
        save_path: Path to save visualization
        
    Returns:
        Explanation dictionary
    """
    explainer = PredictionExplainer(model, class_names)
    explanation = explainer.explain(landmarks)
    
    if save_path:
        explainer.visualize_explanation(landmarks, save_path)
        
    return explanation


if __name__ == "__main__":
    # Test explainability
    print("Testing model explainability...")
    
    from ..models.landmark_model import create_landmark_model
    
    # Create model
    model = create_landmark_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Dummy landmarks
    landmarks = np.random.randn(63).astype(np.float32)
    
    # Explain
    explanation = explain_prediction(model, landmarks)
    
    print("\nPrediction Explanation:")
    print(f"Top prediction: {explanation['predictions'][0]['class']}")
    print(f"Confidence: {explanation['predictions'][0]['confidence']:.4f}")
    print("\nKey landmarks:")
    for lm in explanation['key_landmarks']:
        print(f"  {lm['name']}: {lm['importance']:.4f}")

