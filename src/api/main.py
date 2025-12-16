"""
FastAPI application for ASL gesture classification.

This API provides endpoints to run predictions using the trained ASL model.
"""

from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Constants
MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models"
DEFAULT_MODEL_NAME = "asl_classifier_20251209_213340_final.keras"

CLASS_LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "DELETE"
]

# Global model reference
model: tf.keras.Model | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    
    model_path = MODEL_DIR / DEFAULT_MODEL_NAME
    
    if not model_path.exists():
        # Try to find any .keras file in models directory
        keras_files = list(MODEL_DIR.glob("*.keras"))
        if keras_files:
            model_path = keras_files[-1]  # Use the most recent one
            print(f"Using model: {model_path}")
        else:
            raise FileNotFoundError(
                f"No model found. Expected at {MODEL_DIR / DEFAULT_MODEL_NAME}"
            )
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    yield
    
    # Cleanup
    del model
    print("Model unloaded")


# Create FastAPI app
app = FastAPI(
    title="ASL Gesture Recognition API",
    description="API for classifying American Sign Language (ASL) gestures from MediaPipe hand landmarks",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class LandmarksRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    landmarks: list[float] = Field(
        ...,
        description="Flattened array of 21 hand landmarks × 3 coordinates (x, y, z) = 63 values",
        min_length=63,
        max_length=63,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "landmarks": [0.0] * 63  # Example with zeros
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    """Single prediction result."""
    
    letter: str = Field(..., description="Predicted ASL letter")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    prediction: PredictionResult = Field(..., description="Top prediction")
    top_k: list[PredictionResult] = Field(
        ..., 
        description="Top K predictions with confidence scores"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_input_shape: list | None = None
    model_output_shape: list | None = None


# Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "ASL Gesture Recognition API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and model health status."""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_input_shape=list(model.input_shape) if model else None,
        model_output_shape=list(model.output_shape) if model else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: LandmarksRequest):
    """
    Predict ASL letter from hand landmarks.
    
    The landmarks should be normalized MediaPipe hand landmarks:
    - 21 landmarks per hand
    - Each landmark has (x, y, z) coordinates
    - Total: 63 values (21 × 3)
    
    Returns the predicted letter and confidence scores for top predictions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert landmarks to numpy array and reshape for model
        landmarks_array = np.array(request.landmarks, dtype=np.float32)
        input_tensor = landmarks_array.reshape(1, 63)
        
        # Run prediction
        predictions = model.predict(input_tensor, verbose=0)
        probabilities = predictions[0]
        
        # Get sorted indices by probability
        sorted_indices = np.argsort(probabilities)[::-1]
        
        # Build top-k predictions
        top_k = [
            PredictionResult(
                letter=CLASS_LABELS[idx],
                confidence=float(probabilities[idx])
            )
            for idx in sorted_indices[:5]
        ]
        
        # Top prediction
        best_idx = sorted_indices[0]
        prediction = PredictionResult(
            letter=CLASS_LABELS[best_idx],
            confidence=float(probabilities[best_idx])
        )
        
        return PredictionResponse(
            prediction=prediction,
            top_k=top_k
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/labels", tags=["Info"])
async def get_labels():
    """Get the list of class labels the model can predict."""
    return {"labels": CLASS_LABELS, "count": len(CLASS_LABELS)}












