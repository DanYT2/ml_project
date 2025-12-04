"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { useASLModel } from "@/lib/useASLModel";
import { useMediaPipe } from "@/lib/useMediaPipe";
import { drawHandLandmarks } from "@/lib/drawUtils";

interface CameraProps {
  onPrediction: (letter: string | null, confidence: number, fps: number, handDetected: boolean) => void;
  onModelLoaded: () => void;
  showLandmarks: boolean;
}

export default function Camera({ onPrediction, onModelLoaded, showLandmarks }: CameraProps) {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  
  // Custom hooks for model and MediaPipe
  const { model, isLoading: modelLoading, predict } = useASLModel();
  const { hands, isLoading: mediapipeLoading, extractLandmarks } = useMediaPipe();
  
  // FPS tracking
  const fpsRef = useRef({ frameCount: 0, lastTime: performance.now(), fps: 0 });
  const animationRef = useRef<number>();

  // Process frame
  const processFrame = useCallback(async () => {
    if (!webcamRef.current?.video || !model || !hands) {
      animationRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const video = webcamRef.current.video;
    
    if (video.readyState !== 4) {
      animationRef.current = requestAnimationFrame(processFrame);
      return;
    }

    // Calculate FPS
    const now = performance.now();
    fpsRef.current.frameCount++;
    if (now - fpsRef.current.lastTime >= 1000) {
      fpsRef.current.fps = fpsRef.current.frameCount;
      fpsRef.current.frameCount = 0;
      fpsRef.current.lastTime = now;
    }

    try {
      // Extract landmarks
      const result = await extractLandmarks(video);
      
      if (result && result.landmarks) {
        // Draw landmarks on canvas
        if (canvasRef.current && showLandmarks) {
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) {
            canvasRef.current.width = video.videoWidth;
            canvasRef.current.height = video.videoHeight;
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            drawHandLandmarks(ctx, result.landmarks, canvasRef.current.width, canvasRef.current.height);
          }
        }

        // Predict gesture
        const prediction = await predict(result.normalizedLandmarks);
        
        if (prediction) {
          onPrediction(
            prediction.letter,
            prediction.confidence,
            fpsRef.current.fps,
            true
          );
        } else {
          onPrediction(null, 0, fpsRef.current.fps, true);
        }
      } else {
        // No hand detected
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          }
        }
        onPrediction(null, 0, fpsRef.current.fps, false);
      }
    } catch (error) {
      console.error("Frame processing error:", error);
    }

    animationRef.current = requestAnimationFrame(processFrame);
  }, [model, hands, extractLandmarks, predict, onPrediction, showLandmarks]);

  // Initialize and start processing
  useEffect(() => {
    if (!modelLoading && !mediapipeLoading && model && hands && !isInitialized) {
      setIsInitialized(true);
      onModelLoaded();
      processFrame();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [modelLoading, mediapipeLoading, model, hands, isInitialized, onModelLoaded, processFrame]);

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
    frameRate: { ideal: 30 },
  };

  return (
    <div className="camera-container w-full h-full relative">
      <Webcam
        ref={webcamRef}
        audio={false}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        className="w-full h-full object-cover"
        mirrored
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ transform: "scaleX(-1)" }}
      />
    </div>
  );
}

