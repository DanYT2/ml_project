"use client";

import { useMemo } from "react";
import { Hand, AlertCircle, CheckCircle } from "lucide-react";

interface PredictionDisplayProps {
  letter: string | null;
  confidence: number;
  threshold: number;
}

export default function PredictionDisplay({ 
  letter, 
  confidence, 
  threshold 
}: PredictionDisplayProps) {
  const confidencePercent = Math.round(confidence * 100);
  const isConfident = confidence >= threshold;
  
  const confidenceColor = useMemo(() => {
    if (confidence >= 0.9) return "from-green-500 to-emerald-500";
    if (confidence >= threshold) return "from-primary-500 to-amber-500";
    if (confidence >= 0.5) return "from-amber-500 to-orange-500";
    return "from-red-500 to-rose-500";
  }, [confidence, threshold]);

  const statusColor = useMemo(() => {
    if (confidence >= 0.9) return "text-green-400";
    if (confidence >= threshold) return "text-primary-400";
    if (confidence >= 0.5) return "text-amber-400";
    return "text-red-400";
  }, [confidence, threshold]);

  return (
    <div className="flex items-center gap-6">
      {/* Letter Display */}
      <div className="flex-shrink-0">
        {letter ? (
          <div className="relative">
            <div 
              className={`
                w-20 h-20 rounded-2xl flex items-center justify-center
                font-display text-4xl font-bold
                transition-all duration-300
                ${isConfident 
                  ? 'bg-gradient-to-br from-primary-500 to-primary-600 text-white shadow-lg shadow-primary-500/30 scale-100' 
                  : 'bg-surface-800/50 text-surface-400 border border-surface-700/50 scale-95'
                }
              `}
            >
              {letter === "DELETE" ? "⌫" : letter}
            </div>
            {isConfident && (
              <div className="absolute -top-1 -right-1">
                <CheckCircle className="w-5 h-5 text-green-400" />
              </div>
            )}
          </div>
        ) : (
          <div className="w-20 h-20 rounded-2xl bg-surface-800/30 border border-surface-700/50 flex items-center justify-center">
            <Hand className="w-8 h-8 text-surface-600" />
          </div>
        )}
      </div>

      {/* Confidence Display */}
      <div className="flex-1 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-surface-400">Confidence</span>
          <span className={`text-lg font-mono font-bold ${statusColor}`}>
            {letter ? `${confidencePercent}%` : '--'}
          </span>
        </div>
        
        {/* Confidence Bar */}
        <div className="confidence-bar">
          <div 
            className={`confidence-fill bg-gradient-to-r ${confidenceColor}`}
            style={{ width: `${letter ? confidencePercent : 0}%` }}
          />
        </div>

        {/* Status Message */}
        <div className="flex items-center gap-2 text-sm">
          {letter ? (
            isConfident ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-400" />
                <span className="text-green-400">High confidence - Letter accepted</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-4 h-4 text-amber-400" />
                <span className="text-amber-400">Low confidence - Hold gesture steady</span>
              </>
            )
          ) : (
            <span className="text-surface-500">Position your hand in frame</span>
          )}
        </div>
      </div>

      {/* Top Predictions (could show top 3) */}
      {letter && (
        <div className="hidden md:block flex-shrink-0 text-right">
          <span className="text-xs text-surface-500 block mb-1">Predicted</span>
          <div className="flex gap-1">
            {[letter].map((l, i) => (
              <span 
                key={i}
                className={`
                  inline-flex items-center justify-center w-8 h-8 rounded-lg
                  font-bold text-sm
                  ${i === 0 
                    ? 'bg-primary-500/20 text-primary-400 border border-primary-500/30' 
                    : 'bg-surface-800/50 text-surface-400'
                  }
                `}
              >
                {l === "DELETE" ? "⌫" : l}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}



