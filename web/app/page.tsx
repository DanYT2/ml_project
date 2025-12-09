"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import dynamic from "next/dynamic";
import { 
  Camera as CameraIcon, 
  Settings, 
  Hand, 
  Sparkles,
  Type,
  Trash2,
  RotateCcw,
  Volume2,
  VolumeX,
  Lightbulb,
  ChevronDown
} from "lucide-react";

// Dynamic imports for client-side only components
const Camera = dynamic(() => import("@/components/Camera"), { ssr: false });
const PredictionDisplay = dynamic(() => import("@/components/PredictionDisplay"), { ssr: false });

interface Settings {
  wordCompletion: boolean;
  confidenceThreshold: number;
  soundEnabled: boolean;
  showLandmarks: boolean;
}

export default function Home() {
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentLetter, setCurrentLetter] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [recognizedText, setRecognizedText] = useState<string>("");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [settings, setSettings] = useState<Settings>({
    wordCompletion: true,
    confidenceThreshold: 0.8,
    soundEnabled: false,
    showLandmarks: true,
  });
  const [showSettings, setShowSettings] = useState(false);
  const [fps, setFps] = useState(0);
  const [handDetected, setHandDetected] = useState(false);

  // Refs for debouncing
  const lastLetterRef = useRef<string | null>(null);
  const letterCountRef = useRef(0);
  const requiredCount = 5; // Need to see same letter N times before accepting

  // Handle prediction from camera
  const handlePrediction = useCallback((
    letter: string | null, 
    conf: number, 
    currentFps: number,
    detected: boolean
  ) => {
    setFps(currentFps);
    setHandDetected(detected);
    
    if (!letter || conf < settings.confidenceThreshold) {
      setCurrentLetter(null);
      setConfidence(0);
      letterCountRef.current = 0;
      lastLetterRef.current = null;
      return;
    }

    setCurrentLetter(letter);
    setConfidence(conf);

    // Debounce: require same letter multiple times
    if (letter === lastLetterRef.current) {
      letterCountRef.current++;
    } else {
      letterCountRef.current = 1;
      lastLetterRef.current = letter;
    }

    if (letterCountRef.current === requiredCount) {
      // Accept the letter
      if (letter === "DELETE") {
        setRecognizedText(prev => prev.slice(0, -1));
      } else {
        setRecognizedText(prev => prev + letter);
        
        // Play sound if enabled
        if (settings.soundEnabled) {
          const utterance = new SpeechSynthesisUtterance(letter);
          utterance.rate = 1.5;
          window.speechSynthesis.speak(utterance);
        }
      }
      
      // Reset counter to prevent repeated additions
      letterCountRef.current = -10; // Negative to add delay before next letter
    }
  }, [settings.confidenceThreshold, settings.soundEnabled]);

  // Update word suggestions
  useEffect(() => {
    if (settings.wordCompletion && recognizedText.length > 0) {
      const lastWord = recognizedText.split(" ").pop() || "";
      if (lastWord.length >= 2) {
        // Simple word suggestions (in production, use a proper dictionary API)
        const commonWords = [
          "HELLO", "HELP", "PLEASE", "THANK", "THANKS", "YOU", "YES", "NO",
          "GOOD", "BAD", "HAPPY", "SAD", "LOVE", "LIKE", "WANT", "NEED",
          "WHERE", "WHEN", "WHAT", "WHO", "WHY", "HOW", "HOME", "WORK",
          "FOOD", "WATER", "NAME", "NICE", "MEET", "FRIEND", "FAMILY"
        ];
        const filtered = commonWords.filter(w => 
          w.startsWith(lastWord.toUpperCase()) && w !== lastWord.toUpperCase()
        ).slice(0, 5);
        setSuggestions(filtered);
      } else {
        setSuggestions([]);
      }
    } else {
      setSuggestions([]);
    }
  }, [recognizedText, settings.wordCompletion]);

  // Handle suggestion click
  const handleSuggestionClick = (word: string) => {
    const words = recognizedText.split(" ");
    words[words.length - 1] = word;
    setRecognizedText(words.join(" ") + " ");
    setSuggestions([]);
  };

  // Clear text
  const handleClear = () => {
    setRecognizedText("");
    setSuggestions([]);
  };

  // Delete last character
  const handleDelete = () => {
    setRecognizedText(prev => prev.slice(0, -1));
  };

  // Add space
  const handleSpace = () => {
    setRecognizedText(prev => prev + " ");
  };

  return (
    <main className="min-h-screen p-4 md:p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="flex items-center justify-between mb-8 animate-fade-in">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg shadow-primary-500/25">
              <Hand className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-display font-bold text-white">
                ASL Interpreter
              </h1>
              <p className="text-sm text-surface-400">
                Real-time sign language recognition
              </p>
            </div>
          </div>
          
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="btn-secondary flex items-center gap-2"
          >
            <Settings className="w-4 h-4" />
            <span className="hidden sm:inline">Settings</span>
            <ChevronDown className={`w-4 h-4 transition-transform ${showSettings ? 'rotate-180' : ''}`} />
          </button>
        </header>

        {/* Settings Panel */}
        {showSettings && (
          <div className="glass-card p-6 mb-6 animate-slide-up">
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-primary-400" />
              Settings
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Word Completion Toggle */}
              <label className="flex items-center justify-between p-3 bg-surface-800/30 rounded-xl cursor-pointer hover:bg-surface-800/50 transition-colors">
                <div className="flex items-center gap-3">
                  <Type className="w-4 h-4 text-surface-400" />
                  <span className="text-sm text-surface-200">Word Completion</span>
                </div>
                <input
                  type="checkbox"
                  checked={settings.wordCompletion}
                  onChange={(e) => setSettings(s => ({ ...s, wordCompletion: e.target.checked }))}
                  className="w-5 h-5 rounded accent-primary-500"
                />
              </label>

              {/* Sound Toggle */}
              <label className="flex items-center justify-between p-3 bg-surface-800/30 rounded-xl cursor-pointer hover:bg-surface-800/50 transition-colors">
                <div className="flex items-center gap-3">
                  {settings.soundEnabled ? (
                    <Volume2 className="w-4 h-4 text-surface-400" />
                  ) : (
                    <VolumeX className="w-4 h-4 text-surface-400" />
                  )}
                  <span className="text-sm text-surface-200">Sound</span>
                </div>
                <input
                  type="checkbox"
                  checked={settings.soundEnabled}
                  onChange={(e) => setSettings(s => ({ ...s, soundEnabled: e.target.checked }))}
                  className="w-5 h-5 rounded accent-primary-500"
                />
              </label>

              {/* Show Landmarks */}
              <label className="flex items-center justify-between p-3 bg-surface-800/30 rounded-xl cursor-pointer hover:bg-surface-800/50 transition-colors">
                <div className="flex items-center gap-3">
                  <Sparkles className="w-4 h-4 text-surface-400" />
                  <span className="text-sm text-surface-200">Show Landmarks</span>
                </div>
                <input
                  type="checkbox"
                  checked={settings.showLandmarks}
                  onChange={(e) => setSettings(s => ({ ...s, showLandmarks: e.target.checked }))}
                  className="w-5 h-5 rounded accent-primary-500"
                />
              </label>

              {/* Confidence Threshold */}
              <div className="p-3 bg-surface-800/30 rounded-xl">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-surface-200">Confidence</span>
                  <span className="text-sm text-primary-400 font-mono">
                    {Math.round(settings.confidenceThreshold * 100)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="0.95"
                  step="0.05"
                  value={settings.confidenceThreshold}
                  onChange={(e) => setSettings(s => ({ ...s, confidenceThreshold: parseFloat(e.target.value) }))}
                  className="w-full h-2 bg-surface-700 rounded-full appearance-none cursor-pointer accent-primary-500"
                />
              </div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Camera Section */}
          <div className="lg:col-span-2">
            <div className="glass-card overflow-hidden animate-slide-up">
              {/* Camera Header */}
              <div className="flex items-center justify-between p-4 border-b border-surface-700/30">
                <div className="flex items-center gap-3">
                  <div className={`w-3 h-3 rounded-full ${handDetected ? 'bg-green-500 animate-pulse' : 'bg-surface-600'}`} />
                  <span className="text-sm text-surface-300">
                    {handDetected ? 'Hand Detected' : 'No Hand Detected'}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-surface-400">
                  <span className="font-mono">{fps} FPS</span>
                  {isModelLoaded && (
                    <span className="flex items-center gap-1 text-green-400">
                      <span className="w-2 h-2 bg-green-400 rounded-full" />
                      Model Ready
                    </span>
                  )}
                </div>
              </div>

              {/* Camera View */}
              <div className="aspect-video bg-surface-900 relative">
                <Camera
                  onPrediction={handlePrediction}
                  onModelLoaded={() => {
                    setIsModelLoaded(true);
                    setIsLoading(false);
                  }}
                  showLandmarks={settings.showLandmarks}
                />
                
                {isLoading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-surface-900/80 backdrop-blur-sm">
                    <div className="text-center">
                      <div className="w-16 h-16 border-4 border-primary-500/30 border-t-primary-500 rounded-full animate-spin mx-auto mb-4" />
                      <p className="text-surface-300">Loading AI Model...</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Current Prediction */}
              <div className="p-4 border-t border-surface-700/30">
                <PredictionDisplay 
                  letter={currentLetter} 
                  confidence={confidence}
                  threshold={settings.confidenceThreshold}
                />
              </div>
            </div>
          </div>

          {/* Text Output Section */}
          <div className="lg:col-span-1">
            <div className="glass-card h-full flex flex-col animate-slide-up animate-delay-100">
              {/* Output Header */}
              <div className="flex items-center justify-between p-4 border-b border-surface-700/30">
                <h2 className="font-semibold text-white flex items-center gap-2">
                  <Type className="w-4 h-4 text-primary-400" />
                  Output
                </h2>
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleDelete}
                    className="p-2 hover:bg-surface-700/50 rounded-lg transition-colors"
                    title="Delete last character"
                  >
                    <Trash2 className="w-4 h-4 text-surface-400" />
                  </button>
                  <button
                    onClick={handleClear}
                    className="p-2 hover:bg-surface-700/50 rounded-lg transition-colors"
                    title="Clear all"
                  >
                    <RotateCcw className="w-4 h-4 text-surface-400" />
                  </button>
                </div>
              </div>

              {/* Text Display */}
              <div className="flex-1 p-4 min-h-[200px]">
                <div className="text-2xl font-display font-medium text-white leading-relaxed break-words">
                  {recognizedText || (
                    <span className="text-surface-500 italic">
                      Start signing to see text appear here...
                    </span>
                  )}
                  <span className="inline-block w-0.5 h-6 bg-primary-500 animate-pulse ml-1 align-middle" />
                </div>
              </div>

              {/* Word Suggestions */}
              {settings.wordCompletion && suggestions.length > 0 && (
                <div className="p-4 border-t border-surface-700/30">
                  <p className="text-xs text-surface-500 mb-2">Suggestions</p>
                  <div className="flex flex-wrap gap-2">
                    {suggestions.map((word) => (
                      <button
                        key={word}
                        onClick={() => handleSuggestionClick(word)}
                        className="suggestion-pill"
                      >
                        {word}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Quick Actions */}
              <div className="p-4 border-t border-surface-700/30">
                <div className="flex gap-2">
                  <button
                    onClick={handleSpace}
                    className="flex-1 btn-secondary text-sm"
                  >
                    Space
                  </button>
                  <button
                    onClick={handleDelete}
                    className="flex-1 btn-secondary text-sm"
                  >
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ASL Reference */}
        <div className="mt-8 glass-card p-6 animate-fade-in animate-delay-300">
          <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
            <Hand className="w-4 h-4 text-primary-400" />
            ASL Alphabet Reference
          </h3>
          <div className="grid grid-cols-9 sm:grid-cols-13 md:grid-cols-14 gap-2">
            {[..."ABCDEFGHIJKLMNOPQRSTUVWXYZ", "DEL"].map((letter) => (
              <div
                key={letter}
                className={`aspect-square flex items-center justify-center rounded-lg text-sm font-bold transition-all duration-200 ${
                  currentLetter === (letter === "DEL" ? "DELETE" : letter)
                    ? "bg-primary-500 text-white scale-110 shadow-lg shadow-primary-500/30"
                    : "bg-surface-800/50 text-surface-300 hover:bg-surface-700/50"
                }`}
              >
                {letter}
              </div>
            ))}
          </div>
          <p className="text-xs text-surface-500 mt-4">
            * Letters J and Z require motion gestures. Make a fist for DELETE.
          </p>
        </div>

        {/* Footer */}
        <footer className="mt-8 text-center text-sm text-surface-500 animate-fade-in animate-delay-400">
          <p>
            Built with TensorFlow.js and MediaPipe • 
            <a href="https://github.com" className="text-primary-400 hover:underline ml-1">
              View Source
            </a>
          </p>
        </footer>
      </div>
    </main>
  );
}



