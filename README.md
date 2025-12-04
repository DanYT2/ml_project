# ASL Sign Language Recognition System

A real-time American Sign Language (ASL) alphabet recognition system using deep learning, with a web interface for live gesture-to-text conversion.

## 🎯 Features

- **Real-time ASL alphabet recognition** via webcam
- **Static gesture recognition** (A-Y, excluding J and Z)
- **Dynamic gesture recognition** (J, Z with motion tracking)
- **Word completion** with toggleable autocomplete suggestions
- **Delete gesture** for corrections
- **Model explainability** via Grad-CAM visualization
- **In-browser inference** using TensorFlow.js for <200ms latency

## 📁 Project Structure

```
ml_project/
├── README.md
├── requirements.txt
├── notebooks/                    # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── src/                          # Python ML pipeline
│   ├── data/                     # Data loading & preprocessing
│   ├── models/                   # Model architectures
│   ├── training/                 # Training scripts
│   ├── evaluation/               # Metrics & explainability
│   └── utils/                    # Helper functions
├── scripts/                      # CLI scripts
│   ├── download_dataset.py
│   ├── train_model.py
│   └── export_model.py
├── web/                          # Next.js web application
│   ├── app/                      # App router pages
│   ├── components/               # React components
│   ├── lib/                      # Utility functions
│   └── public/model/             # Exported TF.js model
└── models/                       # Saved model checkpoints
```

## 🚀 Quick Start

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Download ASL Alphabet dataset from Kaggle
# Requires Kaggle API credentials (~/.kaggle/kaggle.json)
python scripts/download_dataset.py
```

### 3. Train Model

```bash
# Train the model
python scripts/train_model.py --epochs 50 --batch-size 32

# Or use the Jupyter notebooks for interactive training
jupyter notebook notebooks/02_model_training.ipynb
```

### 4. Export Model for Web

```bash
# Convert to TensorFlow.js format
python scripts/export_model.py --output web/public/model
```

### 5. Run Web Application

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🧠 Model Architecture

### Hybrid CNN-LSTM Architecture

The model uses a hybrid approach to handle both static and dynamic gestures:

1. **MediaPipe Hand Landmarks**: Extracts 21 3D hand landmarks (63 features)
2. **Convolutional Layers**: Process spatial relationships between landmarks
3. **LSTM Layer**: Captures temporal patterns for dynamic gestures (J, Z)
4. **Dense Layers**: Final classification into 26 letters + delete gesture

```
Input (sequence of landmarks) → Conv1D → Conv1D → LSTM → Dense → Softmax
```

### Data Augmentation

- Random rotation (±15°)
- Random scaling (0.9-1.1x)
- Random translation
- Landmark noise injection
- Temporal augmentation for dynamic gestures

## 📊 Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| Accuracy | >90% | TBD |
| Latency | <200ms | TBD |
| FPS | >30 | TBD |

## 🔧 Configuration

### Training Configuration (`config/training.yaml`)

```yaml
model:
  sequence_length: 30
  num_landmarks: 21
  num_classes: 27  # 26 letters + delete

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10
```

### Web Configuration

Environment variables in `.env.local`:

```
NEXT_PUBLIC_MODEL_PATH=/model/model.json
NEXT_PUBLIC_CONFIDENCE_THRESHOLD=0.8
```

## 🎨 UI Features

- **Minimalist design** with dark/light mode
- **Real-time webcam feed** with hand landmark overlay
- **Letter display** showing recognized characters
- **Word suggestions** with toggleable autocomplete
- **Confidence indicator** for predictions
- **Settings panel** for customization

## 📚 ASL Alphabet Reference

| Letter | Type | Notes |
|--------|------|-------|
| A-I | Static | Single hand pose |
| J | Dynamic | Traces J shape in air |
| K-Y | Static | Single hand pose |
| Z | Dynamic | Traces Z shape in air |
| Delete | Static | Custom gesture (fist) |

## 🔬 Model Explainability

Grad-CAM visualization shows which hand landmarks most influence predictions:

```python
from src.evaluation.explainability import GradCAM

gradcam = GradCAM(model)
heatmap = gradcam.generate_heatmap(input_landmarks, predicted_class)
```

## 🚢 Deployment

### Deploy to Vercel

```bash
cd web
vercel --prod
```

Or connect your GitHub repository to Vercel for automatic deployments.

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

